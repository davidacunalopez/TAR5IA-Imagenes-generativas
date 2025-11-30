"""
Funciones de evaluación, cuantización y análisis DBSCAN
Incluye: extract_embeddings, evaluate_anomaly_detection, quantize_model, dbscan_analysis
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.linalg import inv
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def calculate_mahalanobis_distance(embeddings, mean, cov):
    """
    Calcula la distancia de Mahalanobis para cada embedding.

    Distancia de Mahalanobis: d = sqrt((z - μ)^T Σ^(-1) (z - μ))

    Args:
        embeddings: Array numpy de shape (N, d) con embeddings
        mean: Vector media de shape (d,)
        cov: Matriz de covarianza de shape (d, d)

    Returns:
        distances: Array numpy de shape (N,) con distancias de Mahalanobis
    """
    if embeddings is None or len(embeddings) == 0:
        raise ValueError("❌ ERROR: embeddings no puede estar vacío")
    if mean is None:
        raise ValueError("❌ ERROR: mean no puede ser None")
    if cov is None:
        raise ValueError("❌ ERROR: cov no puede ser None")

    try:
        # Regularización para evitar singularidad
        cov_reg = cov + np.eye(cov.shape[0]) * 1e-6

        # Calcular inversa de la matriz de covarianza
        try:
            cov_inv = inv(cov_reg)
        except np.linalg.LinAlgError as e:
            raise ValueError(f"❌ ERROR: No se pudo invertir la matriz de covarianza: {e}") from e

        # Calcular distancias
        distances = []
        for emb in embeddings:
            diff = emb - mean
            try:
                dist = np.sqrt(diff @ cov_inv @ diff.T)
                if np.isnan(dist) or np.isinf(dist):
                    print(f"  ⚠️ Advertencia: Distancia inválida detectada, usando 0")
                    dist = 0.0
                distances.append(dist)
            except Exception as e:
                print(f"  ⚠️ Advertencia: Error calculando distancia: {e}, usando 0")
                distances.append(0.0)

        return np.array(distances)

    except Exception as e:
        raise RuntimeError(f"❌ ERROR al calcular distancias de Mahalanobis: {e}") from e


def extract_embeddings(model, dataloader, device):
    """
    Extrae embeddings de un dataloader usando el modelo entrenado

    Args:
        model: Modelo entrenado
        dataloader: DataLoader con las imágenes
        device: Dispositivo (cuda/cpu)

    Returns:
        all_embeddings: Array numpy con todos los embeddings
        all_labels: Array numpy con todas las etiquetas (o None)
        all_reconstructions: Lista de reconstrucciones (para autoencoders)
        all_originals: Lista de imágenes originales (para autoencoders)
    """
    if model is None:
        raise ValueError("❌ ERROR: El modelo no puede ser None")

    if dataloader is None:
        raise ValueError("❌ ERROR: El dataloader no puede ser None")

    model.eval()
    all_embeddings = []
    all_labels = []
    all_reconstructions = []
    all_originals = []

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                try:
                    if isinstance(batch, tuple):
                        images, labels = batch
                    else:
                        images = batch
                        labels = None

                    if images is None or images.numel() == 0:
                        print(f"  ⚠️ Advertencia: Batch {batch_idx} está vacío, saltando...")
                        continue

                    images = images.to(device)

                    # Extraer embeddings
                    try:
                        if hasattr(model, 'get_embedding'):
                            embeddings = model.get_embedding(images)
                        elif hasattr(model, 'model') and hasattr(model.model, 'get_embedding'):
                            embeddings = model.model.get_embedding(images)
                        else:
                            if hasattr(model, 'model'):
                                logits, embeddings = model.model(images)
                            else:
                                logits, embeddings = model(images)
                    except Exception as e:
                        print(f"  ❌ Error extrayendo embeddings del batch {batch_idx}: {e}")
                        raise

                    all_embeddings.append(embeddings.cpu().numpy())

                    if labels is not None:
                        all_labels.append(labels.cpu().numpy())

                    # Para autoencoders, guardar reconstrucciones
                    if hasattr(model, 'model') and hasattr(model.model, 'forward'):
                        try:
                            reconstructions = model.model(images)
                            all_reconstructions.append(reconstructions.cpu().numpy())
                            all_originals.append(images.cpu().numpy())
                        except Exception as e:
                            # Si falla la reconstrucción, continuar sin ella
                            pass

                except Exception as e:
                    print(f"  ⚠️ Error procesando batch {batch_idx}: {e}")
                    continue

        if len(all_embeddings) == 0:
            raise ValueError("❌ ERROR: No se pudieron extraer embeddings. El dataloader podría estar vacío.")

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_labels = np.concatenate(all_labels, axis=0) if all_labels else None

        return all_embeddings, all_labels, all_reconstructions, all_originals

    except Exception as e:
        raise RuntimeError(f"❌ ERROR al extraer embeddings: {e}") from e


def estimate_normal_distribution(normal_embeddings):
    """
    Estima la distribución normal (gaussiana multivariada) a partir de embeddings normales.

    Calcula la media μ y la matriz de covarianza Σ:
    μ = (1/N) Σ z_i
    Σ = (1/(N-1)) Σ (z_i - μ)(z_i - μ)^T

    Args:
        normal_embeddings: Array numpy de shape (N, d) con embeddings normales

    Returns:
        mean: Vector media de shape (d,)
        cov: Matriz de covarianza de shape (d, d)
    """
    if normal_embeddings is None or len(normal_embeddings) == 0:
        raise ValueError("❌ ERROR: normal_embeddings no puede estar vacío")

    if len(normal_embeddings.shape) != 2:
        raise ValueError(f"❌ ERROR: normal_embeddings debe ser 2D, pero tiene shape {normal_embeddings.shape}")

    if len(normal_embeddings) < 2:
        raise ValueError(f"❌ ERROR: Se necesitan al menos 2 muestras para calcular covarianza, pero hay {len(normal_embeddings)}")

    try:
        # Media: μ = (1/N) Σ z_i
        mean = np.mean(normal_embeddings, axis=0)

        # Matriz de covarianza: Σ = (1/(N-1)) Σ (z_i - μ)(z_i - μ)^T
        # np.cov usa (N-1) como denominador por defecto
        cov = np.cov(normal_embeddings.T)

        # Validar que la matriz de covarianza es válida
        if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
            raise ValueError("❌ ERROR: La matriz de covarianza contiene NaN o Inf")

        return mean, cov

    except Exception as e:
        raise RuntimeError(f"❌ ERROR al estimar distribución normal: {e}") from e


def evaluate_anomaly_detection(model, normal_dataloader, test_dataloader, device, method="mahalanobis", percentile=95):
    """
    Evalúa la detección de anomalías siguiendo el proceso correcto:
    1. Estima la distribución normal usando el conjunto de validación/entrenamiento (solo datos normales)
    2. Aplica esa distribución al conjunto de prueba para detectar anomalías

    Args:
        model: Modelo entrenado
        normal_dataloader: Dataloader con datos normales (validación o entrenamiento)
        test_dataloader: Dataloader con datos de prueba (normales y anómalos)
        device: Dispositivo (cuda/cpu)
        method: Método de evaluación ("mahalanobis", "euclidean", "reconstruction_loss")
        percentile: Percentil para determinar el umbral (default: 95)

    Returns:
        results: Diccionario con resultados de la evaluación
    """
    # Validación de parámetros
    if model is None:
        raise ValueError("❌ ERROR: El modelo no puede ser None")
    if normal_dataloader is None:
        raise ValueError("❌ ERROR: normal_dataloader no puede ser None")
    if test_dataloader is None:
        raise ValueError("❌ ERROR: test_dataloader no puede ser None")
    if method not in ["mahalanobis", "euclidean", "reconstruction_loss"]:
        raise ValueError(f"❌ ERROR: Método '{method}' no reconocido. Use: 'mahalanobis', 'euclidean', o 'reconstruction_loss'")
    if not (0 < percentile <= 100):
        raise ValueError(f"❌ ERROR: Percentil debe estar entre 0 y 100, pero es {percentile}")

    try:
        # Paso 1: Extraer embeddings del conjunto normal (validación/entrenamiento)
        print("📊 Estimando distribución normal a partir del conjunto de validación/entrenamiento...")
        normal_embeddings, _, normal_reconstructions, normal_originals = extract_embeddings(
            model, normal_dataloader, device
        )

        if len(normal_embeddings) == 0:
            raise ValueError("❌ ERROR: No se pudieron extraer embeddings del conjunto normal")

        # Estimar distribución normal: μ y Σ
        mean, cov = estimate_normal_distribution(normal_embeddings)
        print(f"  ✓ Media (μ) calculada: shape {mean.shape}")
        print(f"  ✓ Matriz de covarianza (Σ) calculada: shape {cov.shape}")

        # Paso 2: Extraer embeddings del conjunto de prueba
        print("\n📊 Extrayendo embeddings del conjunto de prueba...")
        test_embeddings, test_labels, test_reconstructions, test_originals = extract_embeddings(
            model, test_dataloader, device
        )

        if len(test_embeddings) == 0:
            raise ValueError("❌ ERROR: No se pudieron extraer embeddings del conjunto de prueba")

        # Separar embeddings normales y anómalos del conjunto de prueba
        if test_labels is not None:
            test_normal_embeddings = test_embeddings[test_labels == 0]
            test_anomaly_embeddings = test_embeddings[test_labels == 1]

            # Calcular distancias usando la distribución normal estimada
            try:
                if method == "mahalanobis":
                    if len(test_normal_embeddings) > 0:
                        test_normal_distances = calculate_mahalanobis_distance(test_normal_embeddings, mean, cov)
                    else:
                        test_normal_distances = np.array([])

                    if len(test_anomaly_embeddings) > 0:
                        test_anomaly_distances = calculate_mahalanobis_distance(test_anomaly_embeddings, mean, cov)
                    else:
                        test_anomaly_distances = np.array([])

                elif method == "euclidean":
                    if len(test_normal_embeddings) > 0:
                        test_normal_distances = np.linalg.norm(test_normal_embeddings - mean, axis=1)
                    else:
                        test_normal_distances = np.array([])

                    if len(test_anomaly_embeddings) > 0:
                        test_anomaly_distances = np.linalg.norm(test_anomaly_embeddings - mean, axis=1)
                    else:
                        test_anomaly_distances = np.array([])

                elif method == "reconstruction_loss":
                    if len(test_reconstructions) > 0 and len(test_originals) > 0:
                        test_reconstructions = np.concatenate(test_reconstructions, axis=0)
                        test_originals = np.concatenate(test_originals, axis=0)
                        test_normal_recon = test_reconstructions[test_labels == 0]
                        test_normal_orig = test_originals[test_labels == 0]
                        test_anomaly_recon = test_reconstructions[test_labels == 1]
                        test_anomaly_orig = test_originals[test_labels == 1]

                        if len(test_normal_recon) > 0:
                            test_normal_distances = np.mean((test_normal_recon - test_normal_orig) ** 2, axis=(1, 2, 3))
                        else:
                            test_normal_distances = np.array([])

                        if len(test_anomaly_recon) > 0:
                            test_anomaly_distances = np.mean((test_anomaly_recon - test_anomaly_orig) ** 2, axis=(1, 2, 3))
                        else:
                            test_anomaly_distances = np.array([])
                    else:
                        raise ValueError("❌ ERROR: Reconstruction loss requiere reconstrucciones. Asegúrate de usar un modelo autoencoder.")
            except Exception as e:
                raise RuntimeError(f"❌ ERROR al calcular distancias con método '{method}': {e}") from e

            # Determinar umbral usando percentil de las distancias normales del conjunto de validación
            try:
                if method == "mahalanobis":
                    validation_normal_distances = calculate_mahalanobis_distance(normal_embeddings, mean, cov)
                elif method == "euclidean":
                    validation_normal_distances = np.linalg.norm(normal_embeddings - mean, axis=1)
                elif method == "reconstruction_loss":
                    if len(normal_reconstructions) > 0 and len(normal_originals) > 0:
                        normal_reconstructions = np.concatenate(normal_reconstructions, axis=0)
                        normal_originals = np.concatenate(normal_originals, axis=0)
                        validation_normal_distances = np.mean((normal_reconstructions - normal_originals) ** 2, axis=(1, 2, 3))
                    else:
                        raise ValueError("❌ ERROR: Reconstruction loss requiere reconstrucciones del conjunto normal")

                if len(validation_normal_distances) == 0:
                    raise ValueError("❌ ERROR: No se pudieron calcular distancias de validación")

                # Umbral basado en percentil de distancias normales de validación
                threshold = np.percentile(validation_normal_distances, percentile)
                print(f"\n📏 Umbral calculado (percentil {percentile}): {threshold:.4f}")

            except Exception as e:
                raise RuntimeError(f"❌ ERROR al calcular umbral: {e}") from e

            # Clasificar datos de prueba
            if len(test_normal_distances) == 0 and len(test_anomaly_distances) == 0:
                raise ValueError("❌ ERROR: No hay distancias para clasificar")

            all_distances = np.concatenate([test_normal_distances, test_anomaly_distances])
            predictions = (all_distances > threshold).astype(int)
            true_labels = np.concatenate([np.zeros_like(test_normal_distances), np.ones_like(test_anomaly_distances)])

            # Calcular métricas
            try:
                auc_roc = roc_auc_score(true_labels, all_distances)
                auc_pr = average_precision_score(true_labels, all_distances)
            except Exception as e:
                print(f"  ⚠️ Advertencia: Error al calcular métricas: {e}")
                auc_roc = 0.0
                auc_pr = 0.0

            results = {
                'method': method,
                'threshold': threshold,
                'auc_roc': auc_roc,
                'auc_pr': auc_pr,
                'normal_distances': test_normal_distances,
                'anomaly_distances': test_anomaly_distances,
                'all_distances': all_distances,
                'predictions': predictions,
                'true_labels': true_labels,
                'mean': mean,
                'cov': cov,
                'validation_normal_distances': validation_normal_distances
            }
        else:
            # Si no hay labels, solo retornar estadísticas
            mean = np.mean(normal_embeddings, axis=0)
            cov = np.cov(normal_embeddings.T)
            results = {
                'method': method,
                'mean': mean,
                'cov': cov,
                'normal_embeddings': normal_embeddings,
                'test_embeddings': test_embeddings
            }

        return results

    except Exception as e:
        error_msg = f"❌ ERROR en evaluate_anomaly_detection: {e}"
        print(error_msg)
        raise RuntimeError(error_msg) from e


def quantize_model(model, method="dynamic"):
    """
    Cuantiza un modelo PyTorch
    """
    model.eval()

    if method == "dynamic":
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
        )
    elif method == "static":
        # Para cuantización estática, necesitaríamos un dataset de calibración
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
        )
    else:
        raise ValueError(f"Método de cuantización no reconocido: {method}")

    return quantized_model


def compare_model_sizes(original_model, quantized_model):
    """Compara el tamaño de modelos original y cuantizado"""
    def get_model_size(model):
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return param_size + buffer_size

    original_size = get_model_size(original_model)
    quantized_size = get_model_size(quantized_model)

    return {
        'original_size_mb': original_size / (1024 * 1024),
        'quantized_size_mb': quantized_size / (1024 * 1024),
        'compression_ratio': original_size / quantized_size if quantized_size > 0 else 0
    }


def dbscan_analysis(embeddings, eps=0.5, min_samples=5, use_pca=True, pca_components=50,
                    use_tsne=True, tsne_components=2, tsne_perplexity=30):
    """
    Análisis DBSCAN para detección de outliers mediante clustering basado en densidad.

    Proceso:
    1. Reducción de dimensionalidad con PCA (opcional)
    2. Aplicación de DBSCAN para identificar clusters y outliers
    3. Reducción adicional con t-SNE para visualización 2D

    Args:
        embeddings: Array numpy de shape (N, d) con embeddings
        eps: Distancia máxima entre muestras para formar un cluster
        min_samples: Número mínimo de muestras para formar un cluster
        use_pca: Si usar PCA para reducción de dimensionalidad
        pca_components: Número de componentes PCA
        use_tsne: Si usar t-SNE para visualización 2D
        tsne_components: Dimensiones de salida de t-SNE (típicamente 2)
        tsne_perplexity: Perplejidad para t-SNE

    Returns:
        results: Diccionario con resultados del análisis
    """
    print(f"📊 Iniciando análisis DBSCAN...")
    print(f"  Embeddings originales: shape {embeddings.shape}")

    # Reducción de dimensionalidad con PCA
    if use_pca and embeddings.shape[1] > pca_components:
        print(f"  Aplicando PCA: {embeddings.shape[1]} → {pca_components} dimensiones")
        pca = PCA(n_components=pca_components)
        embeddings_reduced = pca.fit_transform(embeddings)
        explained_variance = np.sum(pca.explained_variance_ratio_)
        print(f"  ✓ Varianza explicada por PCA: {explained_variance:.4f} ({explained_variance*100:.2f}%)")
    else:
        embeddings_reduced = embeddings
        pca = None
        explained_variance = 1.0
        print(f"  ⚠️ PCA no aplicado (use_pca=False o dimensión ya es {embeddings.shape[1]})")

    # Aplicar DBSCAN
    print(f"\n  Aplicando DBSCAN (eps={eps}, min_samples={min_samples})...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(embeddings_reduced)

    # Identificar outliers (ruido)
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)
    n_in_clusters = len(clusters) - n_noise

    print(f"  ✓ DBSCAN completado:")
    print(f"    - Clusters encontrados: {n_clusters}")
    print(f"    - Puntos en clusters: {n_in_clusters} ({n_in_clusters/len(clusters)*100:.2f}%)")
    print(f"    - Outliers (ruido): {n_noise} ({n_noise/len(clusters)*100:.2f}%)")

    # Reducción para visualización con t-SNE
    embeddings_2d = None
    if use_tsne:
        print(f"\n  Aplicando t-SNE para visualización 2D...")
        perplexity = min(tsne_perplexity, len(embeddings_reduced) - 1)
        if perplexity > 0:
            tsne = TSNE(n_components=tsne_components, random_state=42, perplexity=perplexity)
            embeddings_2d = tsne.fit_transform(embeddings_reduced)
            print(f"  ✓ t-SNE completado: {embeddings_reduced.shape[1]} → {tsne_components} dimensiones")
        else:
            print(f"  ⚠️ Perplexity inválida ({perplexity}), saltando t-SNE")
    else:
        print(f"  ⚠️ t-SNE deshabilitado (use_tsne=False)")

    results = {
        'clusters': clusters,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'embeddings_reduced': embeddings_reduced,
        'embeddings_2d': embeddings_2d,
        'pca': pca,
        'explained_variance': explained_variance
    }

    return results


def visualize_dbscan_results(dbscan_results, labels=None, save_path=None):
    """
    Visualiza los resultados de DBSCAN de forma completa.

    Muestra:
    1. Clustering DBSCAN (clusters y outliers)
    2. Comparación con ground truth labels
    3. Análisis de distribución de outliers vs normales
    """
    clusters = dbscan_results['clusters']
    embeddings_2d = dbscan_results['embeddings_2d']

    if embeddings_2d is None:
        print("⚠️ No hay visualización 2D disponible (t-SNE no se aplicó)")
        return

    # Crear figura con múltiples subplots para análisis completo
    fig = plt.figure(figsize=(18, 6))
    axes = fig.subplots(1, 3)

    # Visualización por clusters
    unique_clusters = set(clusters)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_clusters)))

    for cluster, color in zip(unique_clusters, colors):
        if cluster == -1:
            # Ruido (outliers)
            mask = clusters == cluster
            axes[0].scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                          c='black', marker='x', s=50, label='Outliers', alpha=0.6)
        else:
            mask = clusters == cluster
            axes[0].scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                          c=[color], s=50, label=f'Cluster {cluster}', alpha=0.6)

    axes[0].set_title(f'DBSCAN Clustering (Clusters: {dbscan_results["n_clusters"]}, Outliers: {dbscan_results["n_noise"]})')
    axes[0].set_xlabel('t-SNE Component 1')
    axes[0].set_ylabel('t-SNE Component 2')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Visualización por labels (si están disponibles)
    if labels is not None:
        normal_mask = labels == 0
        anomaly_mask = labels == 1

        axes[1].scatter(embeddings_2d[normal_mask, 0], embeddings_2d[normal_mask, 1],
                      c='green', s=50, label='Normal', alpha=0.6)
        axes[1].scatter(embeddings_2d[anomaly_mask, 0], embeddings_2d[anomaly_mask, 1],
                      c='red', s=50, label='Anomaly', alpha=0.6)

        axes[1].set_title('Ground Truth Labels')
        axes[1].set_xlabel('t-SNE Component 1')
        axes[1].set_ylabel('t-SNE Component 2')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Visualización adicional: Outliers DBSCAN vs Ground Truth
        outlier_mask = clusters == -1
        in_cluster_mask = clusters != -1

        normal_outliers = (outlier_mask) & (normal_mask)
        anomaly_outliers = (outlier_mask) & (anomaly_mask)
        normal_in_cluster = (in_cluster_mask) & (normal_mask)
        anomaly_in_cluster = (in_cluster_mask) & (anomaly_mask)

        axes[2].scatter(embeddings_2d[normal_in_cluster, 0], embeddings_2d[normal_in_cluster, 1],
                       c='lightgreen', s=30, label='Normal (en cluster)', alpha=0.5, marker='o')
        axes[2].scatter(embeddings_2d[normal_outliers, 0], embeddings_2d[normal_outliers, 1],
                       c='green', s=150, label='Normal (outlier DBSCAN)', alpha=0.8, marker='x', linewidths=2)
        axes[2].scatter(embeddings_2d[anomaly_in_cluster, 0], embeddings_2d[anomaly_in_cluster, 1],
                       c='lightcoral', s=30, label='Anomalía (en cluster)', alpha=0.5, marker='o')
        axes[2].scatter(embeddings_2d[anomaly_outliers, 0], embeddings_2d[anomaly_outliers, 1],
                       c='red', s=150, label='Anomalía (outlier DBSCAN)', alpha=0.8, marker='x', linewidths=2)

        axes[2].set_title('DBSCAN Outliers vs Ground Truth')
        axes[2].set_xlabel('t-SNE Component 1')
        axes[2].set_ylabel('t-SNE Component 2')
        axes[2].legend(loc='best', fontsize=8)
        axes[2].grid(True, alpha=0.3)
    else:
        # Si no hay labels, solo mostrar clustering
        axes[1].axis('off')
        axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Visualización guardada en: {save_path}")

    plt.show()

