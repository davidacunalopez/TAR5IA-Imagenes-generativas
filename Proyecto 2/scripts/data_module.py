"""
DataModule para carga de datos MVTec AD
Incluye: AnomalyDataset, load_dataset_paths, MVTecDataModule
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl


class AnomalyDataset(Dataset):
    """Dataset para cargar imágenes de entrenamiento y prueba"""

    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            return image, self.labels[idx]
        return image


def load_dataset_paths(category_path, split='train', only_good=True):
    """Carga las rutas de las imágenes del dataset"""
    paths = []
    labels = []
    split_path = os.path.join(category_path, split)

    if split == 'train' and only_good:
        # Solo imágenes 'good' en entrenamiento
        good_path = os.path.join(split_path, 'good')
        if os.path.exists(good_path):
            for img_file in os.listdir(good_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    paths.append(os.path.join(good_path, img_file))
                    labels.append(0)  # Se actualizará con el índice de categoría
    else:
        # En test, cargar todas las clases (good y anomalías)
        if os.path.exists(split_path):
            for class_name in os.listdir(split_path):
                class_path = os.path.join(split_path, class_name)
                if os.path.isdir(class_path):
                    for img_file in os.listdir(class_path):
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            paths.append(os.path.join(class_path, img_file))
                            # Label: 0 para 'good', 1 para anomalías
                            labels.append(0 if class_name == 'good' else 1)

    return paths, labels


class MVTecDataModule(pl.LightningDataModule):
    """DataModule para MVTec AD con múltiples categorías"""

    def __init__(self, dataset_path, categories, image_size=128, batch_size=32,
                 num_workers=2, train_split=0.8):
        super().__init__()
        self.dataset_path = dataset_path
        self.categories = categories
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split

        # Transformaciones (normalización a [-1, 1] para compatibilidad con Tanh)
        self.train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.train_paths = []
        self.train_labels = []
        self.val_paths = []
        self.val_labels = []
        self.test_paths = []
        self.test_labels = []

    def setup(self, stage=None):
        """Carga las rutas de las imágenes para todas las categorías"""
        # Validar que el dataset_path existe
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"❌ ERROR: No se encontró el dataset en: {self.dataset_path}\n"
                f"   Por favor, verifica la ruta del dataset."
            )

        all_train_paths = []
        all_train_labels = []
        all_test_paths = []
        all_test_labels = []

        # Cargar datos de todas las categorías
        for cat_idx, category in enumerate(self.categories):
            category_path = os.path.join(self.dataset_path, category)

            # Validar que la categoría existe
            if not os.path.exists(category_path):
                raise FileNotFoundError(
                    f"❌ ERROR: Categoría '{category}' no encontrada en: {category_path}"
                )

            # Entrenamiento (solo 'good')
            train_paths, _ = load_dataset_paths(category_path, split='train', only_good=True)
            if len(train_paths) == 0:
                raise ValueError(
                    f"❌ ERROR: No se encontraron imágenes de entrenamiento para la categoría '{category}'\n"
                    f"   Ruta esperada: {os.path.join(category_path, 'train', 'good')}"
                )
            # Asignar label de categoría
            train_labels = [cat_idx] * len(train_paths)
            all_train_paths.extend(train_paths)
            all_train_labels.extend(train_labels)

            # Prueba (todas las clases)
            test_paths, test_labels = load_dataset_paths(category_path, split='test', only_good=False)
            if len(test_paths) == 0:
                raise ValueError(
                    f"❌ ERROR: No se encontraron imágenes de prueba para la categoría '{category}'\n"
                    f"   Ruta esperada: {os.path.join(category_path, 'test')}"
                )
            all_test_paths.extend(test_paths)
            all_test_labels.extend(test_labels)

        # Dividir entrenamiento en train y validation
        total_train = len(all_train_paths)
        train_size = int(self.train_split * total_train)
        val_size = total_train - train_size

        indices = torch.randperm(total_train).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        self.train_paths = [all_train_paths[i] for i in train_indices]
        self.train_labels = [all_train_labels[i] for i in train_indices]
        self.val_paths = [all_train_paths[i] for i in val_indices]
        self.val_labels = [all_train_labels[i] for i in val_indices]

        self.test_paths = all_test_paths
        self.test_labels = all_test_labels

        # Validación final: verificar que hay datos
        if len(self.train_paths) == 0:
            raise ValueError("❌ ERROR: No se encontraron imágenes de entrenamiento")
        if len(self.val_paths) == 0:
            raise ValueError("❌ ERROR: No se encontraron imágenes de validación")
        if len(self.test_paths) == 0:
            raise ValueError("❌ ERROR: No se encontraron imágenes de prueba")

        print(f"Train: {len(self.train_paths)} imágenes")
        print(f"Validation: {len(self.val_paths)} imágenes")
        print(f"Test: {len(self.test_paths)} imágenes")

    def train_dataloader(self):
        dataset = AnomalyDataset(self.train_paths, labels=self.train_labels, transform=self.train_transform)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        dataset = AnomalyDataset(self.val_paths, labels=self.val_labels, transform=self.val_transform)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        dataset = AnomalyDataset(self.test_paths, labels=self.test_labels, transform=self.test_transform)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

