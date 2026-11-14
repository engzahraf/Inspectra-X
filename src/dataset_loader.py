"""
Dataset Loader Module for Severstal Steel Defect Detection Dataset.

Parses train.csv annotations, aggregates multi-label defect occurrences per image,
and resolves image paths from the raw image repository.
"""

import os
from typing import Dict, List, Optional, Tuple
import pandas as pd


CLASS_ID_TO_NAME = {
    1: 'Crazing',
    2: 'Inclusion',
    3: 'Patches',
    4: 'Pitted_Surface'
}

TARGET_COLUMNS = ['Crazing', 'Inclusion', 'Patches', 'Pitted_Surface']


class SteelDatasetLoader:
    """
    Manages loading, indexing, and multi-label aggregation of the Severstal steel defect dataset.
    """

    def __init__(self, dataset_dir: str):
        """
        Initializes the dataset loader.

        Args:
            dataset_dir: Path to directory containing train.csv and train_images/.
        """
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.csv_path = self._find_file('train.csv')
        self.images_dir = self._find_dir('train_images')

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"train.csv not found in {self.dataset_dir}")
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"train_images directory not found in {self.dataset_dir}")

        self.records: Dict[str, dict] = {}
        self.image_list: List[str] = []
        self._load_and_aggregate()

    def _find_file(self, filename: str) -> str:
        """Finds a file case-insensitively within dataset_dir."""
        for item in os.listdir(self.dataset_dir):
            if item.lower() == filename.lower():
                return os.path.join(self.dataset_dir, item)
        return os.path.join(self.dataset_dir, filename)

    def _find_dir(self, dirname: str) -> str:
        """Finds a subfolder case-insensitively within dataset_dir."""
        for item in os.listdir(self.dataset_dir):
            if item.lower() == dirname.lower() and os.path.isdir(os.path.join(self.dataset_dir, item)):
                return os.path.join(self.dataset_dir, item)
        return os.path.join(self.dataset_dir, dirname)

    def _load_and_aggregate(self) -> None:
        """
        Parses train.csv and matches with all image files in train_images.
        Supports multi-label aggregation where an image may contain multiple defect classes.
        """
        df = pd.read_csv(self.csv_path)

        # Normalize column names if needed
        # Format can be ImageId, ClassId, EncodedPixels or ImageId_ClassId
        if 'ImageId_ClassId' in df.columns:
            df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
            df['ClassId'] = df['ImageId_ClassId'].apply(lambda x: int(x.split('_')[1]))

        # Group RLE by ImageId and ClassId
        csv_grouped: Dict[str, Dict[int, str]] = {}
        for _, row in df.iterrows():
            img_id = str(row['ImageId']).strip()
            class_id = int(row['ClassId'])
            rle = str(row['EncodedPixels']).strip() if pd.notna(row['EncodedPixels']) else ''

            if rle and rle.lower() != 'nan':
                if img_id not in csv_grouped:
                    csv_grouped[img_id] = {}
                csv_grouped[img_id][class_id] = rle

        # Discover all image files physically present on disk
        all_disk_images = sorted([
            f for f in os.listdir(self.images_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        self.image_list = all_disk_images

        # Aggregate records for each image
        for img_id in self.image_list:
            img_path = os.path.join(self.images_dir, img_id)
            rle_map = csv_grouped.get(img_id, {})

            # Build multi-label binary indicators
            labels = {
                'Crazing': 1 if 1 in rle_map else 0,
                'Inclusion': 1 if 2 in rle_map else 0,
                'Patches': 1 if 3 in rle_map else 0,
                'Pitted_Surface': 1 if 4 in rle_map else 0
            }

            self.records[img_id] = {
                'image_id': img_id,
                'image_path': img_path,
                'labels': labels,
                'rle_map': rle_map,
                'has_defect': 1 if len(rle_map) > 0 else 0,
                'defect_count': len(rle_map)
            }

    def __len__(self) -> int:
        return len(self.image_list)

    def get_image_info(self, image_id: str) -> Optional[dict]:
        """Retrieves image metadata, labels, and RLE mappings by ImageId."""
        return self.records.get(image_id)

    def get_defect_free_image_ids(self) -> List[str]:
        """Returns list of all defect-free ImageIds."""
        return [
            img_id for img_id, rec in self.records.items()
            if rec['has_defect'] == 0
        ]

    def get_summary_dataframe(self) -> pd.DataFrame:
        """
        Returns a DataFrame summarizing image IDs and their multi-label target statuses.
        """
        rows = []
        for img_id in self.image_list:
            rec = self.records[img_id]
            rows.append({
                'ImageId': img_id,
                'Crazing': rec['labels']['Crazing'],
                'Inclusion': rec['labels']['Inclusion'],
                'Patches': rec['labels']['Patches'],
                'Pitted_Surface': rec['labels']['Pitted_Surface'],
                'Defect_Count': rec['defect_count'],
                'Has_Defect': rec['has_defect']
            })
        return pd.DataFrame(rows)

