from ..base.labeled import LabeledDataset
from PIL import Image
import numpy as np
import torch
from milliontrees.common.metrics.all_metrics import DetectionAccuracy

class TreeBoxesDataset(LabeledDataset):
    """TreeBoxes labeled dataset."""
    
    _dataset_name = "TreeBoxes"
    _versions_dict = {
        "1.0": {
            "download_url": "",
            "compressed_size": 0,
        }
    }
    
    def _process_labels(self, df: pd.DataFrame) -> None:
        self._y_array = df[["xmin", "ymin", "xmax", "ymax"]].values.astype("float32")
        self._n_classes = 1
        self._y_size = 4
        self.labels = np.zeros(df.shape[0])
        self._metric = DetectionAccuracy()
        self._collate = TreeBoxesDataset._collate_fn
    
    def __getitem__(self, idx):
        """Get a dataset item."""
        x = self.get_input(idx)
        metadata = self._metadata_array[idx]
        y_indices = self._input_lookup[self._input_array[idx]]
        y = self._y_array[y_indices]
        labels = self.labels[y_indices]
        return metadata, x, {"boxes": y, "labels": labels}
        
    def get_input(self, idx):
        """Get input image."""
        img_path = self.data_dir / "images" / self._input_array[idx]
        return Image.open(img_path)
        
    @staticmethod
    def _collate_fn(batch):
        """Custom collate function for detection."""
        images = [item[1] for item in batch]
        targets = [item[2] for item in batch]
        metadata = torch.stack([item[0] for item in batch])
        return metadata, images, targets
