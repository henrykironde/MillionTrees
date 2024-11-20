from ..base.labeled import LabeledDataset
from PIL import Image
import numpy as np

class TreePointsDataset(LabeledDataset):
    """TreePoints labeled dataset."""
    
    _dataset_name = "TreePoints"
    _versions_dict = {
        "1.0": {
            "download_url": "",
            "compressed_size": 0,
        }
    }
    
    def _process_labels(self, df: pd.DataFrame) -> None:
        self._y_array = df[["x", "y"]].values.astype("float32")
        self._n_classes = 1
        self._y_size = 2
    
    def __getitem__(self, idx):
        """Get a dataset item."""
        x = self.get_input(idx)
        metadata = self._metadata_array[idx]
        y_indices = self._input_lookup[self._input_array[idx]]
        y = self._y_array[y_indices]
        return metadata, x, y
        
    def get_input(self, idx):
        """Get input image."""
        img_path = self.data_dir / "images" / self._input_array[idx]
        return Image.open(img_path)
