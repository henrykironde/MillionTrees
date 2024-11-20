from ..base.unlabeled import UnlabeledDataset
from PIL import Image

class TreePolygonsUnlabeledDataset(UnlabeledDataset):
    """TreePolygons unlabeled dataset."""
    
    _dataset_name = "TreePolygons_unlabeled"
    _versions_dict = {
        "1.0": {
            "download_url": "",
            "compressed_size": 0,
        }
    }
    
    def __getitem__(self, idx):
        """Get a dataset item."""
        x = self.get_input(idx)
        metadata = self._metadata_array[idx]
        return metadata, x
        
    def get_input(self, idx):
        """Get input image."""
        img_path = self.data_dir / "images" / self._input_array[idx]
        return Image.open(img_path)
