from ..base.unlabeled import UnlabeledDataset
from PIL import Image

class TreeBoxesUnlabeledDataset(UnlabeledDataset):
    """
    The unlabeled iWildCam2020-milliontrees dataset.
    This is a modified version of the original iWildCam2020 competition dataset.
    Input (x):
        RGB images from camera traps
    Metadata:
        Each image is annotated with the ID of the location (camera trap) it came from.
    Website:
        http://lila.science/datasets/wcscameratraps
        https://library.wcs.org/ScienceData/Camera-Trap-Data-Summary.aspx
    Original publication:
        @misc{wcsdataset,
          title = {Wildlife Conservation Society Camera Traps Dataset},
          howpublished = {\url{http://lila.science/datasets/wcscameratraps}},
        }
    License:
        This dataset is distributed under Community Data License Agreement – Permissive – Version 1.0
        https://cdla.io/permissive-1-0/
    """

    _dataset_name = "TreeBoxes_unlabeled"
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
