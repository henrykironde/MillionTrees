from pathlib import Path
import os
import numpy as np
import torch

class MillionTreesDataset:
    """Shared dataset class for all MillionTrees datasets.

    Each data point in the dataset is an (x, y, metadata) tuple, where:
    - x is the input features
    - y is the target
    - metadata is a vector of relevant information, e.g., domain.
      For convenience, metadata also contains y.
    """
    DEFAULT_SPLITS = {'train': 0, 'val': 1, 'test': 2}
    DEFAULT_SPLIT_NAMES = {
        'train': 'Train',
        'val': 'Validation',
        'test': 'Test'
    }
    DEFAULT_SOURCE_DOMAIN_SPLITS = [0]

    def __init__(self, root_dir, download, split_scheme):
        if len(self._metadata_array.shape) == 1:
            self._metadata_array = self._metadata_array.unsqueeze(1)
        self.check_init()

    def check_init(self):
        """Check that the dataset is properly configured."""
        required_attrs = [
            '_dataset_name', '_data_dir', '_split_scheme', '_split_array',
            '_y_array', '_y_size', '_metadata_fields', '_metadata_array'
        ]
        for attr_name in required_attrs:
            assert hasattr(self, attr_name), f'MillionTreesDataset missing {attr_name}.'

        # Check that data directory exists
        if not os.path.exists(self.data_dir):
            raise ValueError(f'{self.data_dir} does not exist yet.')

        # Check splits match
        assert self.split_dict.keys() == self.split_names.keys()
        
        # Check arrays
        assert isinstance(self.y_array, (np.ndarray, list))
        assert isinstance(self.metadata_array, np.ndarray)
        assert len(self._input_array) == len(self.metadata_array)
        
        # Check metadata
        assert len(self.metadata_array.shape) == 2
        assert len(self.metadata_fields) == self.metadata_array.shape[1]

        # Include y in metadata_fields if y_size == 1
        if self.y_size == 1:
            assert 'y' in self.metadata_fields

    def __len__(self):
        return len(self.y_array)

    def __getitem__(self, idx):
        x = self.get_input(idx)
        y_indices = self._input_lookup[self._input_array[idx]]
        y = self.y_array[y_indices]
        metadata = self.metadata_array[idx]
        targets = {"y": y, "labels": np.zeros(len(y), dtype=int)}
        return metadata, x, targets