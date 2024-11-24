class MillionTreesDataset:
    """Shared dataset class for all MillionTrees datasets."""
    
    def check_init(self):
        """Check dataset configuration."""
        required_attrs = [
            '_dataset_name', '_data_dir', '_split_scheme', '_split_array',
            '_y_array', '_y_size', '_metadata_fields', '_metadata_array'
        ]
        for attr_name in required_attrs:
            assert hasattr(self, attr_name), f'MillionTreesDataset missing {attr_name}.'

        # Check that data directory exists
        if not os.path.exists(self.data_dir):
            raise ValueError(f'{self.data_dir} does not exist yet.')

        # Check splits
        assert self.split_dict.keys() == self.split_names.keys()
        
        # Check arrays
        assert isinstance(self.y_array, (np.ndarray, list))
        assert isinstance(self.metadata_array, np.ndarray)
        assert len(self._input_array) == len(self.metadata_array)
        
        # Check metadata
        assert len(self.metadata_array.shape) == 2
        assert len(self.metadata_fields) == self.metadata_array.shape[1]
