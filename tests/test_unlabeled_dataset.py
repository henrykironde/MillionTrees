import unittest
import pandas as pd
from milliontrees.datasets.base.unlabeled import UnlabeledDataset

class TestUnlabeledDataset(unittest.TestCase):
    
    def setUp(self):
        self.dataset = UnlabeledDataset(root_dir='path/to/data', split_scheme='official')

    def test_process_labels_with_existing_column(self):
        df = pd.DataFrame({
            'y': [1, 0, 1, 0],
            'filename': ['file1', 'file2', 'file3', 'file4']
        })
        self.dataset._process_labels(df)
        self.assertTrue(hasattr(self.dataset, '_y_array'))
        self.assertEqual(self.dataset._y_array.tolist(), [1, 0, 1, 0])
        self.assertEqual(self.dataset._y_size, 1)

    def test_process_labels_with_missing_column(self):
        df = pd.DataFrame({
            'filename': ['file1', 'file2', 'file3', 'file4']
        })
        self.dataset._process_labels(df)
        self.assertTrue(hasattr(self.dataset, '_y_array'))
        self.assertEqual(self.dataset._y_array.tolist(), [0, 0, 0, 0])
        self.assertEqual(self.dataset._y_size, 1)

    def test_process_splits(self):
        df = pd.DataFrame({
            'filename': ['file1', 'file2', 'file3', 'file4']
        })
        self.dataset._process_splits(df)
        self.assertTrue(hasattr(self.dataset, '_split_array'))
        self.assertEqual(self.dataset._split_array.tolist(), [0, 0, 0, 0])  # Assuming split_dict is correct

    def test_process_groups(self):
        df = pd.DataFrame({
            'location': ['A', 'B', 'A', 'C'],
            'filename': ['file1', 'file2', 'file3', 'file4']
        })
        self.dataset._process_groups(df)
        self.assertTrue(hasattr(self.dataset, '_n_groups'))
        self.assertEqual(self.dataset._n_groups, 3)  # A, B, C should yield 3 groups

if __name__ == '__main__':
    unittest.main()
