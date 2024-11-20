from pathlib import Path
from typing import Union, Optional
import pandas as pd
import numpy as np
import torch
from .dataset import MillionTreesDataset
from ..utils import CombinatorialGrouper

class LabeledDataset(MillionTreesDataset):
    """Base class for labeled MillionTrees datasets."""
    
    _metadata_fields = ["source_id"]
    
    _split_dict = {
        'train': 0,
        'val': 1,
        'test': 2,
        'id_val': 3,
        'id_test': 4
    }
    
    _split_names = {
        'train': 'Train',
        'val': 'Validation (OOD/Trans)',
        'test': 'Test (OOD/Trans)',
        'id_val': 'Validation (ID/Cis)',
        'id_test': 'Test (ID/Cis)'
    }
    
    def __init__(
        self, 
        root_dir: Union[str, Path],
        version: Optional[str] = None,
        download: bool = False,
        split_scheme: str = "official"
    ) -> None:
        self._version = version
        self._split_scheme = split_scheme
        if self._split_scheme not in ['official', 'random']:
            raise ValueError(f'Split scheme {self._split_scheme} not recognized')
            
        self._data_dir = Path(self.initialize_data_dir(root_dir, download))
        self._setup_dataset()
        
        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=(['source_id'])
        )
        
        super().__init__(root_dir, download, split_scheme)
        
    def _setup_dataset(self) -> None:
        df = self._load_metadata()
        self._process_splits(df)
        self._process_filenames(df)
        self._process_groups(df)
        self._process_labels(df)
        self._create_metadata_array(df)
    
    def _load_metadata(self) -> pd.DataFrame:
        return pd.read_csv(self._data_dir / f'{self._split_scheme}.csv')
    
    def _process_splits(self, df: pd.DataFrame) -> None:
        unique_files = df.drop_duplicates(subset=['filename'], inplace=False).reset_index(drop=True)
        unique_files['split_id'] = unique_files['split'].apply(lambda x: self._split_dict[x])
        self._split_array = unique_files['split_id'].values
    
    def _process_filenames(self, df: pd.DataFrame) -> None:
        unique_files = df.drop_duplicates(subset=['filename'], inplace=False)
        self._input_array = unique_files.filename
        self._input_lookup = df.groupby('filename').apply(lambda x: x.index.values).to_dict()
    
    def _process_groups(self, df: pd.DataFrame) -> None:
        df["source_id"] = df.source.astype('category').cat.codes
        self._n_groups = max(df['source_id']) + 1
        assert len(np.unique(df['source_id'])) == self._n_groups
    
    def _create_metadata_array(self, df: pd.DataFrame) -> None:
        self._metadata_array = np.stack([df['source_id'].values], axis=1)
