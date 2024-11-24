from .dataset import MillionTreesDataset
from typing import Union, Optional
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from ..utils import CombinatorialGrouper

class UnlabeledDataset(MillionTreesDataset):
    """Base class for unlabeled MillionTrees datasets."""
    
    _metadata_fields = [
        "source_id",
        "location",
        "sequence",
        "datetime",
        "y"
    ]
    
    _split_dict = {"extra_unlabeled": 0}
    _split_names = {"extra_unlabeled": "Extra Unlabeled"}
    
    def __init__(
        self, 
        root_dir: Union[str, Path],
        version: Optional[str] = None,
        download: bool = False,
        split_scheme: str = "official"
    ) -> None:
        self._version = version
        self._split_scheme = split_scheme
        if self._split_scheme != "official":
            raise ValueError(f"Split scheme {self._split_scheme} not recognized")
            
        self._data_dir = Path(self.initialize_data_dir(root_dir, download))
        self._setup_dataset()
        
        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=(["location"])
        )
        
        super().__init__(root_dir, download, split_scheme)
class UnlabeledDataset(MillionTreesDataset):
    """Base class for unlabeled MillionTrees datasets."""
    
    _metadata_fields = ["source_id", "location", "datetime", "y"]
    
    def _setup_dataset(self) -> None:
        df = self._load_metadata()
        self._process_splits(df)
        self._process_filenames(df)
        self._process_groups(df)
        self._process_labels(df)
        self._create_metadata_array(df)
    
    def _process_splits(self, df: pd.DataFrame) -> None:
        df["split_id"] = 0
        self._split_array = df["split_id"].values
    
    def _process_filenames(self, df: pd.DataFrame) -> None:
        self._input_array = df["filename"].values
        self._input_lookup = {fn: [i] for i, fn in enumerate(df["filename"])}
    
    def _process_groups(self, df: pd.DataFrame) -> None:
        self._n_groups = df["location"].nunique()
    
    def _process_labels(self, df: pd.DataFrame) -> None:
        self._y_array = df["y"].values
        self._y_size = 1
    
    def _create_metadata_array(self, df: pd.DataFrame) -> None:
        metadata = [df[field].values for field in self._metadata_fields]
        self._metadata_array = np.stack(metadata, axis=1)
    def _setup_dataset(self) -> None:
        df = self._load_metadata()
        self._process_splits(df)
        self._process_filenames(df)
        self._process_groups(df)
        self._process_labels(df)
        self._create_metadata_array(df)
    
    def _load_metadata(self) -> pd.DataFrame:
        return pd.read_csv(self._data_dir / "metadata.csv")
    
    def _process_splits(self, df: pd.DataFrame) -> None:
        df["split_id"] = 0
        self._split_array = df["split_id"].values
    
    def _process_filenames(self, df: pd.DataFrame) -> None:
        self._input_array = df["filename"].values
        self._input_lookup = {fn: [i] for i, fn in enumerate(df["filename"])}
    
    def _process_groups(self, df: pd.DataFrame) -> None:
        df["source_id"] = df.location.astype('category').cat.codes
        self._n_groups = max(df['source_id']) + 1
        assert len(np.unique(df['source_id'])) == self._n_groups
    
    def _process_labels(self, df: pd.DataFrame) -> None:
        self._y_array = df["y"].values if "y" in df.columns else np.zeros(len(df))
        self._y_size = 1
    
    def _create_metadata_array(self, df: pd.DataFrame) -> None:
        metadata = [df[field].values for field in self._metadata_fields if field in df.columns]
        self._metadata_array = np.stack(metadata, axis=1)    def _setup_dataset(self) -> None:
        df = self._load_metadata()
        self._process_splits(df)
        self._process_filenames(df)
        self._process_groups(df)
        self._process_labels(df)
        self._create_metadata_array(df)
    
    def _load_metadata(self) -> pd.DataFrame:
        return pd.read_csv(self._data_dir / "metadata.csv")
    
    def _process_splits(self, df: pd.DataFrame) -> None:
        df["split_id"] = self._split_dict["extra_unlabeled"]
        self._split_array = df["split_id"].values
    
    def _process_filenames(self, df: pd.DataFrame) -> None:
        self._input_array = df["filename"].values
        self._input_lookup = {fn: [i] for i, fn in enumerate(df["filename"])}
    
    def _process_groups(self, df: pd.DataFrame) -> None:
        df["source_id"] = df.location.astype('category').cat.codes
        self._n_groups = max(df['source_id']) + 1
        assert len(np.unique(df['source_id'])) == self._n_groups
    
    def _process_labels(self, df: pd.DataFrame) -> None:
        self._y_array = df["y"].values if "y" in df.columns else np.zeros(len(df))
        self._y_size = 1
    
    def _create_metadata_array(self, df: pd.DataFrame) -> None:
        metadata = [df[field].values for field in self._metadata_fields if field in df.columns]
        self._metadata_array = np.stack(metadata, axis=1)