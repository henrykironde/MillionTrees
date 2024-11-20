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
