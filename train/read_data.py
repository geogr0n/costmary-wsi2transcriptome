import os
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import torch
from tqdm import tqdm
import h5py


class SuperTileRNADataset(Dataset):
    def __init__(self, csv_path: str, features_path, preload=True):
        self.csv_path = csv_path
        self.features_path = features_path
        if type(csv_path) == str:
            self.data = pd.read_csv(csv_path)
        else:
            self.data = csv_path

        # find the number of genes
        rna_cols = [x for x in self.data.columns if 'rna_' in x]
        self.num_genes = len(rna_cols)

        # find the feature dimension
        row = self.data.iloc[0]
        path = os.path.join(self.features_path, row['tcga_project'], 
                            row['wsi_file_name'], row['wsi_file_name']+'.h5')
        if 'GTEX' not in path:
            path = path.replace('.svs','')
        with h5py.File(path, 'r') as f:
            self.feature_dim = f['cluster_features'].shape[1]
        self._precomputed_inputs = None

        # 预加载所有数据到内存
        self.preloaded = preload
        if preload:
            self._features = []
            self._rna = []
            self._wsi_names = []
            self._projects = []
            rna_matrix = self.data[rna_cols].values.astype(np.float32)

            for idx in tqdm(range(len(self.data)), desc='Preloading data'):
                r = self.data.iloc[idx]
                p = os.path.join(self.features_path, r['tcga_project'],
                                 r['wsi_file_name'], r['wsi_file_name']+'.h5')
                if 'GTEX' not in p:
                    p = p.replace('.svs','')
                try:
                    with h5py.File(p, 'r') as f:
                        feat = torch.tensor(f['cluster_features'][:], dtype=torch.float32)
                except Exception as e:
                    print(e, p)
                    feat = None
                self._features.append(feat)
                self._rna.append(torch.tensor(rna_matrix[idx], dtype=torch.float32))
                self._wsi_names.append(r['wsi_file_name'])
                self._projects.append(r['tcga_project'])

    def __len__(self):
        return self.data.shape[0]

    def set_precomputed_inputs(self, tensors):
        if tensors is None:
            self._precomputed_inputs = None
            return
        if len(tensors) != len(self):
            raise ValueError(f"Expected {len(self)} precomputed tensors, got {len(tensors)}")
        self._precomputed_inputs = tensors

    def __getitem__(self, idx):
        if self.preloaded:
            x = self._precomputed_inputs[idx] if self._precomputed_inputs is not None else self._features[idx]
            return x, self._rna[idx], self._wsi_names[idx], self._projects[idx]

        row = self.data.iloc[idx]
        path = os.path.join(self.features_path, row['tcga_project'],
                            row['wsi_file_name'], row['wsi_file_name']+'.h5')
        rna_data = row[[x for x in row.keys() if 'rna_' in x]].values.astype(np.float32)
        rna_data = torch.tensor(rna_data, dtype=torch.float32)
        try:
            if 'GTEX' not in path:
                path = path.replace('.svs','')
            with h5py.File(path, 'r') as f:
                features = torch.tensor(f['cluster_features'][:], dtype=torch.float32)
        except Exception as e:
            print(e, path)
            features = None

        x = self._precomputed_inputs[idx] if self._precomputed_inputs is not None else features
        return x, rna_data, row['wsi_file_name'], row['tcga_project']
