import numpy as np
import os
import h5py
from sklearn.model_selection import train_test_split, KFold
import torch

def custom_collate_fn(batch):
    """Remove bad entries from the dataloader
    Args:
        batch (torch.Tensor): batch of tensors from the dataaset
    Returns:
        collate: Default collage for the dataloader
    """
    batch = list(filter(lambda x: x[0] is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def filter_no_features(df, feature_path, feature_name):
    print(f'Filtering WSIs that do not have {feature_name} features')
    projects = np.unique(df.tcga_project)
    all_wsis_with_features = []
    remove = []
    for proj in projects:
        wsis_with_features = os.listdir(os.path.join(feature_path, proj))
        for wsi in wsis_with_features:
            try:
                with h5py.File(os.path.join(feature_path, proj, wsi, wsi+'.h5'), "r") as f:
                    cols = list(f.keys())
                    if feature_name not in cols:
                        remove.append(wsi)
            except Exception as e:
                remove.append(wsi)        
        all_wsis_with_features += wsis_with_features
    
    # 统一到“不含 .svs 后缀”的命名空间进行匹配，避免 CSV/目录名后缀不一致导致漏删
    df_wsi_no_ext = df['wsi_file_name'].astype(str).apply(lambda x: x.replace('.svs', ''))
    missing_wsis_no_ext = df_wsi_no_ext[~df_wsi_no_ext.isin(all_wsis_with_features)].values.tolist()
    remove_no_ext = set([str(x).replace('.svs', '') for x in remove] + missing_wsis_no_ext)
    
    print(f'Original shape: {df.shape}')
    df = df[~df_wsi_no_ext.isin(remove_no_ext)].reset_index(drop=True)
    print(f'New shape: {df.shape}')
    return df


def patient_kfold(dataset, n_splits=5, random_state=0, valid_size=0.1):
    """Perform cross-validation with patient split.
    """
    indices = np.arange(len(dataset))

    patients_unique = np.unique(dataset.patient_id)

    skf = KFold(n_splits, shuffle=True, random_state=random_state)
    ind = skf.split(patients_unique)

    train_idx = []
    valid_idx = []
    test_idx = []

    for k, (ind_train, ind_test) in enumerate(ind):

        patients_train = patients_unique[ind_train]
        patients_test = patients_unique[ind_test]

        test_idx.append(indices[np.any(np.array(dataset.patient_id)[:, np.newaxis] ==
                                       np.array(patients_test)[np.newaxis], axis=1)])

        if valid_size > 0:
            patients_train, patients_valid = train_test_split(
                patients_train, test_size=valid_size, random_state=0)
            valid_idx.append(indices[np.any(np.array(dataset.patient_id)[:, np.newaxis] ==
                                            np.array(patients_valid)[np.newaxis], axis=1)])

        train_idx.append(indices[np.any(np.array(dataset.patient_id)[:, np.newaxis] ==
                                        np.array(patients_train)[np.newaxis], axis=1)])

    return train_idx, valid_idx, test_idx


def exists(x):
    return x is not None
