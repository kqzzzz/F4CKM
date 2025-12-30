import os
import h5py
import numpy as np
import pandas as pd
import scipy.io
from typing import Dict

class DatasetParser:
    """MATLAB .mat file dataset parser for CSI data."""
    
    TRANSPOSE_RULES = {
        'CSI': (2, 1, 0),
        'TxPos': (1, 0),
        'RxPos': (1, 0),
        'LastXPts': (1, 0)
    }
    
    CLEAN_RULES = {
        'array_fields': ['RxPos', 'TxPos'],
        'complex_array_fields': ['CSI'],
        'scalar_fields': ['RxID', 'Frequency'],
        'interaction_fields': ['LastXPts']
    }

    def __init__(self, dataset_file: str) -> None:
        self.dataset_file = dataset_file
        self.dataset = self._read_mat_dataset()
        # print(self.dataset['LastXPts'][0].shape)
        # dev = 1

    def _read_mat_dataset(self) -> pd.DataFrame:
        is_v73 = self._is_v73_format()
        
        if is_v73:
            return self._process_hdf5().pipe(self._clean_data)
        else:
            return self._process_legacy_mat().pipe(self._clean_data)

    def _is_v73_format(self) -> bool:
        with open(self.dataset_file, 'rb') as f:
            return f.read(8) == b'MATLAB 7'

    def _process_hdf5(self) -> pd.DataFrame:
        with h5py.File(self.dataset_file, 'r') as f:
            if 'fullDataset' in f:
                return self._parse_full_dataset(f)
            elif all(k in f for k in ['cleanUL', 'cleanDL']):
                return self._merge_ul_dl(f)
            else:
                raise KeyError("HDF5 file missing expected datasets.")

    def _parse_full_dataset(self, f: h5py.File) -> pd.DataFrame:
        dataset = f['fullDataset']
        cols = [f[ref][()].item().decode('utf-8') for ref in dataset.dtype['names']]
        return pd.DataFrame({
            n: np.concatenate([self._transpose_data(dataset[n][i], n) 
                              for i in range(dataset[n].shape[0])])
            for n in cols
        })

    def _merge_ul_dl(self, f: h5py.File) -> pd.DataFrame:
        def process_group(group: h5py.Group) -> Dict[str, list]:
            return {
                col: [self._transpose_data(f[ref], col) 
                      for ref in group[col][0, :]]  # 直接解引用
                for col in group.keys()
            }

        ul_data = process_group(f['cleanUL'])
        dl_data = process_group(f['cleanDL'])
        
        return pd.DataFrame({
            col: ul_data[col] +  dl_data[col] 
            for col in ul_data.keys()
        })

    def _transpose_data(self, dataset: h5py.Dataset, col: str) -> np.ndarray:
        if rule := self.TRANSPOSE_RULES.get(col):
            return dataset[:].transpose(rule)
        return dataset[:]  # 默认不转置

    def _process_legacy_mat(self) -> pd.DataFrame:
        data = scipy.io.loadmat(self.dataset_file)
        key = 'fullDataset' if 'fullDataset' in data else ('cleanUL', 'cleanDL')
        
        if isinstance(key, str):
            dataset = data[key]
            cols = dataset.dtype.names
        else:
            dataset = np.concatenate([data['cleanUL'], data['cleanDL']], axis=0)
            cols = data['cleanUL'].dtype.names

        return pd.DataFrame({
            n: np.concatenate([dataset[n][:, i] for i in range(dataset[n].shape[1])])
            for n in cols
        })

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        
        if 'CSI' in df:
            df['CSIstruct'] = df['CSI'].apply(
                lambda x: [x.shape[0], x.shape[1]-1, x.shape[2]-1] 
                if x is not None and x.ndim == 3 
                else None
            )
        
        # 字段维度处理
        for field in self.CLEAN_RULES['complex_array_fields']:
            if field in df: 
                df[field] = df[field].apply(lambda x: x.view(np.complex128) if x.ndim>1 else np.nan)
        
        # 标量提取
        for field in self.CLEAN_RULES['scalar_fields']:
            if field in df: 
                df[field] = df[field].apply(lambda x: x[0,0] if np.prod(x.shape)==1 else np.nan)
                
        # for field in self.CLEAN_RULES['interaction_fields']:
        #     pass
        
        # 过滤无效数据
        return df[df['TxPos'].notna()]

    def save_to_pkl(self, data_dir: str, surfix: str = "") -> None:
        base_name = os.path.splitext(os.path.basename(self.dataset_file))[0]
        save_path = os.path.join(data_dir, f"{base_name}{surfix}.pkl")
        self.dataset.to_pickle(save_path)

if __name__ == "__main__":
    data_dir = "./datasets/"
    target_files = [
        "conferenceroom_2.4GHz_random10000.mat",
        "conferenceroom_2.4GHz_random2000.mat",
        # "conferenceroom_3.5GHz_random10000.mat",
        # "conferenceroom_3.5GHz_random2000.mat",
        # "conferenceroom_6.7GHz_random10000.mat",
        # "conferenceroom_6.7GHz_random2000.mat",
        # "conferenceroom_28GHz_random10000.mat",
        # "conferenceroom_28GHz_random2000.mat",
        # "bedroom_2.4GHz_random20000.mat",
        # "bedroom_2.4GHz_random4000.mat",
        # "bedroom_3.5GHz_random20000.mat",
        # "bedroom_3.5GHz_random4000.mat",
        # "bedroom_6.7GHz_random20000.mat",
        # "bedroom_6.7GHz_random4000.mat",
        # "office_2.4GHz_random30000.mat",
        # "office_2.4GHz_random6000.mat",
        # "office_3.5GHz_random30000.mat",
        # "office_3.5GHz_random6000.mat",
        # "office_6.7GHz_random30000.mat",
        # "office_6.7GHz_random6000.mat",
        # "conferenceroom_2.4GHz_uniform.mat",
        # "bedroom_2.4GHz_uniform.mat",
        # "office_2.4GHz_uniform.mat",
    ]
    
    for f in target_files:
        parser = DatasetParser(os.path.join(data_dir, f))
        parser.save_to_pkl(data_dir)