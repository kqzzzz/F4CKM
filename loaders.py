from einops import rearrange
import numpy as np
import pandas as pd
from typing import List

class DatasetLoader:
    def __init__(self, dataset_file, norm=True, mask_ratio=0) -> None:
        self.dataset = pd.read_pickle(dataset_file)
        # self.dataset = self.dataset.dropna()
        self.num = (self.dataset['RxID']).size // 2
        self.ud_split = (self.dataset['RxID']).size // 2
        if mask_ratio:
            self.num = int(self.num * mask_ratio)
        self.index = np.arange(0, self.num)
        (self.real_mean_ul,
         self.real_std_ul,
         self.imag_mean_ul,
         self.imag_std_ul,
         self.real_mean_dl,
         self.real_std_dl,
         self.imag_mean_dl,
         self.imag_std_dl) = self.get_stats()
        if norm:
            self.cfr_normalize_all()
    
    def get_stats(self):
        rx_ids = np.arange(self.num)
        data_ul = self.get_uplink_cfr_batch(rx_ids)
        data_dl = self.get_cfr_batch(rx_ids)
        real_ul, imag_ul = np.real(data_ul), np.imag(data_ul)
        real_dl, imag_dl = np.real(data_dl), np.imag(data_dl)
        real_mean_ul, real_std_ul, imag_mean_ul, imag_std_ul = real_ul.mean(), real_ul.std(), imag_ul.mean(), imag_ul.std()
        real_mean_dl, real_std_dl, imag_mean_dl, imag_std_dl = real_dl.mean(), real_dl.std(), imag_dl.mean(), imag_dl.std()
        return real_mean_ul, real_std_ul, imag_mean_ul, imag_std_ul, real_mean_dl, real_std_dl, imag_mean_dl, imag_std_dl
    
    def cfr_normalize_all(self):
        rx_ids = np.arange(self.num)
        Nc, Nr, Nt = self.get_cfr_struct()
        data_ul = self.get_uplink_cfr_batch(rx_ids)
        data_dl = self.get_cfr_batch(rx_ids)
        norm_real_ul = (np.real(data_ul) - self.real_mean_ul) / self.real_std_ul
        norm_imag_ul = (np.imag(data_ul) - self.imag_mean_ul) / self.imag_std_ul
        norm_ul = norm_real_ul + 1j * norm_imag_ul
        norm_real_dl = (np.real(data_dl) - self.real_mean_dl) / self.real_std_dl
        norm_imag_dl = (np.imag(data_dl) - self.imag_mean_dl) / self.imag_std_dl
        norm_dl = norm_real_dl + 1j * norm_imag_dl
        for rx_id in rx_ids:
            self.dataset['CSI'][rx_id][:Nc, :Nr, :Nt] = norm_ul[rx_id]
            self.dataset['CSI'][self.ud_split+rx_id][:Nc, :Nr, :Nt] = norm_dl[rx_id]
    
    def cfr_normalize(self, cfr, dl=True):  # cfr [batch_size, Nc, Nr, Nt]
        if dl:
            norm_real = (np.real(cfr) - self.real_mean_dl) / self.real_std_dl
            norm_imag = (np.imag(cfr) - self.imag_mean_dl) / self.imag_std_dl
        else:
            norm_real = (np.real(cfr) - self.real_mean_ul) / self.real_std_ul
            norm_imag = (np.imag(cfr) - self.imag_mean_ul) / self.imag_std_ul
        norm_cfr = norm_real + 1j * norm_imag
        return norm_cfr
            
    def cfr_restore(self, cfr, dl=True):  # cfr [batch_size, Nc, Nr, Nt]
        if dl:
            res_real = np.real(cfr) * self.real_std_dl + self.real_mean_dl
            res_imag = np.imag(cfr) * self.imag_std_dl + self.imag_mean_dl
        else:
            res_real = np.real(cfr) * self.real_std_ul + self.real_mean_ul
            res_imag = np.imag(cfr) * self.imag_std_ul + self.imag_mean_ul
        res_cfr = res_real + 1j * res_imag
        return res_cfr
    
    def get_cfr_batch(self, rx_ids):
        cfr_lst = []
        Nc, Nr, Nt = self.get_cfr_struct()
        for rx_id in rx_ids:
            cfr_lst.append(self.dataset['CSI'][self.ud_split+rx_id][:Nc, :Nr, :Nt])
        return np.array(cfr_lst)
    
    def get_freq_center_down(self):
        down_c = np.array(self.dataset['Frequency'][self.ud_split], dtype=float)
        return down_c
    
    def get_freq(self):
        up_c = np.array(self.dataset['Frequency'][0], dtype=float)
        down_c = np.array(self.dataset['Frequency'][self.ud_split], dtype=float)
        num_elements = 52
        spacing = 3.125e-4
        up_fc = np.linspace(up_c - (num_elements // 2) * spacing, up_c + (num_elements // 2) * spacing, num_elements)
        down_fc = np.linspace(down_c - (num_elements // 2) * spacing, down_c + (num_elements // 2) * spacing, num_elements)
        fc = np.concatenate((up_fc, down_fc), axis=0).astype(np.float32)
        return fc
    
    def get_cfr_struct(self):
        return self.dataset['CSIstruct'][0]
        
    def get_uplink_cfr_batch(self, rx_ids):
        cfr_lst = []
        Nc, Nr, Nt = self.get_cfr_struct()
        for rx_id in rx_ids:
            cfr_lst.append(self.dataset['CSI'][rx_id][:Nc, :Nr, :Nt])
        return np.array(cfr_lst)
    
    def get_loc(self, dev_type, id):
        if dev_type == "AP":
            dev_entries = self.dataset['TxPos'][id]
        elif dev_type == "STA":
            dev_entries = self.dataset['RxPos'][id][:, -1]
        return dev_entries.T
    
    def get_loc_batch(self, dev_type, ids):
        loc_lst = []
        for i in ids:
            loc_lst.append(self.get_loc(dev_type, i))
        return np.array(loc_lst).astype(np.float32)
    
    def get_aoa(self, id):
        lastXpts = self.get_last_itx_pts(id)
        center = self.get_loc('STA', id)
        aoa = lastXpts-center
        return aoa/np.linalg.norm(aoa,axis=1)[:, None].astype(np.float32)
    
    def get_aoa_batch(self, ids) -> List[np.ndarray]:
        aoa_lst = []
        for i in ids:
            aoa_lst.append(self.get_aoa(i))
        return aoa_lst
    
    def get_last_itx_pts(self, id):
        dev_entries = self.dataset['LastXPts'][id]
        return dev_entries.T
    
    def get_station_ids(self):
        return self.dataset['RxID'].unique()
    
    def get_ap_ids(self):
        return self.dataset['TxID'].unique()
        
    def split_train_val(self, ratio = 0.8, fixed_val = True):
        '''
        Split the dataset into training and validation sets.
        @param ratio: The ratio of the training set to the whole dataset.
        @param fixed_val: If True, the validation set will be fixed to the last 20% of the dataset.
        '''
        all_rx = self.get_station_ids()
        if fixed_val:
            n_valid = int(len(all_rx)*0.2)
            n_train = int(len(all_rx)*ratio)
            self.valset = all_rx[int(-1*n_valid):]
            self.trainset = all_rx[:n_train]
        else:
            n_train = int(len(all_rx)*ratio)
            n_valid = len(all_rx)-n_train
            if n_valid < 1:
                n_valid = 1
                n_train = len(all_rx)-n_valid
            self.valset = all_rx[int(-1*n_valid):]
            self.trainset = all_rx[:int(-1*n_valid)]
            

class ArgosDataLoader:
    def __init__(self, dataset_file='./NeRF2-main/data/MIMO/csidata.npy', norm=True, ratio=0.8, mask_ratio=0, train=True) -> None:
        self.cfr_data = np.load(dataset_file)  # (4000, 8, 52)
        self.cfr_data = rearrange(self.cfr_data[..., None], 'n t c r -> n c r t')
        self.num = int(len(self.cfr_data) * ratio) if train else len(self.cfr_data) - int(len(self.cfr_data) * ratio)
        self.cfr_data = self.cfr_data[:self.num, :, :, :] if train else self.cfr_data[-self.num:, :, :, :]
        self.index = np.arange(0, self.num)
        (self.real_mean_ul,
         self.real_std_ul,
         self.imag_mean_ul,
         self.imag_std_ul,
         self.real_mean_dl,
         self.real_std_dl,
         self.imag_mean_dl,
         self.imag_std_dl) = self.get_stats()
        if norm:
            self.cfr_normalize_all()
    
    def get_stats(self):
        data_ul = self.cfr_data[:, 0:26, :, :]
        data_dl = self.cfr_data[:, 26:52, :, :]
        real_ul, imag_ul = np.real(data_ul), np.imag(data_ul)
        real_dl, imag_dl = np.real(data_dl), np.imag(data_dl)
        real_mean_ul, real_std_ul, imag_mean_ul, imag_std_ul = real_ul.mean(), real_ul.std(), imag_ul.mean(), imag_ul.std()
        real_mean_dl, real_std_dl, imag_mean_dl, imag_std_dl = real_dl.mean(), real_dl.std(), imag_dl.mean(), imag_dl.std()
        return real_mean_ul, real_std_ul, imag_mean_ul, imag_std_ul, real_mean_dl, real_std_dl, imag_mean_dl, imag_std_dl
    
    def cfr_normalize_all(self):
        rx_ids = np.arange(self.num)
        data_ul = self.cfr_data[:, 0:26, :, :]
        data_dl = self.cfr_data[:, 26:52, :, :]
        norm_real_ul = (np.real(data_ul) - self.real_mean_ul) / self.real_std_ul
        norm_imag_ul = (np.imag(data_ul) - self.imag_mean_ul) / self.imag_std_ul
        norm_ul = norm_real_ul + 1j * norm_imag_ul
        norm_real_dl = (np.real(data_dl) - self.real_mean_dl) / self.real_std_dl
        norm_imag_dl = (np.imag(data_dl) - self.imag_mean_dl) / self.imag_std_dl
        norm_dl = norm_real_dl + 1j * norm_imag_dl
        for rx_id in rx_ids:
            self.cfr_data[rx_id, 0:26, :, :] = norm_ul[rx_id]
            self.cfr_data[rx_id, 26:52, :, :] = norm_dl[rx_id]
            
    def cfr_normalize(self, cfr, dl=True):  # cfr [batch_size, Nc, Nr, Nt]
        if dl:
            norm_real = (np.real(cfr) - self.real_mean_dl) / self.real_std_dl
            norm_imag = (np.imag(cfr) - self.imag_mean_dl) / self.imag_std_dl
        else:
            norm_real = (np.real(cfr) - self.real_mean_ul) / self.real_std_ul
            norm_imag = (np.imag(cfr) - self.imag_mean_ul) / self.imag_std_ul
        norm_cfr = norm_real + 1j * norm_imag
        return norm_cfr
    
    def cfr_restore(self, cfr, dl=True):  # cfr [batch_size, Nc, Nr, Nt]
        if dl:
            res_real = np.real(cfr) * self.real_std_dl + self.real_mean_dl
            res_imag = np.imag(cfr) * self.imag_std_dl + self.imag_mean_dl
        else:
            res_real = np.real(cfr) * self.real_std_ul + self.real_mean_ul
            res_imag = np.imag(cfr) * self.imag_std_ul + self.imag_mean_ul
        res_cfr = res_real + 1j * res_imag
        return res_cfr
    
    def get_cfr_batch(self, rx_ids):
        return self.cfr_data[rx_ids, 26:52, :, :]
    
    def get_uplink_cfr_batch(self, rx_ids):
        return self.cfr_data[rx_ids, 0:26, :, :]
    
    def get_freq(self):
        fc = 2.4
        num_elements = 26
        spacing = 3.125e-4
        up_fc = np.linspace(fc - num_elements * spacing, fc, num_elements)
        down_fc = np.linspace(fc, fc + num_elements * spacing, num_elements)
        fc = np.concatenate((up_fc, down_fc), axis=0).astype(np.float32)
        return fc
    
    def get_freq_center_down(self):
        fc = 2.4040625
        return fc
    
    def get_cfr_struct(self):
        shape = list(self.cfr_data.shape)
        shape[-3] = shape[-3] // 2
        return shape
            