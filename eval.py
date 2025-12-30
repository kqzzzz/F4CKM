
import math
import os
import re
from statistics import LinearRegression
import time
import torch
import numpy as np
from train import preliminary
from utils import *
from tqdm import tqdm
from loaders import ArgosDataLoader, DatasetLoader
from pipelines import pipeline
from scipy.spatial.distance import pdist, squareform
from calflops import calculate_flops


def cal_cdf(x):
    if x is not None:
        x = x.flatten()
        sorted_x = np.sort(x)
        cdf = np.arange(1, len(sorted_x) + 1) / len(sorted_x)
        return sorted_x, cdf
    else:
        print('Input is None, CDF calculation skipped.')
        return None, None


def anlys_loader(data_dir):
    loader = DatasetLoader(data_dir)
    n_stas = len(loader.num)
    sta_ids = np.arange(n_stas)
    pts = loader.get_loc_batch("STA", sta_ids)
    up_cfr = loader.get_uplink_cfr_batch(sta_ids)
    down_cfr = loader.get_cfr_batch(sta_ids)
    return n_stas, pts, up_cfr, down_cfr


def example(args, cfg, model, loader, encode, encode_viewdirs, optimizer, synthesizer, device):
    model.eval()
    batch_size = 1
    Nc, Nr, Nt = args.d_input
    n_iters = math.ceil(len(loader.index) / batch_size)
    total_iters = n_iters
    snr_list = torch.zeros([0]).to(device)
    pbar = tqdm(total=total_iters, unit='iter', colour='red', disable=False)
    with torch.no_grad():
        for i in range(n_iters):
            sta_id = i * batch_size + np.arange(min(batch_size, len(loader.index) - i * batch_size))
            (_, _, _, snr, _, _, _) = pipeline(args, cfg, sta_id, loader, model, encode, encode_viewdirs, optimizer, synthesizer, device, mode = 'Eg')
            snr_list = torch.cat([snr_list, snr], dim=0)
            pbar.update(1)
        pbar.close()
        snr_mid, index = torch.median(snr_list, dim=0, keepdim=True)
        index = index.cpu().numpy()
        (input_eg, pred_eg, label_eg, snr_eg,
             shaped_queries, z_vals, viewdirs) = pipeline(args, cfg, index, loader, model, encode, encode_viewdirs, optimizer, synthesizer, device, mode = 'Eg')
            
    return input_eg, pred_eg, label_eg, shaped_queries, z_vals, viewdirs
    

def latency_test(args, cfg, model, loader_test, encode, encode_viewdirs, optimizer, synthesizer, device):
    import time
    model.eval()
    batch_size = args.batchsize
    n_iters = math.ceil(len(loader_test.index) / batch_size)
    total_iters = n_iters
    pbar = tqdm(total=total_iters, unit='iter', colour='red', disable=False)
    with torch.no_grad():
        start = time.process_time()
        for i in range(n_iters):
            sta_id = i * batch_size + np.arange(min(batch_size, len(loader_test.index) - i * batch_size))
            _ = pipeline(args, cfg, sta_id, loader_test, model, encode, encode_viewdirs, optimizer, synthesizer, device, mode='Latency')
            pbar.update(1)
        pbar.close()
        latency_total = time.process_time() - start
    latency_mean = latency_total / n_iters
    return latency_total, latency_mean


def evaluation(args, cfg, model, loader, encode, encode_viewdirs, optimizer, synthesizer, device,
               mode=None,
               esnr_tab=[0, 2, 4, 6, 8, 10, 12]):
    model.eval()
    batch_size = args.batchsize
    Nc, Nr, Nt = args.d_input
    n_iters = math.ceil(len(loader.index) / batch_size)
    if mode == 'esnr':
        total_iters = n_iters * len(esnr_tab)
    elif mode == 'cgm':
        total_iters = n_iters
        pos_list = torch.zeros([0, 3]).to(device)
        gain_pred_list = torch.zeros([0]).to(device)
        gain_gt_list = torch.zeros([0]).to(device)
    else:
        total_iters = n_iters
        snr_list = torch.zeros([0]).to(device)
        sgcs_list = torch.zeros([0]).to(device)
        rate_list = torch.zeros([0]).to(device)
    pbar = tqdm(total=total_iters, unit='iter', colour='red', disable=False)
    with torch.no_grad():
        if mode == 'esnr':
            snr_list = [torch.empty(0).to(device) for _ in esnr_tab]  # 改用列表存储动态数据
            sgcs_list = [torch.empty(0).to(device) for _ in esnr_tab]
            rate_list = [torch.empty(0).to(device) for _ in esnr_tab]
            for j, esnr in enumerate(esnr_tab):
                for i in range(n_iters):
                    sta_id = i * batch_size + np.arange(min(batch_size, len(loader.index) - i * batch_size))
                    (snr, sgcs, rate) = pipeline(args, cfg, sta_id, loader, model, encode, encode_viewdirs, optimizer, synthesizer, device, mode='Eval', esnr=esnr)
                    
                    # 沿第0维度拼接
                    snr_list[j] = torch.cat([snr_list[j], snr.flatten()], dim=0) 
                    sgcs_list[j] = torch.cat([sgcs_list[j], sgcs.flatten()], dim=0)
                    rate_list[j] = torch.cat([rate_list[j], rate.flatten()], dim=0)
                    snr_mean = torch.mean(snr_list[j])
                    sgcs_mean = torch.mean(sgcs_list[j])
                    rate_mean = torch.mean(rate_list[j])
                    pbar.set_postfix({'esnr': f'{esnr} dB', 'psnr': f'{snr_mean:.4f} dB', 'sgcs': f'{sgcs_mean:.4f}', 'rate': f'{rate_mean:.4f} bps/Hz'})
                    pbar.update(1)
            snr_list = torch.stack(snr_list)
            sgcs_list = torch.stack(sgcs_list)
            rate_list = torch.stack(rate_list)
        elif mode == 'cgm':
            for i in range(n_iters):
                sta_id = i * batch_size + np.arange(min(batch_size, len(loader.index) - i * batch_size))
                (pos, gain_pred, gain_gt) = pipeline(args, cfg, sta_id, loader, model, encode, encode_viewdirs, optimizer, synthesizer, device, mode='CGM')
                pos_list = torch.cat([pos_list, pos], dim=0)
                gain_pred_list = torch.cat([gain_pred_list, gain_pred], dim=0)
                gain_gt_list = torch.cat([gain_gt_list, gain_gt], dim=0)
                abs_error = torch.abs(gain_pred_list - gain_gt_list)
                mae = torch.mean(abs_error)
                pbar.set_postfix({'mae': f'{mae:.4f} dB'})
                pbar.update(1)
            pbar.close()
            return pos_list, gain_pred_list, gain_gt_list
        else:
            for i in range(n_iters):
                sta_id = i * batch_size + np.arange(min(batch_size, len(loader.index) - i * batch_size))
                (snr, sgcs, rate) = pipeline(args, cfg, sta_id, loader, model, encode, encode_viewdirs, optimizer, synthesizer, device, mode='Eval')
                
                snr_list = torch.cat([snr_list, snr], dim=0)
                sgcs_list = torch.cat([sgcs_list, sgcs], dim=0)
                rate_list = torch.cat([rate_list, rate], dim=0)
                snr_mean = torch.mean(snr_list)
                sgcs_mean = torch.mean(sgcs_list)
                rate_mean = torch.mean(rate_list)
                pbar.set_postfix({'psnr': f'{snr_mean:.4f} dB', 'sgcs': f'{sgcs_mean:.4f}', 'rate': f'{rate_mean:.4f} bps/Hz'})
                pbar.update(1)
        pbar.close()
        
    return snr_list, sgcs_list, rate_list


def eval_rate_upperbound(args, loader, device):
    batch_size = args.batchsize
    n_iters = math.ceil(len(loader.index) / batch_size)
    total_iters = n_iters
    rate_list = torch.zeros([0]).to(device)
    pbar = tqdm(total=total_iters, unit='iter', colour='red', disable=False)
    with torch.no_grad():
        for i in range(n_iters):
            sta_id = i * batch_size + np.arange(min(batch_size, len(loader.index) - i * batch_size))
            target_cfr = torch.tensor(loader.get_cfr_batch(sta_id), dtype=torch.complex64).to(device)  # [batch_size, Nc, Nr, Nt]
            target_cfr = loader.cfr_restore(target_cfr, dl=True)
            _, rate = calculate_rate(target_cfr, target_cfr, SNR_dB=10, precoding_type='dft')
            rate_list = torch.cat([rate_list, rate], dim=0)
            rate_mean = torch.mean(rate_list)
            pbar.set_postfix({'perfect rate': f'{rate_mean:.4f} bps/Hz'})
            pbar.update(1)
        pbar.close()
    pfm = {
            'rate_list': rate_list.cpu().numpy()
        }
    savedir = f'./performance/{args.env}/SE_perfectCSI.npy'
    np.save(savedir, pfm)
    print(f'==> Simulation results saved to: {savedir}')
    
    
def parse_log_file(log_path):
    train_entries = {}
    valid_entries = {}
    
    if 'NeWRF' in log_path:
        pattern = re.compile(
            r'^.*\[INFO\] - (Training|Validating) \|\| Epoch \[(\d+)\/\d+\].*?'
            r'Loss_Fine=([\d.eE+-]+) \|\| SNR_Fine=([\d.-]+) dB'
            )
    elif 'NeRF2' in log_path:
        # if 'argos' in log_path:
        #     pattern = re.compile(
        #     r'^.*INFO - Epoch: (\d+).*?'
        #     r'SNR_Mean = ([\d.-]+) dB.*?'
        #     )
        # else:
        #     pattern = re.compile(
        #     r'^.*INFO - Epoch (\d+) Performance: SNR_Mean = ([\d.-]+) dB'
        #     )
        pattern = re.compile(
            r'^.*INFO - (Training|Validating) \|\| Epoch: (\d+).*?'
            r'SNR_Mean = ([\d.-]+) dB.*?'
            )
    elif 'FIRE' in log_path:
        if 'argos' in log_path:
            pattern = re.compile(
            r'^.*INFO - Training \|\| Epoch \[(\d+)\/\d+\] \|\| SNR_Mean = ([\d.-]+) dB.*?'
            )
        else:
            pattern = re.compile(
            r'^.*INFO - Epoch: (\d+) \|\| SNR_Mean = ([\d.-]+) dB.*?'
            )
    else:
        pattern = re.compile(
            r'^.*\[INFO\] - (Training|Validating) \|\| Epoch \[(\d+)\/\d+\].*?'
            r'SNR_Mean=([\d.-]+) dB'
            )
    
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            match = pattern.match(line)
            if match:
                if 'NeRF2' in log_path:
                    entry_type = match.group(1)
                    epoch = int(match.group(2))
                    snr = float(match.group(3))
                    if entry_type == 'Training':
                        train_entries[epoch] = snr, snr
                    else:
                        valid_entries[epoch] = snr, snr
                elif 'FIRE' in log_path:
                    epoch = int(match.group(1))
                    snr = float(match.group(2))
                    train_entries[epoch] = snr, snr
                    valid_entries[epoch] = snr, snr
                elif 'NeWRF' in log_path:
                    entry_type = match.group(1)
                    epoch = int(match.group(2))
                    snr = float(match.group(4))
                    if entry_type == 'Training':
                        train_entries[epoch] = snr, snr
                    else:
                        valid_entries[epoch] = snr, snr
                else:
                    entry_type = match.group(1)
                    epoch = int(match.group(2))
                    snr = float(match.group(3))
                    if entry_type == 'Training':
                        train_entries[epoch] = snr, snr
                    else:
                        valid_entries[epoch] = snr, snr
    
    # 生成有序数组
    epochs = sorted(train_entries.keys())
    train_array = np.array([train_entries[e] for e in epochs])
    valid_array = np.array([valid_entries[e] for e in epochs])
    
    return train_array, valid_array


def parse_trans_log_file(log_path):
    entries = {}
    
    if 'NeRF2' in log_path:
        pattern = re.compile(
            r'^.*INFO - Validating \|\| Epoch: (\d+).*?'
            r'SNR_Mean = ([\d.-]+) dB.*?'
            r'SNR_Mid = ([\d.-]+) dB.*?'
            )
    elif 'NeWRF' in log_path:
        pattern = re.compile(
            r'^.*\[INFO\] - Validating \|\| Epoch \[(\d+)\/\d+\].*?'
            r'SNR_Coarse=([\d.-]+) dB.*?'
            r'SNR_Fine=([\d.-]+) dB.*?'
            )
    elif 'FIRE' in log_path:
        pattern = re.compile(
            r'^.*INFO - Validating \|\| Epoch \[(\d+)\/\d+\].*?'
            r'SNR_Mean: ([\d.-]+) dB.*?'
            r'SNR_Mid: ([\d.-]+) dB.*?'
            )
    else:
        pattern = re.compile(
            r'^.*\[INFO\] - Validating \|\| Epoch \[(\d+)\/\d+\].*?'
            r'SNR_Mean=([\d.-]+) dB.*?'
            r'SNR_Median=([\d.-]+) dB.*?'
            )
    
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            match = pattern.match(line)
            if match:
                if 'NeRF2' in log_path:
                    epoch = int(match.group(1))
                    snr_mean = float(match.group(2))
                    snr_mid = float(match.group(3))
                    entries[epoch] = snr_mean, snr_mid
                elif 'NeWRF' in log_path:
                    epoch = int(match.group(1))
                    snr_coarse = float(match.group(2))
                    snr_fine = float(match.group(3))
                    entries[epoch] = snr_coarse, snr_fine
                elif 'FIRE' in log_path:
                    epoch = int(match.group(1))
                    snr_mean = float(match.group(2))
                    snr_mid = float(match.group(3))
                    entries[epoch] = snr_mean, snr_mid
                else:
                    epoch = int(match.group(1))
                    snr_mean = float(match.group(2))
                    snr_mid = float(match.group(3))
                    entries[epoch] = snr_mean, snr_mid
    
    # 生成有序数组
    epochs = sorted(entries.keys())
    entries_array = np.array([entries[e] for e in epochs])
    
    return entries_array


def eval_fast_learning_performance():
    """NeRF2"""
    logpath_nerf2_argos = './NeRF2-main/logs/MIMO/mimo-csi-exp1/logger_argos.log'
    train_nerf2_argos, val_nerf2_argos = parse_log_file(logpath_nerf2_argos)
    pfm_nerf2 = {'train': train_nerf2_argos, 'val': val_nerf2_argos}
    savedir_nerf2 = f'./performance/argos/fastlearning_2.4GHz_MIMO_nerf2.npy'
    np.save(savedir_nerf2, pfm_nerf2)
    print(f'==> Simulation results saved to: {savedir_nerf2}')
    """FIRE"""
    logpath_fire_argos = './FIRE-main/logs/fire_logger_argos_2.4GHz.log'
    train_fire_argos, val_fire_argos = parse_log_file(logpath_fire_argos)
    pfm_fire = {'train': train_fire_argos, 'val': val_fire_argos}
    savedir_fire = f'./performance/argos/fastlearning_2.4GHz_MIMO_fire.npy'
    np.save(savedir_fire, pfm_fire)
    print(f'==> Simulation results saved to: {savedir_fire}')
    """F4CKM w/o SFG sampling and shaping filter"""
    logpath_n_wosfg_argos = './logger/argos/2.4GHz_SF_FB_Argos_n_wosfg_2025-12-13_15-04-17.log'
    train_n_wosfg_argos, val_n_wosfg_argos = parse_log_file(logpath_n_wosfg_argos)
    pfm_n_wosfg = {'train': train_n_wosfg_argos, 'val': val_n_wosfg_argos}
    savedir_n_wosfg = f'./performance/argos/fastlearning_2.4GHz_SF_FB_n_wosfg.npy'
    np.save(savedir_n_wosfg, pfm_n_wosfg)
    print(f'==> Simulation results saved to: {savedir_n_wosfg}')
    """F4CKM w/o shaping filter"""
    logpath_n_wsfg_argos = './logger/argos/2.4GHz_SF_FB_Argos_n_wonsfg_2025-12-12_23-20-50.log'
    train_n_wsfg_argos, val_n_wsfg_argos = parse_log_file(logpath_n_wsfg_argos)
    pfm_n_wsfg = {'train': train_n_wsfg_argos, 'val': val_n_wsfg_argos}
    savedir_n_wsfg = f'./performance/argos/fastlearning_2.4GHz_SF_FB_n_wsfg.npy'
    np.save(savedir_n_wsfg, pfm_n_wsfg)
    print(f'==> Simulation results saved to: {savedir_n_wsfg}')
    """F4CKM w/o SFG sampling"""
    logpath_sd_wosfg_argos = './logger/argos/2.4GHz_SF_FB_Argos_sd_wosfg_2025-12-08_10-52-18.log'
    train_sd_wosfg_argos, val_sd_wosfg_argos = parse_log_file(logpath_sd_wosfg_argos)
    pfm_sd_wosfg = {'train': train_sd_wosfg_argos, 'val': val_sd_wosfg_argos}
    savedir_sd_wosfg = f'./performance/argos/fastlearning_2.4GHz_SF_FB_sd_wosfg.npy'
    np.save(savedir_sd_wosfg, pfm_sd_wosfg)
    print(f'==> Simulation results saved to: {savedir_sd_wosfg}')
    """F4CKM full model"""
    logpath_sd_wsfg_argos = './logger/argos/2.4GHz_SF_FB_Argos_sd_wonsfg_2025-12-08_10-52-14.log'
    train_sd_wsfg_argos, val_sd_wsfg_argos = parse_log_file(logpath_sd_wsfg_argos)
    pfm_sd_wsfg = {'train': train_sd_wsfg_argos, 'val': val_sd_wsfg_argos}
    savedir_sd_wsfg = f'./performance/argos/fastlearning_2.4GHz_SF_FB_sd_wsfg.npy'
    np.save(savedir_sd_wsfg, pfm_sd_wsfg)
    print(f'==> Simulation results saved to: {savedir_sd_wsfg}')


def eval_transfer_performance():
    """NeRF2"""
    logpath_nerf2_bedroom = './NeRF2-main/logs/MIMO/mimo-csi-exp2-2.4GHz/logger_bedroom_trans.log'
    logpath_nerf2_office = './NeRF2-main/logs/MIMO/mimo-csi-exp2-2.4GHz/logger_office_trans.log'
    pfm_nerf2_bedroom = parse_trans_log_file(logpath_nerf2_bedroom)
    pfm_nerf2_office = parse_trans_log_file(logpath_nerf2_office)
    savedir_nerf2_bedroom = f'./performance/bedroom/trans_2.4GHz_MIMO_nerf2.npy'
    savedir_nerf2_office = f'./performance/office/trans_2.4GHz_MIMO_nerf2.npy'
    np.save(savedir_nerf2_bedroom, pfm_nerf2_bedroom)
    np.save(savedir_nerf2_office, pfm_nerf2_office)
    print(f'==> Simulation results saved to: {savedir_nerf2_bedroom}')
    print(f'==> Simulation results saved to: {savedir_nerf2_office}')
    logpath_nerf2_bedroom = './NeRF2-main/logs/MIMO/mimo-csi-exp2-2.4GHz/logger_bedroom.log'
    logpath_nerf2_office = './NeRF2-main/logs/MIMO/mimo-csi-exp2-2.4GHz/logger_office.log'
    _, pfm_nerf2_bedroom = parse_log_file(logpath_nerf2_bedroom)
    _, pfm_nerf2_office = parse_log_file(logpath_nerf2_office)
    savedir_nerf2_bedroom = f'./performance/bedroom/ntrans_2.4GHz_MIMO_nerf2.npy'
    savedir_nerf2_office = f'./performance/office/ntrans_2.4GHz_MIMO_nerf2.npy'
    np.save(savedir_nerf2_bedroom, pfm_nerf2_bedroom)
    np.save(savedir_nerf2_office, pfm_nerf2_office)
    print(f'==> Simulation results saved to: {savedir_nerf2_bedroom}')
    print(f'==> Simulation results saved to: {savedir_nerf2_office}')
    """NeWRF"""
    logpath_newrf_bedroom = './NeWRF-main/logger/bedroom/2.4GHz_MIMO_NeWRF_trans_2025-11-30_20-42-11.log'
    logpath_newrf_office = './NeWRF-main/logger/office/2.4GHz_MIMO_NeWRF_trans_2025-12-01_11-30-54.log'
    pfm_newrf_bedroom = parse_trans_log_file(logpath_newrf_bedroom)
    pfm_newrf_office = parse_trans_log_file(logpath_newrf_office)
    savedir_newrf_bedroom = f'./performance/bedroom/trans_2.4GHz_MIMO_newrf.npy'
    savedir_newrf_office = f'./performance/office/trans_2.4GHz_MIMO_newrf.npy'
    np.save(savedir_newrf_bedroom, pfm_newrf_bedroom)
    np.save(savedir_newrf_office, pfm_newrf_office)
    print(f'==> Simulation results saved to: {savedir_newrf_bedroom}')
    print(f'==> Simulation results saved to: {savedir_newrf_office}')
    logpath_newrf_bedroom = './NeWRF-main/logger/bedroom/2.4GHz_MIMO_NeWRF_2025-05-09_22-50-10.log'
    logpath_newrf_office = './NeWRF-main/logger/office/2.4GHz_MIMO_NeWRF_2025-05-23_17-42-50.log'
    _, pfm_newrf_bedroom = parse_log_file(logpath_newrf_bedroom)
    _, pfm_newrf_office = parse_log_file(logpath_newrf_office)
    savedir_newrf_bedroom = f'./performance/bedroom/ntrans_2.4GHz_MIMO_newrf.npy'
    savedir_newrf_office = f'./performance/office/ntrans_2.4GHz_MIMO_newrf.npy'
    np.save(savedir_newrf_bedroom, pfm_newrf_bedroom)
    np.save(savedir_newrf_office, pfm_newrf_office)
    print(f'==> Simulation results saved to: {savedir_newrf_bedroom}')
    print(f'==> Simulation results saved to: {savedir_newrf_office}')
    """FIRE"""
    logpath_fire_bedroom = './FIRE-main/logs/fire_logger_bedroom_2.4GHz_trans.log'
    logpath_fire_office = './FIRE-main/logs/fire_logger_office_2.4GHz_trans.log'
    pfm_fire_bedroom = parse_trans_log_file(logpath_fire_bedroom)
    pfm_fire_office = parse_trans_log_file(logpath_fire_office)
    savedir_fire_bedroom = f'./performance/bedroom/trans_2.4GHz_MIMO_fire.npy'
    savedir_fire_office = f'./performance/office/trans_2.4GHz_MIMO_fire.npy'
    np.save(savedir_fire_bedroom, pfm_fire_bedroom)
    np.save(savedir_fire_office, pfm_fire_office)
    print(f'==> Simulation results saved to: {savedir_fire_bedroom}')
    print(f'==> Simulation results saved to: {savedir_fire_office}')
    logpath_fire_bedroom = './FIRE-main/logs/fire_logger_bedroom_2.4GHz.log'
    logpath_fire_office = './FIRE-main/logs/fire_logger_office_2.4GHz.log'
    _, pfm_fire_bedroom = parse_log_file(logpath_fire_bedroom)
    _, pfm_fire_office = parse_log_file(logpath_fire_office)
    savedir_fire_bedroom = f'./performance/bedroom/ntrans_2.4GHz_MIMO_fire.npy'
    savedir_fire_office = f'./performance/office/ntrans_2.4GHz_MIMO_fire.npy'
    np.save(savedir_fire_bedroom, pfm_fire_bedroom)
    np.save(savedir_fire_office, pfm_fire_office)
    print(f'==> Simulation results saved to: {savedir_fire_bedroom}')
    print(f'==> Simulation results saved to: {savedir_fire_office}')
    """F4CKM"""
    logpath_f4ckm_bedroom = './logger/bedroom/2.4GHz_SF_FB_sd_trans_2025-11-27_15-52-46.log'
    logpath_f4ckm_office = './logger/office/2.4GHz_SF_FB_sd_trans_2025-11-28_10-26-16.log'
    pfm_f4ckm_bedroom = parse_trans_log_file(logpath_f4ckm_bedroom)
    pfm_f4ckm_office = parse_trans_log_file(logpath_f4ckm_office)
    savedir_f4ckm_bedroom = f'./performance/bedroom/trans_2.4GHz_SF_FB_sd.npy'
    savedir_f4ckm_office = f'./performance/office/trans_2.4GHz_SF_FB_sd.npy'
    np.save(savedir_f4ckm_bedroom, pfm_f4ckm_bedroom)
    np.save(savedir_f4ckm_office, pfm_f4ckm_office)
    print(f'==> Simulation results saved to: {savedir_f4ckm_bedroom}')
    print(f'==> Simulation results saved to: {savedir_f4ckm_office}')
    logpath_f4ckm_bedroom = './logger/bedroom/2.4GHz_SF_FB_sd_2025-05-18_12-42-30.log'
    logpath_f4ckm_office = './logger/office/2.4GHz_SF_FB_sd_2025-05-19_09-55-59.log'
    _, pfm_f4ckm_bedroom = parse_log_file(logpath_f4ckm_bedroom)
    _, pfm_f4ckm_office = parse_log_file(logpath_f4ckm_office)
    savedir_f4ckm_bedroom = f'./performance/bedroom/ntrans_2.4GHz_SF_FB_sd.npy'
    savedir_f4ckm_office = f'./performance/office/ntrans_2.4GHz_SF_FB_sd.npy'
    np.save(savedir_f4ckm_bedroom, pfm_f4ckm_bedroom)
    np.save(savedir_f4ckm_office, pfm_f4ckm_office)
    print(f'==> Simulation results saved to: {savedir_f4ckm_bedroom}')
    print(f'==> Simulation results saved to: {savedir_f4ckm_office}')


def eval_sample_resolution_performance(args, cfg):
    args.env = 'argos'
    args.data_tag = 'argos'
    args.model_tag = '2.4GHz_SF_FB_Argos_lite'
    args.d_input = [26, 1, 8]
    # logger
    log_dir = f'./logger/argos'
    current_time = time.localtime()
    time_string = time.strftime("%Y-%m-%d_%H-%M-%S", current_time)
    log_path = os.path.join(log_dir, f"sampling_resolution_test_{time_string}.log")
    file_logger, disp_logger = init_logger(cfg, log_path)
    disp_logger.info('\n****************************** Logger ******************************')
    disp_logger.info(f'==> Logger saved at: {log_path}')
    # key configurations
    disp_logger.info('\n****************************** Key Configurations ******************************')
    log = '\n'.join([
        f'batch size: {cfg.training.batch_size}',
        f'd input: {args.d_input}',
    ])
    disp_logger.info(log)
    # hardware
    disp_logger.info('\n****************************** Hardware ******************************')
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    disp_logger.info(f"==> Using {device}")
    # random number generator
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    disp_logger.info('\n****************************** RNG ******************************')
    disp_logger.info(f'==> Using seed:{seed}')
    # dataloader
    dataset_file = './NeRF2-main/data/MIMO/csidata.npy'
    loader_test = ArgosDataLoader(dataset_file=dataset_file, train=False)
    disp_logger.info('\n****************************** Dataloader ******************************')
    disp_logger.info(f'==> dataset: {dataset_file}')

    radial_range_tab = ['3.0', '5.0', '7.0', '9.0']
    n_rays_tab = ['16', '24', '32', '40']
    n_samples_tab = ['16', '32', '64', '128']
    snr_list = [torch.empty(0).to(device) for _ in range(len(radial_range_tab)*len(n_rays_tab)*len(n_samples_tab))]
    sgcs_list = [torch.empty(0).to(device) for _ in range(len(radial_range_tab)*len(n_rays_tab)*len(n_samples_tab))]
    rate_list = [torch.empty(0).to(device) for _ in range(len(radial_range_tab)*len(n_rays_tab)*len(n_samples_tab))]
    for i, radial_range in enumerate(radial_range_tab):
        for j, n_rays in enumerate(n_rays_tab):
            for k, n_samples in enumerate(n_samples_tab):
                cfg.sampling.n_rays = int(n_rays)
                cfg.sampling.n_samples = int(n_samples)
                cfg.sampling.far = float(radial_range)
                ckpt_argos = f'ckpt/argos/2.4GHz_SF_FB_Argos_lite_sd_{n_rays}R{n_samples}S{radial_range}F_ckpt_best.pt'
                (model, encode, encode_viewdirs, optimizer, scheduler, synthesizer, history_epoch, best_score) = init_models(args, cfg, device, disp_logger, ckpt_argos)
                (snr_list_permodel, sgcs_list_permodel, rate_list_permodel) = evaluation(args, cfg, model, loader_test, encode, encode_viewdirs, optimizer, synthesizer, device)
                index = i * len(n_rays_tab) * len(n_samples_tab) + j * len(n_samples_tab) + k
                snr_list[index] = snr_list_permodel
                sgcs_list[index] = sgcs_list_permodel
                rate_list[index] = rate_list_permodel
                disp_logger.info(f'==> Sampling resolution test completed for {radial_range}m, {n_rays} rays, {n_samples} samples.')
                disp_logger.info(f'==> PSNR: {torch.mean(snr_list_permodel):.4f} dB, SGCS: {torch.mean(sgcs_list_permodel):.4f}, Rate: {torch.mean(rate_list_permodel):.4f} bps/Hz')
    snr_list = torch.stack(snr_list)
    sgcs_list = torch.stack(sgcs_list)
    rate_list = torch.stack(rate_list)
    pfm_argos = {
        'snr_list': snr_list.cpu().numpy(),
        'sgcs_list': sgcs_list.cpu().numpy(),
        'rate_list': rate_list.cpu().numpy()
    }
    savedir_argos = f'./performance/argos/2.4GHz_SF_FB_Argos_lite_sd_SS.npy'
    np.save(savedir_argos, pfm_argos)
    disp_logger.info(f'==> Simulation results saved to: {savedir_argos}')


def eval_sample_resolution_performance_single_model(args, cfg):
    args.env = 'argos'
    args.data_tag = 'argos'
    args.model_tag = '2.4GHz_SF_FB_Argos_lite'
    args.d_input = [26, 1, 8]
    # logger
    log_dir = f'./logger/argos'
    current_time = time.localtime()
    time_string = time.strftime("%Y-%m-%d_%H-%M-%S", current_time)
    log_path = os.path.join(log_dir, f"sampling_resolution_single_model_test_{time_string}.log")
    file_logger, disp_logger = init_logger(cfg, log_path)
    disp_logger.info('\n****************************** Logger ******************************')
    disp_logger.info(f'==> Logger saved at: {log_path}')
    # key configurations
    disp_logger.info('\n****************************** Key Configurations ******************************')
    log = '\n'.join([
        f'batch size: {cfg.training.batch_size}',
        f'd input: {args.d_input}',
    ])
    disp_logger.info(log)
    # hardware
    disp_logger.info('\n****************************** Hardware ******************************')
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    disp_logger.info(f"==> Using {device}")
    # random number generator
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    disp_logger.info('\n****************************** RNG ******************************')
    disp_logger.info(f'==> Using seed:{seed}')
    # dataloader
    dataset_file = './NeRF2-main/data/MIMO/csidata.npy'
    loader_test = ArgosDataLoader(dataset_file=dataset_file, train=False)
    disp_logger.info('\n****************************** Dataloader ******************************')
    disp_logger.info(f'==> dataset: {dataset_file}')

    ckpt_argos = f'ckpt/argos/2.4GHz_SF_FB_Argos_lite_sd_40R128S9.0F_ckpt_best.pt'
    radial_range_tab = ['3.0', '5.0', '7.0', '9.0']
    n_rays_tab = ['16', '24', '32', '40']
    n_samples_tab = ['16', '32', '64', '128']
    snr_list = [torch.empty(0).to(device) for _ in range(len(radial_range_tab)*len(n_rays_tab)*len(n_samples_tab))]
    sgcs_list = [torch.empty(0).to(device) for _ in range(len(radial_range_tab)*len(n_rays_tab)*len(n_samples_tab))]
    rate_list = [torch.empty(0).to(device) for _ in range(len(radial_range_tab)*len(n_rays_tab)*len(n_samples_tab))]
    for i, radial_range in enumerate(radial_range_tab):
        for j, n_rays in enumerate(n_rays_tab):
            for k, n_samples in enumerate(n_samples_tab):
                cfg.sampling.n_rays = int(n_rays)
                cfg.sampling.n_samples = int(n_samples)
                cfg.sampling.far = float(radial_range)
                disp_logger.info('\n************************************************************************')
                disp_logger.info(f'\n==> Testing with {radial_range}m, {n_rays} rays, {n_samples} samples using single model: {ckpt_argos}')
                (model, encode, encode_viewdirs, optimizer, scheduler, synthesizer, history_epoch, best_score) = init_models(args, cfg, device, disp_logger, ckpt_argos)
                (snr_list_permodel, sgcs_list_permodel, rate_list_permodel) = evaluation(args, cfg, model, loader_test, encode, encode_viewdirs, optimizer, synthesizer, device)
                index = i * len(n_rays_tab) * len(n_samples_tab) + j * len(n_samples_tab) + k
                snr_list[index] = snr_list_permodel
                sgcs_list[index] = sgcs_list_permodel
                rate_list[index] = rate_list_permodel
                disp_logger.info(f'==> Sampling resolution test completed for {radial_range}m, {n_rays} rays, {n_samples} samples.')
                disp_logger.info(f'==> PSNR: {torch.mean(snr_list_permodel):.4f} dB, SGCS: {torch.mean(sgcs_list_permodel):.4f}, Rate: {torch.mean(rate_list_permodel):.4f} bps/Hz')
    snr_list = torch.stack(snr_list)
    sgcs_list = torch.stack(sgcs_list)
    rate_list = torch.stack(rate_list)
    pfm_argos = {
        'snr_list': snr_list.cpu().numpy(),
        'sgcs_list': sgcs_list.cpu().numpy(),
        'rate_list': rate_list.cpu().numpy()
    }
    savedir_argos = f'./performance/argos/2.4GHz_SF_FB_Argos_lite_sd_SS_single_model.npy'
    np.save(savedir_argos, pfm_argos)
    disp_logger.info(f'==> Simulation results saved to: {savedir_argos}')
    

def eval_channel_gain_map(args, cfg):
    args.data_tag = '2.4GHz_uniform'
    args.model_tag = '2.4GHz_SF_FB'
    args.d_input = [52, 4, 16]
    # logger
    log_dir = f'./logger/{args.env}'
    current_time = time.localtime()
    time_string = time.strftime("%Y-%m-%d_%H-%M-%S", current_time)
    log_path = os.path.join(log_dir, f"{args.model_tag}_{args.filter}_{time_string}_test_CGM.log")
    file_logger, disp_logger = init_logger(cfg, log_path)
    disp_logger.info('\n****************************** Logger ******************************')
    disp_logger.info(f'==> Logger saved at: {log_path}')
    # key configurations
    disp_logger.info('\n****************************** Key Configurations ******************************')
    log = '\n'.join([
        f'batch size: {cfg.training.batch_size}',
        f'd input: {args.d_input}',
    ])
    disp_logger.info(log)
    # hardware
    disp_logger.info('\n****************************** Hardware ******************************')
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    disp_logger.info(f"==> Using {device}")
    # random number generator
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    disp_logger.info('\n****************************** RNG ******************************')
    disp_logger.info(f'==> Using seed:{seed}')
    # dataloader
    data_dir = "./simulator/datasets/"
    dataset_test = os.path.join(data_dir, f"{args.env}_{args.data_tag}.pkl")
    loader_test = DatasetLoader(dataset_test, mask_ratio=args.mask_ratio)
    disp_logger.info('\n****************************** Dataloader ******************************')
    disp_logger.info(f'==> testing dataset: {dataset_test}')

    ckpt = f'ckpt/{args.env}/{args.model_tag}_{args.filter}_ckpt_best.pt'
    (model, encode, encode_viewdirs, optimizer, scheduler, synthesizer, history_epoch, best_score) = init_models(args, cfg, device, disp_logger, ckpt)
    (pos_list, gain_pred_list, gain_gt_list) = evaluation(args, cfg, model, loader_test, encode, encode_viewdirs, optimizer, synthesizer, device, mode='cgm')
    pfm = {
        'pos_list': pos_list.cpu().numpy(),
        'gain_pred_list': gain_pred_list.cpu().numpy(),
        'gain_gt_list': gain_gt_list.cpu().numpy(),
    }
    savedir = f'./performance/{args.env}/2.4GHz_SF_FB_CGM.npy'
    np.save(savedir, pfm)
    disp_logger.info(f'==> Simulation results saved to: {savedir}')
    
        
def load_performance(path, type='sys'):
    if os.path.exists(path):
        if type == 'sys':
            pfm = np.load(path, allow_pickle=True).item()
            snr_list, sgcs_list = pfm['snr_list'], pfm['sgcs_list']
            print(f'==> Simulation results loaded successfully from: {path}')
            return snr_list, sgcs_list
        elif type == 'psnr':
            pfm = np.load(path, allow_pickle=True).item()
            snr_list = pfm['snr_list']
            print(f'==> Simulation results loaded successfully from: {path}')
            return snr_list
        elif type == 'sgcs':
            pfm = np.load(path, allow_pickle=True).item()
            sgcs_list = pfm['sgcs_list']
            print(f'==> Simulation results loaded successfully from: {path}')
            return sgcs_list
        elif type == 'rate':
            pfm = np.load(path, allow_pickle=True).item()
            rate_list = pfm['rate_list']
            print(f'==> Simulation results loaded successfully from: {path}')
            return rate_list
        elif type == 'filter':
            pfm = np.load(path, allow_pickle=True).item()
            train, val = pfm['train'], pfm['val']
            print(f'==> Simulation results loaded successfully from: {path}')
            return train, val
        elif type == 'esnr':
            pfm = np.load(path, allow_pickle=True).item()
            snr_list, sgcs_list, rate_list = pfm['snr_list'], pfm['sgcs_list'], pfm['rate_list']
            print(f'==> Simulation results loaded successfully from: {path}')
            return snr_list, sgcs_list, rate_list
        elif type == 'fl':
            pfm = np.load(path, allow_pickle=True).item()
            train, val = pfm['train'], pfm['val']
            print(f'==> Simulation results loaded successfully from: {path}')
            return train, val
        elif type == 'eg':
            pfm = np.load(path, allow_pickle=True).item()
            input, pred, label = pfm['input_eg'], pfm['pred_eg'], pfm['label_eg']
            print(f'==> Simulation results loaded successfully from: {path}')
            return input, pred, label
        elif type == 'trans':
            pfm = np.load(path, allow_pickle=True)
            print(f'==> Simulation results loaded successfully from: {path}')
            return pfm
        elif type == 'cgm':
            pfm = np.load(path, allow_pickle=True).item()
            pos_list, gain_pred_list, gain_gt_list = pfm['pos_list'], pfm['gain_pred_list'], pfm['gain_gt_list']
            print(f'==> Simulation results loaded successfully from: {path}')
            return pos_list, gain_pred_list, gain_gt_list
    else:
        print(f'==> Data path does not exist: {path}')
        return None, None
    
    
def calculate_boxplot(data):
    upper_quartile = np.percentile(data, 75)
    lower_quartile = np.percentile(data, 25)
    IQR = upper_quartile - lower_quartile
    upper_whisker = upper_quartile + 1.5 * IQR
    lower_whisker = lower_quartile - 1.5 * IQR
    median = np.median(data)
    # print(f'upper_whisker: {upper_whisker:.2f}')
    # print(f'upper_quartile: {upper_quartile:.2f}')
    print(f'median: {median:.2f}')
    # print(f'lower_quartile: {lower_quartile:.2f}')
    # print(f'lower_whisker: {lower_whisker:.2f}')
    return upper_whisker, upper_quartile, median, lower_quartile, lower_whisker


def build_heatmap_from_samples(
    pos_list,
    gain_pred,
    gain_gt,
    fc,
    c=299792458,
    fill_value=-100.0,
    auto_range=True,
    x_range=None,
    y_range=None,
    force_grid_alignment=True
):

    x = np.asarray(pos_list)[:, 1].flatten()
    y = np.asarray(pos_list)[:, 0].flatten()
    gain_pred = np.asarray(gain_pred).flatten()
    gain_gt = np.asarray(gain_gt).flatten()
    dx = dy = c / fc
    
    if auto_range:
        if x_range is None:
            x_min, x_max = x.min(), x.max()
        else:
            x_min, x_max = x_range
        if y_range is None:
            y_min, y_max = y.min(), y.max()
        else:
            y_min, y_max = y_range
    else:
        assert x_range is not None and y_range is not None, "x_range/y_range required when auto_range=False"
        x_min, x_max = x_range
        y_min, y_max = y_range

    nx = int(round((x_max - x_min) / dx)) + 1
    ny = int(round((y_max - y_min) / dy)) + 1
    x_grid = np.linspace(x_min, x_max, nx)
    y_grid = np.linspace(y_min, y_max, ny)
    
    gain_pred_map = np.full((nx, ny), fill_value, dtype=np.float32)
    gain_gt_map   = np.full((nx, ny), fill_value, dtype=np.float32)
    
    if force_grid_alignment:
        ix = np.rint((x - x_min) / dx).astype(int)
        iy = np.rint((y - y_min) / dy).astype(int)
    else:
        ix = np.argmin(np.abs(x_grid[:, None] - x[None, :]), axis=0)
        iy = np.argmin(np.abs(y_grid[:, None] - y[None, :]), axis=0)
    
    valid_mask = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ix, iy = ix[valid_mask], iy[valid_mask]
    gp, gt = gain_pred[valid_mask], gain_gt[valid_mask]
    gain_pred_map[ix, iy] = gp
    gain_gt_map[ix, iy] = gt
    extent = [0, (y_max-y_min), 0, (x_max-x_min)]
    print(f"Heatmap grid: {nx} × {ny}")
    print(f"Sampled points: {len(pos_list)} → Mapped: {int(nx * ny)}")

    return {
        'gain_pred_map': gain_pred_map,
        'gain_gt_map':   gain_gt_map,
        'x_grid': x_grid,
        'y_grid': y_grid,
        'extent': extent,
        'dx': dx,
        'fill_value': fill_value
    }
    
    
def eval_flops(args, cfg, model):
    model.eval().to('cpu')
    n_rays = cfg.sampling.n_rays
    n_samples = cfg.sampling.n_samples
    chunksize = int(n_rays * n_samples)
    Nc, Nr, Nt = args.d_input
    input_shape = (chunksize, int(2*Nc), Nr, Nt)
    flops, macs, params = calculate_flops(model=model, input_shape=input_shape, output_as_string=True, output_precision=4, print_detailed=False, print_results=False)
    return flops, macs, params
        
        
        
        
        

"""The functions below have been deprecated."""
