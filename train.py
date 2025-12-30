# %%
import math
import sys
import time
import torch
from torch import nn
from pipelines import pipeline
import numpy as np
from utils import *
from tqdm import tqdm, trange
from loaders import ArgosDataLoader, DatasetLoader
import os, random, string
from omegaconf import OmegaConf
from argparse import ArgumentParser
from tqdm_logger import TqdmLogger
import logging
import os


# %%
def preliminary(args, cfg, data_dir, save_dir=None, test=False, test_tag=None):
    # logger
    log_dir = f'./logger/{args.env}'
    current_time = time.localtime()
    time_string = time.strftime("%Y-%m-%d_%H-%M-%S", current_time)
    trans_suffix = "_trans" if args.trans else ""
    ss_suffix = f'_{args.n_rays}R{args.n_samples}S{args.radial_max}F' if args.sample_res else ''
    ablation_suffix = "" if args.ablation == 'n' else f"_wo{args.ablation}"
    log_path = os.path.join(log_dir, f"{args.model_tag}_{args.filter}{ss_suffix}{trans_suffix}{ablation_suffix}_{time_string}.log")
    if test:
        log_path = os.path.join(log_dir, f"{args.model_tag}_{args.filter}_{time_string}_test_{test_tag}.log")
    file_logger, disp_logger = init_logger(cfg, log_path)
    disp_logger.info('\n****************************** Logger ******************************')
    disp_logger.info(f'==> Logger saved at: {log_path}')
    # key configurations
    disp_logger.info('\n****************************** Key Configurations ******************************')
    cfg.sampling.n_rays = args.n_rays
    cfg.sampling.n_samples = args.n_samples
    cfg.sampling.far = args.radial_max
    if args.ablation == 'sfg':
        cfg.sampling.fibonacci = False
        cfg.sampling.num_theta_samples = 36
        cfg.sampling.num_phi_samples = 18
        cfg.sampling.n_samples = 16
        cfg.training.batch_size = 1
    elif args.ablation == 'nsfg':
        cfg.sampling.fibonacci = True
        cfg.sampling.n_rays = 648
        cfg.sampling.n_samples = 16
        cfg.training.batch_size = 1
    log = '\n'.join([
        f'ablation: {args.ablation}',
        f'model tag: {args.model_tag}',
        f'filter: {args.filter}',
        f'data tag: {args.data_tag}',
        f'mask ratio: {args.mask_ratio}',
        f'metrics define: {args.define}',
        f'lr: {args.lr}',
        f'batch size: {cfg.training.batch_size}',
        f'epochs: {args.epochs}',
        f'd input: {args.d_input}',
        f'known_aoa: {cfg.sampling.known_aoa}',
        f'radial range: {cfg.sampling.far}',
        f'n_samples: {cfg.sampling.n_samples}',
        f'SFG: {cfg.sampling.fibonacci}',
        f'n_rays: {cfg.sampling.n_rays}',
        f'theta_resolution: {cfg.sampling.num_theta_samples}',
        f'phi_resolution: {cfg.sampling.num_phi_samples}',
        f'DoA noise: {cfg.sampling.doa_noise}',
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
    if args.data_tag == 'argos':
        dataset_file = './NeRF2-main/data/MIMO/csidata.npy'
        loader_train = ArgosDataLoader(dataset_file=dataset_file, train=True, ratio=0.8, mask_ratio=args.mask_ratio)
        loader_test = ArgosDataLoader(dataset_file=dataset_file, train=False)
        disp_logger.info('\n****************************** Dataloader ******************************')
        disp_logger.info(f'==> dataset: {dataset_file}')
    else:
        train_size = {'conferenceroom': 10000, 'bedroom': 20000, 'office': 30000}
        test_size = {'conferenceroom': 2000, 'bedroom': 4000, 'office': 6000}
        dataset_train = os.path.join(data_dir, f"{args.env}_{args.data_tag}{train_size[args.env]}.pkl")
        dataset_test = os.path.join(data_dir, f"{args.env}_{args.data_tag}{test_size[args.env]}.pkl")
        loader_train = DatasetLoader(dataset_train, mask_ratio=args.mask_ratio)
        loader_test = DatasetLoader(dataset_test, mask_ratio=args.mask_ratio)
        disp_logger.info('\n****************************** Dataloader ******************************')
        disp_logger.info(f'==> training dataset: {dataset_train} || size: {len(loader_train.index)}')
        disp_logger.info(f'==> testing dataset: {dataset_test} || size: {len(loader_test.index)}')
    # save path
    if save_dir:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        disp_logger.info('\n****************************** Save Path ******************************')
        disp_logger.info(f"==> Checkpoint will be saved at: {save_dir}/")
    
    return loader_train, loader_test, file_logger, disp_logger, device


def train_step(args, cfg, model, loader, encode, encode_viewdirs, optimizer, scheduler, synthesizer, device, file_logger, disp_logger, epoch, history_epoch):
    model.train()
    batch_size = cfg.training.batch_size
    n_iters = math.ceil(len(loader.index) / batch_size)
    random_idx = np.random.choice(loader.index, len(loader.index))
    pbar = tqdm(total=n_iters, unit='iter', colour='green')
    loss_total = 0
    snr_list = torch.zeros([0]).to(device)
    epoch_start_time = time.time()
    for i in range(n_iters):
        # Run the training pipeline
        # sta_id = np.random.choice(loader.index, cfg.training.batch_size)
        sta_id = random_idx[i * batch_size:(i * batch_size + min(batch_size, len(loader.index) - i * batch_size))]
        (loss, snr) = pipeline(args, cfg, sta_id, loader, model, encode, encode_viewdirs, optimizer, synthesizer, device, mode='Train')
        
        loss_total += loss
        snr_list = torch.cat([snr_list, snr], dim=0)
        loss_mean = loss_total / (i + 1)
        snr_mean = torch.mean(snr_list)
        pbar.set_description(f"Epoch [{epoch + history_epoch + 1}/{args.epochs + history_epoch}] || Training")
        pbar.set_postfix({
            "|| LR": f"{scheduler.get_last_lr()[0]:.3e}",
            '|| loss_mean': f'{loss_mean:.4f}',
            '|| snr_mean': f'{snr_mean:.4f} dB'})
        pbar.update(1)
    pbar.close()
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    mins, secs = divmod(epoch_duration, 60)
    epoch_duration_str = f"{int(mins):02d}:{int(secs):02d}"
    # output log
    log = ' || '.join(['Training',
        f'Epoch [{epoch + history_epoch + 1}/{args.epochs + history_epoch}]',
        f'LR={scheduler.get_last_lr()[0]:.3e}',
        f'Loss_Mean={loss_mean:.4f}',
        f'SNR_Mean={snr_mean:.4f} dB',
        f'Epoch_Duration={epoch_duration_str} mins:secs',
    ])
    file_logger.info(log)
    
    return snr_list


def test_step(args, cfg, model, loader, encode, encode_viewdirs, optimizer, scheduler, synthesizer, device, file_logger, disp_logger, epoch, best_score, history_epoch):
    model.eval()
    # batch_size = cfg.training.batch_size
    batch_size = cfg.training.batch_size
    n_iters = math.ceil(len(loader.index) / batch_size)
    pbar = tqdm(total=n_iters, unit='iter', colour='yellow', disable=False)
    snr_list = torch.zeros([0]).to(device)
    sgcs_list = torch.zeros([0]).to(device)
    loss_total = 0
    epoch_start_time = time.time()
    with torch.no_grad():
        for i in range(n_iters):
            sta_id = i * batch_size + np.arange(min(batch_size, len(loader.index) - i * batch_size))
            (loss, snr, sgcs) = pipeline(args, cfg, sta_id, loader, model, encode, encode_viewdirs, optimizer, synthesizer, device, mode='Validate')
            
            loss_total += loss
            snr_list = torch.cat([snr_list, snr], dim=0)
            sgcs_list = torch.cat([sgcs_list, sgcs], dim=0)
            snr_mean = torch.mean(snr_list)
            snr_mid = torch.median(snr_list)
            sgcs_mean = torch.mean(sgcs_list)
            sgcs_mid = torch.median(sgcs_list)
            pbar.set_description(f"Epoch [{epoch + history_epoch + 1}/{args.epochs + history_epoch}] || Validating")
            pbar.set_postfix({
            '|| snr_mean': f'{snr_mean:.4f} dB',
            '|| snr_mid': f'{snr_mid:.4f} dB',
            '|| sgcs_mean': f'{sgcs_mean:.4f}',
            '|| sgcs_mid': f'{sgcs_mid:.4f}',})
            pbar.update(1)
        pbar.close()
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    mins, secs = divmod(epoch_duration, 60)
    epoch_duration_str = f"{int(mins):02d}:{int(secs):02d}"
    # output log
    log = ' || '.join(['Validating',
        f'Epoch [{epoch + history_epoch + 1}/{args.epochs + history_epoch}]',
        f'SNR_Mean={snr_mean:.4f} dB',
        f'SNR_Median={snr_mid:.4f} dB',
        f'SGCS_Mean={sgcs_mean:.4f}',
        f'SGCS_Median={sgcs_mid:.4f}',
        f'Epoch_Duration={epoch_duration_str} mins:secs',
    ])
    file_logger.info(log)
    # save a checkpoint at given rate
    if (epoch+1) % 10 == 0:
        disp_logger.info('Saving...')
        trans_suffix = "_trans" if args.trans else ""
        ss_suffix = f'_{args.n_rays}R{args.n_samples}S{args.radial_max}F' if args.sample_res else ''
        ablation_suffix = "" if args.ablation == 'n' else f"_wo{args.ablation}"
        save_path = os.path.join(save_dir, f"{args.model_tag}_{args.filter}{ss_suffix}{trans_suffix}{ablation_suffix}_ckpt_epoch_{history_epoch + epoch + 1}.pt")
        save_ckpt(model, optimizer, scheduler, save_path, history_epoch=(history_epoch+epoch+1), best_score=snr_mean)
        disp_logger.info(f"Checkpoint saved at: {save_path}")
    # save best sheckpoint
    disp_logger.info('Best performance verification...')
    if snr_mean > best_score:
        disp_logger.info('Yes. Saving...')
        trans_suffix = "_trans" if args.trans else ""
        ss_suffix = f'_{args.n_rays}R{args.n_samples}S{args.radial_max}F' if args.sample_res else ''
        ablation_suffix = "" if args.ablation == 'n' else f"_wo{args.ablation}"
        if cfg.training.overwrite:
            save_path = os.path.join(save_dir, f'{args.model_tag}_{args.filter}{ss_suffix}{trans_suffix}{ablation_suffix}_ckpt_best.pt')
        else:
            save_path = os.path.join(save_dir, f'{args.model_tag}_{args.filter}{ss_suffix}{trans_suffix}{ablation_suffix}_ckpt_best_epoch_{history_epoch+epoch+1}.pt')
        save_ckpt(model, optimizer, scheduler, save_path, history_epoch=(history_epoch+epoch+1), best_score=snr_mean)
        disp_logger.info(f"Best checkpoint saved at: {save_path} || Best SNR_Mean: {snr_mean:.4f} dB")
        best_score = snr_mean
    else:
        disp_logger.info('No.')
        
    return loss_total, snr_list, best_score


def main(args, cfg, data_dir, save_dir, ckpt=None):
    # Preliminary
    (loader_train, loader_test, file_logger, disp_logger, device) = preliminary(args, cfg, data_dir, save_dir)
    
    # Initialize models
    (model, encode, encode_viewdirs, optimizer, scheduler, synthesizer, history_epoch, best_score) = init_models(args, cfg, device, disp_logger, ckpt)

    for epoch in range(args.epochs):
        _ = train_step(args, cfg, model, loader_train, encode, encode_viewdirs, optimizer, scheduler,
                            synthesizer, device, file_logger, disp_logger, epoch, history_epoch)
        (loss_total, _, best_score) = test_step(args, cfg, model, loader_test, encode, encode_viewdirs, optimizer, scheduler,
                                                        synthesizer, device, file_logger, disp_logger, epoch, best_score, history_epoch)
        if scheduler is not None:
            scheduler.step(loss_total)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, choices=['conferenceroom', 'bedroom', 'office', 'argos'], default='conferenceroom')
    parser.add_argument('--ckpt', type=str, default='best')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--d_input', nargs='*', type=int, default=[52, 4, 16])
    parser.add_argument('--n_rays', type=int, default=24, help='Number of rays per batch')
    parser.add_argument('--n_samples', type=int, default=32, help='Samples per ray')
    parser.add_argument('--radial_max', type=float, default=9.0, help='Max sampling distance (meters)')
    parser.add_argument('--mask_ratio', type=float, default=0, help='mask training data in argos dataset')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--filter', type=str, default='sd')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0001)
    parser.add_argument('--define', type=str, default='ours')
    parser.add_argument('-mt', '--model_tag', type=str, default='28GHz_SF_FB_lite')
    parser.add_argument('-dt', '--data_tag', type=str, default='28GHz_random')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--ablation', type=str, default='n', choices=['n', 'fca', 'cnn', 'sfg', 'nsfg'])
    parser.add_argument('--mode', type=str, default='fb', choices=['fb', 'eg', 'esnr', 'latency', 'trans', 'flop', 'rate'])
    parser.add_argument('-t', '--trans', action='store_true', default=False, help='fine-tune the conferenceroom model on new scene')
    parser.add_argument('-ss', '--sample_res', action='store_true', default=False, help='record sampling resolution')
    parser.add_argument('-r', '--retrain', action='store_true', default=False)
    args = parser.parse_args()
    
    cfg = OmegaConf.load('./config/default.yaml')
    data_dir = "./simulator/datasets/"
    save_dir = os.path.join(cfg.training.save_dir, f'{args.env}')

    if args.trans:
        ckpt_fname = f'ckpt/conferenceroom/{args.model_tag}_{args.filter}_ckpt_{args.ckpt}.pt'
        if args.retrain:
            ckpt_fname = f'ckpt/{args.env}/{args.model_tag}_{args.filter}_trans_ckpt_{args.ckpt}.pt'
    elif args.retrain:
        ablation_suffix = "" if args.ablation == 'n' else f"_wo{args.ablation}"
        ss_suffix = f'_{args.n_rays}R{args.n_samples}S{args.radial_max}F' if args.sample_res else ''
        ckpt_fname = f'ckpt/{args.env}/{args.model_tag}_{args.filter}{ss_suffix}{ablation_suffix}_ckpt_{args.ckpt}.pt'
    else:
        ckpt_fname = None

    main(args, cfg, data_dir=data_dir, save_dir=save_dir, ckpt=ckpt_fname)
