from argparse import ArgumentParser
import sys
import time
from eval import eval_channel_gain_map, eval_flops, eval_fast_learning_performance, eval_rate_upperbound, eval_sample_resolution_performance, eval_sample_resolution_performance_single_model, eval_transfer_performance, evaluation, example, latency_test
import os
import numpy as np
from omegaconf import OmegaConf
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import matplotlib.pyplot as plt
import seaborn as sns
from loaders import DatasetLoader
from train import preliminary
from utils import init_models

parser = ArgumentParser()
parser.add_argument('--env', type=str, choices=['conferenceroom', 'bedroom', 'office', 'argos'], default='conferenceroom')
parser.add_argument('--ckpt', type=str, default='best_epoch')
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--d_input', nargs='*', type=int, default=[52, 4, 16])
parser.add_argument('--n_rays', type=int, default=24, help='Number of rays per batch')
parser.add_argument('--n_samples', type=int, default=32, help='Samples per ray')
parser.add_argument('--radial_max', type=float, default=9.0, help='Max sampling distance (meters)')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batchsize', type=int, default=1)
parser.add_argument('--mask_ratio', type=float, default=0, help='mask training data in argos dataset')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--filter', type=str, default='sd')
parser.add_argument('--mode', type=str, default='eg', choices=['fb', 'eg', 'esnr', 'latency', 'trans', 'flop', 'rate', 'ss', 'sss', 'cgm'])
parser.add_argument('--define', type=str, default='nerf')
parser.add_argument('-wd', '--weight_decay', type=float, default=0.0001)
parser.add_argument('-mt', '--model_tag', type=str, default='2.4GHz_SF_FB')
parser.add_argument('-dt', '--data_tag', type=str, default='2.4GHz_random')
parser.add_argument('-t', '--trans', action='store_true', default=False, help='do not activate this flag')
parser.add_argument('-r', '--retrain', action='store_true', default=False, help='do not activate this flag')
parser.add_argument('-ss', '--sample_res', action='store_true', default=False, help='record sampling resolution')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--ablation', type=str, default='n', choices=['n', 'fca', 'cnn', 'sfg', 'nsfg', 'fl'])

args = parser.parse_args()
cfg = OmegaConf.load('./config/default.yaml')

if args.mode == 'trans':
    eval_transfer_performance()
    sys.exit(0)
if args.ablation == 'fl':
    eval_fast_learning_performance()
    sys.exit(0)
if args.mode == 'ss':
    eval_sample_resolution_performance(args, cfg)
    sys.exit(0)
if args.mode == 'sss':
    eval_sample_resolution_performance_single_model(args, cfg)
    sys.exit(0)
if args.mode == 'cgm':
    eval_channel_gain_map(args, cfg)
    sys.exit(0)
    
test_tag_tab = {
    'fb': 'FB',
    'eg': 'EG',
    'esnr': 'ESNR',
    'latency': 'LATENCY',
    'trans': 'TRANS',
    'flop': 'FLOP',
    'rate': 'RATE',
    'ss': 'SS'
}
test_tag = test_tag_tab[args.mode]
data_dir = "./simulator/datasets/"

(_, loader_test, file_logger, disp_logger, device) = preliminary(args, cfg, data_dir, test=True, test_tag=test_tag)

if args.mode == 'rate':
    eval_rate_upperbound(args, loader_test, device)
    sys.exit(0)

mode=args.mode
esnr_tab = [0, 3, 6, 9, 12, 15, 18]

# conference model
if args.env == 'conferenceroom':
    ablation_suffix = "" if args.ablation == 'n' else f"_wo{args.ablation}"
    ckpt_conference = f'ckpt/conferenceroom/{args.model_tag}_{args.filter}{ablation_suffix}_ckpt_best.pt'
    (model, encode, encode_viewdirs, optimizer, scheduler, synthesizer, history_epoch, best_score) = init_models(args, cfg, device, disp_logger, ckpt_conference)
    if mode == 'eg':
        input_eg, pred_eg, label_eg, shaped_queries, z_vals, viewdirs = example(args, cfg, model, loader_test, encode, encode_viewdirs, optimizer, synthesizer, device)
        eg_conference = {
            'input_eg': input_eg.cpu().numpy(),
            'pred_eg': pred_eg.cpu().numpy(),
            'label_eg': label_eg.cpu().numpy(),
            'shaped_queries': shaped_queries.cpu().numpy(),
            'z_vals': z_vals.cpu().numpy(),
            'viewdirs': viewdirs.cpu().numpy()
        }
        savedir_conference = f'./performance/conferenceroom/{args.model_tag}_{args.filter}_eg.npy'
        np.save(savedir_conference, eg_conference)
        disp_logger.info(f'==> Simulation results saved to: {savedir_conference}')
    elif mode == 'latency':
        total_list = []
        mean_list = []
        for _ in range(10):
            latency_total, latency_mean = latency_test(args, cfg, model, loader_test, encode, encode_viewdirs, optimizer, synthesizer, device)
            total_list.append(latency_total)
            mean_list.append(latency_mean)
        avg_total = sum(total_list) / len(total_list)
        avg_mean = sum(mean_list) / len(mean_list)
        mins, secs = divmod(avg_total, 60)
        latency_total_str = f"{int(mins)} m {int(secs)} s"
        latency_mean_ms = avg_mean * 1000
        latency_mean_str = f"{latency_mean_ms:.3f} ms"
        disp_logger.info(f'==> Total Runtime (avg): {latency_total_str} || Avg Runtime: {latency_mean_str} || Batchsize: {args.batchsize}')
    elif mode == 'flop':
        flops, macs, params = eval_flops(args, cfg, model)
        disp_logger.info(f'==> FLOPs: {flops} || MACs: {macs} || Params: {params}')
    else:
        (snr_list_conference, sgcs_list_conference, rate_list_conference) = evaluation(args, cfg, model, loader_test, encode, encode_viewdirs, optimizer, synthesizer, device,
                                                                mode=mode, esnr_tab=esnr_tab)
        pfm_conference = {
            'snr_list': snr_list_conference.cpu().numpy(),
            'sgcs_list': sgcs_list_conference.cpu().numpy(),
            'rate_list': rate_list_conference.cpu().numpy()
        }
        if mode == 'fb':
            savedir_conference = f'./performance/conferenceroom/{args.model_tag}_{args.filter}{ablation_suffix}.npy'
        else:
            savedir_conference = f'./performance/conferenceroom/{args.model_tag}_{args.filter}{ablation_suffix}_{test_tag}.npy'
        np.save(savedir_conference, pfm_conference)
        disp_logger.info(f'==> Simulation results saved to: {savedir_conference}')
# bedroom model
elif args.env == 'bedroom':
    ablation_suffix = "" if args.ablation == 'n' else f"_wo{args.ablation}"
    if mode == 'trans':
        ckpt_bedroom = f'ckpt/conferenceroom/{args.model_tag}_{args.filter}{ablation_suffix}_ckpt_best.pt'
    else:
        ckpt_bedroom = f'ckpt/bedroom/{args.model_tag}_{args.filter}{ablation_suffix}_ckpt_best.pt'
    (model, encode, encode_viewdirs, optimizer, scheduler, synthesizer, history_epoch, best_score) = init_models(args, cfg, device, disp_logger, ckpt_bedroom)
    (snr_list_bedroom, sgcs_list_bedroom, rate_list_bedroom) = evaluation(args, cfg, model, loader_test, encode, encode_viewdirs, optimizer, synthesizer, device,
                                                       mode=mode, esnr_tab=esnr_tab)
    pfm_bedroom = {
        'snr_list': snr_list_bedroom.cpu().numpy(),
        'sgcs_list': sgcs_list_bedroom.cpu().numpy(),
        'rate_list': rate_list_bedroom.cpu().numpy()
    }
    if mode == 'fb':
        savedir_bedroom = f'./performance/bedroom/{args.model_tag}_{args.filter}{ablation_suffix}.npy'
    else:
        savedir_bedroom = f'./performance/bedroom/{args.model_tag}_{args.filter}{ablation_suffix}_{test_tag}.npy'

    np.save(savedir_bedroom, pfm_bedroom)
    disp_logger.info(f'==> Simulation results saved to: {savedir_bedroom}')
# office model
elif args.env =='office':
    ablation_suffix = "" if args.ablation == 'n' else f"_wo{args.ablation}"
    if mode == 'trans':
        ckpt_office = f'ckpt/conferenceroom/{args.model_tag}_{args.filter}{ablation_suffix}_ckpt_best.pt'
    else:
        ckpt_office = f'ckpt/office/{args.model_tag}_{args.filter}{ablation_suffix}_ckpt_best.pt'
    (model, encode, encode_viewdirs, optimizer, scheduler, synthesizer, history_epoch, best_score) = init_models(args, cfg, device, disp_logger, ckpt_office)
    (snr_list_office, sgcs_list_office, rate_list_office) = evaluation(args, cfg, model, loader_test, encode, encode_viewdirs, optimizer, synthesizer, device,
                                                     mode=mode, esnr_tab=esnr_tab)
    pfm_office = {
        'snr_list': snr_list_office.cpu().numpy(),
        'sgcs_list': sgcs_list_office.cpu().numpy(),
        'rate_list': rate_list_office.cpu().numpy()
    }
    ablation_suffix = "" if args.ablation == 'n' else f"_wo{args.ablation}"
    if mode == 'fb':
        savedir_office = f'./performance/office/{args.model_tag}_{args.filter}{ablation_suffix}.npy'
    else:
        savedir_office = f'./performance/office/{args.model_tag}_{args.filter}{ablation_suffix}_{test_tag}.npy'
    np.save(savedir_office, pfm_office)
    disp_logger.info(f'==> Simulation results saved to: {savedir_office}')
# argos model
elif args.env =='argos':
    ablation_suffix = "" if args.ablation == 'n' else f"_wo{args.ablation}"
    ckpt_argos = f'ckpt/argos/{args.model_tag}_{args.filter}{ablation_suffix}_ckpt_best.pt'
    (model, encode, encode_viewdirs, optimizer, scheduler, synthesizer, history_epoch, best_score) = init_models(args, cfg, device, disp_logger, ckpt_argos)
    if mode == 'eg':
        input_eg, pred_eg, label_eg = example(args, cfg, model, loader_test, encode, encode_viewdirs, optimizer, synthesizer, device)
        eg_argos = {
            'input_eg': input_eg.cpu().numpy(),
            'pred_eg': pred_eg.cpu().numpy(),
            'label_eg': label_eg.cpu().numpy()
        }
        savedir_argos = f'./performance/argos/{args.model_tag}_{args.filter}_eg.npy'
        np.save(savedir_argos, eg_argos)
        disp_logger.info(f'==> Simulation results saved to: {savedir_argos}')
    elif mode == 'latency':
        total_list = []
        mean_list = []
        for _ in range(10):
            latency_total, latency_mean = latency_test(args, cfg, model, loader_test, encode, encode_viewdirs, optimizer, synthesizer, device)
            total_list.append(latency_total)
            mean_list.append(latency_mean)
        avg_total = sum(total_list) / len(total_list)
        avg_mean = sum(mean_list) / len(mean_list)
        mins, secs = divmod(avg_total, 60)
        latency_total_str = f"{int(mins)} m {int(secs)} s"
        latency_mean_ms = avg_mean * 1000
        latency_mean_str = f"{latency_mean_ms:.3f} ms"
        disp_logger.info(f'==> Total Runtime (avg): {latency_total_str} || Avg Runtime: {latency_mean_str} || Batchsize: {args.batchsize}')
    elif mode == 'flop':
        flops, macs, params = eval_flops(args, cfg, model)
        disp_logger.info(f'==> FLOPs: {flops} || MACs: {macs} || Params: {params}')
    else:
        (snr_list_argos, sgcs_list_argos, rate_list_argos) = evaluation(args, cfg, model, loader_test, encode, encode_viewdirs, optimizer, synthesizer, device,
                                                        mode=mode, esnr_tab=esnr_tab)
        pfm_argos = {
            'snr_list': snr_list_argos.cpu().numpy(),
            'sgcs_list': sgcs_list_argos.cpu().numpy(),
            'rate_list': rate_list_argos.cpu().numpy()
        }
        if mode == 'fb':
            savedir_argos = f'./performance/argos/{args.model_tag}_{args.filter}{ablation_suffix}.npy'
        else:
            savedir_argos = f'./performance/argos/{args.model_tag}_{args.filter}{ablation_suffix}_{test_tag}.npy'
        np.save(savedir_argos, pfm_argos)
        disp_logger.info(f'==> Simulation results saved to: {savedir_argos}')