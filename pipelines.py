
import logging
from torch import nn
import torch
import numpy as np
from ray_gen import RayGenerator
from utils import calculate_channel_gain, calculate_nmse, calculate_rate, calculate_sgcs_with_samples, clip_rays, get_chunks, sample_and_prepare_batches

def pipeline_batch_shaped_queries(
    args, 
    model: nn.Module,
    n_samples: int,
    batches_uplink_cfrs: torch.Tensor,
    batches_z_vals: torch.Tensor,
    batches_viewdirs: torch.Tensor,
):
    Nc, Nr, Nt = args.d_input
    shaped_queries = []
    for batch_uplink_cfrs, batch_z_vals, batch_viewdirs in zip(batches_uplink_cfrs, batches_z_vals, batches_viewdirs):
        shaped_queries.append(model.cfr_encoder(batch_uplink_cfrs, batch_z_vals, batch_viewdirs, filter=args.filter))       # (chunksize, Nc, Nr, Nt)
    z_vals = torch.cat(batches_z_vals, dim=0).view(-1, n_samples, 2 * Nc, Nr, Nt)
    shaped_queries = torch.cat(shaped_queries, dim=0).view(-1, n_samples, Nc, Nr, Nt)
    viewdir_shape = batches_viewdirs[0].shape[1:]
    viewdirs = torch.cat(batches_viewdirs, dim=0).view(-1, n_samples, *viewdir_shape)
    return shaped_queries, z_vals, viewdirs


def pipeline_batch(
    args, 
    model: nn.Module,
    f: torch.Tensor,
    n_samples: int,
    batches_uplink_cfrs: torch.Tensor,
    batches_z_vals: torch.Tensor,
    batches_viewdirs: torch.Tensor,
    synthesis_fn, 
    n_rays_lst,
):
    predictions = []
    for batch_uplink_cfrs, batch_z_vals, batch_viewdirs in zip(batches_uplink_cfrs, batches_z_vals, batches_viewdirs):
        if args.ablation == 'fca':
            predictions.append(model(uplink_cfr=batch_uplink_cfrs, z_vals=batch_z_vals, viewdirs=batch_viewdirs, filter=args.filter))
        else:
            predictions.append(model(uplink_cfr=batch_uplink_cfrs, z_vals=batch_z_vals, viewdirs=batch_viewdirs, fc_vec=f, filter=args.filter))
    cfr_size_real = batches_z_vals[0].shape[1:]
    z_vals = torch.cat(batches_z_vals, dim=0).view(-1, n_samples, cfr_size_real[0], cfr_size_real[1], cfr_size_real[2])
    if args.ablation == 'cnn':
        channel_size = predictions[0].shape[1]
        raw = torch.cat(predictions, dim=0).reshape(-1, cfr_size_real[1], cfr_size_real[2], channel_size).permute(0, 3, 1, 2) # [B, C, H, W]
        raw = raw.reshape(-1, n_samples, channel_size, cfr_size_real[1], cfr_size_real[2])
    else:
        cfr_size_pred = predictions[0].shape[1:]
        raw = torch.cat(predictions, dim=0).view(-1, n_samples, cfr_size_pred[0], cfr_size_pred[1], cfr_size_pred[2])
    syn_cfr = synthesis_fn.synthesize(raw, z_vals, f, cfr_size_real[0]//2, n_rays_lst)
    return syn_cfr


def pipeline(args, cfg, sta_id, loader, model, encode, encode_viewdirs, optimizer, synthesizer, device, mode="Train", esnr='inf'):
    # Pick a station sample from the training set / validation set
    if mode == 'Train':
        model.train()
        
    elif mode == 'Eval':
        model.eval()

    # Sampling configurations
    n_samples=cfg.sampling.n_samples
    chunksize=cfg.training.chunksize
    near = cfg.sampling.near
    far = cfg.sampling.far
    
    # Get ground truth CFR and Station Location
    # sta_loc = torch.tensor(loader.get_loc_batch("STA", sta_id)).to(device)
    sta_loc = torch.zeros((len(sta_id), 3))

    # Get Rays for backtracing
    if args.data_tag != 'argos':
        aoa_lst = loader.get_aoa_batch(sta_id)
        aoa_lst = clip_rays(aoa_lst, cfg.sampling.doa_clip)
    rays_os = []
    rays_ds = []
    n_rays_lst = []
    cfr_size_complex = loader.get_cfr_struct()
    cfr_size_real = [2 * cfr_size_complex[-3]] + list(cfr_size_complex[-2:])
    uplink_cfr = torch.zeros([0,*cfr_size_real]).to(device)
    for i in range(len(sta_id)):
        if args.data_tag != 'argos':
            aoa = aoa_lst[i]
            ray_gen = RayGenerator(sta_loc[i], cfg, device, torch.tensor(aoa, dtype=torch.float32))
        ray_gen = RayGenerator(sta_loc[i], cfg, device)
        rays_o, rays_d = ray_gen.get_rays()
        rays_os.append(rays_o)
        rays_ds.append(rays_d)
        uplink_cfr_i = torch.tensor(loader.get_uplink_cfr_batch([sta_id[i]])).to(device)
        if esnr != 'inf':
            uplink_cfr_i = loader.cfr_restore(uplink_cfr_i, dl=False)
            uplink_cfr_i = torch.stack([torch.real(uplink_cfr_i), torch.imag(uplink_cfr_i)], dim=2).flatten(start_dim=1, end_dim=2).to(torch.float32)
            Nc, Nr, Nt = cfr_size_complex[-3], cfr_size_complex[-2], cfr_size_complex[-1]
            true_power = torch.sum((uplink_cfr_i[:, :Nc, :, :]**2 + uplink_cfr_i[:, Nc:, :, :]**2), dim=(1, 2, 3)) / (Nc * Nr * Nt)
            n_std = torch.sqrt(true_power / 10 ** (esnr/ 10.0) / 2)
            noise = torch.randn(uplink_cfr_i.shape).to(uplink_cfr_i.device) * n_std
            uplink_cfr_i += noise
            uplink_cfr_i = uplink_cfr_i[:, 0::2, :, :] + 1j * uplink_cfr_i[:, 1::2, :, :]
            uplink_cfr_i = loader.cfr_normalize(uplink_cfr_i, dl=False)
        uplink_cfr_i = torch.stack([torch.real(uplink_cfr_i), torch.imag(uplink_cfr_i)], dim=2).flatten(start_dim=1, end_dim=2).to(torch.float32)
        uplink_cfr_i = uplink_cfr_i.unsqueeze(0).repeat(rays_o.shape[0],cfg.sampling.n_samples,1,1,1).view(-1,*cfr_size_real)
        uplink_cfr = torch.cat([uplink_cfr, uplink_cfr_i], dim=0)
        assert(rays_o.shape == rays_d.shape) # [n_rays, 3]
        n_rays_lst.append(rays_o.shape[0])

    batches_uplink_cfrs = get_chunks(uplink_cfr, chunksize=cfg.training.chunksize)
    rays_o = torch.cat(rays_os, dim=0)
    rays_d = torch.cat(rays_ds, dim=0)

    fc = torch.tensor(loader.get_freq()).to(device)
    fc_down = loader.get_freq_center_down()
    # Sampling the rays
    batches_viewdirs, batches_z_vals = sample_and_prepare_batches(fc_down, rays_o, rays_d, near, far,
                                                                  viewdirs_encoding_fn=encode_viewdirs,
                                                                  n_samples=n_samples,
                                                                  cfr_size=cfr_size_real,
                                                                  chunksize=chunksize)
    # Train model
    syn_cfr = pipeline_batch(args, model, fc, n_samples, batches_uplink_cfrs, batches_z_vals, batches_viewdirs, synthesizer, n_rays_lst)
        
    if mode == "Train":
        target_cfr = torch.tensor(loader.get_cfr_batch(sta_id), dtype=torch.complex64).to(device)  # [batch_size, Nc, Nr, Nt]
        loss, snr = calculate_nmse(syn_cfr, target_cfr, define=args.define)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        # res_cfr = loader.cfr_restore(target_cfr, dl=True)
        return float(loss), snr
    elif mode == "Validate":
        target_cfr = torch.tensor(loader.get_cfr_batch(sta_id), dtype=torch.complex64).to(device)  # [batch_size, Nc, Nr, Nt]
        loss, snr = calculate_nmse(syn_cfr, target_cfr, define=args.define)
        sgcs = calculate_sgcs_with_samples(syn_cfr, target_cfr)
        return float(loss), snr, sgcs
    elif mode == "Eval":
        syn_cfr = loader.cfr_restore(syn_cfr, dl=True)
        target_cfr = torch.tensor(loader.get_cfr_batch(sta_id), dtype=torch.complex64).to(device)  # [batch_size, Nc, Nr, Nt]
        target_cfr = loader.cfr_restore(target_cfr, dl=True)
        _, snr = calculate_nmse(syn_cfr, target_cfr, define=args.define)
        sgcs = calculate_sgcs_with_samples(syn_cfr, target_cfr)
        _, rate = calculate_rate(syn_cfr, target_cfr, SNR_dB=10, precoding_type='dft')
        return snr, sgcs, rate
    elif mode == "Eg":
        syn_cfr = loader.cfr_restore(syn_cfr, dl=True)
        target_cfr = torch.tensor(loader.get_cfr_batch(sta_id), dtype=torch.complex64).to(device)  # [batch_size, Nc, Nr, Nt]
        target_cfr = loader.cfr_restore(target_cfr, dl=True)
        uplink_cfr = torch.tensor(loader.get_uplink_cfr_batch(sta_id), dtype=torch.complex64).to(device)
        uplink_cfr = loader.cfr_restore(uplink_cfr, dl=False)
        _, snr = calculate_nmse(syn_cfr, target_cfr, define=args.define)
        shaped_queries, z_vals, viewdirs = pipeline_batch_shaped_queries(args, model, n_samples, batches_uplink_cfrs, batches_z_vals, batches_viewdirs)
        return uplink_cfr, syn_cfr, target_cfr, snr, shaped_queries, z_vals, viewdirs
    elif mode == 'Latency':
        return syn_cfr
    elif mode == 'CGM':
        syn_cfr = loader.cfr_restore(syn_cfr, dl=True)
        target_cfr = torch.tensor(loader.get_cfr_batch(sta_id), dtype=torch.complex64).to(device)  # [batch_size, Nc, Nr, Nt]
        target_cfr = loader.cfr_restore(target_cfr, dl=True)
        gain_pred = calculate_channel_gain(syn_cfr)
        gain_gt = calculate_channel_gain(target_cfr)
        pos = torch.tensor(loader.get_loc_batch("STA", sta_id), dtype=torch.float32).to(device)
        return pos, gain_pred, gain_gt
    

