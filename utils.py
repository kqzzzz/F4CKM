import numpy as np
from omegaconf import OmegaConf
import torch
from typing import Optional, Tuple, List, Union, Callable
import logging
from models_ablation import F4CKM_WO_CNN, F4CKM_WO_FCA
from samplers import sample_stratified, sample_hierarchical
import logging.config
from models import * 
from synthesizers import Synthesizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from encoders import PositionalEncoder


def calculate_nmse(H_est: torch.Tensor, H_true: torch.Tensor, define='nerf') -> float:
        
    error = torch.abs(H_est - H_true) ** 2  # (batchsize, Nc, Nr, Nt)
    true_energy = torch.abs(H_true) ** 2  # (batchsize, Nc, Nr, Nt)
    if define == 'nerf':
        nmse_per_sample = torch.mean(error, dim=(1,2,3)) / torch.mean(true_energy, dim=(1,2,3))
        nmse_loss = torch.mean(nmse_per_sample)
        snr_per_sample = -10. * torch.log10(nmse_per_sample)
    else:
        error_per_sample_sc = torch.sum(error, dim=(2, 3))  # (batchsize, Nc)
        energy_per_sample_sc = torch.sum(true_energy, dim=(2, 3))  # (batchsize, Nc)
        nmse_per_sample_sc = error_per_sample_sc / (energy_per_sample_sc + 1e-12)  # (batchsize, Nc)

        nmse_per_sample = torch.mean(nmse_per_sample_sc, dim=1)  # (batchsize,)
        nmse_loss = torch.mean(nmse_per_sample)
        snr_per_sample = -10. * torch.log10(nmse_per_sample)  # (batchsize,)
    
    return nmse_loss, snr_per_sample


def calculate_sgcs_with_samples(H_est: torch.Tensor, H_true: torch.Tensor) -> tuple:
    
    batch_size, num_subcarriers, _, _ = H_est.shape
    device = H_true.device
    
    if len(H_est) < len(H_true):
        H_true = H_true[:len(H_est), :, :, :]
        
    sgcs_samples = torch.zeros(batch_size, dtype=torch.float32, device=device)

    for b in range(batch_size):
        sgcs_sample = torch.tensor(0.0, dtype=torch.float32, device=device)
        
        for t in range(num_subcarriers):
            H_true_sub = H_true[b, t]  # (Nr, Nt)
            H_est_sub = H_est[b, t]    # (Nr, Nt)

            # calculate the ideal eigen vecor
            _, _, Vh_true = torch.linalg.svd(H_true_sub, full_matrices=False)
            w_true = Vh_true[0].conj()
            _, _, Vh_est = torch.linalg.svd(H_est_sub, full_matrices=False)
            w_est = Vh_est[0].conj()
            # calculate SGCS
            inner_product = torch.vdot(w_true.flatten(), w_est.flatten())
            numerator = torch.abs(inner_product) ** 2
            denominator = (torch.norm(w_true)**2) * (torch.norm(w_est)**2)
            sgcs_sample += (numerator / denominator).float()

        sgcs_samples[b] = sgcs_sample / num_subcarriers
    
    return sgcs_samples


def calculate_rate(H_est, H_true, alpha=0, SNR_dB=10, precoding_type='dft'):
    if H_true.shape != H_est.shape:
        raise ValueError("H_true and H_est must have the same shape")
    
    # Normalize H_true and H_est
    pl = torch.sqrt((H_true.abs() ** 2).mean(dim=(1, 2, 3), keepdim=True))
    H_true = H_true / pl
    H_est = H_est / pl
    
    SNR_linear = 10 ** (SNR_dB / 10)
    batchsize, Nc, Nr, Nt = H_true.shape
    rate_list = torch.empty(batchsize, device=H_true.device, dtype=torch.float32)
    
    if precoding_type == 'mrt':
        # Batch processing for MRT
        if Nr == 1:
            # MRT for Nr = 1
            H_est_flat = H_est.view(batchsize, Nc, -1)  # Flatten H_est to [batchsize, Nc, Nt]
            P = H_est_flat.conj() / torch.norm(H_est_flat, dim=2, keepdim=True)  # Normalize precoder
            H_eq = (H_true.view(batchsize, Nc, -1) * P).sum(dim=2)  # Equivalent channel
            channel_gain = torch.abs(H_eq) ** 2
            rate_list = torch.log2(1 + SNR_linear * channel_gain).mean(dim=1)  # Average over Nc
        else:
            # MRT for Nr > 1
            H_est_H = H_est.conj().transpose(-1, -2)  # [batchsize, Nc, Nt, Nr]
            P = H_est_H / torch.norm(H_est_H, dim=2, keepdim=True)  # Normalize MRT precoder
            H_eq = torch.matmul(H_true, P)  # [batchsize, Nc, Nr, Nr]
            matrix = torch.eye(Nr, dtype=torch.complex64, device=H_true.device) + \
                     SNR_linear * torch.matmul(H_eq, H_eq.conj().transpose(-1, -2))
            det_matrix = torch.linalg.det(matrix).real
            rate_list = torch.log2(det_matrix).mean(dim=1)  # Average over Nc
    
    elif precoding_type == 'dft':
        F = torch.fft.fft(torch.eye(Nt, dtype=torch.cfloat, device=H_true.device), dim=0) / np.sqrt(Nt)
        W = torch.fft.fft(torch.eye(Nr, dtype=torch.cfloat, device=H_true.device), dim=0) / np.sqrt(Nr)
        # Batch processing for DFT
        for b in range(batchsize):
            G_3d = (W.conj().T @ H_est[b] @ F).abs()**2          # [Nc, Nr, Nt]
            G_all = G_3d.mean(dim=0)                             # [Nr, Nt]
            best_gain, idx = G_all.view(-1).max(0)
            best_rx, best_tx = divmod(idx.item(), Nt)

            p_tx = F[:, best_tx].view(Nt, 1)
            p_rx = W[:, best_rx].view(Nr, 1)
            h_eq_all = (p_rx.conj().T @ H_true[b] @ p_tx).squeeze()   # [Nc]
            rate_bc = torch.log2(1 + SNR_linear * h_eq_all.abs()**2).mean().item()

            rate_list[b] = rate_bc
    
    effective_rate_list = (1 - alpha) * rate_list
    return effective_rate_list, rate_list


def calculate_channel_gain(H: torch.Tensor) -> float:
    energy_per_sample_sc = torch.sum(torch.abs(H)**2, dim=(2, 3))   # (batchsize, Nc)
    energy_per_sample = torch.mean(energy_per_sample_sc, dim=1)            # (batchsize,)
    channel_gain_dB = 10 * torch.log10(energy_per_sample)              # (batchsize,)
    return channel_gain_dB


def clip_rays(aoa_list, aoa_clip):
    
    if aoa_clip == 'inf':
        return aoa_list
    else:
        for i, aoa in enumerate(aoa_list):
            if len(aoa) > aoa_clip:
                aoa_list[i] = aoa[:aoa_clip]
        return aoa_list


def fibonacci_sphere_sampling(n):

    if n == 0:
        return np.zeros((0, 3))
    if n == 1:
        return np.array([[0.0, 1.0, 0.0]])
    
    golden_angle = np.pi * (3 - np.sqrt(5))
    indices = np.arange(n, dtype=np.float64)
    
    y = 1 - 2 * indices / (n - 1)
    
    radius = np.sqrt(1 - y**2)
    theta = golden_angle * indices
    
    x = radius * np.cos(theta)
    z = radius * np.sin(theta)
    
    return np.column_stack((x, y, z))


def get_chunks(
    inputs: torch.Tensor,
    chunksize: int = 2**15
) -> List[torch.Tensor]:
    r"""
    Divide an input into chunks.
    Borrowed from: Mason McGough 
    Source: https://towardsdatascience.com/its-nerf-from-nothing-build-a-vanilla-nerf-with-pytorch-7846e4c45666
    """
    output = [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]
    # if len(output[-1]) < chunksize:
    #     output = output[:-1]
    
    return output


def prepare_chunks(
    points: torch.Tensor,
    encoding_function: Callable[[torch.Tensor], torch.Tensor],
    chunksize: int = 2**15
) -> List[torch.Tensor]:
    r"""
    Encode and chunkify points to prepare for NeRF model.
    Borrowed from: Mason McGough 
    Source: https://towardsdatascience.com/its-nerf-from-nothing-build-a-vanilla-nerf-with-pytorch-7846e4c45666
    """
    points = points.reshape((-1, 3))
    points = encoding_function(points)
    points = get_chunks(points, chunksize=chunksize)
    return points


import numpy as np

def correct_z_vals(z_vals, rays_d, fc_down):

    n_ray, n_sample, n_c, n_r, n_t = z_vals.shape
    spacing = 299792458 / (fc_down * 1e9) / 2
    rx_m = int(torch.sqrt(torch.tensor(n_r)))
    rx_n = rx_m

    if rx_m == 1 and rx_n == 1:
        return z_vals

    i, j = torch.meshgrid(torch.arange(-rx_m // 2, rx_m // 2), torch.arange(-rx_n // 2, rx_n // 2))
    rx_positions = torch.stack((i * spacing, j * spacing, torch.zeros_like(i)), dim=-1).view(-1, 3).to(rays_d.device)
    array_center = torch.mean(rx_positions, dim=0)
    vectors_to_center = array_center - rx_positions

    z_vals_corrected = z_vals.clone()

    for ray_idx, direction in enumerate(rays_d):
        direction = direction / torch.norm(direction)
        path_length_corrections = torch.sum(vectors_to_center * direction.view(1, 3), dim=1).view(1, 1, -1, 1)

        z_vals_corrected[ray_idx, :, :, :, :] += path_length_corrections.expand(n_sample, n_c, n_r, n_t)

    return z_vals_corrected


def prepare_zvals_chunks(
    fc_down,
    rays_o: torch.Tensor, # [n_ray, 3]
    rays_d: torch.Tensor, # [n_ray, 3]
    near: float,
    far: float,
    n_samples: int,
    cfr_size: np.array = [104, 4, 16],
    chunksize: int = 2048,
    inverse_depth: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:

    # Grab samples for space integration along ray
    t_vals = torch.linspace(0., 1., n_samples, device=rays_o.device)
    if not inverse_depth:
        # Sample linearly between `near` and `far`
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        # Sample linearly in inverse depth (disparity)
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    # Make sure the z_vals starts from 0
    z_vals[0] = 1e-10
    z_vals = z_vals[None, :, None, None, None].expand(rays_o.shape[0],n_samples,*cfr_size)  # (n_ray, n_sample, Nc, Nr, Nt)
    z_vals = correct_z_vals(z_vals, rays_d, fc_down).reshape(-1, *cfr_size)
    z_vals = get_chunks(z_vals, chunksize=chunksize)

    return z_vals


def prepare_viewdirs_chunks(
    rays_d: torch.Tensor,
    encoding_function: Callable[[torch.Tensor], torch.Tensor],
    n_samples: int = 128,
    cfr_size: np.array = [104, 4, 16],
    chunksize: int = 2048
) -> List[torch.Tensor]:

    # Prepare the viewdirs
    viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    viewdirs = viewdirs[:, None, :, None, None].expand(viewdirs.shape[0],n_samples,viewdirs.shape[1],cfr_size[1],cfr_size[2]).permute(0,1,3,4,2)
    trans_shape = viewdirs.shape
    viewdirs = encoding_function(viewdirs.reshape((-1, 3)))
    new_shape = (trans_shape[0]*trans_shape[1],) + trans_shape[2:4] + (viewdirs.shape[-1],)
    viewdirs = viewdirs.reshape(new_shape).permute(0,3,1,2)
    viewdirs = get_chunks(viewdirs, chunksize=chunksize)
    return viewdirs


def sample_and_prepare_batches(
    fc_down,
    rays_o: torch.Tensor, # [n_theta, n_phi, 3]
    rays_d: torch.Tensor, # [n_theta, n_phi, 3]
    near: float,
    far: float,
    viewdirs_encoding_fn: Optional[Callable[[
        torch.Tensor], torch.Tensor]] = None,
    n_samples: int = 128,
    cfr_size: np.array = [104, 4, 16],
    chunksize: int = 2**15,
):
    # Sample query points along each ray.
    batches_z_vals = prepare_zvals_chunks(fc_down, rays_o, rays_d, near, far, n_samples, cfr_size, chunksize)
    # Prepare batches.
    if viewdirs_encoding_fn is not None:
        batches_viewdirs = prepare_viewdirs_chunks(rays_d,
                                                    encoding_function=viewdirs_encoding_fn,
                                                    chunksize=chunksize,
                                                    n_samples=n_samples,
                                                    cfr_size=cfr_size)
    else:
        batches_viewdirs = [None] * len(batches_z_vals)

    return batches_viewdirs, batches_z_vals


def initialize_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0, std=0.01)
        # nn.init.uniform_(module.weight, a=-0.01, b=0.01)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, mean=0, std=0.01)
        # nn.init.uniform_(module.weight, a=-0.01, b=0.01)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


def init_models(args, cfg, device, disp_logger, ckpt=None):
    r"""
    Initialize models, encoders, and optimizer for NeRF training.
    """
    disp_logger.info('\n****************************** Model ******************************')
    if ckpt is None:
        if disp_logger is not None:
            disp_logger.info("==> Model initializing")
    else:
        if disp_logger is not None:
            disp_logger.info(f"==> Loading models from checkpoint: {ckpt}")
        d = torch.load(ckpt, map_location=device, weights_only=True)
        
    # Encoders
    encoder = PositionalEncoder(cfg.encoder.d_input, cfg.encoder.n_freqs, log_space=cfg.encoder.log_space)
    def encode(x): return encoder(x)

    # View direction encoders
    if cfg.encoder.use_viewdirs:
        encoder_viewdirs = PositionalEncoder(cfg.encoder.d_input, cfg.encoder.n_freqs_views,
                                             log_space=cfg.encoder.log_space)

        def encode_viewdirs(x): return encoder_viewdirs(x)
        d_viewdirs = encoder_viewdirs.d_output
    else:
        encode_viewdirs = None
        d_viewdirs = None
    
    # Models
    if args.mode == 'flop':
        if 'lite' in args.model_tag:
            model = F4CKM_FLOP(d_input=args.d_input, n_samples=cfg.sampling.n_samples, n_blocks=cfg.models.n_blocks_lite, n_filters=cfg.models.n_filters_lite, d_viewdirs=d_viewdirs)            
        else:
            model = F4CKM_FLOP(d_input=args.d_input, n_samples=cfg.sampling.n_samples, n_blocks=cfg.models.n_blocks, n_filters=cfg.models.n_filters, d_viewdirs=d_viewdirs)
    elif args.ablation == 'n' or args.ablation == 'sfg' or args.ablation == 'nsfg':
        if 'lite' in args.model_tag:
            model = F4CKM(d_input=args.d_input, n_samples=cfg.sampling.n_samples, n_blocks=cfg.models.n_blocks_lite, n_filters=cfg.models.n_filters_lite, d_viewdirs=d_viewdirs)
        else:
            model = F4CKM(d_input=args.d_input, n_samples=cfg.sampling.n_samples, n_blocks=cfg.models.n_blocks, n_filters=cfg.models.n_filters, d_viewdirs=d_viewdirs)
    elif args.ablation == 'fca':
        model = F4CKM_WO_FCA(d_input=args.d_input, n_samples=cfg.sampling.n_samples, n_blocks=cfg.models.n_blocks, n_filters=cfg.models.n_filters, d_viewdirs=d_viewdirs)
    elif args.ablation == 'cnn':
        model = F4CKM_WO_CNN(d_input=args.d_input, n_samples=cfg.sampling.n_samples, n_blocks=cfg.models.n_blocks, n_filters=cfg.models.n_filters, d_viewdirs=d_viewdirs)
    model.apply(initialize_weights)
    if ckpt is not None:
        pretrained_state_dict = d['model_state_dict']
        new_state_dict = model.state_dict()
        # Only load matched keys
        for key in new_state_dict.keys():
            if key in pretrained_state_dict and new_state_dict[key].shape == pretrained_state_dict[key].shape:
                new_state_dict[key] = pretrained_state_dict[key]
        model.load_state_dict(new_state_dict, strict=False)

    model.to(device)
    model_params = list(model.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    disp_logger.info(f"==> Total Parameters: {total_params/1e6:.2f} M")
    disp_logger.info(f"==> Trainable Parameters: {trainable_params/1e6:.2f} M")

    # Synthesizer
    # Set the synthesizer for the corresponding model
    synthesizer = Synthesizer()
    # Optimizer
    if cfg.optimizer.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model_params, lr=args.lr, weight_decay = args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model_params, lr=args.lr, weight_decay = args.weight_decay)
    if args.retrain:
        optimizer.load_state_dict(d['optimizer_state_dict'])
    # optimizer.param_groups[0]['weight_decay'] = cfg.optimizer.weight_decay
    logging.debug(optimizer)
    if cfg.optimizer.use_lr_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, 'min', 
                                      patience = cfg.optimizer.scheduler_patience, 
                                      factor=cfg.optimizer.scheduler_factor, 
                                      min_lr = cfg.optimizer.min_lr)
    else:
        scheduler = None
    if args.retrain:
        scheduler.load_state_dict(d['scheduler_state_dict'])
    # History training iterations
    if args.retrain:
        history_epoch = d['history_epoch']
    else:
        history_epoch = 0
    # Best performance
    if args.retrain:
        best_score = d['best_score']
    else:
        best_score = 0

    return model, encode, encode_viewdirs, optimizer, scheduler, synthesizer, history_epoch, best_score


def init_logger(cfg, log_path):
    logging_cfg = OmegaConf.to_container(cfg.get("logging", {}), resolve=True)
    logging_cfg["handlers"]["file"]["filename"] = log_path
    logging.config.dictConfig(logging_cfg)
    file_logger = logging.getLogger("file_logger")
    disp_logger = logging.getLogger("disp_logger")
    return file_logger, disp_logger
    
    
def save_ckpt(model, optimizer, scheduler, save_path, history_epoch, best_score):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'history_epoch': history_epoch,
        'best_score': best_score
    }, save_path)
