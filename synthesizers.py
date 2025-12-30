import torch
from torch import nn

class Synthesizer:
    def __init__(self) -> None:
        self.synthesize_strategy = self.volume_render_channel
        self.speedOfLight = 299792458
        self.return_bdc = False

    def synthesize(
            self,     
            raw: torch.Tensor,
            z_vals: torch.Tensor,
            fc: torch.Tensor,
            nc: int,
            ray_batches = None
        ):
        return self.synthesize_strategy(raw, z_vals, fc, nc, ray_batches)

    ####################### Helper functions #########################
    def cumprod_exclusive(
        self,
        tensor: torch.Tensor,
        dim
    ) -> torch.Tensor:
        # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
        cumprod = torch.cumprod(tensor, dim)
        # "Roll" the elements along dimension 'dim' by 1 element.
        cumprod = torch.roll(cumprod, 1, dim)
        # Replace the first element by "0".
        n_rays, n_samples, n_carriers, Nr, Nt = tensor.size()
        cumprod[:, 0, :, :, :] = torch.zeros([n_rays, n_carriers, Nr, Nt])

        return cumprod
    
    #####################################################################
    def volume_render_channel(
        self,
        raw: torch.Tensor,
        z_vals: torch.Tensor,
        fc: torch.Tensor,
        nc: int,
        ray_batches
    ):
        # difference between consecutive elements of `z_vals`. [n_rays, n_samples]
        alpha_raw = raw[..., int(2*nc):int(3*nc), :, :]
        z_vals = z_vals[..., :nc, :, :]
        dists = z_vals[:, 1:, ...] - z_vals[:, :-1, ...]
        dists = torch.cat([dists, 1e-10 * torch.ones_like(dists[:, :1, ...])], dim=1)
        # alpha = 1.0 - torch.exp(- alpha_raw * dists) 
        # alpha = 1.0 - torch.sigmoid(- alpha_raw)
        # alpha = 1.0 - torch.exp(- alpha_raw * 7e-2)
        alpha = 1.0 - torch.exp(- alpha_raw)
        # weights = torch.mean(alpha * self.cumprod_exclusive((1. - alpha) + 1e-10, dim=1), dim=-1)

        fc_down = fc[nc:int(2*nc)].view(1, 1, nc, 1, 1).expand(alpha.shape)
        phs_shift = torch.exp(-1j * (2 * torch.pi * fc_down * 1e9/self.speedOfLight) * dists)
        # compute weight for each sample along each ray. [n_rays, n_samples]
        coeffs = alpha * self.cumprod_exclusive((1. - alpha) * phs_shift + 1e-10, dim=1)
        amp_decay = self.speedOfLight/(z_vals * fc_down * 1e9 * 4 * torch.pi)
        # re_ch = raw[..., 0:52, :, :]  # [n_rays, n_samples, n_carriers]
        # im_ch = raw[..., 52:104, :, :]     # [n_rays, n_samples, n_carriers]
        re_ch = raw[..., 0:nc, :, :]  # [n_rays, n_samples, n_carriers]
        im_ch = raw[..., nc:int(2*nc), :, :]     # [n_rays, n_samples, n_carriers]
        # produce CFR
        sum_along_rays = torch.sum((re_ch + 1j * im_ch) * amp_decay * coeffs, dim=1)  # [n_rays, n_carriers]
        grouped_rays = torch.split(sum_along_rays, ray_batches)
        syn_cfr = torch.stack([torch.sum(grouped_rays[i], dim = 0) for i in range(len(grouped_rays))], dim=0)
        return syn_cfr
    
    

