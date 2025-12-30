import torch
from torch import nn
from typing import Optional, Tuple, List


class Dir_Fusion(nn.Module):
    def __init__(
        self,
        d_viewdirs: int = 27,
        d_cfr: int = 52,
        n_layers: int = 3,
    ):
        super(Dir_Fusion, self).__init__()
        
        self.fuse1 = nn.Sequential(
            nn.Conv2d(in_channels=d_viewdirs+d_cfr, out_channels=d_viewdirs, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(d_viewdirs // 3, d_viewdirs),
            nn.GELU()
        )
        self.fuse2 = ResBlock(in_channels=d_viewdirs, hid_channels=d_viewdirs, kernel_size=3, stride=1)
        self.fuse3 = ResBlock(in_channels=d_viewdirs, hid_channels=d_viewdirs, kernel_size=3, stride=1)
    
    def forward(self, viewdirs, cfr):
        x = torch.cat([viewdirs, cfr], dim=1)  # (chunksize, d_viewdirs+Nc, Nr, Nt)
        x = self.fuse1(x)  # (chunksize, d_viewdirs, Nr, Nt)
        x = self.fuse2(x)  # (chunksize, d_viewdirs, Nr, Nt)
        x = self.fuse3(x)  # (chunksize, d_viewdirs, Nr, Nt)
        return x
    

class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        hid_channels: int = 128,
        kernel_size: int = 3,
        stride: int = 1,
    ):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hid_channels, kernel_size=kernel_size, stride=stride, padding=1, bias=False)
        self.ln1 = nn.GroupNorm(1, hid_channels)
        self.conv2 = nn.Conv2d(hid_channels, hid_channels, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.ln2 = nn.GroupNorm(1, hid_channels)
        self.relu = nn.GELU()
        if stride != 1 or in_channels != hid_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, hid_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(1, hid_channels)
            )
        else:
            self.shortcut = nn.Sequential()
            
    def forward(self, x):
        out = self.relu(self.ln1(self.conv1(x)))
        out = self.ln2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class FreqAT(nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        fc_channels: int = 26,
        hid_channels: int = 128,
    ):
        super(FreqAT, self).__init__()
        self.fc_channels = fc_channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels + fc_channels, hid_channels)
        self.fc2 = nn.Linear(hid_channels, hid_channels)
        self.split = nn.Linear(hid_channels, 2*in_channels)
        self.ln1 = nn.GroupNorm(1, hid_channels)
        self.ln2 = nn.GroupNorm(1, hid_channels)
        self.relu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
            
    def forward(self, inputs, fc_vec):
        N, C, H, W = inputs.shape                                               # (chunksize, n_inputs, Nr, Nt)
        w = self.pool(inputs).squeeze()                                         # (chunksize, n_inputs)
        fc_vec = fc_vec[None, self.fc_channels:].expand(N, self.fc_channels)    # (chunksize, n_fc)
        w = torch.cat((w, fc_vec), dim=1)                                       # (chunksize, n_inputs + n_fc)
        w = self.relu(self.ln1(self.fc1(w)))                                    # (chunksize, n_filters)
        w = self.relu(self.ln2(self.fc2(w)))                                    # (chunksize, n_filters)
        w = self.split(w)                                                       # (chunksize, 2 * n_inputs)
        alpha, beta = w[:, :C], w[:, C:]
        out = inputs * alpha.view(N, C, 1, 1) + beta[..., None, None].expand(inputs.shape)      # (chunksize, n_inputs)
        return out


class CFR_ShapingFilter(nn.Module):
    def __init__(
        self,
        d_input: tuple = (104, 4, 16),
        d_shaping: int = 128,
        n_samples: int = 128,
        d_viewdirs: int = 27
    ):
        super(CFR_ShapingFilter, self).__init__()
        
        self.n_samples = n_samples
        self.d_viewdirs = d_viewdirs
        (self.Nc, self.Nr, self.Nt) = d_input
        
        self.voxel_shaping = nn.Sequential(
            nn.Conv2d(in_channels=self.n_samples+self.n_samples, out_channels=d_shaping, kernel_size=3, padding=1, stride=1),
            nn.GroupNorm(d_shaping // 4, d_shaping),
            nn.GELU(),
            nn.Conv2d(in_channels=d_shaping, out_channels=self.n_samples, kernel_size=3, padding=1, stride=1),
            nn.GroupNorm(self.n_samples // 4, self.n_samples),
            nn.GELU(),
        )
        self.dir_shaping = nn.Sequential(
            nn.Conv2d(in_channels=self.Nc+self.d_viewdirs, out_channels=d_shaping, kernel_size=3, padding=1, stride=1),
            nn.GroupNorm(d_shaping // 4, d_shaping),
            nn.GELU(),
            nn.Conv2d(in_channels=d_shaping, out_channels=self.Nc, kernel_size=3, padding=1, stride=1),
            nn.GroupNorm(self.Nc // 4, self.Nc),
            nn.GELU(),
        )
        self.out = nn.Conv2d(in_channels=self.Nc, out_channels=self.Nc//2, kernel_size=3, stride=1, padding=1)
        # self.out2 = nn.Conv2d(in_channels=self.Nc, out_channels=self.Nc//2, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x, z_vals, viewdirs, filter='sd'):                            # (chunksize, 2Nc, Nr, Nt)
        """
            x : (chunksize, 2Nc, Nr, Nt). Uplink CFRs
            z_vals : (chunksize, 2Nc, Nr, Nt). voxel indicator
            viewdirs : (chunksize, d_viewdirs, Nr, Nt). Ray-direction indicator
        """
        if 'd' in filter:
            x = torch.cat([x, viewdirs], dim=1)                                     # (chunksize, 2Nc+d_viewdirs, Nr, Nt)
            x = self.dir_shaping(x)                                                 # (chunksize, 2Nc, Nr, Nt)
        if 's' in filter:
            x = x.view(-1, self.n_samples, self.Nc, self.Nr * self.Nt)              # (n_rays, n_samples, 2Nc, NrNt)
            z_vals = z_vals.view(-1, self.n_samples, self.Nc, self.Nr * self.Nt)    # (n_rays, n_samples, 2Nc, NrNt)
            x = torch.cat([x, z_vals], dim=1)                                       # (n_rays, 2*n_samples, 2Nc, NrNt)
            x = self.voxel_shaping(x)                                               # (n_rays, n_samples, 2Nc, NrNt)
            x = x.view(-1, self.Nc, self.Nr, self.Nt)                               # (chunksize, 2Nc, Nr, Nt)
        
        x = self.out(x)                                                             # (chunksize, Nc, Nr, Nt)
        return x
    

class F4CKM(nn.Module):

    def __init__(
        self,
        d_input: tuple = (52, 4, 16),
        n_samples: int = 128,
        n_blocks: int = 3,
        n_filters: int = 128,
        d_viewdirs: Optional[int] = None
    ):
        super().__init__()
        (self.Nc, self.Nr, self.Nt) = d_input
        self.d_input = (2 * self.Nc, self.Nr, self.Nt)
        self.n_samples = n_samples
        # d_input[0] *= 2  # real and image part concatenated
        self.act = nn.GELU()
        self.d_viewdirs = d_viewdirs
        self.d_freq = 0

        # pre-attenuation layers
        self.cfr_encoder = CFR_ShapingFilter(n_samples=n_samples, d_input=self.d_input)                                             # (chunksize, Nc, Nr, Nt)
        self.dir_fusion = Dir_Fusion(d_viewdirs=d_viewdirs, d_cfr=self.Nc)                                                          # (chunksize, d_viewdirs, Nr, Nt)
        
        # attenuation layers
        self.conv1 = nn.Sequential(                                                                                                 # (chunksize, n_filters, Nr, Nt)
            nn.Conv2d(in_channels=self.Nc, out_channels=n_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(n_filters // 4, n_filters),
            nn.GELU()
        )
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(ResBlock(in_channels=n_filters, hid_channels=n_filters, kernel_size=3, stride=1))
            self.blocks.append(FreqAT(in_channels=n_filters, fc_channels=self.Nc, hid_channels=n_filters))
        self.alpha_out = nn.Conv2d(in_channels=n_filters, out_channels=self.Nc, kernel_size=3, stride=1, padding=1)                 # (chunksize, Nc, Nr, Nt)
        self.downsampling = nn.AdaptiveAvgPool2d(1)
        
        # bottleneck layers
        if self.d_viewdirs is not None:
            self.branch = nn.ModuleList()
            self.branch.append(ResBlock(in_channels=n_filters+self.d_viewdirs, hid_channels=n_filters, kernel_size=3, stride=1))    # (chunksize, n_filters, Nr, Nt)
            self.branch.append(FreqAT(in_channels=n_filters, fc_channels=self.Nc, hid_channels=n_filters))                          # (chunksize, n_filters, Nr, Nt)
            
        self.realimag_filter = nn.ModuleList()
        self.realimag_filter.append(ResBlock(in_channels=n_filters, hid_channels=2*n_filters, kernel_size=3, stride=1))             # (chunksize, 2*n_filters, Nr, Nt)
        self.realimag_filter.append(FreqAT(in_channels=2*n_filters, fc_channels=self.Nc, hid_channels=n_filters))                   # (chunksize, 2*n_filters, Nr, Nt)
        
        self.output = nn.Conv2d(in_channels=2*n_filters, out_channels=2*self.Nc, kernel_size=3, stride=1, padding=1)                # (chunksize, 2Nc, Nr, Nt)
    
    def forward(
        self,
        uplink_cfr: Optional[torch.Tensor] = None,          # (chunksize, 2Nc, Nr, Nt)
        z_vals: Optional[torch.Tensor] = None,              # (chunksize, 2Nc, Nr, Nt)
        viewdirs: Optional[torch.Tensor] = None,            # (chunksize, d_viewdirs, Nr, Nt)
        fc_vec: Optional[torch.Tensor] = None,              # (2Nc)
        filter: str = 'sd',
    ) -> torch.Tensor:
        r"""
        Forward pass with optional view direction.
        """
        # Cannot use viewdirs if instantiated with d_viewdirs = None
        if self.d_viewdirs is None and viewdirs is not None:
            raise ValueError(
                'Cannot input x_direction if d_viewdirs was not given.')

        # Pre-process
        x = self.cfr_encoder(uplink_cfr, z_vals, viewdirs, filter=filter)
        
        # Apply forward pass up to bottleneck
        x = self.conv1(x)
        for i, block in enumerate(self.blocks):
            if i % 2 == 0:
                x = block(x)
            else:
                x = block(x, fc_vec)

        # Apply bottleneck
        if self.d_viewdirs is not None:
            # Split alpha from network output
            alpha = self.alpha_out(x)                               # (chunksize, Nc, Nr, Nt)
            # alpha = self.downsampling(alpha)                        # (chunksize, Nc)
            # alpha = alpha.expand(uplink_cfr.shape[0], self.Nc, self.Nr, self.Nt)  # (chunksize, Nc, Nr, Nt)

            # Pass through bottleneck to get real and imag parts
            # viewdirs = self.dir_fusion(viewdirs, cfr_feature2)    # (chunksize, d_viewdirs, Nr, Nt)
            x = torch.concat([x, viewdirs], dim=-3)                 # (chunksize, d_viewdirs+n_filters, Nr, Nt)
            for i, block in enumerate(self.branch):
                if i % 2 == 0:
                    x = block(x)
                else:
                    x = block(x, fc_vec)                            # (chunksize, n_filters, Nr, Nt)
            for i, block in enumerate(self.realimag_filter):
                if i % 2 == 0:
                    x = block(x)
                else:
                    x = block(x, fc_vec)                            # (chunksize, 2*n_filters, Nr, Nt)
            x = self.output(x)                                      # (chunksize, 2Nc, Nr, Nt)

            # Concatenate alphas to output
            x = torch.concat([x, alpha], dim=-3)                    # (chunksize, 3Nc)
        else:
            # Simple output
            x = self.realimag_filter(x)                             # (chunksize, 2*n_filter, Nr, Nt)
            x = self.output(x)                                      # (chunksize, 3Nc, Nr, Nt)
        return x


class Renderer(torch.nn.Module):
    def __init__(
        self,
        d_input: tuple = (52, 4, 16),
        n_samples: int = 128,
    ):
        super(Renderer, self).__init__()
        (self.Nc, self.Nr, self.Nt) = d_input
        self.d_input = (2 * self.Nc, self.Nr, self.Nt)
        self.n_samples = n_samples
    
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
    
    def forward(self, x, z_vals, fc_vec):
        x = x.reshape(-1, self.n_samples, 3 * self.Nc, self.Nr, self.Nt)
        alpha_raw = x[..., int(2*self.Nc):int(3*self.Nc), :, :]
        z_vals = z_vals.reshape(-1, self.n_samples, 2 * self.Nc, self.Nr, self.Nt)
        z_vals = z_vals[..., :self.Nc, :, :]
        dists = z_vals[:, 1:, ...] - z_vals[:, :-1, ...]
        dists = torch.cat([dists, 1e-10 * torch.ones_like(dists[:, :1, ...])], dim=1)
        alpha = 1.0 - torch.exp(- alpha_raw)
        fc_down = fc_vec[self.Nc:int(2*self.Nc)].view(1, 1, self.Nc, 1, 1).expand(alpha.shape)
        phs_shift = torch.exp(-1j * (2 * torch.pi * fc_down * 1e9/299792458) * dists)
        coeffs = alpha * self.cumprod_exclusive((1. - alpha) * phs_shift + 1e-10, dim=1)
        amp_decay = 299792458/(z_vals * fc_down * 1e9 * 4 * torch.pi)
        re_ch = x[..., 0:self.Nc, :, :]
        im_ch = x[..., self.Nc:int(2*self.Nc), :, :]
        syn_cfr = torch.sum((re_ch + 1j * im_ch) * amp_decay * coeffs, dim=1)
        return syn_cfr
    
    
class F4CKM_FLOP(F4CKM):
    def __init__(
        self,
        d_input: tuple = (52, 4, 16),
        n_samples: int = 128,
        n_blocks: int = 3,
        n_filters: int = 128,
        d_viewdirs: Optional[int] = None
    ):
        super().__init__(
            d_input=d_input,
            n_samples=n_samples,
            n_blocks=n_blocks,
            n_filters=n_filters,
            d_viewdirs=d_viewdirs
        )
        self.renderer = Renderer(d_input=d_input, n_samples=n_samples)
        
    def forward(
        self,
        uplink_cfr: Optional[torch.Tensor] = None,          # (chunksize, 2Nc, Nr, Nt)
    ) -> torch.Tensor:
        chunksize = uplink_cfr.shape[0]
        (Nc, Nr, Nt) = (self.Nc, self.Nr, self.Nt)
        z_vals = torch.randn([chunksize, (2*Nc), Nr, Nt])
        viewdirs = torch.randn([chunksize, 27, Nr, Nt])
        fc_vec = torch.randn([2*Nc])
        filter = 'sd'

        x = self.cfr_encoder(uplink_cfr, z_vals, viewdirs, filter=filter)
        x = self.conv1(x)
        for i, block in enumerate(self.blocks):
            if i % 2 == 0:
                x = block(x)
            else:
                x = block(x, fc_vec)
        if self.d_viewdirs is not None:
            alpha = self.alpha_out(x)                               # (chunksize, Nc, Nr, Nt)
            x = torch.concat([x, viewdirs], dim=-3)                 # (chunksize, d_viewdirs+n_filters, Nr, Nt)
            for i, block in enumerate(self.branch):
                if i % 2 == 0:
                    x = block(x)
                else:
                    x = block(x, fc_vec)                            # (chunksize, n_filters, Nr, Nt)
            for i, block in enumerate(self.realimag_filter):
                if i % 2 == 0:
                    x = block(x)
                else:
                    x = block(x, fc_vec)                            # (chunksize, 2*n_filters, Nr, Nt)
            x = self.output(x)                                      # (chunksize, 2Nc, Nr, Nt)
            x = torch.concat([x, alpha], dim=-3)                    # (chunksize, 3Nc)
        else:
            x = self.realimag_filter(x)                             # (chunksize, 2*n_filter, Nr, Nt)
            x = self.output(x)                                      # (chunksize, 3Nc, Nr, Nt)
        
        syn_cfr = self.renderer(x, z_vals, fc_vec)
        
        return syn_cfr


