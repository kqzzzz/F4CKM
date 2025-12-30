import torch
from torch import nn
from typing import Optional, Tuple, List
from models import CFR_ShapingFilter, Dir_Fusion, FreqAT, ResBlock


class ResBlock_MLP(nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        hid_channels: int = 128,
    ):
        super(ResBlock_MLP, self).__init__()
        self.linear1 = nn.Linear(in_channels, hid_channels, bias=False)
        self.ln1 = nn.GroupNorm(1, hid_channels)
        self.linear2 = nn.Linear(hid_channels, hid_channels, bias=False)
        self.ln2 = nn.GroupNorm(1, hid_channels)
        self.relu = nn.GELU()
        if in_channels != hid_channels:
            self.shortcut = nn.Sequential(
                nn.Linear(in_channels, hid_channels, bias=False),
                nn.GroupNorm(1, hid_channels)
            )
        else:
            self.shortcut = nn.Sequential()
            
    def forward(self, x):
        out = self.relu(self.ln1(self.linear1(x)))
        out = self.ln2(self.linear2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out
    

class FreqAT_MLP(nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        fc_channels: int = 26,
        hid_channels: int = 128,
    ):
        super(FreqAT_MLP, self).__init__()
        self.fc_channels = fc_channels
        self.fc1 = nn.Linear(in_channels + fc_channels, hid_channels)
        self.fc2 = nn.Linear(hid_channels, hid_channels)
        self.split = nn.Linear(hid_channels, 2*in_channels)
        self.ln1 = nn.GroupNorm(1, hid_channels)
        self.ln2 = nn.GroupNorm(1, hid_channels)
        self.relu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
            
    def forward(self, inputs, fc_vec):
        N, C = inputs.shape                                                     # (N*Nr*Nt, n_inputs)
        fc_vec = fc_vec[None, self.fc_channels:].expand(N, self.fc_channels)    # (N*Nr*Nt, n_fc)
        w = torch.cat((inputs, fc_vec), dim=1)                                  # (N*Nr*Nt, n_inputs + n_fc)
        w = self.relu(self.ln1(self.fc1(w)))                                    # (N*Nr*Nt, n_filters)
        w = self.relu(self.ln2(self.fc2(w)))                                    # (N*Nr*Nt, n_filters)
        w = self.split(w)                                                       # (N*Nr*Nt, 2 * n_inputs)
        alpha, beta = w[:, :C], w[:, C:]
        out = inputs * alpha + beta                                             # (N*Nr*Nt, n_inputs)
        return out
    
        
class F4CKM_WO_FCA(nn.Module):

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
        # d_input[0] *= 2  # real and image part concatenated
        self.act = nn.GELU()
        self.d_viewdirs = d_viewdirs
        self.d_freq = 0


        # pre-attenuation layers
        self.cfr_encoder = CFR_ShapingFilter(n_samples=n_samples, d_input=self.d_input)                                                  # (chunksize, Nc, Nr, Nt)
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
        self.alpha_out = nn.Conv2d(in_channels=n_filters, out_channels=self.Nc, kernel_size=3, stride=1, padding=1)                 # (chunksize, Nc, Nr, Nt)
        self.downsampling = nn.AdaptiveAvgPool2d(1)
        
        # bottleneck layers
        if self.d_viewdirs is not None:
            self.branch = nn.ModuleList()
            self.branch.append(ResBlock(in_channels=n_filters+self.d_viewdirs, hid_channels=n_filters, kernel_size=3, stride=1))    # (chunksize, n_filters, Nr, Nt)
            
        self.realimag_filter = nn.ModuleList()
        self.realimag_filter.append(ResBlock(in_channels=n_filters, hid_channels=2*n_filters, kernel_size=3, stride=1))             # (chunksize, 2*n_filters, Nr, Nt)
        
        self.output = nn.Conv2d(in_channels=2*n_filters, out_channels=2*self.Nc, kernel_size=3, stride=1, padding=1)                # (chunksize, 2Nc, Nr, Nt)
        
    def forward(
        self,
        uplink_cfr: Optional[torch.Tensor] = None,          # (chunksize, 2Nc, Nr, Nt)
        z_vals: Optional[torch.Tensor] = None,              # (chunksize, 2Nc, Nr, Nt)
        viewdirs: Optional[torch.Tensor] = None,            # (chunksize, d_viewdirs, Nr, Nt)
        filter: str = 'sd'
    ) -> torch.Tensor:

        # Pre-process
        x = self.cfr_encoder(uplink_cfr, z_vals, viewdirs, filter=filter)
        
        # Apply forward pass up to bottleneck
        x = self.conv1(x)
        for i, block in enumerate(self.blocks):
            x = block(x)

        # Apply bottleneck
        if self.d_viewdirs is not None:
            # Split alpha from network output
            alpha = self.alpha_out(x)                               # (chunksize, Nc, Nr, Nt)

            # Pass through bottleneck to get real and imag parts
            x = torch.concat([x, viewdirs], dim=-3)                 # (chunksize, d_viewdirs+n_filters, Nr, Nt)
            for i, block in enumerate(self.branch):
                x = block(x)
            for i, block in enumerate(self.realimag_filter):
                x = block(x)
            x = self.output(x)                                      # (chunksize, 2Nc, Nr, Nt)

            # Concatenate alphas to output
            x = torch.concat([x, alpha], dim=-3)                    # (chunksize, 3Nc)
        else:
            # Simple output
            x = self.realimag_filter(x)                             # (chunksize, 2*n_filter, Nr, Nt)
            x = self.output(x)                                      # (chunksize, 3Nc, Nr, Nt)
        return x
    

class F4CKM_WO_CNN(nn.Module):

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
        # d_input[0] *= 2  # real and image part concatenated
        self.act = nn.GELU()
        self.d_viewdirs = d_viewdirs
        self.d_freq = 0


        # pre-attenuation layers
        self.cfr_encoder = CFR_ShapingFilter(n_samples=n_samples, d_input=self.d_input)                                     # (N, Nc, Nr, Nt)
        
        # attenuation layers
        self.mlp1 = nn.Sequential(                                                                                          # (N*Nr*Nt, n_filters)
            nn.Linear(in_features=self.Nc, out_features=n_filters, bias=False),
            nn.GroupNorm(n_filters // 4, n_filters),
            nn.GELU()
        )
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(ResBlock_MLP(in_channels=n_filters, hid_channels=n_filters))                                 # (N*Nr*Nt, n_filters)
            self.blocks.append(FreqAT_MLP(in_channels=n_filters, fc_channels=self.Nc, hid_channels=n_filters))                  # (N*Nr*Nt, n_filters)
        self.alpha_out = nn.Linear(in_features=n_filters, out_features=self.Nc)                                             # (N*Nr*Nt, Nc)
        
        # bottleneck layers
        if self.d_viewdirs is not None:
            self.branch = nn.ModuleList()
            self.branch.append(ResBlock_MLP(in_channels=n_filters+self.d_viewdirs, hid_channels=n_filters))                 # (N*Nr*Nt, n_filters)
            self.branch.append(FreqAT_MLP(in_channels=n_filters, fc_channels=self.Nc, hid_channels=n_filters))                  # (N*Nr*Nt, n_filters)
            
        self.realimag_filter = nn.ModuleList()
        self.realimag_filter.append(ResBlock_MLP(in_channels=n_filters, hid_channels=2*n_filters))                          # (N*Nr*Nt, 2*n_filters)
        self.realimag_filter.append(FreqAT_MLP(in_channels=2*n_filters, fc_channels=self.Nc, hid_channels=n_filters))                    # (N*Nr*Nt, 2*n_filters)
        
        self.output = nn.Linear(in_features=2*n_filters, out_features=2*self.Nc)                                            # (N*Nr*Nt, 2Nc)
        
    def forward(
        self,
        uplink_cfr: Optional[torch.Tensor] = None,          # (N, 2Nc, Nr, Nt)
        z_vals: Optional[torch.Tensor] = None,              # (N, 2Nc, Nr, Nt)
        viewdirs: Optional[torch.Tensor] = None,            # (N, d_viewdirs, Nr, Nt)
        fc_vec: Optional[torch.Tensor] = None,              # (2Nc)
        filter: str = 'sd'
    ) -> torch.Tensor:

        # Pre-process
        x = self.cfr_encoder(uplink_cfr, z_vals, viewdirs, filter=filter)   # (N, Nc, Nr, Nt)
        x = x.permute(0, 2, 3, 1).reshape(-1, self.Nc)                      # (N*Nr*Nt, Nc)
        
        # Apply forward pass up to bottleneck
        x = self.mlp1(x)                                                    # (N*Nr*Nt, n_filters)
        for i, block in enumerate(self.blocks):
            if i % 2 == 0:
                x = block(x)
            else:
                x = block(x, fc_vec)                                        # (N*Nr*Nt, n_filters)

        # Apply bottleneck
        if self.d_viewdirs is not None:
            # Split alpha from network output
            alpha = self.alpha_out(x)                                       # (N*Nr*Nt, Nc)

            # Pass through bottleneck to get real and imag parts
            viewdirs = viewdirs.permute(0, 2, 3, 1).reshape(-1, self.d_viewdirs)        # (N*Nr*Nt, d_viewdirs)
            x = torch.concat([x, viewdirs], dim=1)                                     # (N*Nr*Nt, d_viewdirs+n_filters)
            for i, block in enumerate(self.branch):
                if i % 2 == 0:
                    x = block(x)
                else:
                    x = block(x, fc_vec)                                                # (N*Nr*Nt, n_filters)
            for i, block in enumerate(self.realimag_filter):
                if i % 2 == 0:
                    x = block(x)
                else:
                    x = block(x, fc_vec)                                                # (N*Nr*Nt, 2*n_filters)
            x = self.output(x)                                                          # (N*Nr*Nt, 2Nc)

            # Concatenate alphas to output
            x = torch.concat([x, alpha], dim=1)                                        # (N*Nr*Nt, 3Nc)
        else:
            # Simple output
            x = self.realimag_filter(x)                                                 # (N*Nr*Nt, 2*n_filter)
            x = self.output(x)                                                          # (N*Nr*Nt, 2Nc)
        return x