"""
2D KAN with Residual Connections (lmKAN-Res)
=============================================

Implements a novel 2D KAN architecture with nested residual connections based on:

KAN_2D_res(x) = Σ_k W^(l-2)_vk * B_v(x_v) * B_k(Σ_u W^(l-1)_kuf * B_u(x_u) * B_f(Σ_j W^l_fj * B_j(x_j)))

This creates a hierarchical structure where:
- Level 0: 1D KAN basis functions on input x → intermediate representation
- Level 1: 2D coupling between original x and Level 0 output (residual)
- Level 2: 2D coupling between original x and Level 1 output (residual)

The key insight is that original inputs x appear at ALL levels via residual 
connections, allowing information to flow through and improving gradient flow.

Author: Generated for KAN research
"""

import torch
import torch.nn as nn
import numpy as np
import math


def get_borders_cdf_grid(n_chunks):
    """
    Get non-uniform grid borders using inverse Laplace CDF.
    
    Parameters:
    -----------
    n_chunks : int
        Number of grid chunks (grid will have n_chunks + 1 border points)
        
    Returns:
    --------
    list
        Grid border values in original x-space
    """
    def inverse_grid_function(x):
        """Inverse Laplace CDF: maps [0, 1] -> R"""
        if x <= 0.5:
            return np.log(2.0 * x)
        else:
            return -np.log(2.0 * (1.0 - x))
    
    chunk_size = 1.0 / n_chunks
    borders = []
    for i in range(1, n_chunks):
        level_now = i * chunk_size
        borders.append(inverse_grid_function(level_now))
    
    left_most = borders[0] - (borders[1] - borders[0])
    right_most = borders[-1] + (borders[-1] - borders[-2])
    return [left_most] + borders + [right_most]


def laplace_cdf(x):
    """
    Apply Laplace CDF: maps R -> [0, 1]
    
    cdf(x) = 0.5 * exp(-|x|) for x <= 0
           = 1 - 0.5 * exp(-|x|) for x > 0
    """
    exp_neg_abs = torch.exp(-torch.abs(x))
    return torch.where(x > 0, 1.0 - 0.5 * exp_neg_abs, 0.5 * exp_neg_abs)


class BasisFunction1D(nn.Module):
    """
    1D Basis Function Layer using linear interpolation on CDF grid.
    
    Computes: B_j(x_j) with linear interpolation between grid points.
    """
    
    def __init__(self, num_grids, input_dim, output_dim, init_scale=0.1):
        super().__init__()
        
        self.num_grids = num_grids
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Parameters: [num_grids + 1, output_dim, input_dim]
        self.func_parameter = nn.Parameter(
            torch.empty(num_grids + 1, output_dim, input_dim)
        )
        nn.init.uniform_(
            self.func_parameter,
            -init_scale / math.sqrt(input_dim),
            init_scale / math.sqrt(input_dim)
        )
        
        # Grid borders
        borders = get_borders_cdf_grid(num_grids)
        self.register_buffer('borders', torch.tensor(borders, dtype=torch.float32))
        
        # Precompute inverse chunk lengths
        inverse_lengths = []
        for i in range(num_grids):
            inverse_lengths.append(1.0 / (borders[i + 1] - borders[i]))
        self.register_buffer('inverse_chunk_lengths', torch.tensor(inverse_lengths, dtype=torch.float32))
    
    def forward(self, x):
        """
        Forward pass for 1D basis functions.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape [input_dim, batch_size] (batch-last layout)
            
        Returns:
        --------
        torch.Tensor
            Output of shape [output_dim, batch_size]
        """
        batch_size = x.shape[1]
        device = x.device
        dtype = x.dtype
        
        # Apply Laplace CDF
        x_cdf = laplace_cdf(x)
        
        # Find grid cell indices
        x_scaled = x_cdf * self.num_grids
        indices = torch.clamp(x_scaled.long(), 0, self.num_grids - 1)
        
        # Get left borders and inverse lengths
        left = self.borders[indices]
        inv_len = self.inverse_chunk_lengths[indices]
        
        # Compute interpolation weight (delta = position within grid cell)
        delta = (x - left) * inv_len
        
        # Clamp indices for right neighbor
        indices_next = torch.clamp(indices + 1, 0, self.num_grids)
        
        # Perform interpolation for all dimensions efficiently
        # indices: [input_dim, batch_size]
        # We need: [output_dim, batch_size]
        
        output = torch.zeros(self.output_dim, batch_size, device=device, dtype=dtype)
        
        for in_idx in range(self.input_dim):
            idx = indices[in_idx]           # [batch_size]
            idx_next = indices_next[in_idx] # [batch_size]
            d = delta[in_idx]               # [batch_size]
            
            # Get values at left and right grid points: [output_dim, batch_size]
            val_left = self.func_parameter[idx, :, in_idx].T   # [output_dim, batch_size]
            val_right = self.func_parameter[idx_next, :, in_idx].T
            
            # Linear interpolation and accumulate
            output += (1 - d) * val_left + d * val_right
        
        return output


class BasisFunction2D(nn.Module):
    """
    2D Basis Function Layer using bilinear interpolation on CDF grid.
    
    Computes: B_u(x_u) * B_f(z_f) with bilinear interpolation.
    This couples two different inputs x and z in 2D space.
    """
    
    def __init__(self, num_grids, input_dim_x, input_dim_z, output_dim, init_scale=0.1):
        super().__init__()
        
        self.num_grids = num_grids
        self.input_dim_x = input_dim_x  # Dimension of residual input x
        self.input_dim_z = input_dim_z  # Dimension of previous layer output z
        self.output_dim = output_dim
        
        # Parameters: [num_grids+1, num_grids+1, output_dim, input_dim_x, input_dim_z]
        # Each (x, z) pair has a 2D lookup table for each output dimension
        self.func_parameter = nn.Parameter(
            torch.empty(num_grids + 1, num_grids + 1, output_dim, input_dim_x, input_dim_z)
        )
        nn.init.uniform_(
            self.func_parameter,
            -init_scale / math.sqrt(input_dim_x * input_dim_z),
            init_scale / math.sqrt(input_dim_x * input_dim_z)
        )
        
        # Grid borders
        borders = get_borders_cdf_grid(num_grids)
        self.register_buffer('borders', torch.tensor(borders, dtype=torch.float32))
        
        # Precompute inverse chunk lengths
        inverse_lengths = []
        for i in range(num_grids):
            inverse_lengths.append(1.0 / (borders[i + 1] - borders[i]))
        self.register_buffer('inverse_chunk_lengths', torch.tensor(inverse_lengths, dtype=torch.float32))
    
    def forward(self, x, z):
        """
        Forward pass for 2D basis functions coupling x and z.
        
        Parameters:
        -----------
        x : torch.Tensor
            Residual input of shape [input_dim_x, batch_size]
        z : torch.Tensor
            Previous layer output of shape [input_dim_z, batch_size]
            
        Returns:
        --------
        torch.Tensor
            Output of shape [output_dim, batch_size]
        """
        batch_size = x.shape[1]
        device = x.device
        dtype = x.dtype
        
        # Apply Laplace CDF to both inputs
        x_cdf = laplace_cdf(x)  # [input_dim_x, batch_size]
        z_cdf = laplace_cdf(z)  # [input_dim_z, batch_size]
        
        # Find grid cell indices
        x_scaled = x_cdf * self.num_grids
        z_scaled = z_cdf * self.num_grids
        
        idx_x = torch.clamp(x_scaled.long(), 0, self.num_grids - 1)
        idx_z = torch.clamp(z_scaled.long(), 0, self.num_grids - 1)
        
        # Get left borders and inverse lengths
        left_x = self.borders[idx_x]
        left_z = self.borders[idx_z]
        inv_len_x = self.inverse_chunk_lengths[idx_x]
        inv_len_z = self.inverse_chunk_lengths[idx_z]
        
        # Compute interpolation weights
        delta_x = (x - left_x) * inv_len_x
        delta_z = (z - left_z) * inv_len_z
        
        # Clamp indices for next
        idx_x_next = torch.clamp(idx_x + 1, 0, self.num_grids)
        idx_z_next = torch.clamp(idx_z + 1, 0, self.num_grids)
        
        # Initialize output
        output = torch.zeros(self.output_dim, batch_size, device=device, dtype=dtype)
        
        # Bilinear interpolation: sum over all (x, z) pairs
        for ix in range(self.input_dim_x):
            for iz in range(self.input_dim_z):
                # Indices for this pair
                i_x, i_x_n = idx_x[ix], idx_x_next[ix]      # [batch_size]
                i_z, i_z_n = idx_z[iz], idx_z_next[iz]      # [batch_size]
                
                # Interpolation weights
                dx = delta_x[ix]  # [batch_size]
                dz = delta_z[iz]  # [batch_size]
                
                w00 = (1 - dx) * (1 - dz)  # [batch_size]
                w01 = (1 - dx) * dz
                w10 = dx * (1 - dz)
                w11 = dx * dz
                
                # Get parameter values at 4 corners: [output_dim, batch_size]
                p00 = self.func_parameter[i_x, i_z, :, ix, iz].T
                p01 = self.func_parameter[i_x, i_z_n, :, ix, iz].T
                p10 = self.func_parameter[i_x_n, i_z, :, ix, iz].T
                p11 = self.func_parameter[i_x_n, i_z_n, :, ix, iz].T
                
                # Bilinear interpolation and accumulate
                output += w00 * p00 + w01 * p01 + w10 * p10 + w11 * p11
        
        return output


class LookupKAN2D_Residual(nn.Module):
    """
    2D KAN with Residual Connections.
    
    Implements the nested residual formula:
    KAN_2D_res(x) = Σ W^(l-2) * B(x) * B(Σ W^(l-1) * B(x) * B(Σ W^l * B(x)))
    
    Architecture:
    - Level 0 (innermost): 1D KAN on input x
        z0 = Σ_j W^l_fj * B_j(x_j)
        
    - Level 1 (middle): 2D coupling between x and z0
        z1 = Σ_uf W^(l-1)_kuf * B_u(x_u) * B_f(z0_f)
        
    - Level 2 (outer): 2D coupling between x and z1
        z2 = Σ_vk W^(l-2)_vk * B_v(x_v) * B_k(z1_k)
    
    The residual connections (x appearing at all levels) improve gradient
    flow and allow the network to learn complex hierarchical transformations.
    
    Parameters:
    -----------
    num_grids : int
        Number of grid cells per dimension for basis functions
    input_dim : int
        Input dimension (must be even for compatibility)
    hidden_dim : int
        Hidden dimension between layers
    output_dim : int
        Output dimension
    num_levels : int
        Number of residual coupling levels (default: 2)
    init_scale : float
        Scale for parameter initialization
    """
    
    def __init__(
        self, 
        num_grids, 
        input_dim, 
        hidden_dim, 
        output_dim,
        num_levels=2,
        init_scale=0.1
    ):
        super().__init__()
        
        self.num_grids = num_grids
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_levels = num_levels
        
        # Level 0: 1D basis functions on input x
        # z0 = Σ_j W^l_fj * B_j(x_j)
        self.level0 = BasisFunction1D(
            num_grids=num_grids,
            input_dim=input_dim,
            output_dim=hidden_dim,
            init_scale=init_scale
        )
        
        # Level 1 to num_levels-1: 2D coupling layers
        # Each couples original x with previous layer output
        self.coupling_layers = nn.ModuleList()
        
        prev_dim = hidden_dim
        for level in range(1, num_levels):
            # For the last level, output is output_dim, otherwise hidden_dim
            if level == num_levels - 1:
                out_d = output_dim
            else:
                out_d = hidden_dim
            
            self.coupling_layers.append(
                BasisFunction2D(
                    num_grids=num_grids,
                    input_dim_x=input_dim,
                    input_dim_z=prev_dim,
                    output_dim=out_d,
                    init_scale=init_scale
                )
            )
            prev_dim = out_d
    
    def forward(self, x):
        """
        Forward pass with residual coupling.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape [input_dim, batch_size] (batch-last layout)
            
        Returns:
        --------
        torch.Tensor
            Output of shape [output_dim, batch_size]
        """
        # Level 0: 1D KAN
        z = self.level0(x)  # [hidden_dim, batch_size]
        
        # Levels 1+: 2D coupling with residual connection from x
        for coupling in self.coupling_layers:
            z = coupling(x, z)
        
        return z
    
    def get_regularization(self):
        """
        Compute regularization loss for smooth basis functions.
        
        Returns:
        --------
        torch.Tensor
            Scalar regularization loss
        """
        reg = 0.0
        
        # Level 0: 1D smoothness
        f0 = self.level0.func_parameter
        # Second derivative approximation
        f0_xx = f0[2:, :, :] - 2 * f0[1:-1, :, :] + f0[:-2, :, :]
        reg += torch.mean(f0_xx ** 2)
        
        # 2D coupling layers
        for coupling in self.coupling_layers:
            f = coupling.func_parameter
            # Second derivatives in both dimensions
            f_xx = f[2:, :, :, :, :] - 2 * f[1:-1, :, :, :, :] + f[:-2, :, :, :, :]
            f_yy = f[:, 2:, :, :, :] - 2 * f[:, 1:-1, :, :, :] + f[:, :-2, :, :, :]
            # Mixed derivative
            f_xy = (f[2:, 2:, :, :, :] - f[2:, :-2, :, :, :] - 
                    f[:-2, 2:, :, :, :] + f[:-2, :-2, :, :, :]) / 4
            
            reg += torch.mean(f_xx ** 2) + torch.mean(f_yy ** 2) + 2 * torch.mean(f_xy ** 2)
        
        return reg


class LookupKAN2D_Residual_Efficient(nn.Module):
    """
    Memory-efficient version of 2D KAN with Residual Connections.
    
    Uses shared 2D lookup tables for input pairs instead of full Cartesian
    product, reducing memory from O(input_dim^2) to O(input_dim).
    
    Parameters:
    -----------
    num_grids : int
        Number of grid cells per dimension
    input_dim : int
        Input dimension (must be even)
    hidden_dim : int
        Hidden dimension (should be even for 2D coupling)
    output_dim : int
        Output dimension
    init_scale : float
        Initialization scale
    """
    
    def __init__(
        self,
        num_grids,
        input_dim,
        hidden_dim,
        output_dim,
        init_scale=0.1
    ):
        super().__init__()
        
        if input_dim % 2 != 0:
            raise ValueError("input_dim must be divisible by 2")
        if hidden_dim % 2 != 0:
            raise ValueError("hidden_dim must be divisible by 2")
        
        self.num_grids = num_grids
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_input_pairs = input_dim // 2
        self.n_hidden_pairs = hidden_dim // 2
        
        # Level 0: Standard 2D KAN on input pairs (x_{2i}, x_{2i+1})
        # Shape: [num_grids+1, num_grids+1, hidden_dim, n_input_pairs]
        self.level0_params = nn.Parameter(
            torch.empty(num_grids + 1, num_grids + 1, hidden_dim, self.n_input_pairs)
        )
        nn.init.uniform_(
            self.level0_params,
            -init_scale / math.sqrt(input_dim),
            init_scale / math.sqrt(input_dim)
        )
        
        # Level 1: 2D coupling between input x and level0 output z
        # For efficiency, couple pairs: (x_{2i}, x_{2i+1}) with (z_{2j}, z_{2j+1})
        # Shape: [num_grids+1, num_grids+1, output_dim, n_input_pairs, n_hidden_pairs]
        self.level1_params = nn.Parameter(
            torch.empty(
                num_grids + 1, num_grids + 1, 
                output_dim, self.n_input_pairs, self.n_hidden_pairs
            )
        )
        nn.init.uniform_(
            self.level1_params,
            -init_scale / math.sqrt(input_dim * hidden_dim / 4),
            init_scale / math.sqrt(input_dim * hidden_dim / 4)
        )
        
        # Grid borders
        borders = get_borders_cdf_grid(num_grids)
        self.register_buffer('borders', torch.tensor(borders, dtype=torch.float32))
        
        # Inverse chunk lengths
        inverse_lengths = []
        for i in range(num_grids):
            inverse_lengths.append(1.0 / (borders[i + 1] - borders[i]))
        self.register_buffer('inverse_chunk_lengths', torch.tensor(inverse_lengths, dtype=torch.float32))
    
    def _bilinear_interp_2d(self, x1, x2, params):
        """
        Bilinear interpolation on 2D grid for paired inputs.
        
        Parameters:
        -----------
        x1 : torch.Tensor
            First input of pairs, shape [n_pairs, batch_size]
        x2 : torch.Tensor
            Second input of pairs, shape [n_pairs, batch_size]
        params : torch.Tensor
            Parameters, shape [g+1, g+1, output_dim, n_pairs, ...]
            
        Returns:
        --------
        torch.Tensor
            Interpolated output
        """
        batch_size = x1.shape[1]
        n_pairs = x1.shape[0]
        device = x1.device
        dtype = x1.dtype
        
        # Apply Laplace CDF
        x1_cdf = laplace_cdf(x1)
        x2_cdf = laplace_cdf(x2)
        
        # Grid indices
        x1_scaled = x1_cdf * self.num_grids
        x2_scaled = x2_cdf * self.num_grids
        
        idx1 = torch.clamp(x1_scaled.long(), 0, self.num_grids - 1)
        idx2 = torch.clamp(x2_scaled.long(), 0, self.num_grids - 1)
        
        # Interpolation weights
        left1 = self.borders[idx1]
        left2 = self.borders[idx2]
        inv_len1 = self.inverse_chunk_lengths[idx1]
        inv_len2 = self.inverse_chunk_lengths[idx2]
        
        delta1 = (x1 - left1) * inv_len1
        delta2 = (x2 - left2) * inv_len2
        
        idx1_next = torch.clamp(idx1 + 1, 0, self.num_grids)
        idx2_next = torch.clamp(idx2 + 1, 0, self.num_grids)
        
        # Bilinear weights
        w00 = (1 - delta1) * (1 - delta2)
        w01 = (1 - delta1) * delta2
        w10 = delta1 * (1 - delta2)
        w11 = delta1 * delta2
        
        return idx1, idx2, idx1_next, idx2_next, w00, w01, w10, w11
    
    def forward(self, x):
        """
        Forward pass with efficient 2D residual coupling.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input, shape [input_dim, batch_size]
            
        Returns:
        --------
        torch.Tensor
            Output, shape [output_dim, batch_size]
        """
        batch_size = x.shape[1]
        device = x.device
        dtype = x.dtype
        
        # Split input into pairs
        x_pairs = x.view(self.n_input_pairs, 2, batch_size)
        x1 = x_pairs[:, 0, :]  # [n_input_pairs, batch_size]
        x2 = x_pairs[:, 1, :]  # [n_input_pairs, batch_size]
        
        # Level 0: Standard 2D KAN on input pairs
        idx1, idx2, idx1n, idx2n, w00, w01, w10, w11 = self._bilinear_interp_2d(x1, x2, self.level0_params)
        
        z0 = torch.zeros(self.hidden_dim, batch_size, device=device, dtype=dtype)
        for p in range(self.n_input_pairs):
            i1, i2 = idx1[p], idx2[p]
            i1n, i2n = idx1n[p], idx2n[p]
            
            p00 = self.level0_params[i1, i2, :, p].T
            p01 = self.level0_params[i1, i2n, :, p].T
            p10 = self.level0_params[i1n, i2, :, p].T
            p11 = self.level0_params[i1n, i2n, :, p].T
            
            z0 += w00[p] * p00 + w01[p] * p01 + w10[p] * p10 + w11[p] * p11
        
        # Level 1: 2D coupling between x pairs and z0 pairs
        # z0 is [hidden_dim, batch_size], reshape to pairs
        z0_pairs = z0.view(self.n_hidden_pairs, 2, batch_size)
        z1_x = z0_pairs[:, 0, :]  # [n_hidden_pairs, batch_size]
        z2_x = z0_pairs[:, 1, :]  # [n_hidden_pairs, batch_size]
        
        # Get interpolation info for z0 pairs
        idxz1, idxz2, idxz1n, idxz2n, wz00, wz01, wz10, wz11 = self._bilinear_interp_2d(z1_x, z2_x, None)
        
        # Combine x-pair info with z-pair info
        output = torch.zeros(self.output_dim, batch_size, device=device, dtype=dtype)
        
        for px in range(self.n_input_pairs):
            for pz in range(self.n_hidden_pairs):
                # x-pair indices and weights
                i1x, i2x = idx1[px], idx2[px]
                i1xn, i2xn = idx1n[px], idx2n[px]
                
                # z-pair indices and weights  
                i1z, i2z = idxz1[pz], idxz2[pz]
                i1zn, i2zn = idxz1n[pz], idxz2n[pz]
                
                # Combined 4D interpolation weights (x1, x2) × (z1, z2)
                # For simplicity, we use (x1, z1) 2D lookup
                wx = w00[px] + w01[px] + w10[px] + w11[px]  # Marginalize x pair
                wz = wz00[pz] + wz01[pz] + wz10[pz] + wz11[pz]  # Marginalize z pair
                
                # Use (x1, z1) indices for the 2D lookup
                p00 = self.level1_params[i1x, i1z, :, px, pz].T
                p01 = self.level1_params[i1x, i1zn, :, px, pz].T
                p10 = self.level1_params[i1xn, i1z, :, px, pz].T
                p11 = self.level1_params[i1xn, i1zn, :, px, pz].T
                
                # Use x-z bilinear weights
                w_xz_00 = w00[px] * wz00[pz]
                w_xz_01 = w00[px] * wz10[pz]
                w_xz_10 = w10[px] * wz00[pz]
                w_xz_11 = w10[px] * wz10[pz]
                
                w_sum = w_xz_00 + w_xz_01 + w_xz_10 + w_xz_11
                if torch.any(w_sum > 0):
                    output += (w_xz_00 * p00 + w_xz_01 * p01 + w_xz_10 * p10 + w_xz_11 * p11)
        
        return output
    
    def get_regularization(self):
        """Compute smoothness regularization."""
        reg = 0.0
        
        for params in [self.level0_params, self.level1_params]:
            f_xx = params[2:, :, ...] - 2 * params[1:-1, :, ...] + params[:-2, :, ...]
            f_yy = params[:, 2:, ...] - 2 * params[:, 1:-1, ...] + params[:, :-2, ...]
            reg += torch.mean(f_xx ** 2) + torch.mean(f_yy ** 2)
        
        return reg


class LookupKAN2D_ResidualModel(nn.Module):
    """
    Complete 2D KAN Residual model with BatchNorm wrapper.
    
    Handles batch-first to batch-last conversion and optional normalization.
    
    Parameters:
    -----------
    num_grids : int
        Grid resolution
    input_dim : int
        Input dimension
    hidden_dim : int
        Hidden dimension for residual layers
    output_dim : int
        Final output dimension
    num_levels : int
        Number of residual coupling levels
    use_batchnorm : bool
        Whether to apply batch normalization to input
    """
    
    def __init__(
        self,
        num_grids,
        input_dim,
        hidden_dim,
        output_dim,
        num_levels=2,
        use_batchnorm=True
    ):
        super().__init__()
        
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.bn = nn.BatchNorm1d(input_dim, affine=False)
        
        self.kan = LookupKAN2D_Residual(
            num_grids=num_grids,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_levels=num_levels
        )
    
    def forward(self, x):
        """
        Forward pass with batch-first input.
        
        Parameters:
        -----------
        x : torch.Tensor
            Shape [batch_size, input_dim]
            
        Returns:
        --------
        torch.Tensor
            Shape [batch_size, output_dim]
        """
        if self.use_batchnorm:
            x = self.bn(x)
        x = self.kan(x.T).T  # Convert to/from batch-last
        return x
    
    def get_regularization(self):
        return self.kan.get_regularization()
    
    def get_hessian_regularization(self):
        """Alias for compatibility with train_model function."""
        return self.kan.get_regularization()


def demo():
    """Quick demonstration of the residual 2D KAN layer."""
    print("=" * 60)
    print("2D KAN with Residual Connections - Demo")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create model
    model = LookupKAN2D_Residual(
        num_grids=8,
        input_dim=32,
        hidden_dim=16,
        output_dim=8,
        num_levels=2
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    batch_size = 64
    x = torch.randn(32, batch_size, device=device)  # [input_dim, batch_size]
    
    with torch.no_grad():
        y = model(x)
    
    print(f"Input shape:  {list(x.shape)} [input_dim, batch_size]")
    print(f"Output shape: {list(y.shape)} [output_dim, batch_size]")
    
    # Test regularization
    reg = model.get_regularization()
    print(f"Regularization loss: {reg.item():.6f}")
    
    # Test gradient flow
    x.requires_grad_(True)
    y = model(x)
    loss = y.sum()
    loss.backward()
    print(f"Gradient check: x.grad exists = {x.grad is not None}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    demo()

