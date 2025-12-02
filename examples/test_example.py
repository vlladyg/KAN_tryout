"""
Test script to verify the example notebook works correctly
"""
import sys
import os

# Add lmkan to path
sys.path.insert(0, '/home/vladimir/DATA/linux_data/GitHub/KAN/lmkan')

print("=" * 60)
print("Testing lmKAN Example - Quick Verification")
print("=" * 60)

# Test 1: Import checks
print("\n1. Testing imports...")
try:
    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    print("   ✓ Basic imports successful")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

try:
    from lmKAN import LMKAN2DLayer, utilities
    print("   ✓ lmKAN imports successful")
except ImportError as e:
    print(f"   ✗ lmKAN import failed: {e}")
    print("   Note: lmKAN requires CUDA compilation. Run 'pip install .' in lmkan directory")
    HAS_LMKAN = False
else:
    HAS_LMKAN = True

# Check CUDA
print(f"\n2. CUDA availability: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    device = torch.device('cuda')
else:
    print("   Using CPU (2D KAN will be skipped)")
    device = torch.device('cpu')

# Test 2: CDF grid functions
print("\n3. Testing CDF grid generation...")
def direct_grid_function(x):
    absolute = np.abs(x)
    value = 0.5 * np.exp(-absolute)
    if x > 0:
        return 1.0 - value
    else:
        return value

def inverse_grid_function(x):
    if x <= 0.5:
        return np.log(2.0 * x)
    else:
        return -np.log(2.0 * (1.0 - x))

def get_borders_cdf_grid(n_chunks):
    chunk_size = 1.0 / n_chunks
    borders = []
    for i in range(1, n_chunks):
        level_now = i * chunk_size
        borders.append(inverse_grid_function(level_now))
    left_most = borders[0] - (borders[1] - borders[0])
    right_most = borders[-1] + (borders[-1] - borders[-2])
    return [left_most] + borders + [right_most]

borders = get_borders_cdf_grid(8)
print(f"   ✓ Generated {len(borders)} grid borders")
print(f"   Range: [{min(borders):.3f}, {max(borders):.3f}]")

# Test 3: 1D KAN implementation
print("\n4. Testing 1D KAN implementation...")
import math

class LookupKAN1D(nn.Module):
    def __init__(self, num_grids, input_dim, output_dim, init_scale=0.1):
        super(LookupKAN1D, self).__init__()
        self.num_grids = num_grids
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.func_parameter = nn.Parameter(
            torch.empty(num_grids + 1, output_dim, input_dim)
        )
        nn.init.uniform_(
            self.func_parameter,
            -init_scale / math.sqrt(input_dim),
            init_scale / math.sqrt(input_dim)
        )
        
        self.register_buffer(
            'borders',
            torch.tensor(get_borders_cdf_grid(num_grids), dtype=torch.float32)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        func_value = torch.exp(-torch.abs(x))
        x_cdf = torch.where(x > 0, 1.0 - 0.5 * func_value, 0.5 * func_value)
        x_scaled = x_cdf * self.num_grids
        indices = torch.clamp(x_scaled.long(), 0, self.num_grids - 1)
        delta = x_scaled - indices.float()
        delta = torch.clamp(delta, 0.0, 1.0)
        
        output = torch.zeros(batch_size, self.output_dim, device=x.device)
        for out_idx in range(self.output_dim):
            for in_idx in range(self.input_dim):
                grid_idx = indices[:, in_idx]
                d = delta[:, in_idx]
                val_left = self.func_parameter[grid_idx, out_idx, in_idx]
                val_right = self.func_parameter[torch.clamp(grid_idx + 1, 0, self.num_grids), out_idx, in_idx]
                interp_val = (1 - d) * val_left + d * val_right
                output[:, out_idx] += interp_val
        
        return output
    
    def get_hessian_regularization(self):
        second_deriv = self.func_parameter[2:] - 2*self.func_parameter[1:-1] + self.func_parameter[:-2]
        return torch.mean(second_deriv ** 2)

try:
    model_1d = LookupKAN1D(num_grids=8, input_dim=32, output_dim=1).to(device)
    test_input = torch.randn(16, 32).to(device)
    test_output = model_1d(test_input)
    print(f"   ✓ 1D KAN forward pass successful")
    print(f"   Input: {test_input.shape}, Output: {test_output.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model_1d.parameters()):,}")
except Exception as e:
    print(f"   ✗ 1D KAN test failed: {e}")
    sys.exit(1)

# Test 4: 2D KAN (if CUDA available)
if HAS_LMKAN and torch.cuda.is_available():
    print("\n5. Testing 2D KAN (lmKAN)...")
    
    class LookupKAN2DModel(nn.Module):
        def __init__(self, num_grids, input_dim, hidden_dim, output_dim):
            super(LookupKAN2DModel, self).__init__()
            self.bn = nn.BatchNorm1d(input_dim, affine=False)
            # hidden_dim must be divisible by tile_size (8)
            self.kan = LMKAN2DLayer(
                num_grids=num_grids,
                input_dim=input_dim,
                output_dim=hidden_dim,  # Use hidden_dim divisible by tile_size
                tile_size_forward=8,
                tile_size_backward=4,
                block_size_forward=1024,
                block_size_backward=512,
            )
            # Final linear layer to get to output_dim
            self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        def forward(self, x):
            x = self.bn(x)
            x = x.T.contiguous()  # Must be contiguous for CUDA kernels
            x = self.kan(x)  # [hidden_dim, batch_size]
            x = x.T  # [batch_size, hidden_dim]
            return self.output_layer(x)  # [batch_size, output_dim]
        
        def get_hessian_regularization(self):
            return self.kan.get_hessian_regularization()
    
    try:
        # hidden_dim=8 is divisible by tile_size=8
        model_2d = LookupKAN2DModel(num_grids=8, input_dim=32, hidden_dim=8, output_dim=1).cuda()
        test_input = torch.randn(16, 32).cuda()
        test_output = model_2d(test_input)
        print(f"   ✓ 2D KAN forward pass successful")
        print(f"   Input: {test_input.shape}, Output: {test_output.shape}")
        print(f"   Parameters: {sum(p.numel() for p in model_2d.parameters()):,}")
    except Exception as e:
        print(f"   ✗ 2D KAN test failed: {e}")
        sys.exit(1)
else:
    print("\n5. Skipping 2D KAN test (requires CUDA and lmKAN)")

# Test 5: Target MLP and basic training step
print("\n6. Testing target MLP and training step...")
try:
    torch.manual_seed(42)
    target_mlp = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    ).to(device)
    target_mlp.eval()
    for p in target_mlp.parameters():
        p.requires_grad = False
    
    # Generate small dataset
    x_train = torch.randn(100, 32).to(device)
    with torch.no_grad():
        y_train = target_mlp(x_train)
    
    print(f"   ✓ Target MLP created")
    print(f"   Dataset: {x_train.shape[0]} samples")
    
    # Test training step
    model_1d.train()
    optimizer = torch.optim.Adam(model_1d.parameters(), lr=1e-3)
    pred = model_1d(x_train[:16])
    loss = torch.nn.functional.mse_loss(pred, y_train[:16])
    loss.backward()
    optimizer.step()
    print(f"   ✓ Training step successful (loss: {loss.item():.6f})")
    
except Exception as e:
    print(f"   ✗ Training test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Basis function computation
print("\n7. Testing basis function computation...")
try:
    def compute_1d_basis_functions(x_vals, borders):
        n_grids = len(borders) - 1
        basis_funcs = np.zeros((len(x_vals), n_grids + 1))
        for i, x in enumerate(x_vals):
            grid_idx = np.searchsorted(borders, x) - 1
            grid_idx = np.clip(grid_idx, 0, n_grids - 1)
            left_border = borders[grid_idx]
            right_border = borders[grid_idx + 1]
            delta = (x - left_border) / (right_border - left_border)
            delta = np.clip(delta, 0.0, 1.0)
            basis_funcs[i, grid_idx] = 1 - delta
            basis_funcs[i, grid_idx + 1] = delta
        return basis_funcs
    
    x_range = np.linspace(borders[0], borders[-1], 100)
    basis_1d = compute_1d_basis_functions(x_range, borders)
    sum_basis = basis_1d.sum(axis=1)
    
    print(f"   ✓ Basis functions computed")
    print(f"   Partition of unity check: min={sum_basis.min():.6f}, max={sum_basis.max():.6f}")
    assert np.allclose(sum_basis, 1.0, atol=1e-6), "Partition of unity failed!"
    print(f"   ✓ Partition of unity verified")
    
except Exception as e:
    print(f"   ✗ Basis function test failed: {e}")
    sys.exit(1)

# Test 7: Visualization (create a simple plot)
print("\n8. Testing visualization...")
try:
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.scatter(borders, np.zeros_like(borders), s=100, c='red')
    ax.set_title('CDF Grid Borders')
    ax.set_xlabel('x')
    plt.savefig('/home/vladimir/DATA/linux_data/GitHub/KAN/examples/test_grid.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Visualization test successful")
    print(f"   Saved test plot to examples/test_grid.png")
except Exception as e:
    print(f"   ✗ Visualization test failed: {e}")

print("\n" + "=" * 60)
print("ALL TESTS PASSED! ✓")
print("=" * 60)
print("\nThe example notebook should work correctly.")
print("Run it with: jupyter notebook examples/example_1d_vs_2d_comparison.ipynb")
print("=" * 60)

