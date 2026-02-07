import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patheffects import withStroke
from scipy import interpolate

def interpolate_refractive_index(file_path, lambda_range_tensor):
    """
    Interpolates refractive index (n + ik) from a CSV file.
    Args:
        file_path: Path to CSV with columns 'wl', 'n', 'k'.
        lambda_range_tensor: Tensor of wavelengths to query.
    Returns:
        Complex numpy array of refractive indices.
    """
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found. Returning dummy data.")
        return np.ones_like(lambda_range_tensor.cpu().numpy()) * 1.5

    wl = data['wl'].values
    n = data['n'].values
    k = data['k'].values
    
    # Interpolation
    interp_n = interpolate.InterpolatedUnivariateSpline(wl, n, k=3)
    interp_k = interpolate.InterpolatedUnivariateSpline(wl, k, k=3)
    
    # Convert input tensor to numpy for scipy
    x_query = lambda_range_tensor.detach().cpu().numpy()
    
    n_real = interp_n(x_query)
    n_imag = interp_k(x_query)
    
    return n_real - 1j * n_imag

def plot_transmittance_map(T_matrix, lambda_range, theta_range_deg, 
                           title="Transmittance", save_path=None):
    """
    Plots a 2D heatmap of transmittance.
    """
    plt.figure(figsize=(8, 6))
    
    # Convert inputs to numpy
    if isinstance(T_matrix, torch.Tensor): T_matrix = T_matrix.detach().cpu().numpy()
    if isinstance(lambda_range, torch.Tensor): lambda_range = lambda_range.detach().cpu().numpy()
    if isinstance(theta_range_deg, torch.Tensor): theta_range_deg = theta_range_deg.detach().cpu().numpy()

    extent = (theta_range_deg.min(), theta_range_deg.max(),
              lambda_range.min(), lambda_range.max())

    plt.imshow(T_matrix, extent=extent, aspect='auto', origin='lower', cmap='magma', vmin=0, vmax=1)
    plt.colorbar(label='Transmittance')
    plt.xlabel('Incident Angle (degree)')
    plt.ylabel('Wavelength (μm)')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot to {save_path}")
    plt.show()

def plot_loss(history, save_path=None):
    plt.figure(figsize=(8, 5))
    plt.plot(history, label='Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Optimization Convergence')
    plt.legend()
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path)
    plt.show()