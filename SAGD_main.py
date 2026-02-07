import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm

# Import local modules
from src.tmm_layer import TMMLayer
from src.utils import interpolate_refractive_index, plot_transmittance_map, plot_loss

# --- Configuration ---
CFG = {
    'layer_num': 35,
    'num_epochs': 200,          # Reduced to 200 as requested
    'base_lr': 1e-4,            # Slightly higher LR for fewer epochs
    'rho': 0.05,                # Sharpness Aware Minimization radius
    'd_ini_min': 0.0,
    'd_ini_max': 0.5,           # Microns
    'data_dir': './data',
    'output_dir': './results',
    'device': 'cpu'
}

def regular_loss(d_list):
    """Penalty for negative thickness."""
    zero_val = torch.tensor(0.0, dtype=d_list.dtype, device=d_list.device)
    # Heavy penalty if thickness < 0
    return torch.sum(torch.where(d_list < 0, -1000.0 * d_list, zero_val))

def main():
    os.makedirs(CFG['output_dir'], exist_ok=True)
    device = torch.device(CFG['device'])
    print(f"Running on: {device}")

    # --- 1. Prepare Optimization Data (Low Res for Speed) ---
    print(">>> Preparing Training Data...")
    lambdas_train = torch.linspace(0.620, 0.630, 6).to(device)
    theta_rad_train = torch.linspace(0, 0.6435, 10).to(device) # ~0 to 36.8 deg
    
    # Target: Sin^2 variation with angle
    sin_sq = torch.sin(theta_rad_train) ** 2
    target_T = (sin_sq - sin_sq.min()) / (sin_sq.max() - sin_sq.min())
    target_T = target_T.repeat(len(lambdas_train), 1).to(torch.float32).to(device)

    # Load Materials (Interpolate for training wavelengths)
    ri_sio2_path = os.path.join(CFG['data_dir'], "SiO2_Thinfilm.csv")
    ri_tio2_path = os.path.join(CFG['data_dir'], "TiO2_Thinfilm.csv")
    
    n_sio2 = torch.tensor(interpolate_refractive_index(ri_sio2_path, lambdas_train), dtype=torch.complex64).to(device)
    n_tio2 = torch.tensor(interpolate_refractive_index(ri_tio2_path, lambdas_train), dtype=torch.complex64).to(device)

    # Stack layers (Alternating TiO2/SiO2)
    n_layers_train = torch.stack([n_tio2 if i % 2 == 0 else n_sio2 for i in range(CFG['layer_num'])])
    
    # Ambient and Substrate
    n_0 = torch.ones_like(lambdas_train, dtype=torch.complex64).to(device)   # Air
    n_G = torch.ones_like(lambdas_train, dtype=torch.complex64).to(device)   # Air/Glass
    d_sub = torch.tensor([0.0]).to(device)

    # --- 2. Initialize Model ---
    print(">>> Initializing Model...")
    # Random initial thickness
    d_ini = CFG['d_ini_min'] + (CFG['d_ini_max'] - CFG['d_ini_min']) * torch.rand(CFG['layer_num'], dtype=torch.float32)
    tmm_model = TMMLayer(d_ini.to(device))
    
    optimizer = optim.SGD([tmm_model.d_list], lr=CFG['base_lr'], momentum=0.9)
    loss_fn = nn.MSELoss()
    
    loss_history = []
    best_loss = float('inf')
    best_d_list = None

    # --- 3. Optimization Loop (SAGD) ---
    print(f">>> Starting Optimization ({CFG['num_epochs']} epochs, Rho={CFG['rho']})...")
    
    pbar = tqdm(range(CFG['num_epochs']))
    for epoch in pbar:
        tmm_model.d_list.requires_grad_(True)
        
        # --- SAM Step 1: Ascent (Find Perturbation) ---
        optimizer.zero_grad()
        
        # Forward pass (S and P pol)
        T_s, _, _, _, _ = tmm_model(n_layers_train, n_0, n_G, theta_rad_train, lambdas_train, 's', d_sub)
        T_p, _, _, _, _ = tmm_model(n_layers_train, n_0, n_G, theta_rad_train, lambdas_train, 'p', d_sub)
        
        loss_val = loss_fn(T_s.float(), target_T) + loss_fn(T_p.float(), target_T)
        reg = regular_loss(tmm_model.d_list)
        (loss_val + reg).backward()
        
        # Calculate Epsilon (Perturbation)
        with torch.no_grad():
            grad_norm = torch.norm(tmm_model.d_list.grad)
            scale = CFG['rho'] / (grad_norm + 1e-12)
            epsilon = tmm_model.d_list.grad * scale
            
            # Save original weights and apply perturbation
            d_original = tmm_model.d_list.data.clone()
            tmm_model.d_list.add_(epsilon)

        # --- SAM Step 2: Descent (Update at adversarial point) ---
        optimizer.zero_grad()
        
        T_s_adv, _, _, _, _ = tmm_model(n_layers_train, n_0, n_G, theta_rad_train, lambdas_train, 's', d_sub)
        T_p_adv, _, _, _, _ = tmm_model(n_layers_train, n_0, n_G, theta_rad_train, lambdas_train, 'p', d_sub)
        
        loss_adv = loss_fn(T_s_adv.float(), target_T) + loss_fn(T_p_adv.float(), target_T)
        reg_adv = regular_loss(tmm_model.d_list)
        
        total_loss = loss_adv + reg_adv
        total_loss.backward()
        
        # Restore original weights and apply gradients
        with torch.no_grad():
            tmm_model.d_list.copy_(d_original)
            
            # Save best result
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_d_list = d_original.clone()
        
        optimizer.step()
        
        loss_history.append(total_loss.item())
        pbar.set_description(f"Loss: {total_loss.item():.5f}")

    # Plot Convergence
    plot_loss(loss_history, save_path=os.path.join(CFG['output_dir'], 'convergence.png'))
    print(f">>> Optimization Done. Best Loss: {best_loss:.5f}")

    # --- 4. High-Resolution Simulation & Visualization ---
    print(">>> Running High-Resolution Simulation on Best Result...")
    
    # Define Dense Grid
    lambda_dense = torch.linspace(0.600, 0.650, 200).to(device)  # Wider range, higher res
    theta_dense_rad = torch.linspace(0, np.pi/4, 200).to(device) # 0 to 45 deg, higher res
    theta_dense_deg = theta_dense_rad * 180 / np.pi

    # Interpolate RI for dense grid
    n_sio2_dense = torch.tensor(interpolate_refractive_index(ri_sio2_path, lambda_dense), dtype=torch.complex64).to(device)
    n_tio2_dense = torch.tensor(interpolate_refractive_index(ri_tio2_path, lambda_dense), dtype=torch.complex64).to(device)
    
    n_layers_dense = torch.stack([n_tio2_dense if i % 2 == 0 else n_sio2_dense for i in range(CFG['layer_num'])])
    n_0_dense = torch.ones_like(lambda_dense, dtype=torch.complex64).to(device)
    n_G_dense = torch.ones_like(lambda_dense, dtype=torch.complex64).to(device)

    # Load best weights into model
    final_model = TMMLayer(best_d_list.to(device))

    # Run Forward Pass (No Grad needed)
    with torch.no_grad():
        T_s_high, _, _, _, _ = final_model(n_layers_dense, n_0_dense, n_G_dense, theta_dense_rad, lambda_dense, 's', d_sub)
        T_p_high, _, _, _, _ = final_model(n_layers_dense, n_0_dense, n_G_dense, theta_dense_rad, lambda_dense, 'p', d_sub)

    # Plot
    plot_transmittance_map(T_s_high, lambda_dense, theta_dense_deg, 
                          title="High-Res S-Polarization Transmittance",
                          save_path=os.path.join(CFG['output_dir'], 'final_spectrum_s.png'))
    
    plot_transmittance_map(T_p_high, lambda_dense, theta_dense_deg, 
                          title="High-Res P-Polarization Transmittance",
                          save_path=os.path.join(CFG['output_dir'], 'final_spectrum_p.png'))

    print(">>> Process Complete. Results saved in 'results/' folder.")

if __name__ == '__main__':
    main()