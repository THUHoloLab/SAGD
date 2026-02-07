import torch
import torch.nn as nn
import sys

class TMMLayer(nn.Module):
    def __init__(self, d_list_tensor):
        """
        Initialize the TMM Layer.
        
        Args:
            d_list_tensor (Tensor): Initial thicknesses of the layers.
                                    Must be a tensor to maintain the computation graph.
        """
        super(TMMLayer, self).__init__()
        # Use the passed tensor directly to maintain gradient flow
        self.d_list = d_list_tensor
        self.layer_num = len(d_list_tensor)

    def _correct_forward_angle(self, n, theta_guess):
        """
        Ensures the calculated angle corresponds to the forward-propagating wave.
        Handles complex refractive indices and evanescent waves.
        """
        EPSILON = 1e-12

        ncostheta = n * torch.cos(theta_guess)
        
        # Check if wave is evanescent or lossy (imaginary part is significant)
        is_evanescent_or_lossy = torch.abs(ncostheta.imag) > EPSILON
        
        # Direction logic
        forward_if_lossy = ncostheta.imag > 0
        forward_if_not_lossy = ncostheta.real > 0
        
        is_forward = torch.where(is_evanescent_or_lossy, forward_if_lossy, forward_if_not_lossy)
        
        # Correct the angle if not forward
        corrected_theta = torch.where(is_forward, theta_guess, torch.pi - theta_guess)
        
        return corrected_theta
    
    def _coh_tmm_compute(self, n_all, d_all, lambda_grid, theta0_grid, polar):
        """
        Vectorized Coherent TMM Computation.
        
        Returns:
            T, R, A, t, r (Transmittance, Reflectance, Absorption, trans coeff, ref coeff)
        """
        N = d_all.shape[0] - 2  # Number of thin film layers

        # --- Step 1: Snell's Law for angles in all layers ---
        sin_theta_all = (n_all[0] * torch.sin(theta0_grid)) / n_all
        theta_all_guess = torch.arcsin(sin_theta_all)

        # Correct input and output layer angles
        theta_all = theta_all_guess.clone()
        theta_all[0] = self._correct_forward_angle(n_all[0], theta_all_guess[0])
        theta_all[-1] = self._correct_forward_angle(n_all[-1], theta_all_guess[-1])

        # --- Step 2: Wave vectors and Phase ---
        kz_all = 2 * torch.pi * n_all * torch.cos(theta_all) / lambda_grid.unsqueeze(0)
        delta = kz_all[1:-1] * d_all[1:-1] # Phase shift for film layers only

        n_i, n_f = n_all[:-1], n_all[1:]
        th_i, th_f = theta_all[:-1], theta_all[1:]

        # Fresnel coefficients
        if polar == 's':
            r_if = (n_i * torch.cos(th_i) - n_f * torch.cos(th_f)) / (n_i * torch.cos(th_i) + n_f * torch.cos(th_f))
            t_if = (2 * n_i * torch.cos(th_i)) / (n_i * torch.cos(th_i) + n_f * torch.cos(th_f))
        elif polar == 'p':
            r_if = (n_f * torch.cos(th_i) - n_i * torch.cos(th_f)) / (n_f * torch.cos(th_i) + n_i * torch.cos(th_f))
            t_if = (2 * n_i * torch.cos(th_i)) / (n_f * torch.cos(th_i) + n_i * torch.cos(th_f))
        else: 
            raise ValueError("Polarization must be 's' or 'p'")

        # --- Step 3: Matrix Multiplication ---
        # Initialize System Matrix Mtilde
        Mtilde = torch.zeros(lambda_grid.shape + (2, 2), dtype=torch.complex64, device=n_all.device)
        Mtilde[..., 0, 0], Mtilde[..., 0, 1] = 1, r_if[0]
        Mtilde[..., 1, 0], Mtilde[..., 1, 1] = r_if[0], 1
        Mtilde /= t_if[0].unsqueeze(-1).unsqueeze(-1)

        # Loop through layers
        for i in range(N):
            # Propagation Matrix
            P_i = torch.zeros_like(Mtilde)
            P_i[..., 0, 0] = torch.exp(-1j * delta[i])
            P_i[..., 1, 1] = torch.exp(1j * delta[i])
            
            # Interface Matrix
            I_i1 = torch.zeros_like(Mtilde)
            I_i1[..., 0, 0], I_i1[..., 0, 1] = 1, r_if[i+1]
            I_i1[..., 1, 0], I_i1[..., 1, 1] = r_if[i+1], 1
            
            M_layer = (P_i @ I_i1) / t_if[i+1].unsqueeze(-1).unsqueeze(-1)
            Mtilde = Mtilde @ M_layer

        # --- Step 4: Final Coefficients ---
        r = Mtilde[..., 1, 0] / Mtilde[..., 0, 0]
        t = 1 / Mtilde[..., 0, 0]
        
        R = torch.abs(r)**2
        
        # Energy flux correction for Transmittance
        if polar == 's':
            T_num = (n_all[-1] * torch.cos(theta_all[-1])).real
            T_den = (n_all[0] * torch.cos(theta_all[0])).real
        else: # 'p'
            T_num = (n_all[-1] * torch.cos(theta_all[-1]).conj()).real
            T_den = (n_all[0] * torch.cos(theta_all[0]).conj()).real
        
        T = torch.abs(t)**2 * (T_num / T_den)
        A = 1 - R - T
        
        return T, R, A, t, r
    
    def forward(self, n_list, n_0, n_sub, theta0_range, lambda_range, polar, d_sub=0):
        """
        Forward pass for TMM.
        Prepares grids and expands tensors before calling compute.
        """
        # Create Meshgrid
        lambda_grid, theta0_grid = torch.meshgrid(lambda_range, theta0_range, indexing='ij')
        
        # Expand refractive indices to grid shape
        n_0_grid = n_0.unsqueeze(1).expand_as(lambda_grid)
        n_sub_grid = n_sub.unsqueeze(1).expand_as(lambda_grid)
        
        N = self.d_list.shape[0]
        n_list_expanded = n_list.unsqueeze(2).expand(N, *lambda_grid.shape)
        
        # Concatenate all N (Ambient + Layers + Substrate)
        n_all = torch.cat([n_0_grid.unsqueeze(0), n_list_expanded, n_sub_grid.unsqueeze(0)], dim=0)

        # Prepare thickness tensor (Infinite for ambient/substrate)
        inf_tensor = torch.tensor([float('inf')], device=self.d_list.device)
        d_all = torch.cat([inf_tensor, self.d_list, inf_tensor], dim=0).unsqueeze(1).unsqueeze(1)
        
        return self._coh_tmm_compute(n_all, d_all, lambda_grid, theta0_grid, polar)