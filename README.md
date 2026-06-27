# Sharpness-Aware Minimization for Robust Inverse Design of Nonlocal Thin Films

PyTorch implementation for the robust inverse design of angle- and wavelength-dependent optical thin-film responses using a differentiable transfer-matrix method (TMM) and sharpness-aware gradient descent (SAGD).

This repository accompanies the paper:

> Y. Ma, G. Hu, and L. Cao, **Sharpness-Aware Minimization for Robust Inverse Design of Nonlocal Thin Films**, *Laser & Photonics Reviews* (2026). DOI: [10.1002/lpor.71366](https://doi.org/10.1002/lpor.71366)

## Overview

Multilayer thin films can exhibit strongly nonlocal optical responses, where the transmission and reflection depend on both wavelength and incident angle. This code provides a compact differentiable framework for designing such responses by optimizing the physical layer thicknesses directly.

The core idea is to combine:

- **Differentiable coherent TMM** for forward simulation of multilayer films.
- **Automatic differentiation in PyTorch** for thickness-gradient calculation.
- **Sharpness-aware optimization** to improve robustness against fabrication perturbations.
- **Angle–wavelength response visualization** for checking the designed nonlocal optical transfer function.

The default example optimizes a 35-layer alternating TiO2/SiO2 film to approximate a target angular transmittance profile over a narrow wavelength band.

## Features

- **Differentiable multilayer optics**  
  Coherent TMM is implemented as a `torch.nn.Module`, allowing gradients to propagate through the layer thicknesses.

- **s- and p-polarization support**  
  The forward model computes transmittance, reflectance, absorption, and complex transmission/reflection coefficients for both polarizations.

- **Sharpness-aware gradient descent**  
  The optimization loop first perturbs the thickness vector along the gradient direction and then updates the design using the loss evaluated at the perturbed point.

- **Material-dispersion support**  
  Refractive indices are loaded from CSV files and interpolated over the wavelength grid.

- **GPU-ready implementation**  
  The simulation is written in PyTorch and can run on CUDA by changing the device setting.

- **Built-in visualization**  
  The example script generates convergence curves and high-resolution angle–wavelength transmittance maps.

## Repository Structure

```text
SAGD/
├── SAGD_main.py              # Main optimization example
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── data/                     # Refractive-index data files
│   ├── SiO2_Thinfilm.csv
│   └── TiO2_Thinfilm.csv
├── results/                  # Generated figures and optimization outputs
└── src/
    ├── tmm_layer.py          # Differentiable coherent TMM layer
    └── utils.py              # Refractive-index interpolation and plotting tools
```

If your local repository keeps `tmm_layer.py` and `utils.py` in the root directory rather than in `src/`, either move them into `src/` or update the import statements in `SAGD_main.py`.

## Installation

Clone the repository and install the required Python packages:

```bash
git clone https://github.com/THUHoloLab/SAGD.git
cd SAGD
pip install -r requirements.txt
```

The minimal dependencies are:

```text
numpy
matplotlib
torch
scipy
pandas
tqdm
```

For GPU acceleration, install a CUDA-enabled PyTorch build following the official PyTorch installation instructions, then set the device in `SAGD_main.py`:

```python
CFG = {
    ...
    'device': 'cuda'
}
```

## Material Data Format

Place material dispersion files in the `data/` directory. Each CSV file should contain three columns:

```text
wl,n,k
0.600,1.46,0.0000
0.610,1.46,0.0000
...
```

where:

- `wl` is the wavelength in micrometers.
- `n` is the real part of the refractive index.
- `k` is the extinction coefficient.

The default script expects:

```text
data/SiO2_Thinfilm.csv
data/TiO2_Thinfilm.csv
```

Please check the complex-index sign convention in `src/utils.py` before using new material data.

## Quick Start

Run the default optimization example:

```bash
python SAGD_main.py
```

The script will:

1. Load the TiO2 and SiO2 refractive-index data.
2. Initialize a 35-layer alternating TiO2/SiO2 multilayer.
3. Optimize the layer thicknesses using SAGD.
4. Simulate the optimized film on a dense angle–wavelength grid.
5. Save the output figures in `results/`.

Expected output files include:

```text
results/convergence.png
results/final_spectrum_s.png
results/final_spectrum_p.png
```

## Default Optimization Example

The main configuration is defined in `SAGD_main.py`:

```python
CFG = {
    'layer_num': 35,
    'num_epochs': 200,
    'base_lr': 1e-4,
    'rho': 0.05,
    'd_ini_min': 0.0,
    'd_ini_max': 0.5,
    'data_dir': './data',
    'output_dir': './results',
    'device': 'cpu'
}
```

The default training grid is:

```python
lambdas_train = torch.linspace(0.620, 0.630, 6)
theta_rad_train = torch.linspace(0, 0.6435, 10)
```

The target response is a normalized angular `sin^2(theta)` transmittance profile:

```python
sin_sq = torch.sin(theta_rad_train) ** 2
target_T = (sin_sq - sin_sq.min()) / (sin_sq.max() - sin_sq.min())
target_T = target_T.repeat(len(lambdas_train), 1)
```

The loss includes both polarizations:

```python
loss = MSE(T_s, target_T) + MSE(T_p, target_T) + regularization
```

## How SAGD Is Implemented

For each epoch, the optimizer performs two forward/backward passes:

1. **Ascent step**  
   Compute the thickness gradient and perturb the design by

   ```python
   epsilon = rho * grad / (||grad|| + 1e-12)
   ```

2. **Descent step**  
   Evaluate the loss at the perturbed design, restore the original thicknesses, and update them using the sharpness-aware gradient.

This encourages convergence toward flatter minima in the thickness landscape, which can improve tolerance to fabrication errors.

## Customizing the Design Target

To design a different nonlocal thin-film response, modify three parts of `SAGD_main.py`:

### 1. Wavelength and angle grids

```python
lambdas_train = torch.linspace(lambda_min, lambda_max, num_lambda)
theta_rad_train = torch.linspace(theta_min, theta_max, num_theta)
```

### 2. Target transfer function

Replace the default `sin^2(theta)` target with your desired 2D response:

```python
target_T = your_target_function(lambdas_train, theta_rad_train)
```

The target tensor should have the shape:

```text
[num_wavelengths, num_angles]
```

### 3. Material stack

The default stack alternates TiO2 and SiO2:

```python
n_layers_train = torch.stack([
    n_tio2 if i % 2 == 0 else n_sio2
    for i in range(CFG['layer_num'])
])
```

You can replace this with another material sequence or add more materials if their dispersion data are available.

## Core API

### `TMMLayer`

```python
from src.tmm_layer import TMMLayer

model = TMMLayer(d_ini)
T, R, A, t, r = model(
    n_list,
    n_0,
    n_sub,
    theta0_range,
    lambda_range,
    polar='s'
)
```

Returns:

- `T`: transmittance
- `R`: reflectance
- `A`: absorption, calculated as `1 - R - T`
- `t`: complex transmission coefficient
- `r`: complex reflection coefficient

### `interpolate_refractive_index`

```python
from src.utils import interpolate_refractive_index

n_material = interpolate_refractive_index(
    './data/SiO2_Thinfilm.csv',
    lambda_range_tensor
)
```

### Visualization

```python
from src.utils import plot_transmittance_map, plot_loss

plot_loss(loss_history, save_path='./results/convergence.png')
plot_transmittance_map(T, lambdas, theta_deg, save_path='./results/final_spectrum.png')
```

## Notes and Practical Tips

- Thicknesses are represented in micrometers in the default script.
- Angles are represented in radians during simulation and converted to degrees only for plotting.
- The default `regular_loss` penalizes negative thickness values but does not impose an upper thickness bound.
- Increasing `rho` strengthens the sharpness-aware perturbation but may make optimization less stable.
- Increasing the number of wavelengths and angles improves design fidelity but increases memory and runtime.
- For reproducibility, you may set random seeds before initializing the thickness vector.

Example:

```python
import torch
import numpy as np

torch.manual_seed(0)
np.random.seed(0)
```

## Citation

If this code is useful for your research, please cite:

```bibtex
@article{ma2026sharpness,
  title   = {Sharpness-Aware Minimization for Robust Inverse Design of Nonlocal Thin Films},
  author  = {Ma, Yuchen and Hu, Guangwei and Cao, Liangcai},
  journal = {Laser \& Photonics Reviews},
  year    = {2026},
  doi     = {10.1002/lpor.71366}
}
```

## License

Please refer to the `LICENSE` file in this repository. If no license is provided, all rights are reserved by the authors.

## Contact

For questions, bug reports, or collaboration inquiries, please open an issue in this repository.
