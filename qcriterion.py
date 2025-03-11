import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Parameters and Domain Settings
# -------------------------------
N = 128              # Grid size
L = 2 * np.pi        # Domain length (periodic domain)
dx = L / N
x = np.linspace(0, L, N, endpoint=False)
y = np.linspace(0, L, N, endpoint=False)
X, Y = np.meshgrid(x, y)

# -------------------------------
# Fourier Space Setup
# -------------------------------
kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi
ky = np.fft.fftfreq(N, d=dx) * 2 * np.pi
kx, ky = np.meshgrid(kx, ky)
k2 = kx**2 + ky**2
k = np.sqrt(k2)

# -----------------------------------------------------
# Amplitude Scaling for a Turbulent Field (Kolmogorov scaling)
# -----------------------------------------------------
amplitude = np.zeros_like(k)
nonzero = k > 0
amplitude[nonzero] = k[nonzero]**(-11/6)

# Generate a random phase field for each Fourier mode
random_phase = np.exp(1j * 2 * np.pi * np.random.rand(N, N))
base_field = amplitude * random_phase

# -----------------------------------------------
# Construct the Velocity Field in Fourier Space
# -----------------------------------------------
u_hat_x = base_field.copy()
u_hat_y = base_field.copy()

# Enforce Incompressibility: Project out the component along k.
k2_nozero = np.where(k2 == 0, 1.0, k2)  # Avoid division by zero
divergence = (kx * u_hat_x + ky * u_hat_y) / k2_nozero
u_hat_x -= kx * divergence
u_hat_y -= ky * divergence

# Transform to Physical Space
u_x = np.real(np.fft.ifft2(u_hat_x))
u_y = np.real(np.fft.ifft2(u_hat_y))

# -------------------------------
# Compute the Velocity Gradients
# -------------------------------
# Using np.gradient (assumes uniform grid spacing dx)
dux_dx, dux_dy = np.gradient(u_x, dx, dx)
duy_dx, duy_dy = np.gradient(u_y, dx, dx)

# -------------------------------
# Compute the Strain Rate Tensor (S) and Vorticity Tensor (Ω)
# -------------------------------
S_xx = dux_dx
S_yy = duy_dy
S_xy = 0.5 * (dux_dy + duy_dx)

# For 2D, the vorticity tensor has one independent component:
omega = duy_dx - dux_dy  # Scalar vorticity
Omega_xy = 0.5 * omega

# Compute norms:
norm_S2 = S_xx**2 + 2 * S_xy**2 + S_yy**2  # ||S||²
norm_Omega2 = 2 * Omega_xy**2               # ||Ω||²

# -------------------------------
# Calculate the Q-Criterion
# -------------------------------
Q = 0.5 * (norm_Omega2 - norm_S2)

# -------------------------------
# Visualization of the Q-Criterion
# -------------------------------
plt.figure(figsize=(6, 5))
contour = plt.contourf(X, Y, Q, levels=50, cmap='RdBu_r')
plt.colorbar(contour, label='Q-criterion')
plt.title('Q-criterion of the Synthetic Turbulent Velocity Field')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()
