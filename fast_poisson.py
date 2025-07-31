"""
fast_poisson.py
---------------
Call `poisson_height(p, q)` where      p = -nx/nz   and   q = -ny/nz
Both p & q are float32 H×W arrays (masked values can be 0).
Returns a height map z, normalised to [0,1].

Requires only NumPy + SciPy.
"""
import numpy as np
from scipy.fft import fft2, ifft2, fftfreq

def poisson_height(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    assert p.shape == q.shape
    H, W = p.shape
    # 1. frequency grids
    fy, fx = np.meshgrid(fftfreq(H), fftfreq(W), indexing='ij')
    fx, fy = fx.astype(np.float32), fy.astype(np.float32)

    # 2. forward FFT of slopes
    P = fft2(p);  Q = fft2(q)

    # 3. solve −(kx²+ky²)·Z = i*kx*P + i*ky*Q
    denom = (fx**2 + fy**2)
    denom[0,0] = 1.0                         # avoid divide-by-0 at DC
    Z = (1j*fx)*P + (1j*fy)*Q
    Z /= denom

    # 4. inverse FFT → heights
    z = np.real(ifft2(Z))
    z -= z.min();  z /= z.ptp() + 1e-8       # [0,1] normalisation
    return z.astype(np.float32)
