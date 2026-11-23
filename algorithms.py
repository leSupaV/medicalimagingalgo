import numpy as np
from scipy.fft import fft, ifft, fftfreq

def reconstruct_fbp(sinogram, angles_deg, t_vals, output_size=128, zero_pad=True, window="hamming"):
    num_angles, num_rays = sinogram.shape
    dt = t_vals[1] - t_vals[0]
    padded_len = max(64, int(2 ** np.ceil(np.log2(2 * num_rays)))) if zero_pad else num_rays
    
    padded_sino = np.zeros((num_angles, padded_len), dtype=np.float64)
    padded_sino[:, :num_rays] = sinogram
    
    freqs = fftfreq(padded_len, d=dt)
    ramp_filter = np.abs(freqs)
    norm_freqs = fftfreq(padded_len)
    
    if window == "hamming":
        win = 0.54 + 0.46 * np.cos(2.0 * np.pi * norm_freqs)
    elif window == "cosine":
        win = np.cos(np.pi * norm_freqs)
    else:
        win = np.ones(padded_len)
        
    sino_fft = fft(padded_sino, axis=1)
    filtered_sino = np.real(ifft(sino_fft * (ramp_filter * win), axis=1))[:, :num_rays]
    
    recon = np.zeros((output_size, output_size), dtype=np.float64)
    coords = np.linspace(-1.0, 1.0, output_size)
    X, Y = np.meshgrid(coords, coords)
    d_theta = np.pi / num_angles
    
    for i, theta in enumerate(np.deg2rad(angles_deg)):
        t_grid = X * np.cos(theta) + Y * np.sin(theta)
        recon += np.interp(t_grid, t_vals, filtered_sino[i, :], left=0.0, right=0.0)
        
    return recon * d_theta

def reconstruct_sart(sinogram, angles_deg, t_vals, output_size=128, iterations=1, 
                     relaxation=1.0, use_hamming=True, non_negative=True):
    num_angles, num_rays = sinogram.shape
    coords = np.linspace(-1.0, 1.0, output_size)
    delta = coords[1] - coords[0]
    ds = 0.5 * delta
    
    X, Y = np.meshgrid(coords, coords)
    recon_circle = (X**2 + Y**2) <= 1.0
    g = np.zeros((output_size, output_size), dtype=np.float64)
    
    step = max(1, int(round(num_angles * (73.8 / 180.0))))
    while np.gcd(step, num_angles) != 1 and step < num_angles:
        step += 1
    angle_order = [(k * step) % num_angles for k in range(num_angles)]
    
    num_samples = int(np.ceil(2.0 / ds)) + 1
    s_vals = np.linspace(-1.0, 1.0, num_samples)
    T_grid, S_grid = np.meshgrid(t_vals, s_vals, indexing='ij')
    inside_circle = (T_grid**2 + S_grid**2) <= 1.0
    L_i = np.sum(inside_circle, axis=1) * ds
    
    ray_weights = np.zeros_like(T_grid)
    for r in range(num_rays):
        valid_idx = np.where(inside_circle[r])[0]
        M_i = len(valid_idx)
        if M_i > 1:
            ray_weights[r, valid_idx] = (0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(M_i) / (M_i - 1))) if use_hamming else 1.0
        elif M_i == 1:
            ray_weights[r, valid_idx] = 1.0

    angles_rad = np.deg2rad(angles_deg)
    for _ in range(iterations):
        for idx in angle_order:
            theta = angles_rad[idx]
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            col_f = ((T_grid * cos_t - S_grid * sin_t) + 1.0) / delta
            row_f = ((T_grid * sin_t + S_grid * cos_t) + 1.0) / delta
            
            c0, r0 = np.floor(col_f).astype(np.int32), np.floor(row_f).astype(np.int32)
            c1, r1 = c0 + 1, r0 + 1
            valid = inside_circle & (c0 >= 0) & (c1 < output_size) & (r0 >= 0) & (r1 < output_size)
            
            dc, dr = col_f - c0, row_f - r0
            w00 = np.where(valid, (1.0 - dc) * (1.0 - dr), 0.0)
            w10 = np.where(valid, dc * (1.0 - dr), 0.0)
            w01 = np.where(valid, (1.0 - dc) * dr, 0.0)
            w11 = np.where(valid, dc * dr, 0.0)
            
            c0_s, c1_s = np.clip(c0, 0, output_size - 1), np.clip(c1, 0, output_size - 1)
            r0_s, r1_s = np.clip(r0, 0, output_size - 1), np.clip(r1, 0, output_size - 1)
            
            f_interp = w00 * g[r0_s, c0_s] + w10 * g[r0_s, c1_s] + w01 * g[r1_s, c0_s] + w11 * g[r1_s, c1_s]
            q_i = np.sum(f_interp, axis=1) * ds
            
            residual = np.zeros(num_rays, dtype=np.float64)
            active = L_i > 1e-6
            residual[active] = (sinogram[idx, active] - q_i[active]) / L_i[active]
            
            weighted_corr = residual[:, None] * ray_weights * ds
            weight_term = ray_weights * ds
            
            num_acc, den_acc = np.zeros_like(g), np.zeros_like(g)
            for w_c, r_i, c_i in [(w00, r0_s, c0_s), (w10, r0_s, c1_s), (w01, r1_s, c0_s), (w11, r1_s, c1_s)]:
                np.add.at(num_acc, (r_i, c_i), weighted_corr * w_c)
                np.add.at(den_acc, (r_i, c_i), weight_term * w_c)
                
            upd = (den_acc > 1e-8) & recon_circle
            g[upd] += relaxation * (num_acc[upd] / den_acc[upd])
            if non_negative:
                g = np.maximum(g, 0.0)
    return g