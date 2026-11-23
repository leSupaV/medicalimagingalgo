import numpy as np

SHEPP_LOGAN_TABLE_3_1 = [
    (0.0,   0.0,     0.92,   0.69,    90.0,  2.00),
    (0.0,  -0.0184,  0.874,  0.6624,  90.0, -0.98),
    (0.22,  0.0,     0.31,   0.11,    72.0, -0.02),
    (-0.22, 0.0,     0.41,   0.16,   108.0, -0.02),
    (0.0,   0.35,    0.25,   0.21,    90.0,  0.01),
    (0.0,   0.1,     0.046,  0.046,    0.0,  0.01),
    (0.0,  -0.1,     0.046,  0.046,    0.0,  0.01),
    (-0.08,-0.605,   0.046,  0.023,    0.0,  0.01),
    (0.0,  -0.605,   0.023,  0.023,    0.0,  0.01),
    (0.06, -0.605,   0.046,  0.023,   90.0,  0.01),
]

def generate_phantom_image(grid_size=128, ellipses=SHEPP_LOGAN_TABLE_3_1):
    coords = np.linspace(-1.0, 1.0, grid_size)
    X, Y = np.meshgrid(coords, coords)
    img = np.zeros((grid_size, grid_size), dtype=np.float64)
    for x0, y0, A, B, alpha_deg, rho in ellipses:
        alpha = np.deg2rad(alpha_deg)
        dx, dy = X - x0, Y - y0
        x_rot = dx * np.cos(alpha) + dy * np.sin(alpha)
        y_rot = -dx * np.sin(alpha) + dy * np.cos(alpha)
        img[(x_rot**2 / A**2) + (y_rot**2 / B**2) <= 1.0] += rho
    return img

def analytical_sinogram(angles_deg, num_rays=127, t_max=1.0, ellipses=SHEPP_LOGAN_TABLE_3_1):
    t_vals = np.linspace(-t_max, t_max, num_rays)
    angles_rad = np.deg2rad(angles_deg)
    sinogram = np.zeros((len(angles_rad), num_rays), dtype=np.float64)
    for i, theta in enumerate(angles_rad):
        proj_angle = np.zeros(num_rays, dtype=np.float64)
        for x0, y0, A, B, alpha_deg, rho in ellipses:
            alpha = np.deg2rad(alpha_deg)
            theta_eff = theta - alpha
            t_shift = t_vals - (x0 * np.cos(theta) + y0 * np.sin(theta))
            a_sq = (A * np.cos(theta_eff))**2 + (B * np.sin(theta_eff))**2
            inside = t_shift**2 < a_sq
            proj_angle[inside] += (2.0 * rho * A * B / a_sq) * np.sqrt(a_sq - t_shift[inside]**2)
        sinogram[i, :] = proj_angle
    return sinogram, t_vals