import io
import numpy as np
from PIL import Image

def prepare_grid_image(arr_2d, grid_size):
    """Resizes a 2D array to grid_size x grid_size, normalizes to [0, 1], and masks to unit circle."""
    img_pil = Image.fromarray(arr_2d.astype(np.float64)).resize((grid_size, grid_size), Image.Resampling.BILINEAR)
    img = np.flipud(np.array(img_pil, dtype=np.float64))

    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())

    coords = np.linspace(-1.0, 1.0, grid_size)
    X, Y = np.meshgrid(coords, coords)
    img[(X**2 + Y**2) > 1.0] = 0.0
    return img

def load_custom_image(file_bytes, file_name, grid_size):
    buf = io.BytesIO(file_bytes)
    if file_name.endswith(".npy"):
        arr = np.load(buf).astype(np.float64)
        if arr.ndim == 3:
            arr = arr.mean(axis=-1)
    else:
        arr = np.array(Image.open(buf).convert("L"), dtype=np.float64)
    return prepare_grid_image(arr, grid_size)

def load_medical_database_slice(sample_index=0, grid_size=128):
    """Fetches a real 2D axial CT slice from the OrganAMNIST clinical dataset."""
    from medmnist import OrganAMNIST
    dataset = OrganAMNIST(split="test", download=True, size=128)
    img_array, _ = dataset[int(sample_index) % len(dataset)]
    return prepare_grid_image(img_array.squeeze(), grid_size)

def forward_project_image(image, angles_deg, num_rays=127, t_max=1.0):
    """Discrete Radon transform via bilinear ray integration (Kak & Slaney Eq. 7.25)."""
    grid_size = image.shape[0]
    coords = np.linspace(-1.0, 1.0, grid_size)
    delta = coords[1] - coords[0]
    ds = 0.5 * delta

    t_vals = np.linspace(-t_max, t_max, num_rays)
    num_samples = int(np.ceil(2.0 * np.sqrt(2.0) / ds)) + 1
    s_vals = np.linspace(-np.sqrt(2.0), np.sqrt(2.0), num_samples)
    T_grid, S_grid = np.meshgrid(t_vals, s_vals, indexing="ij")

    sinogram = np.zeros((len(angles_deg), num_rays), dtype=np.float64)
    for i, theta in enumerate(np.deg2rad(angles_deg)):
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        col_f = ((T_grid * cos_t - S_grid * sin_t) + 1.0) / delta
        row_f = ((T_grid * sin_t + S_grid * cos_t) + 1.0) / delta

        c0, r0 = np.floor(col_f).astype(np.int32), np.floor(row_f).astype(np.int32)
        c1, r1 = c0 + 1, r0 + 1
        valid = (c0 >= 0) & (c1 < grid_size) & (r0 >= 0) & (r1 < grid_size)

        dc, dr = col_f - c0, row_f - r0
        w00 = np.where(valid, (1.0 - dc) * (1.0 - dr), 0.0)
        w10 = np.where(valid, dc * (1.0 - dr), 0.0)
        w01 = np.where(valid, (1.0 - dc) * dr, 0.0)
        w11 = np.where(valid, dc * dr, 0.0)

        c0_s, c1_s = np.clip(c0, 0, grid_size - 1), np.clip(c1, 0, grid_size - 1)
        r0_s, r1_s = np.clip(r0, 0, grid_size - 1), np.clip(r1, 0, grid_size - 1)

        f_interp = (
            w00 * image[r0_s, c0_s]
            + w10 * image[r0_s, c1_s]
            + w01 * image[r1_s, c0_s]
            + w11 * image[r1_s, c1_s]
        )
        sinogram[i, :] = np.sum(f_interp, axis=1) * ds

    return sinogram, t_vals