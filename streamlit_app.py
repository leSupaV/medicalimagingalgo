import time
import numpy as np
import streamlit as st
from PIL import Image

from phantom_data import PHANTOM_PRESETS, generate_phantom_image, analytical_sinogram
from clinical_loader import load_custom_image, load_medical_database_slice, forward_project_image
from algorithms import reconstruct_fbp, reconstruct_sart

# ==============================================================================
# UI RENDERING HELPERS
# ==============================================================================
def render_equal_square_image(img, vmin, vmax, flip_vertical=True, display_px=512):
    if flip_vertical:
        img = np.flipud(img)
    clipped = np.clip((img - vmin) / max(1e-8, (vmax - vmin)), 0.0, 1.0)
    gray_uint8 = (clipped * 255).astype(np.uint8)
    pil_square = Image.fromarray(gray_uint8).resize((display_px, display_px), Image.Resampling.NEAREST)
    return np.array(pil_square, dtype=np.uint8)

def render_metric_card(title, value, bg_color="#262730"):
    st.markdown(
        f"""
        <div style="
            background-color: {bg_color};
            border-radius: 8px;
            height: 112px;
            padding: 12px 14px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            align-items: center;
            text-align: center;
            box-sizing: border-box;
        ">
            <div style="
                width: 100%;
                font-size: 0.74rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                color: #ffffff;
                line-height: 1.25;
                padding-bottom: 8px;
                border-bottom: 1px solid rgba(255, 255, 255, 0.22);
                display: flex;
                align-items: center;
                justify-content: center;
                min-height: 36px;
            ">
                {title}
            </div>
            <div style="
                width: 100%;
                font-size: 1.45rem;
                font-weight: 900;
                color: #ffffff;
                line-height: 1.1;
                padding-top: 4px;
            ">
                {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

@st.dialog("Fullscreen Image View", width="large")
def open_fullscreen_modal(title, img_uint8):
    st.subheader(title)
    st.image(img_uint8, use_container_width=True)

# ==============================================================================
# CACHED DATA PIPELINES
# ==============================================================================
@st.cache_data
def get_analytical_data(preset_name, n, n_ang, n_ray):
    ellipses = PHANTOM_PRESETS[preset_name]
    ang = np.linspace(0, 180, n_ang, endpoint=False)
    gt = generate_phantom_image(n, ellipses)
    sino, t = analytical_sinogram(ang, num_rays=n_ray, t_max=1.0, ellipses=ellipses)
    return gt, sino, ang, t

@st.cache_data
def get_uploaded_data(file_bytes, file_name, n, n_ang, n_ray):
    ang = np.linspace(0, 180, n_ang, endpoint=False)
    gt = load_custom_image(file_bytes, file_name, n)
    sino, t = forward_project_image(gt, ang, num_rays=n_ray, t_max=1.0)
    return gt, sino, ang, t

@st.cache_data
def get_database_data(sample_idx, n, n_ang, n_ray):
    ang = np.linspace(0, 180, n_ang, endpoint=False)
    gt = load_medical_database_slice(sample_idx, n)
    sino, t = forward_project_image(gt, ang, num_rays=n_ray, t_max=1.0)
    return gt, sino, ang, t

# ==============================================================================
# MAIN WORKSTATION LAYOUT
# ==============================================================================
st.set_page_config(page_title="Computed Tomography Reconstruction Workstation", layout="wide")

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
    </style>
    <div style="text-align: center; padding: 0.25rem 0 1.25rem 0;">
        <h1 style="margin-bottom: 0; font-weight: 800; color: #ffffff;">Computed Tomography Reconstruction Workstation</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Input Source")
    source_mode = st.selectbox(
        "Phantom Source",
        ["Textbook Phantom", "Clinical Database Slice", "Upload Custom Image"],
        help="Select between analytical textbook phantoms, real axial scans from OrganAMNIST, or a custom uploaded file.",
    )

    uploaded_file = None
    db_sample_idx = 0
    phantom_choice = "Shepp-Logan Head Phantom (Table 3.1)"

    if source_mode == "Textbook Phantom":
        phantom_choice = st.selectbox(
            "Phantom Variant",
            list(PHANTOM_PRESETS.keys()),
            help="Select the standard Shepp-Logan phantom or the Chapter 7 subdural hematoma variant.",
        )
    elif source_mode == "Clinical Database Slice":
        db_sample_idx = st.number_input(
            "Database Slice Index",
            min_value=0,
            max_value=500,
            value=0,
            step=1,
            help="Selects an axial computed tomography slice from the OrganAMNIST clinical dataset.",
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload Image",
            type=["png", "jpg", "jpeg", "npy"],
            help="Upload a PNG, JPG, JPEG, or NumPy (.npy) file to simulate forward projections and reconstruct it.",
        )

    st.divider()
    st.header("Scanner Geometry")
    grid_size = st.selectbox(
        "Reconstruction Grid",
        [64, 128, 256],
        index=1,
        help="Pixel resolution (N × N) of the reconstructed cross-sectional image.",
    )
    num_angles = st.slider(
        "Projection Angles",
        15,
        180,
        100,
        step=5,
        help="Total number of evenly spaced X-ray projection views acquired across 180 degrees.",
    )
    num_rays = st.slider(
        "Rays per Projection",
        31,
        255,
        127,
        step=2,
        help="Number of parallel detector measurements sampled at each projection angle.",
    )

    st.divider()
    st.header("Artifact & Noise Injection")
    noise_level = st.slider(
        "Detector Noise",
        0.0,
        0.05,
        0.0,
        step=0.005,
        help="Standard deviation of additive Gaussian noise injected into the sinogram measurements.",
    )
    ring_artifact = st.checkbox(
        "Ring Artifact",
        value=False,
        help="Simulates a miscalibrated detector element across all angles, producing concentric rings.",
    )
    bad_det = num_rays // 2 + 15 if num_rays > 40 else num_rays // 2

    st.divider()
    st.header("Reconstruction Method")
    algo = st.radio(
        "Algorithm",
        ["Side-by-Side", "Filtered Backprojection", "Simultaneous Algebraic Reconstruction"],
        index=0,
        help="Compare both reconstruction methods side-by-side (4 equal views) or inspect a single method.",
    )

    if algo in ("Filtered Backprojection", "Side-by-Side"):
        zero_pad = st.checkbox(
            "Zero-Pad Projections",
            value=True,
            help="Pads projections with zeros before the Fast Fourier Transform to prevent circular convolution dishing artifacts.",
        )
        fbp_window = st.selectbox(
            "Filter Window",
            ["hamming", "cosine", "ram-lak"],
            help="Frequency-domain window multiplied with the ramp filter to balance edge sharpness against high-frequency noise.",
        )

    if algo in ("Simultaneous Algebraic Reconstruction", "Side-by-Side"):
        sart_iters = st.slider(
            "Iterations",
            1,
            5,
            1,
            help="Number of complete passes through all projection angles.",
        )
        sart_hamming = st.checkbox(
            "Longitudinal Ray Hamming Window",
            value=True,
            help="Weights samples near the middle of each ray more heavily than noisy boundary pixels.",
        )
        sart_nonneg = st.checkbox(
            "Enforce Non-Negativity",
            value=True,
            help="Clips negative attenuation values to zero after each projection angle update.",
        )

    st.divider()
    st.header("Contrast Window")
    if source_mode == "Textbook Phantom":
        preset = st.selectbox(
            "Display Range",
            ["Soft Tissue [0.95 – 1.05]", "Full Range [0.0 – 2.0]"],
            help="Narrow window reveals subtle low-contrast brain structures; full range shows the entire skull and interior.",
        )
        vmin, vmax = (0.95, 1.05) if "Soft" in preset else (0.0, 2.0)
    else:
        vmin, vmax = st.slider(
            "Custom Contrast Range",
            0.0,
            1.0,
            (0.0, 1.0),
            step=0.05,
            help="Adjust minimum and maximum intensity clipping thresholds for display.",
        )

# Load Selected Phantom / Clinical Scan
if source_mode == "Clinical Database Slice":
    try:
        ground_truth, clean_sino, angles, t_vals = get_database_data(db_sample_idx, grid_size, num_angles, num_rays)
    except Exception as exc:
        st.error(f"Database load failed ({exc}). Falling back to Textbook Phantom.")
        ground_truth, clean_sino, angles, t_vals = get_analytical_data(phantom_choice, grid_size, num_angles, num_rays)
elif source_mode == "Upload Custom Image" and uploaded_file is not None:
    ground_truth, clean_sino, angles, t_vals = get_uploaded_data(
        uploaded_file.getvalue(), uploaded_file.name, grid_size, num_angles, num_rays
    )
else:
    ground_truth, clean_sino, angles, t_vals = get_analytical_data(phantom_choice, grid_size, num_angles, num_rays)

# Inject Artifacts
sinogram = clean_sino.copy()
if noise_level > 0:
    sinogram += np.random.default_rng(42).normal(0.0, noise_level, size=sinogram.shape)
if ring_artifact:
    sinogram[:, bad_det] *= 0.985

# Run Reconstructions
results = {}
if algo in ("Filtered Backprojection", "Side-by-Side"):
    t0 = time.perf_counter()
    img = reconstruct_fbp(sinogram, angles, t_vals, output_size=grid_size, zero_pad=zero_pad, window=fbp_window)
    results["Filtered Backprojection"] = {
        "img": img,
        "ms": (time.perf_counter() - t0) * 1000,
        "rmse": np.sqrt(np.mean((ground_truth - img)**2)),
        "info": "Analytical reconstruction computed by ramp-filtering projections in the frequency domain and backprojecting across the image plane.",
    }

if algo in ("Simultaneous Algebraic Reconstruction", "Side-by-Side"):
    with st.spinner("Running iterations..."):
        t0 = time.perf_counter()
        img = reconstruct_sart(
            sinogram,
            angles,
            t_vals,
            output_size=grid_size,
            iterations=sart_iters,
            use_hamming=sart_hamming,
            non_negative=sart_nonneg,
        )
        results["Simultaneous Algebraic Reconstruction"] = {
            "img": img,
            "ms": (time.perf_counter() - t0) * 1000,
            "rmse": np.sqrt(np.mean((ground_truth - img)**2)),
            "info": "Iterative algebraic reconstruction updating bilinear pixel weights simultaneously for each projection angle.",
        }

# Equal-Height Gray KPI Cards
total_cards = len(results) * 2 + 1
gray_shades = ["#262730", "#2f313c", "#383a47", "#414453", "#4a4e5f"]

metric_cols = st.columns(total_cards)
with metric_cols[0]:
    render_metric_card("VIEWS × RAYS", f"{num_angles} × {num_rays}", bg_color=gray_shades[0])

col_idx = 1
for name, d in results.items():
    with metric_cols[col_idx]:
        render_metric_card(
            f"{name} Time",
            f"{d['ms']:.1f} ms",
            bg_color=gray_shades[min(col_idx, len(gray_shades) - 1)],
        )
    with metric_cols[col_idx + 1]:
        render_metric_card(
            f"{name} Root Mean Square Error",
            f"{d['rmse']:.4f}",
            bg_color=gray_shades[min(col_idx + 1, len(gray_shades) - 1)],
        )
    col_idx += 2

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
st.subheader("Tomographic Image Views")

cards = [
    {
        "title": "Input Sinogram",
        "img": render_equal_square_image(sinogram, sinogram.min(), sinogram.max(), flip_vertical=False),
        "info": "Raw 2D projection data plotted across detector offset (horizontal) and scan angle (vertical).",
        "key": "sinogram_zoom",
    },
    {
        "title": "Ground Truth Phantom",
        "img": render_equal_square_image(ground_truth, vmin, vmax, flip_vertical=True),
        "info": "Reference cross-sectional attenuation distribution used to generate the simulated X-ray projections.",
        "key": "gt_zoom",
    },
]

for idx, (name, d) in enumerate(results.items()):
    cards.append(
        {
            "title": name,
            "img": render_equal_square_image(d["img"], vmin, vmax, flip_vertical=True),
            "info": d["info"],
            "key": f"recon_zoom_{idx}",
        }
    )

img_cols = st.columns(len(cards))
for col, card in zip(img_cols, cards):
    with col:
        with st.container(border=True):
            header_col, info_col = st.columns([0.86, 0.14], vertical_alignment="center")
            header_col.markdown(
                f"<div style='min-height: 44px; display: flex; align-items: center; font-weight: 700; line-height: 1.25;'>{card['title']}</div>",
                unsafe_allow_html=True,
            )
            info_col.markdown("", help=card["info"])
            st.image(card["img"], use_container_width=True)
            if st.button("Expand Fullscreen", key=card["key"], use_container_width=True):
                open_fullscreen_modal(card["title"], card["img"])