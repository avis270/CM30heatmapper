import io, os, tempfile, zipfile
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe for packaging/headless
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import streamlit as st


# ---------- Helper Functions ----------
def detect_plate_type(raw: str) -> str:
    """Detect plate type by reading the <vessel Type> section."""
    for line in raw.splitlines():
        if line.strip().endswith("well"):
            return line.strip().split(",")[-1].lower()  # e.g. '96well'
    return "96well"  # default fallback


def parse_file(uploaded_file):
    """Extract well data from CM30 CSV file."""
    raw = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    plate_type = detect_plate_type(raw)

    # Decide dimensions
    plate_dims = {
        "6well": (2, 3),
        "12well": (3, 4),
        "24well": (4, 6),
        "96well": (8, 12),
    }
    if plate_type not in plate_dims:
        raise ValueError(f"Unsupported plate type: {plate_type}")

    # Extract colony forming results / single results
    data = {}
    current_well = None
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("Well"):
            current_well = line.replace("Well", "").split(",")[0]
            data[current_well] = []
        elif current_well and ("," in line):
            parts = line.split(",")
            if len(parts) >= 4 and parts[1] and parts[2].replace(".", "").isdigit():
                # Time, Confluency, Count
                try:
                    data[current_well].append(
                        {
                            "time": parts[1],
                            "confluency": float(parts[2]),
                            "count": float(parts[3]),
                        }
                    )
                except ValueError:
                    continue

    if not data:
        raise ValueError("Parsed no usable data from file")

    return plate_type, plate_dims[plate_type], data


def plot_plate(plate_shape, data, time_index, vmin_color, vmax_color):
    """Generate a heatmap of the plate at a specific time index."""
    rows, cols = plate_shape
    fig, ax = plt.subplots(figsize=(cols, rows))
    cmap = mcolors.LinearSegmentedColormap.from_list("custom", [vmin_color, vmax_color])

    for i, well in enumerate(sorted(data.keys())):
        r, c = divmod(i, cols)
        if time_index < len(data[well]):
            val = data[well][time_index]["confluency"]
        else:
            val = 0
        rect = patches.Rectangle((c, rows - r - 1), 1, 1,
                                 facecolor=cmap(val / 100),
                                 edgecolor="black")
        ax.add_patch(rect)
        ax.text(c + 0.5, rows - r - 0.5, f"{val:.1f}%", ha="center", va="center", fontsize=8)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.axis("off")
    plt.tight_layout()
    return fig


# ---------- Streamlit App ----------
st.title("CM30 Plate Heatmapper — Multi-Plate Support")

uploaded_file = st.file_uploader("Upload CM30 CSV", type="csv")

if uploaded_file:
    try:
        plate_type, plate_shape, data = parse_file(uploaded_file)
        st.success(f"Detected plate type: **{plate_type}** ({plate_shape[0]}x{plate_shape[1]})")

        # Number of timepoints (take from first well)
        n_timepoints = max(len(v) for v in data.values())
        time_index = st.slider("Timepoint Index", 0, n_timepoints - 1, 0)

        vmin_color = st.color_picker("Color for 0% confluency", "#ffffff")
        vmax_color = st.color_picker("Color for 100% confluency", "#ff0000")

        fig = plot_plate(plate_shape, data, time_index, vmin_color, vmax_color)
        st.pyplot(fig)

        # Download buttons
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=220)
        st.download_button("Download Current Timepoint (PNG)",
                           data=buf.getvalue(),
                           file_name=f"plate_{plate_type}_t{time_index}.png",
                           mime="image/png")

        # Zip of all timepoints
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            for i in range(n_timepoints):
                fig = plot_plate(plate_shape, data, i, vmin_color, vmax_color)
                img_bytes = io.BytesIO()
                fig.savefig(img_bytes, format="png", dpi=220)
                zf.writestr(f"plate_t{i}.png", img_bytes.getvalue())
        st.download_button("Download All Timepoints (ZIP)",
                           data=zip_buf.getvalue(),
                           file_name=f"plate_{plate_type}_all_timepoints.zip",
                           mime="application/zip")

    except Exception as e:
        st.error(f"Could not parse file: {e}")
