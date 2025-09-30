import io, os, re
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe for packaging/headless
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import streamlit as st

# ------------------------
# Plate layouts definition
# ------------------------
PLATE_LAYOUTS = {
    "6well":   (["A", "B"], list(range(1, 4))),
    "12well":  (["A", "B", "C"], list(range(1, 5))),
    "24well":  (["A", "B", "C", "D"], list(range(1, 7))),
    "96well":  (list("ABCDEFGH"), list(range(1, 13))),
}

# ------------------------
# Plate detection
# ------------------------
def detect_plate_type(lines):
    """Detect plate type from the <vessel Type> section or file content."""
    for line in lines:
        lower = line.lower()
        if "well" in lower:
            tokens = re.split(r"[\t,]", line.strip())
            for t in tokens:
                t_clean = t.strip().lower()
                if t_clean in PLATE_LAYOUTS:
                    return t_clean
    return None

# ------------------------
# File parsing
# ------------------------
def parse_file(uploaded_file):
    content = uploaded_file.read().decode("utf-8", errors="ignore")
    lines = content.splitlines()

    plate_type = detect_plate_type(lines)
    if not plate_type:
        raise ValueError("Unsupported or undetected plate type in file.")

    # Parse <Single Result> or <Colony Forming Result>
    data = {}
    times = []
    current_well = None
    in_results = False
    for line in lines:
        line = line.strip()
        if line.startswith("<Single Result>") or line.startswith("<Colony Forming Result>"):
            in_results = True
            continue
        if not in_results or not line:
            continue

        if line.startswith("Well"):
            current_well = line.split()[0]
            data[current_well] = []
            continue

        if current_well and not line.startswith("Passage#"):
            parts = line.split(",")
            if len(parts) >= 4:
                try:
                    t = parts[1].strip()
                    conf = float(parts[2])
                    data[current_well].append((t, conf))
                    if t not in times:
                        times.append(t)
                except ValueError:
                    pass

    return plate_type, data, times

# ------------------------
# Heatmap rendering
# ------------------------
def render_heatmap(plate_type, data, timepoint, cmap="viridis", vmin=0, vmax=100):
    rows, cols = PLATE_LAYOUTS[plate_type]
    fig, ax = plt.subplots(figsize=(len(cols), len(rows)))

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap)

    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            well_id = f"Well{row}{col}"
            val = None
            if well_id in data:
                # find closest match to selected timepoint
                for t, conf in data[well_id]:
                    if t == timepoint:
                        val = conf
                        break
            color = cmap(norm(val)) if val is not None else "black"
            rect = patches.Rectangle((j, i), 1, 1, facecolor=color, edgecolor="white")
            ax.add_patch(rect)

            # Label with confluency % or "NA"
            label = f"{val:.1f}" if val is not None else "NA"
            ax.text(j + 0.5, i + 0.5, label, ha="center", va="center", color="white", fontsize=8)

    ax.set_xlim(0, len(cols))
    ax.set_ylim(0, len(rows))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()
    return fig

# ------------------------
# Streamlit app
# ------------------------
st.title("Well Plate Heatmap Viewer")

uploaded_file = st.file_uploader("Upload a plate CSV", type=["csv", "txt"])
if uploaded_file:
    try:
        plate_type, data, times = parse_file(uploaded_file)
        st.success(f"Detected plate type: {plate_type}")

        # Timepoint slider
        if times:
            idx = st.slider("Select timepoint index", 0, len(times)-1, 0)
            timepoint = times[idx]
            st.write(f"Selected time: **{timepoint}**")
        else:
            timepoint = None

        vmin = st.number_input("Minimum value (color scale)", value=0)
        vmax = st.number_input("Maximum value (color scale)", value=100)
        cmap = st.selectbox("Color map", ["viridis", "plasma", "inferno", "magma", "cividis"])

        if timepoint:
            fig = render_heatmap(plate_type, data, timepoint, cmap=cmap, vmin=vmin, vmax=vmax)
            st.pyplot(fig)

            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            st.download_button("Download Heatmap", buf.getvalue(), file_name="heatmap.png", mime="image/png")

    except Exception as e:
        st.error(f"Could not parse file: {e}")
