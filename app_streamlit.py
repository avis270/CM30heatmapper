import io, os, tempfile, re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import streamlit as st

# ---------------------------
# Detect plate type
# ---------------------------
def detect_plate_type(text):
    if "6well" in text.lower():
        return "6well"
    if "12well" in text.lower():
        return "12well"
    if "24well" in text.lower():
        return "24well"
    if "96well" in text.lower():
        return "96well"
    # fallback: check well naming
    if re.search(r"Well[A-H]\d+", text):
        return "96well"
    numbers = re.findall(r"Well\d+", text)
    if numbers:
        unique = len(set(numbers))
        if unique == 6: return "6well"
        if unique == 12: return "12well"
        if unique == 24: return "24well"
    return "unknown"

# ---------------------------
# Parse CM30 file robustly
# ---------------------------
def parse_cm30_file(uploaded_file):
    try:
        raw = uploaded_file.read().decode("utf-8", errors="ignore")
    except Exception:
        raw = uploaded_file.read().decode("latin-1", errors="ignore")

    plate_type = detect_plate_type(raw)

    # Find result section
    m = re.search(r"<(Colony Forming Result|Single Result)>", raw)
    if not m:
        raise ValueError("Could not find results section")

    # Extract lines after that section
    section = raw[m.end():].strip().splitlines()
    rows = []
    current_well = None

    for line in section:
        if not line.strip():
            continue
        if line.startswith("Well"):
            current_well = line.strip()
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        passage, time, confl, count = parts[:4]
        rows.append({
            "Well": current_well,
            "Passage": passage,
            "Time": time,
            "Confluency": pd.to_numeric(confl, errors="coerce"),
            "Count": pd.to_numeric(count, errors="coerce")
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Parsed no data rows")

    return plate_type, df

# ---------------------------
# Heatmap plotting
# ---------------------------
def plot_heatmap(plate_type, df, cmap_min="#0000ff", cmap_max="#ff0000"):
    dims = {"6well": (2,3), "12well": (3,4), "24well": (4,6), "96well": (8,12)}
    if plate_type not in dims:
        raise ValueError(f"Unsupported plate type: {plate_type}")
    rows, cols = dims[plate_type]

    summary = df.groupby("Well")["Confluency"].last()
    norm = mcolors.Normalize(vmin=summary.min(skipna=True), vmax=summary.max(skipna=True))
    cmap = mcolors.LinearSegmentedColormap.from_list("custom", [cmap_min, cmap_max])

    fig, ax = plt.subplots(figsize=(cols, rows))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.axis("off")

    wells = sorted(summary.index)
    for i, well in enumerate(wells):
        r, c = divmod(i, cols)
        val = summary[well]
        color = cmap(norm(val)) if pd.notnull(val) else "lightgray"
        rect = patches.Rectangle((c, rows-1-r), 1, 1, facecolor=color, edgecolor="black")
        ax.add_patch(rect)
        ax.text(c+0.5, rows-1-r+0.5,
                f"{val:.1f}" if pd.notnull(val) else "NA",
                ha="center", va="center", fontsize=8, color="white")
    return fig

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("CM30 Heatmapper (Iteration 1, Fixed Parser)")

uploaded_file = st.file_uploader("Upload CM30 CSV file", type=["csv"])
cmap_min = st.color_picker("Low value color", "#0000ff")
cmap_max = st.color_picker("High value color", "#ff0000")

if uploaded_file:
    try:
        plate_type, df = parse_cm30_file(uploaded_file)
        st.success(f"Detected plate type: {plate_type}")
        fig = plot_heatmap(plate_type, df, cmap_min, cmap_max)
        st.pyplot(fig)
    except Exception as e:
        st.error(str(e))
        st.text("Preview of first 30 lines:")
        uploaded_file.seek(0)
        st.text(uploaded_file.read(2000).decode("utf-8", errors="ignore"))
