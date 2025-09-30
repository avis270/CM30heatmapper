import io
import os
import tempfile
import zipfile
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe for packaging/headless
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import streamlit as st

# Plate layouts for supported types
PLATE_LAYOUTS = {
    "6well":   (["A", "B"], list(range(1, 4))),
    "12well":  (["A", "B", "C"], list(range(1, 5))),
    "24well":  (["A", "B", "C", "D"], list(range(1, 7))),
    "96well":  (list("ABCDEFGH"), list(range(1, 13))),
}

def normalize_well_name(well_name, plate_type):
    """Convert CM30 well labels into standard A1/B2 format."""
    if well_name.startswith("Well"):
        core = well_name[4:]  # strip "Well"
        if core.isdigit():
            # Numbered wells (seen in 6-well plates)
            rows, cols = PLATE_LAYOUTS[plate_type]
            num = int(core)
            r = (num - 1) // len(cols)
            c = (num - 1) % len(cols) + 1
            return f"{rows[r]}{c}"
        else:
            # Already like A2, C7 etc.
            return core
    return well_name

def detect_plate_type(lines):
    """Detect plate type from the <vessel Type> section."""
    for line in lines:
        if "well" in line.lower():
            tokens = line.strip().replace("\t", ",").split(",")
            for t in tokens:
                if "well" in t.lower():
                    return t.lower().strip()
    return None

def parse_cm30_file(file_content):
    """Parse CM30 CSV-like file into dataframe with confluency values."""
    lines = file_content.splitlines()
    plate_type = detect_plate_type(lines)
    if plate_type not in PLATE_LAYOUTS:
        raise ValueError(f"Unsupported or undetected plate type: {plate_type}")

    data = {}
    current_well = None
    in_results = False

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("<Single Result>") or line.startswith("<Colony Forming Result>"):
            in_results = True
            continue
        if in_results and line.startswith("Well"):
            current_well = line.split(",")[0].strip()
            data[current_well] = []
            continue
        if current_well and re.match(r"^\d+", line):  # data rows start with a number
            parts = re.split(r"[\t,]", line)
            if len(parts) >= 4:
                try:
                    conf = float(parts[2])
                    data[current_well].append(conf)
                except ValueError:
                    pass

    if not data:
        raise ValueError("No confluency data found")

    df = pd.DataFrame([
        {
            "Well": normalize_well_name(well, plate_type),
            "Confluency": pd.Series(vals).mean()
        }
        for well, vals in data.items()
    ])

    return df, plate_type

def plot_plate(df, plate_type):
    """Render a plate heatmap given dataframe and plate type."""
    rows, cols = PLATE_LAYOUTS[plate_type]

    fig, ax = plt.subplots(figsize=(len(cols), len(rows)))
    cmap = plt.cm.viridis
    norm = mcolors.Normalize(vmin=0, vmax=100)

    # Draw each well
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            well_id = f"{row}{col}"
            val = df.loc[df["Well"] == well_id, "Confluency"]
            if not val.empty:
                color = cmap(norm(val.values[0]))
            else:
                color = "black"  # no data
            rect = patches.Rectangle((j, i), 1, 1, facecolor=color, edgecolor="white")
            ax.add_patch(rect)
            ax.text(j + 0.5, i + 0.5, well_id, ha="center", va="center", color="white")

    ax.set_xlim(0, len(cols))
    ax.set_ylim(0, len(rows))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{plate_type.upper()} Plate Confluency Heatmap")
    plt.tight_layout()
    return fig

# Streamlit app
st.title("CM30 Plate Viewer")

uploaded_file = st.file_uploader("Upload a CM30 export file", type=["csv", "txt"])

if uploaded_file:
    try:
        content = uploaded_file.read().decode("utf-8-sig")
        df, plate_type = parse_cm30_file(content)
        st.write("Parsed Data:", df)
        fig = plot_plate(df, plate_type)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Could not parse file: {e}")
