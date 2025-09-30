import io, os, tempfile, zipfile
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe for packaging/headless
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import streamlit as st

# Plate layouts: row letters × column counts
PLATE_LAYOUTS = {
    "6well":   (["A", "B"], 3),
    "12well":  (["A", "B", "C"], 4),
    "24well":  (["A", "B", "C", "D"], 6),
    "96well":  (list("ABCDEFGH"), 12),
}

def parse_cm30_file(uploaded_file):
    """Parse CM30 CSV export into a tidy dataframe with Well + Confluency."""
    text = uploaded_file.read().decode("utf-8", errors="ignore")
    lines = text.splitlines()

    # Detect plate type
    plate_type = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Plate"):
            parts = line.split(",")
            if len(parts) >= 6:
                plate_type = parts[5].strip()
                break
    if plate_type not in PLATE_LAYOUTS:
        raise ValueError(f"Unsupported or unknown plate type: {plate_type}")

    # Find where results start
    result_idx = None
    result_type = None
    for i, line in enumerate(lines):
        if line.strip() == "<Single Result>":
            result_idx, result_type = i, "single"
            break
        if line.strip() == "<Colony Forming Result>":
            result_idx, result_type = i, "colony"
            break
    if result_idx is None:
        raise ValueError("No results section (<Single Result> or <Colony Forming Result>) found.")

    # Parse section after results
    data = {}
    well_name, collecting = None, False
    for line in lines[result_idx+1:]:
        if line.startswith("Well"):
            well_name = line.strip()
            data[well_name] = []
            collecting = True
            continue
        if collecting and line.strip() == "":
            well_name, collecting = None, False
            continue
        if collecting and well_name:
            parts = line.split(",")
            if len(parts) >= 4:
                try:
                    conf = float(parts[2])
                    data[well_name].append(conf)
                except ValueError:
                    continue

    # Aggregate (mean confluency per well)
    df = pd.DataFrame([
        {"Well": well, "Confluency": pd.Series(vals).mean()}
        for well, vals in data.items()
    ])

    return df, plate_type


def generate_plate_layout(plate_type, data_df):
    """Return dataframe with all wells for given plate_type merged with data_df."""
    rows, cols = PLATE_LAYOUTS[plate_type]
    wells = [f"{r}{c}" for r in rows for c in range(1, cols+1)]
    full_df = pd.DataFrame({"Well": wells})
    merged = full_df.merge(data_df, on="Well", how="left")
    return merged, rows, cols


def plot_plate_
