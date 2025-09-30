import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------
# Parser with plate-type detection
# ------------------------
def detect_plate_type(lines):
    for i, line in enumerate(lines):
        if "<vessel Type>" in line:
            # Plate type usually appears 2 lines down in Model column
            return lines[i+2].split("\t")[-1].strip()
    return None

def parse_cm30_file(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    plate_type = detect_plate_type(lines)

    # Pick correct section based on plate type
    if plate_type in ["6well", "12well", "24well"]:
        start_tag = "<Colony Forming Result>"
    else:  # assume 96well
        start_tag = "<Single Result>"

    data = []
    well = None
    in_data_block = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith(start_tag):
            in_data_block = True
            continue

        # Detect new well section
        if in_data_block and line.startswith("Well"):
            well = line.strip()
            continue

        # Skip header row
        if line.startswith("Passage#"):
            continue

        # Parse actual data rows
        if in_data_block and well and not line.startswith("Well"):
            parts = line.split("\t")
            if len(parts) >= 4:
                try:
                    _, time, conf, count = parts[:4]
                    data.append({
                        "Well": well,
                        "Time": time,
                        "Confluency": float(conf),
                        "Count": float(count)
                    })
                except ValueError:
                    pass  # skip rows that don't parse cleanly

    df = pd.DataFrame(data)
    if not df.empty:
        df["TimepointIndex"] = df.groupby("Well").cumcount()
    return df, plate_type

# ------------------------
# Streamlit UI
# ------------------------
st.title("CM30 Heatmapper (Multi-Plate Version)")

uploaded_file = st.file_uploader("Upload CM30 CSV file", type="csv")

if uploaded_file is not None:
    # Save temp and parse
    with open("temp.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())

    df, plate_type = parse_cm30_file("temp.csv")

    if df.empty:
        st.error("Could not parse file.")
    else:
        st.success(f"Detected plate type: **{plate_type}**")
        st.write("Preview of parsed data:")
        st.dataframe(df.head())

        # Select wells
        wells = df["Well"].unique()
        selected_wells = st.multiselect("Select wells to display", wells, default=wells[:min(6, len(wells))])

        # Plot confluency over time
        if selected_wells:
            fig, ax = plt.subplots(figsize=(8, 5))
            for w in selected_wells:
                subset = df[df["Well"] == w]
                ax.plot(subset["TimepointIndex"], subset["Confluency"], marker="o", label=w)
            ax.set_xlabel("Timepoint Index")
            ax.set_ylabel("Confluency (%)")
            ax.legend()
            st.pyplot(fig)


