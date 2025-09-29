import io, os, tempfile, zipfile
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe for packaging/headless
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import streamlit as st

ROW_ORDER = list("ABCDEFGH")      # A at top
COL_ORDER = list(range(1, 13))    # 1..12

def parse_csv(file):
    """Parse your original export format into a tidy df with Well/Time/Confluency/T_index/ReferenceTime."""
    text = file.read().decode("utf-8", errors="ignore")
    lines = text.splitlines()

    recs, current_well = [], None
    for line in lines:
        line = line.strip()
        if line.startswith("Well"):
            current_well = line.upper()   # e.g., WELLA1
            continue
        if line.startswith("Passage#"):
            continue
        # data rows start with a digit
        if current_well and line and line[0].isdigit():
            parts = line.split(",")
            if len(parts) >= 3:
                try:
                    ts = parts[1]
                    conf = float(parts[2])
                    recs.append({"Well": current_well, "Time": ts, "Confluency": conf})
                except:
                    pass

    df = pd.DataFrame(recs)
    if df.empty:
        return None

    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time", "Confluency"]).sort_values(["Well", "Time"]).copy()

    # T_index counts 1..n *per well* (even if timestamps differ slightly)
    df["T_index"] = df.groupby("Well").cumcount() + 1

    # Use Well A1 (or first well alphabetically) as the "reference time" for labeling each T_index
    ref_well = "WELLA1" if "WELLA1" in df["Well"].unique() else sorted(df["Well"].unique())[0]
    ref_times = (
        df.loc[df["Well"] == ref_well, ["T_index", "Time"]]
          .drop_duplicates("T_index")
          .rename(columns={"Time": "ReferenceTime"})
    )
    df = df.merge(ref_times, on="T_index", how="left")
    return df

def draw_plate(df_slice, vmax=100, figsize=(12, 7.5)):
    """96-well graphic: circles, thin black stroke, integer labels, A1 top-left, legend on the right."""
    val_by_well = {w: v for w, v in zip(df_slice["Well"], df_slice["Confluency"])}

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 12); ax.set_ylim(0, 8)

    # Put ticks at well centers, align labels
    ax.set_xticks([j + 0.5 for j in range(12)])
    ax.set_xticklabels([str(c) for c in COL_ORDER])
    ax.set_yticks([i + 0.5 for i in range(8)])
    ax.set_yticklabels(ROW_ORDER)

    ax.invert_yaxis()  # A row on top
    ax.set_aspect("equal", adjustable="box")

    norm = mcolors.Normalize(vmin=0, vmax=max(1.0, float(vmax)))
    cmap = plt.cm.Reds

    for i, row_letter in enumerate(ROW_ORDER):
        for j, col_number in enumerate(COL_ORDER):
            well = f"WELL{row_letter}{col_number}"
            conf = val_by_well.get(well, None)
            cx, cy = j + 0.5, i + 0.5

            if conf is None or pd.isna(conf) or conf < 0:
                face = (0, 0, 0, 1)  # black for missing/invalid
                label_txt = ""
            else:
                face = cmap(norm(conf))
                label_txt = f"{int(round(conf))}%"

            circ = patches.Circle((cx, cy), 0.4, facecolor=face, edgecolor="black", linewidth=0.7)
            ax.add_patch(circ)
            if label_txt:
                ax.text(cx, cy, label_txt, ha="center", va="center", fontsize=8, color="black")

    # Title = reference time for this T_index (from A1 or first well)
    if not df_slice["ReferenceTime"].isna().all():
        rt = df_slice["ReferenceTime"].iloc[0]
        ax.set_title(f"96-Well Plate Confluency — {rt.strftime('%Y-%m-%d %H:%M')}")
    else:
        ax.set_title("96-Well Plate Confluency")

    # Legend
    sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.Reds); sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("% Confluency (max color = target)")

    ax.grid(False)
    plt.tight_layout()
    return fig

def save_all_zip(df, vmax=100, dpi=220):
    """Export one PNG per timepoint into a ZIP; filenames include reference time when available."""
    tmpdir = tempfile.mkdtemp(prefix="plate_pngs_")
    zip_path = os.path.join(tmpdir, "all_timepoints_png.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for t_idx in sorted(df["T_index"].unique()):
            df_slice = df[df["T_index"] == t_idx]
            fig = draw_plate(df_slice, vmax=vmax)
            if not df_slice["ReferenceTime"].isna().all():
                rt = df_slice["ReferenceTime"].iloc[0]
                fname = f"T{t_idx:02d}_{rt.strftime('%Y%m%d_%H%M')}.png"
            else:
                fname = f"T{t_idx:02d}.png"
            out_path = os.path.join(tmpdir, fname)
            fig.savefig(out_path, dpi=220, bbox_inches="tight")
            plt.close(fig)
            z.write(out_path, arcname=fname)
    return zip_path

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="CM30 Heatmapper", layout="wide")
st.title("CM30 Heatmapper — 96-Well Plate Confluency")

uploaded = st.file_uploader("Upload original CSV", type=["csv"])
target = st.number_input("Target confluency (%)", value=100, step=5)

if uploaded is None:
    st.info("Upload your CSV to begin.")
else:
    df = parse_csv(uploaded)
    if df is None or df.empty:
        st.error("Could not parse the CSV. Make sure it uses the expected export format.")
    else:
        max_t = int(df["T_index"].max())
        t_idx = st.slider("Timepoint Index", 1, max_t, 1)

        # Show the “A1 time” label for this T_index
        ref_time = df.loc[df["T_index"] == t_idx, "ReferenceTime"].dropna()
        if not ref_time.empty:
            st.write(f"**T{t_idx} — {ref_time.iloc[0].strftime('%Y-%m-%d %H:%M')}**")
        else:
            st.write(f"**T{t_idx} — (no ref time)**")

        # Plot
        fig = draw_plate(df[df["T_index"] == t_idx], vmax=target)
        st.pyplot(fig, use_container_width=True)

        # Download current PNG (fixed 220 DPI for sharpness)
        png_buf = io.BytesIO()
        fig.savefig(png_buf, dpi=220, format="png", bbox_inches="tight")
        st.download_button(
            "⬇️ Download current timepoint (PNG)",
            data=png_buf.getvalue(),
            file_name=f"timepoint_{t_idx}.png",
            mime="image/png"
        )

        # Download all timepoints as ZIP
        if st.button("📦 Create ZIP of all timepoints"):
            zpath = save_all_zip(df, vmax=target, dpi=220)
            with open(zpath, "rb") as f:
                st.download_button(
                    "Download ZIP",
                    data=f,
                    file_name="all_timepoints.zip",
                    mime="application/zip"
                )
