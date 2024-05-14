#%%
import pandas as pd
from pathlib import Path
import pathlib
import cv2

def filter_species(maxn_df):
    # Keep "other" class since they also want non-shark classifications
    elasmobranch_species = ["Carcharhinus falciformis", "Carcharhinus albimarginatus", "Carcharhinus galapaguensis", "Sphyrna lewini", "Carcharhinus limbatus", "Galeocerdo cuvier", "Rhizoprionodon longurio"]
    maxn_df["species"] = maxn_df["species"].apply(lambda s: s if s in elasmobranch_species else "other")
    return maxn_df


def load_maxn_revilla(path):
    maxn_df = pd.read_excel(path, "Datos crudo")
    maxn_df.head()
    maxn_df = maxn_df[maxn_df["Area"] == "Revillagigedo"]

    video_root = Path("/Volumes/Expansion/Expedicion Revillagigedo/")
    search_depth = 3
    search_pattern = pathlib.os.sep.join(["*"] * search_depth)
    video_folders = video_root.glob(search_pattern)
    code2video = {v.stem.strip().lower(): v for v in video_folders}

    def calculate_video_path(row):
        video_code = row.Cod
        video_name = row["NOMBRE VIDEO"]
        clean_code = video_code.strip().lower()
        if clean_code not in code2video:
            print(f"invalid code: {clean_code}")
            return None

        video_folder = code2video[clean_code]
        video_paths = video_folder.rglob(f"*{video_name}")
        filename = next(video_paths, None)
        while filename and filename.stem.startswith("."):
            filename = next(video_paths, None)
        return filename

    maxn_df["video_path"] = maxn_df.apply(calculate_video_path, axis=1)
    maxn_df["chapter_name"] = maxn_df["NOMBRE VIDEO"]
    maxn_df["species"] = maxn_df["NOMBRE CIENTIFICO"]
    maxn_df = maxn_df[["video_path", "chapter_name", "species", "MINUTO. INICIAL"]]
    maxn_df = maxn_df.dropna()
    maxn_df["time_seconds"] = maxn_df["MINUTO. INICIAL"].apply(lambda time: time.hour * 60 + time.minute)
    maxn_df = maxn_df[["video_path", "chapter_name", "species", "time_seconds"]]

    maxn_df = filter_species(maxn_df)

    maxn_df.to_csv("maxn.csv", index=False)


load_maxn_revilla("~/Desktop/revilla.xlsx")
    


# %%
