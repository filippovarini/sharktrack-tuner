#%%
import pandas as pd
from pathlib import Path
import pathlib
import cv2

def load_maxn_revilla(path):
    maxn_df = pd.read_excel(path, "Datos crudo")
    maxn_df.head()
    maxn_df = maxn_df[maxn_df["Area"] == "Revillagigedo"]

    video_root = Path("/Volumes/Expansion/Expedicion Revillagigedo/")
    search_depth = 3
    search_pattern = pathlib.os.sep.join(["*"] * search_depth)
    videos = video_root.glob(search_pattern)
    code2video = {v.stem.strip().lower(): v for v in videos}

    maxn_df["video_path"] = maxn_df["Cod"].apply(lambda code: code2video.get(code.strip().lower(), None))
    maxn_df["chapter_name"] = maxn_df["NOMBRE VIDEO"]
    maxn_df["species"] = maxn_df["NOMBRE CIENTIFICO"]
    maxn_df = maxn_df[["video_path", "chapter_name", "species", "MINUTO. INICIAL"]]
    maxn_df = maxn_df.dropna()
    maxn_df["time_seconds"] = maxn_df["MINUTO. INICIAL"].apply(lambda time: time.hour * 60 + time.minute)
    maxn_df = maxn_df[["video_path", "chapter_name", "species", "time_seconds"]]

    maxn_df.to_csv("maxn.csv", index=False)


load_maxn_revilla("~/Desktop/revilla.xlsx")
    


# %%
