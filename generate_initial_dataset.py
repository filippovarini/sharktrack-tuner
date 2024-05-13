#%%
from pathlib import Path
import cv2
import pandas as pd

def seek_video(video_path, time_seconds):
    vidcap = cv2.VideoCapture(video_path)
    vidcap.set(cv2.CAP_PROP_POS_MSEC, time_seconds*1000)
    return vidcap

def maxn_df_sanity_check(maxn_df: pd.DataFrame):
    # check video path correct
    sample_video_path = Path(maxn_df[0]["video_path"]) / maxn_df[0]["chapter_name"]
    sample_video_time = maxn_df[0]["time_seconds"]
    vidcap = seek_video(sample_video_path)
    ret, _ = vidcap.read()
    print(ret)
    assert ret, f"Can't read video {sample_video_path}"

    assert not maxn_df.isnull(), "MaxN file can't contain null values"  

# %%
def invert_ratio(species_df):
    # We want rarely occurring species to have a higher number of frames of representation
    # to achieve data balance
    species_df["inverted_ratio"] = 1 / species_df["species_ratio"]
    species_df["inverted_ratio"] = species_df["inverted_ratio"] / species_df["inverted_ratio"].sum()

def calculate_frames_extraction_ratio(maxn_df: pd.DataFrame):
    avg_frames_per_maxn = 20
    total_frames = len(maxn_df) * avg_frames_per_maxn

    species_df = maxn_df.groupby(["species"]).agg(n=("species", "count"))
    species_df = species_df.reset_index()
    species_df["species_ratio"] = species_df["n"] / species_df["n"].sum()
    invert_ratio(species_df)
    species_df["frames_per_maxn"] = (species_df["inverted_ratio"] * total_frames / species_df["n"]).astype(int)
    
    maxn_df = pd.merge(maxn_df, species_df[["species", "frames_per_maxn"]], on='species', how='inner')
    print(maxn_df)

#%%

def extract_frames(maxn_df: pd.DataFrame, dataset_path: Path):
    # extract frames given ratio
    # saves csv mapping image id to video path, time and species 
    maxn_df_sanity_check(maxn_df)
    extraction_guide = calculate_frames_extraction_ratio(maxn_df)
    extraction_log = []

    for index, row in extraction_guide.iterrows():
        video_path = Path(row["video_path"]) / row["chapter_name"]
        time_start = max(0, row["time_seconds"] - row["frames_per_maxn"] // 2)
        curr_time = time_start
        vidcap = seek_video(video_path, time_start)

        ret = True

        while ret and curr_time - time_start < row["frames_per_maxn"]:
            ret, frame = vidcap.read()

            extraction_log.append({
                "video_path": str(video_path),
                "time": curr_time,
                "species": row["species"],
                "image_id": len(extraction_log)
            })
            cv2.imwrite(dataset_path / "images", frame)
            curr_time += 1
        

    extraction_df = pd.DataFrame(extraction_log)
    extraction_df.to_csv(dataset_path / "log.csv")

    return True

maxn_df = pd.read_csv("maxn_df.csv")
dataset_path = Path("initial_dataset")
extract_frames("maxn.csv", )



# %%
