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
    ret, _ = seek_video(sample_video_path, sample_video_time)
    assert ret, f"Can't read video {sample_video_path}"

    assert not maxn_df.isnull(), "MaxN file can't contain null values"
    

# %%
def calculate_inverted_species_occurrance_ratio(species_occurrance_ratio: pd.Series):
    inverted_species_occurrance_ratio = 1 / species_occurrance_ratio
    normalised_species_occurrance_ratio = inverted_species_occurrance_ratio / inverted_species_occurrance_ratio.sum()
    return normalised_species_occurrance_ratio

def calculate_frames_extraction_ratio(maxn_df: pd.DataFrame):
    avg_frames_per_maxn = 20
    total_frames = len(maxn_df) * avg_frames_per_maxn

    species_occurrance_ratio = maxn_df.value_counts("species") / len(maxn_df)
    species_occurrance_ratio = calculate_frames_extraction_ratio(species_occurrance_ratio)
    frames_per_species = species_occurrance_ratio * total_frames
    print(frames_per_species.sum())
    

    

    





calculate_frames_extraction_ratio(pd.read_csv("maxn.csv"))
# %%
