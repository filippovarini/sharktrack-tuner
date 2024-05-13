from pathlib import Path
import cv2

def seek_video(video_path, time_seconds):
    vidcap = cv2.VideoCapture(video_path)
    vidcap.set(cv2.CAP_PROP_POS_MSEC, time_seconds*1000)
    return vidcap

def maxn_df_sanity_check(maxn_df):
    sample_video_path = Path(maxn_df[0]["video_path"]) / maxn_df[0]["chapter_name"]
    samplle_video_time = maxn_df[0]["time_seconds"]
    vidcap = seek_video(sample_video_path)
    
