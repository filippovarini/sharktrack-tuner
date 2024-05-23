from utils.config import Config
from utils.path_resolver import construct_new_folder, setup_yolov8_dataset
from pathlib import Path
import cv2
import pandas as pd
from ultralytics import YOLO

def seek_video(video_path, time_seconds):
    vidcap = cv2.VideoCapture(str(video_path))
    vidcap.set(cv2.CAP_PROP_POS_MSEC, time_seconds*1000)
    return vidcap

def maxn_df_sanity_check(maxn_df: pd.DataFrame):
    # check video path correct
    sample_video_path = Path(maxn_df["video_path"].iloc[0])
    sample_video_time = maxn_df["time_seconds"].iloc[0]
    vidcap = seek_video(sample_video_path, sample_video_time)
    ret, _ = vidcap.read()
    assert ret, f"Can't read video {sample_video_path}"

    assert maxn_df.isnull().sum().sum() == 0, f"MaxN file can't contain null values {maxn_df.isnull().sum()}"  

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
    return maxn_df

def extract_frames(maxn_df: pd.DataFrame, dataset_path: Path, images_path: Path):
    maxn_df_sanity_check(maxn_df)
    extraction_guide = calculate_frames_extraction_ratio(maxn_df)
    extraction_log = []

    for _, row in extraction_guide.iterrows():
        video_path = Path(row["video_path"])
        time_start = max(0, row["time_seconds"] - row["frames_per_maxn"] // 2)
        curr_time = time_start
        vidcap = seek_video(video_path, time_start)
        video_fps = vidcap.get(cv2.CAP_PROP_FPS)

        i = 0

        ret, frame = vidcap.read()

        while ret and curr_time - time_start < row["frames_per_maxn"]:
            i += 1
            if i % int(video_fps) == 0:
                extraction_log.append({
                    "video_path": str(video_path),
                    "time": curr_time,
                    "species": row["species"],
                    "image_id": len(extraction_log)
                })
                image_path = str(images_path / f"{len(extraction_log)}.jpg") 
                print(f"Saving image {image_path}")
                cv2.imwrite(image_path, frame)
                curr_time += 1
            ret, frame = vidcap.read()
        

    extraction_df = pd.DataFrame(extraction_log)
    extraction_df.to_csv(dataset_path / "log.csv")

    return True

def run_sharktrack_preinference(folder: Path):
    model = YOLO("./models/sharktrack.pt")
    model(str(folder), save_txt=True, project=str(folder.parent / "labels"))

def create_dataset_folder():
    root_path = Config.get_preliminary_dataset_path()

    new_path = construct_new_folder(root_path)
    new_path.mkdir()

    return new_path

def generate_initial_dataset(maxn_df_path: str):
    path = create_dataset_folder()

    yolo_path = path / "yolo"
    yolo_path.mkdir()
    setup_yolov8_dataset(yolo_path)

    maxn_df = pd.read_csv(maxn_df_path)
    images_folder = yolo_path / "train" / "images"
    extract_frames(maxn_df, path, images_folder)

    run_sharktrack_preinference(images_folder)


generate_initial_dataset("maxn/maxn_revilla.csv")
