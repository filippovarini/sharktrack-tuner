#%%
from pathlib import Path
import cv2
import pandas as pd
import yaml
from ultralytics import YOLO
import roboflow

#%%
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

#%%
def extract_frames(maxn_df: pd.DataFrame, dataset_path: Path):
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

        ret = True

        while ret and curr_time - time_start < row["frames_per_maxn"]:
            ret, frame = vidcap.read()
            if i % video_fps == 0:
                extraction_log.append({
                    "video_path": str(video_path),
                    "time": curr_time,
                    "species": row["species"],
                    "image_id": len(extraction_log)
                })
                image_path = str(dataset_path / "images" / f"{len(extraction_log)}.jpg") 
                print(f"Saving image {image_path}")
                cv2.imwrite(image_path, frame)
                curr_time += 1
        

    extraction_df = pd.DataFrame(extraction_log)
    extraction_df.to_csv(dataset_path / "log.csv")

    return True

#%%

def setup_yolov8_dataset(root: Path, yaml_name='data_config.yaml'):    
    # Creating directories for train, val, test, and their images subdirectories
    for set_type in ['train', 'val', 'test']:
        (root / set_type / 'images').mkdir(parents=True, exist_ok=True)
        (root / set_type / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Define the data structure for the YAML file
    data = {
        'path': str(root), 
        'train': str(root / 'train'),
        'val': str(root / 'val'),
        'test': str(root / 'test'),
        'names': {0: 'elasmobranch'}
    }
    
    # Write the YAML file
    yaml_file = root / yaml_name
    with open(yaml_file, 'w') as file:
        yaml.dump(data, file, sort_keys=False)
    
    print(f"Directory structure and YAML configuration file created at {root}")

def run_sharktrack_preinference(folder: Path):
    model = YOLO("./models/sharktrack.pt")
    model(str(folder), save_txt=True)

dataset_path_root = Path("/Users/filippovarini/Desktop/SharkTrack_workdir/sharktrack-tuner/initial_dataset")
run_sharktrack_preinference(dataset_path / "images")


# %%
def create_initial_dataset(name="preliminary"):
    root_path = Path("data")
    path = root_path / name
    new_path = path
    i = 0
    while new_path.exists():
        i += 1
        new_path = path.parent / name + str(i)
    
    new_path.mkdir()

    setup_yolov8_dataset(root_path)

