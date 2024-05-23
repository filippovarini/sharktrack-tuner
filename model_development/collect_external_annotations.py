#%%
import pandas as pd
import yaml
import sys
import cv2
from pathlib import Path
from typing import List
sys.path.append("utils")
from path_resolver import setup_yolov8_dataset
from time_processor import ms_to_string, string_to_ms
from video_processor import extract_frame_at_time

#%%
def update_annotations(external_annotations_path: str, original_output_path: str):
    print("Updating annotations of original output...")
    with open(external_annotations_path, "r") as f:
        external_annotations = yaml.load(f, Loader=yaml.SafeLoader)

    new_output = pd.read_csv(original_output_path)
    new_output["label"] = new_output.apply(lambda x: external_annotations.get(x.track_id, x.label), axis=1)

    return new_output, list(set(external_annotations.values())) + ["elasmobranch"]

# %%
def filter_used_sightings(new_output_df: pd.DataFrame, preliminary_dataset_df: pd.DataFrame):
    print("Removing sightings already included in the dataset...")
    used_sightings = set((row['video_path'], ms_to_string(row['time'] * 1000)) for _, row in preliminary_dataset_df.iterrows())
    filtered_df = new_output_df[~new_output_df.apply(lambda row: (row['video_path'], row['time']) in used_sightings, axis=1)]
    print(f"Removed {len(new_output_df) - len(filtered_df)} rows")
    images_saved_so_far = preliminary_dataset_df["image_id"].max()
    return filtered_df, images_saved_so_far

def convert_xyxy_to_yolo(boxes):
    new_boxes = []
    for box in boxes:
        width, heigth = box[4], box[5]
        xc = (box[1] + box[0]) / 2 / width
        yc = (box[3] + box[2]) / 2 / heigth
        w = (box[1] - box[0])  / width
        h = (box[3] - box[2])  / heigth
        new_boxes.append([xc, yc, w, h])
    return new_boxes


# %%
def construct_yolov8_dataset(video_prefix: Path, filtered_df: pd.DataFrame, dataset_path: Path, classes: List, images_saved_so_far: int):
    class_names = {i:label for i, label in enumerate(classes)}
    setup_yolov8_dataset(dataset_path, class_names)

    annotated_sightings = filtered_df.groupby(["video_path", "time"], as_index=False).filter(lambda x: any([c in set(x.label)for c in classes if c not in ["elasmobranch", "ESU"]]))
    annotated_sightings_1fps = annotated_sightings.groupby(["video_path", "time", "track_id"], as_index=False).agg({
        "xmin": "first",
        "xmax": "first",
        "ymin": "first",
        "ymax": "first",
        "w": "first",
        "h": "first",
        "label": "first"
        })

    
    image_id = images_saved_so_far

    for group in annotated_sightings_1fps.groupby(["video_path", "time"]):
        video_path = group[0][0]
        full_video_path = str(video_prefix / video_path)
        time = group[0][1]
        boxes = convert_xyxy_to_yolo(group[1][["xmin", "xmax", "ymin", "ymax", "w", "h"]].values.tolist())
        labels = group[1]["label"].values.tolist()
        assert set(labels) != {"elasmobranch"}, f"Label contains only elasmobranch!"

        try:
            image_id = image_id+1
            image = extract_frame_at_time(full_video_path, time_ms=string_to_ms(time))
            
            cv2.imwrite(str(dataset_path / "train" / "images" / f"{image_id}.jpg"), image)

            # write annotations]
            label_path = dataset_path / "train" / "labels" / f"{image_id}.txt"
            with open(str(label_path), 'w') as f:
                for i, box in enumerate(boxes):
                    box = [classes.index(labels[i])] + box
                    f.write(" ".join([str(round(b, 4)) for b in box]) + '\n')
            print(f"Read success! {full_video_path}")
        except Exception as e:
            print(e)
            print(f"{full_video_path} couldn't be read")

new_output_df, classes = update_annotations("deployment_specific_data/annotations.yaml", "deployment_specific_data/output.csv")
preliminary_dataset_df = pd.read_csv("deployment_specific_data/preliminary_dataset.csv")
filtered_df, images_saved_so_far = filter_used_sightings(new_output_df, preliminary_dataset_df)
video_prefix = Path("/Volumes/Expansion/Expedicion Revillagigedo/")
dataset_path = Path("data/external_annotation/revilla1")
construct_yolov8_dataset(video_prefix, filtered_df, dataset_path, classes, images_saved_so_far)





# %%
