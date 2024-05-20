# %%
from roboflow import Roboflow
from pathlib import Path
import yaml
import getpass
import numpy as np
import cv2
import sys
sys.path.append("utils")
from utils.config import Config
from utils.path_resolver import construct_new_folder

def setup_database_management():
    api_key = getpass.getpass("Enter your Roboflow API key: ")
    rf = Roboflow(api_key=api_key)
    return rf

def update_splits_path(data_yaml_payh: Path):
    data = None
    with open(str(data_yaml_payh), "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    data["path"] = str(data_yaml_payh.parent.absolute())
    data["train"] = "./train"
    data["test"] = "./test"
    data["val"] = "./valid"
    print(data)
    with open(str(data_yaml_payh), "w") as f:
        yaml.dump(data, f)

def download_dataset(workspace: str, project: str, version: int, annotation_format: str):
    rf = setup_database_management()
    project = rf.workspace(workspace).project(project)
    dataset_version = project.version(version)

    download_location = construct_new_folder(Config.get_development_dataset_path() / f"v{version}data")
    download_location = construct_new_folder(download_location / Config.get_object_detection_dataset_name())
    dataset_version.download(annotation_format, location=str(download_location))

    data_yaml_path = download_location / "data.yaml"
    update_splits_path(data_yaml_path)

    return data_yaml_path

def construct_image_classification_folder_structure(object_detection_data_yaml_path: Path):
    object_detection_location = object_detection_data_yaml_path.parent
    image_classification_location = construct_new_folder(object_detection_location.parent / Config.get_image_classification_dataset_name())

    labels = []
    with open(str(object_detection_data_yaml_path), "r") as f:
        data_config = yaml.load(f, Loader=yaml.SafeLoader)
        labels = data_config["names"]
    
    for label in labels:
        (image_classification_location / label).mkdir(parents=True)
    
    return image_classification_location, labels

def extract_boxes(imagefile: Path, labelfile: Path):
    boxes = []
    label_ids = []

    image = cv2.imread(str(imagefile))
    image_w, image_h, _ = image.shape

    with open(str(labelfile), "r") as f:
        instances = f.readlines()
        print(f"Extracting {len(instances)} patches from {imagefile}...")
        for instance in instances:
            annotations = instance.split(" ")
            label_ids.append(int(annotations[0]))
            x_centre = float(annotations[1]) * image_w
            y_centre = float(annotations[2]) * image_h
            box_w = float(annotations[3]) * image_w
            box_h = float(annotations[4]) * image_h

            xmin = int(x_centre - box_w / 2)
            xmax = int(x_centre + box_w / 2)
            ymin = int(y_centre - box_h / 2)
            ymax = int(y_centre + box_h / 2)

            boxes.append(image[ymin:ymax, xmin:xmax])
    
    return boxes, label_ids

def construct_image_classification_dataset(object_detection_data_yaml_path: Path) -> Path:
    image_classification_folder, labels = construct_image_classification_folder_structure(object_detection_data_yaml_path)
    return image_classification_folder

    object_detection_dataset = object_detection_data_yaml_path.parent
    for imagefile in object_detection_dataset.rglob("*.jpg"):
        labelfile = imagefile.parent.parent / "labels" / (imagefile.stem + ".txt")
        if not labelfile.exists():
            print(f"Image {imagefile} does not have annotations {labelfile}")
            continue

        boxes, label_ids = extract_boxes(imagefile, labelfile)
        assert len(boxes) == len(label_ids), f"Different number of boxes {len(boxes)} and label ids {len(label_ids)}"
        for i, (box, label_id) in enumerate(zip(boxes, label_ids)):
            label = labels[label_id]
            new_imagefile_folder = image_classification_folder / label
            assert new_imagefile_folder.exists(), f"Folder {new_imagefile_folder} shold have already been created"
            new_imagefile = imagefile.stem + f"_patch{i}" + imagefile.suffix
            cv2.imwrite(str(new_imagefile_folder / new_imagefile), box)

def aggregate_all_classes(data_yaml_path: Path):
    aggregated_class = "0"
    with open(str(data_yaml_path)) as f:
        data_config = yaml.load(f, Loader=yaml.SafeLoader)
    
    data_config["names"] = [Config.get_sharktrack_class()]
    data_config["nc"] = 1

    with open(str(data_yaml_path), "w") as f:
        yaml.dump(data_config, f)
    
    splits = ["train", "valid", "test"]

    for split in splits:
        split_path = data_yaml_path.parent / split
        for label_file in split_path.rglob("*.txt"):
            print(f"Aggregating classes for file {label_file}")
            new_instances = []
            with open(str(label_file), "r") as f:
                instances = f.readlines()
                new_instances = [f"{aggregated_class} {' '.join(instance.split(' ')[1:])}" for instance in instances]

            with open(str(label_file), "w") as f:
                f.writelines(new_instances)
