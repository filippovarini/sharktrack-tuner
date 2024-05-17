#%%
from model_development.dataset_management import download_dataset, aggregate_all_classes, construct_image_classification_dataset
from pathlib import Path

#%%
# Download dataset
workspace = "msc-fish-seychelles"
project = "seychelles"
version = 35
object_detection_dataset = download_dataset(workspace, project, version, "yolov8")

#%%
# Construct Image Classification Dataset
construct_image_classification_dataset(object_detection_dataset / "data.yaml")
# %%
