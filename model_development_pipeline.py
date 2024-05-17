#%%
from model_development.dataset_management import download_dataset, aggregate_all_classes, construct_image_classification_dataset
from model_development.sharktrack_trainer import SharkTrackTrainer
from pathlib import Path

#%%
# Download dataset
workspace = "msc-fish-seychelles"
project = "seychelles"
version = 35
object_detection_yaml = download_dataset(workspace, project, version, "yolov8")

#%%
# Construct Image Classification Dataset
construct_image_classification_dataset(object_detection_yaml)
# %%
aggregate_all_classes(object_detection_yaml)

# %%
trainer = SharkTrackTrainer(project_name="Revillagigedo")
trainer.train(object_detection_yaml)

# %%
