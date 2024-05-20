# %%
from model_development.dataset_management import download_dataset, aggregate_all_classes, construct_image_classification_dataset
from model_development.sharktrack_trainer import SharkTrackTrainer
from model_development.train_image_classifier import main as train_image_classifier
from pathlib import Path

# %%
# Download dataset
workspace = "msc-fish-seychelles"
project = "seychelles"
version = 35
object_detection_yaml = download_dataset(workspace, project, version, "yolov8")

# %%
# Construct Image Classification Dataset
image_classification_folder = construct_image_classification_dataset(
    object_detection_yaml
)

# %%
# Aggregate the object_detection classes in a single "elasmobranch" class
aggregate_all_classes(object_detection_yaml)

# Train the image classifier
train_image_classifier(str(image_classification_folder))

# %%
# Train single class SharkTrack detector
trainer = SharkTrackTrainer(project_name="Revillagigedo")
model_accuracy = trainer.train(object_detection_yaml)

# %%
