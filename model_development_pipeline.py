# %%
from model_development.dataset_management import download_dataset, aggregate_all_classes, construct_image_classification_dataset
from model_development.sharktrack_trainer import SharkTrackTrainer
from model_development.train_image_classifier import main as train_image_classifier
import wandb

# # %%
# Download dataset
workspace = "sharktrackpelagios-giyh4"
project = "sharktrack_revilla"
version = 3
object_detection_yaml = download_dataset(workspace, project, version, "yolov8")

# %%
# Construct Image Classification Dataset
image_classification_folder = construct_image_classification_dataset(object_detection_yaml)

# %%
# Aggregate the object_detection classes in a single "elasmobranch" class
aggregated_path_yaml = aggregate_all_classes(object_detection_yaml)

# %%
# Train single class SharkTrack detector
trainer = SharkTrackTrainer(project_name="Revillagigedo")
single_class_accuracy = trainer.train(
    aggregated_path_yaml, 
    **{
        "name": "/".join(aggregated_path_yaml.parts[-3:-1]),
        "epochs": 200,
        "patience": 20,
    })

# %%
# Train multi class SharkTrack detector
from pathlib import Path
object_detection_yaml = Path("/vol/biomedic3/bglocker/ugproj2324/fv220/dev/sharktrack-tuner/data/development/v3data_remove_full_esu/object_detection/data.yaml")
trainer = SharkTrackTrainer(project_name="Revillagigedo")
multi_class_accuracy = trainer.train(
    object_detection_yaml, 
    previous_model="yolov8s",
    **{
        "name": "/".join(object_detection_yaml.parts[-3:-1]),
        "epochs": 500,
        "patience": 50,
        "single_cls": False,
    })


# %%
# Train the image classifier
image_classifier_accuracy, *other_results = train_image_classifier(
    image_classification_folder,
    num_epochs=50
    )
if image_classifier_accuracy:
    image_classifier_accuracy = image_classifier_accuracy.item()

# %%
# Log Results
# start a new wandb run to track this script
wandb.init(
    project=project,
    config={
        "aggregated_path": str(aggregated_path_yaml),
        "image_classification_folder": str(image_classification_folder),
        "object_detection_yaml": str(object_detection_yaml)
    }
)

wandb.log({
    "single_class_accuracy": single_class_accuracy,
    "image_classifier_accuracy": image_classifier_accuracy
})
