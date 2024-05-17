#%%
from ultralytics import YOLO
from pathlib import Path
import sys

sys.path.append("utils")
from config import Config

class SharkTrackTrainer():
    def __init__(self, project_name: str) -> None:
        self.train_params = {
            "epochs" : 1,
            "imgsz" : 640,
            "patience" : 100,
            "batch" : 32,
            "conf" : 0.2,
            "name": project_name,
            "project": str(Config.get_project_folder()),
            "verbose": True,
            "save_period": 50
        }

    def train(self, dataset_yaml: Path, previous_model: Path = None, **kwargs):
        self.train_params.update(kwargs)

        previous_model_path = str(previous_model) if previous_model else str(Config.get_sharktrack_model())
        model = YOLO(previous_model_path)

        results = model.train(
            data=str(dataset_yaml),
            **self.train_params,
        )

        map50_score = results.results_dict["metrics/mAP50(B)"]
        return map50_score
