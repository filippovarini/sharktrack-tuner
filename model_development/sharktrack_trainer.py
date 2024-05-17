from ultralytics import YOLO
from pathlib import Path
import sys

sys.path.append("utils")
from config import Config

class SharkTrackTrainer():
    def __init__(self, project_name: str) -> None:
        self.train_params = {
            "epochs" : 500,
            "imgsz" : 640,
            "patience" : 100,
            "batch" : 64,
            "iou_association_threshold" : 0.5,
            "conf_threshold" : 0.2,
            "name": project_name,
            "project": str(Config.get_project_folder()),
            "verbose": True,
            "save_period": 50
        }

    def train(self, dataset_yaml: Path, previous_model: Path = None, **kwargs):
        self.train_params.update(kwargs)

        previous_model_path = str(previous_model) if previous_model else str(Config.get_sharktrack_model())
        model = YOLO(previous_model_path)

        model.train(
            data=str(dataset_yaml),
            **kwargs,
        )

def validate_sharktrack(dataset_path):
    pass



# # Get mAP
# model_folder = os.path.join(params['project_folder'], params['name'])
# assert os.path.exists(model_folder), 'Model folder does not exist'
# results_path = os.path.join(model_folder, 'results.csv')
# assert os.path.exists(results_path), 'Results file does not exist'
# results = pd.read_csv(results_path)
# results.columns = results.columns.str.strip()
# best_mAP = results['metrics/mAP50(B)'].max()


# # track
# model_path = os.path.join(model_folder, 'weights', 'best.pt')
# assert os.path.exists(model_path), 'Model file does not exist'
# mota, motp, idf1, track_time, track_device = evaluate(
#   model_path, 
#   params['conf_threshold'], 
#   params['iou_association_threshold'],
#   params['imgsz'],
#   params['tracker'],
#   params['project_folder']
# )

# # Log on wandb
# wandb.init(project="SharkTrack", name=params['name'], config=params, job_type="training")
# wandb.log({'mAP': best_mAP, 'mota': mota, 'motp': motp, 'idf1': idf1, 'track_time': track_time, 'track_device': track_device})
# wandb.finish()