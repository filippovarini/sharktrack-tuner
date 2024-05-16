#%%
from utils.config import Config
from utils.path_resolver import construct_new_folder
from roboflow import Roboflow
from pathlib import Path
import yaml
import getpass

def setup_database_management():
    api_key = getpass.getpass("Enter your Roboflow API key: ")
    rf = Roboflow(api_key=api_key)
    return rf

def download_dataset(workspace: str, project: str, version: int, annotation_format: str):
    rf = setup_database_management()
    project = rf.workspace(workspace).project(project)
    version = project.version(version)

    download_location = construct_new_folder(Config.get_development_dataset_path())
    version.download(annotation_format, location=str(download_location))

    return download_location

def aggregate_all_classes(data_yaml_path: Path):
    aggregated_class = "0"
    with open(str(data_yaml_path)) as f:
        data_config = yaml.load(f, Loader=yaml.SafeLoader)
    
    data_config["names"] = [Config.get_sharktrack_class()]
    data_config["nc"] = 1

    with open(str(data_yaml_path), "w") as f:
        yaml.dump(data_config, f)
    
    splits = ["train", "val", "test"]

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


    

# download_dataset("msc-fish-seychelles", "seychelles", 35, "yolov8")
aggregate_all_classes(Path("/vol/biomedic3/bglocker/ugproj2324/fv220/dev/sharktrack-tuner/data/development1/data.yaml"))


# %%
