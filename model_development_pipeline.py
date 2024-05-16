#%%
from utils.config import Config
from roboflow import Roboflow
import getpass

def setup_database_management():
    api_key = getpass.getpass("Enter your Roboflow API key: ")
    rf = Roboflow(api_key=api_key)
    return rf

def download_dataset(workspace: str, project: str, version: int, annotation_format: str):
    rf = setup_database_management()
    project = rf.workspace(workspace).project(project)
    version = project.version(version)
    version.download(annotation_format, location=str(Config.get_development_dataset_path()))

download_dataset("msc-fish-seychelles", "seychelles", 35, "yolov5")


# %%
