from pathlib import Path
import yaml

def construct_new_folder(path: Path):
    assert path.suffix == "", f"Path must be of a folder, not a filename {path}"
    new_path = path
    i = 0
    while new_path.exists():
        i += 1
        new_path = path.parent / (path.stem + str(i))

    return new_path

def setup_yolov8_dataset(root: Path, class_names = {0: 'elasmobranch'}, yaml_name='data_config.yaml'):    
    # Creating directories for train, val, test, and their images subdirectories
    for set_type in ['train', 'val', 'test']:
        (root / set_type / 'images').mkdir(parents=True, exist_ok=True)
        (root / set_type / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Define the data structure for the YAML file
    data = {
        'path': str(root), 
        'train': str(root / 'train'),
        'val': str(root / 'val'),
        'test': str(root / 'test'),
        'names': class_names
    }
    
    # Write the YAML file
    yaml_file = root / yaml_name
    with open(yaml_file, 'w') as file:
        yaml.dump(data, file, sort_keys=False)
    
    print(f"Directory structure and YAML configuration file created at {root}")