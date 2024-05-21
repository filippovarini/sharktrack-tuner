from pathlib import Path

class Config():
    data_folder = Path("data")
    
    @classmethod
    def get_preliminary_dataset_path(self) -> Path:
        return self.data_folder / "preliminary"
    
    @classmethod
    def get_development_dataset_path(self) -> Path:
        return self.data_folder / "development"
    
    @classmethod
    def get_object_detection_dataset_name(self) -> Path:
        return "object_detection"

    @classmethod
    def get_image_classification_dataset_name(self) -> Path:
        return "image_classification"
    
    @classmethod
    def get_sharktrack_class(self) -> Path:
        return "detection"
    
    @classmethod
    def get_project_folder(self) -> Path:
        return Path("models")

    @classmethod
    def get_sharktrack_model(self) -> Path:
        return self.get_project_folder() / "sharktrack.pt"
    
    @classmethod
    def get_sharktrack_single_cls_name(self) -> Path:
        return "single_cls"

    @classmethod
    def get_sharktrack_multi_cls_name(self) -> Path:
        return "multi_cls"

    
    

    