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
        return "elasmobranch"
    