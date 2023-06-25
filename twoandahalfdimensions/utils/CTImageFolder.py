from monai.data.image_dataset import ImageDataset
from pathlib import Path
from numpy import array
from copy import deepcopy


class CTImageFolder(ImageDataset):
    def __init__(self, folder: Path, *args, **kwargs) -> None:
        class_folders = [f for f in folder.iterdir() if f.is_dir()]
        paths, labels = [], []
        for label, cf in enumerate(class_folders):
            ct_files = [p for p in cf.rglob("*.nii") if p.is_file()]
            paths.extend(ct_files)
            labels.extend([deepcopy(label) for _ in ct_files])
        super().__init__(image_files=paths, labels=array(labels), *args, **kwargs)
