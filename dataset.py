
import cv2
import torch
import typing as t

from pathlib import Path
from torch.utils.data import Dataset

Label = t.Dict[str, t.Any]


def extract_labels(labels: t.Dict[str, t.Any]) -> t.Tuple[Label, Label]:
    image_ids = labels["images"]
    
    keypoints = {}
    bounding_boxes = {}

    for label in labels["annotations"]:
        idx = str(label["image_id"])
        image_name = image_ids[idx]
        box = label["bbox"]
        key = label["keypoints"]

        bounding_boxes[image_name] = box
        keypoints[image_name] = key

    return keypoints, bounding_boxes



class AnimalPose(Dataset):
    def __init__(self, folder: Path, keypoints: Label, bounding_boxes: Label, transform=None):
        self.folder = folder
        self.images = list(self.folder.rglob("*.jpg"))
        self.keypoints = keypoints
        self.bounding_boxes = bounding_boxes
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int) -> t.Tuple[torch.Tensor, t.Dict[str, t.Any]]:
        image = cv2.imread(str(self.images[index]))
        file_name = self.images[index].name
        keypoint= self.keypoints[file_name]
        boxes = self.bounding_boxes[file_name]
        
        sample = {
            "image": image,
            "keypoints": keypoint,
            "bounding": boxes,
        }
        if self.transform:
            sample = self.transform(
                image=sample["image"],
                keypoints=sample["keypoints"],

            )

        return sample

