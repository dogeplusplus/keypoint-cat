import os
import cv2
import json
import torch
import random
import albumentations as A
import matplotlib.pyplot as plt

from albumentations.pytorch import ToTensorV2
from pathlib import Path
from torch.utils.data import DataLoader


from dataset import AnimalPose, extract_labels
from models.keypoint_rcnn import KeypointRCNN


COLOR_MAP = {
    "[0, 1]": (0, 255, 0),  # left_eye right_eye
    "[0, 2]": (0, 255, 0),  # left_eye nose
    "[1, 2]": (0, 255, 0),  # right_eye nose
    "[0, 3]": (0, 255, 0),  # left_eye left_ear
    "[1, 4]": (0, 255, 0),  # right_eye, right_ear
    "[2, 17]": (0, 125, 0),  # nose throat
    "[18, 19]": (0, 0, 255),  # withers tailbase
    "[5, 9]": (255, 0, 0),  # left_front_elbow left_front_knee
    "[6, 10]": (255, 0, 0),  # right_front_elbow right_front_knee
    "[7, 11]": (255, 0, 0),  # left_back_elbow left_back_knee
    "[8, 12]": (255, 0, 0),  # right_back_elbow right_back_knee
    "[9, 13]": (255, 0, 0),  # left_front_knee left_front_paw
    "[10, 14]": (255, 0, 0),  # right_front_knee right_front_paw
    "[11, 15]": (255, 0, 0),  # left_back_knee left_back_paw
    "[12, 16]": (255, 0, 0),  # right_back_knee right_back_paw
    "[17, 5]": (0, 0, 255),  # throat left_front_elbow
    "[17, 6]": (0, 0, 255),  # throat right_front_elbow
    "[19, 7]": (0, 0, 255),  # throat left_back_elbow
    "[19, 8]": (0, 0, 255),  # throat right_back_elbow
    "[17, 18]": (0, 0, 255),  # throat withers
}

SKELETON = [
    [0, 1],
    [0, 2],
    [1, 2],
    [0, 3],
    [1, 4],
    [2, 17],
    [18, 19],
    [5, 9],
    [6, 10],
    [7, 11],
    [8, 12],
    [9, 13],
    [10, 14],
    [11, 15],
    [12, 16],
    [17, 5],
    [17, 6],
    [19, 7],
    [19, 8],
    [17, 18],
]


def render_keypoints(image, keypoints, skeleton):
    for edge in skeleton:
        edge_color = COLOR_MAP[str(edge)]
        start = keypoints[edge[0]]
        end = keypoints[edge[1]]
        # Either of the points are not visible
        if start == [0, 0, 0] or end == [0, 0, 0]:
            continue
        image = cv2.line(image, tuple(start[:2]), tuple(end[:2]), edge_color, thickness=3)
    for point in keypoints:
        # Point is not visible
        if point == [0, 0, 0]:
            continue
        x, y, v = point
        if v == 1:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        image = cv2.circle(image, (x, y), 4, color, -1)

    plt.imshow(image)
    plt.show()


def main():
    keypoints_path = Path("../../Downloads/keypoints.json")

    with open(keypoints_path) as f:
        labels = json.load(f)
        keypoints, bounding_boxes = extract_labels(labels)

    folder = "../../Downloads/animal_pose"
    images = os.listdir(folder)

    image_path = random.choice(images)
    image = cv2.imread(os.path.join(folder, image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mobile_net = torch.hub.load("pytorch/vision:v0.10.0", "mobilenet_v2", pretrained=True)
    backbone = list(mobile_net.children())[0]
    NUM_KEYPOINTS = 40

    image = torch.zeros((1, 3, 224, 224))
    k = 9
    head_neurons = [16, 32, 64]
    dense_backbone_neurons = [4, 8, 16]
    net = KeypointRCNN(
        backbone,
        k,
        NUM_KEYPOINTS,
        head_neurons,
        dense_backbone_neurons,
    )

    # 20 pairs of x, y coordinates
    preprocessing = A.Compose([
        A.Normalize(),
        A.Resize(224, 224),
        ToTensorV2(),
    ],
        keypoint_params=A.KeypointParams(format="xy")
    )

    dataset = AnimalPose(
        Path(folder),
        keypoints,
        bounding_boxes,
        preprocessing,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=2,
    )

    for batch in data_loader:
        keypoint_mask, bounding_boxes, class_labels = net(batch["image"])


if __name__ == "__main__":
    main()
