import torch

from models.keypoint_rcnn import KeyPointHead, BoxHead, ClassHead


def test_keypoint_head_shape():
    head = KeyPointHead(256, 17)
    x = torch.zeros((2, 256, 14, 14))
    y = head(x)
    assert y.shape == (2, 17, 56, 56)


def test_box_head_shape():
    head = BoxHead([4, 8, 16])
    x = torch.zeros((2, 4))
    y = head(x)
    assert y.shape == (2, 8)

def test_class_head_shape():
    head = ClassHead([4, 8, 16])
    x = torch.zeros((2, 4))
    y = head(x)
    assert y.shape == (2, 2)


