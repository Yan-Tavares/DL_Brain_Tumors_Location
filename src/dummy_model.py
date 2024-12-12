import torch
import os

class DummyModel:
    def __init__(self, avg_Y):
        self.avg_Y = avg_Y

    def forward(self, X):
        batch_size = X.size(0)
        return self.avg_Y.repeat(batch_size, 1)


def calculate_avg_Y(train_dir):
    total_Y = None
    count = 0

    for file in os.listdir(train_dir):
        Y_bbox = torch.load(os.path.join(train_dir, file))[1]
        if total_Y is None:
            total_Y = Y_bbox.sum(dim=0)
        else:
            total_Y += Y_bbox.sum(dim=0)
        count += Y_bbox.size(0)

    avg_Y = total_Y / count
    return avg_Y
