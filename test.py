import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from src.convolution_network import CNN
import pickle
import os

############################################
# Settings
############################################
inspection = False
dummy_prediction = True
eval_IOU = True
img_inspect = -7

############################################

# Load trained model with parameters
with open('models/cnn_params.pkl', 'rb') as f:
    params = pickle.load(f)

net = CNN(**params)
net.load_state_dict(torch.load('models/CNN_Maxima.pth'))

############################################
# Inspect prediction
############################################

num_batches = len([name for name in os.listdir('data/tensors/unseen')])
print(f"Number of batches available: {num_batches}")

if inspection:

    # Get the list of files in the unseen directory
    unseen_dir = 'data/tensors/unseen'
    files = os.listdir(unseen_dir)

    # Sort the files to ensure they are in order
    files.sort()
    
    # Get the last file in the sorted list
    last_file = files[-1]
    print(f"Batch being inspected: {last_file}")

    # Load the tensors from the last file
    X, Y_bbox = torch.load(os.path.join(unseen_dir, last_file))[:2]

    Y_hat = net.forward(X[img_inspect:img_inspect+1]).squeeze().detach()
    Y_label = Y_bbox[img_inspect:img_inspect+1].squeeze().detach()

    print(f"Normilized outputs:")
    print(f"Y_hat: {Y_hat}")
    print(f"Y_label: {Y_label}\n")

    # Convert tensor to numpy array and then to image
    image = X[img_inspect].squeeze()

    # Denormalize pixel values
    image = image * 250

    print(f"Image being inspected: {img_inspect}")
    print(f"Inspection image shape: {image.shape}")
    print(f"Inspection image min pixel value: {image.min()}, max: {image.max()}")

    # Get image height and width
    H = image.shape[0]
    W = image.shape[1]

    # Denormalize the bounding boxes

    Y_hat[0] = Y_hat[0] * W
    Y_hat[1] = Y_hat[1] * H
    Y_hat[2] = Y_hat[2] * W
    Y_hat[3] = Y_hat[3] * H

    Y_label[0] = Y_label[0] * W
    Y_label[1] = Y_label[1] * H
    Y_label[2] = Y_label[2] * W
    Y_label[3] = Y_label[3] * H


    # Create a figure
    fig = px.imshow(image, color_continuous_scale='gray')

    # Add the predicted bounding box
    fig.add_shape(
        type="rect",
        x0=Y_hat[0],
        y0=Y_hat[1],
        x1=Y_hat[0] + Y_hat[2],
        y1=Y_hat[1] + Y_hat[3],
        line=dict(color="red", width=2),
        name="Predicted Box"
    )

    # Add the ground truth bounding box
    fig.add_shape(
        type="rect",
        x0= Y_label[0],
        y0= Y_label[1],
        x1= Y_label[0] + Y_label[2],
        y1= Y_label[1] + Y_label[3],
        line=dict(color="green", width=2),
        name="Ground Truth Box"
    )

    # Show the figure
    fig.show()

    del X, Y_bbox, Y_hat, Y_label, image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

############################################
# Dummy prediction
############################################
from src.dummy_model import DummyModel, calculate_avg_Y

if dummy_prediction:
    # Calculate the average Y values from the training set
    avg_Y = calculate_avg_Y('data/tensors/train')
    print(f"Average Y values: {avg_Y}")

    dummy_model = DummyModel(avg_Y.to(device))

############################################
# Performance evaluation on test set
############################################

def IOU(Y_hat, Y_bbox):

    # Extract the coordinates
    x1_hat, y1_hat, w1_hat, h1_hat = Y_hat[:, 0], Y_hat[:, 1], Y_hat[:, 2], Y_hat[:, 3]
    x1_bbox, y1_bbox, w1_bbox, h1_bbox = Y_bbox[:, 0], Y_bbox[:, 1], Y_bbox[:, 2], Y_bbox[:, 3]

    # Compute the coordinates for intersection calculation
    x1_inter = torch.max(x1_hat, x1_bbox)
    y1_inter = torch.max(y1_hat, y1_bbox)
    x2_inter = torch.min(x1_hat + w1_hat, x1_bbox + w1_bbox)
    y2_inter = torch.min(y1_hat + h1_hat, y1_bbox + h1_bbox)

    # Compute the area of the intersection rectangle
    inter_area = torch.clamp(x2_inter - x1_inter, min=0) * torch.clamp(y2_inter - y1_inter, min=0)

    # Compute the area of both the prediction and ground truth rectangles
    area_hat = w1_hat * h1_hat
    area_bbox = w1_bbox * h1_bbox

    # Compute the area of the union
    union_area = area_hat + area_bbox - inter_area

    # Compute the IoU
    iou = inter_area / union_area

    return iou

if eval_IOU:

    net.to(device)
    net.eval()
    total_iou = 0.0
    total_iou_dummy = 0.0
    with torch.no_grad():  # Disable gradient computation
        for file in os.listdir('data/tensors/unseen'):
            X, Y_bbox = torch.load(f'data/tensors/unseen/{file}')[:2]
            X, Y_bbox = X.to(device), Y_bbox.to(device)

            # Forward pass
            Y_hat = net.forward(X)
            Y_hat_dummy = dummy_model.forward(X)

            # Compute IoU
            batch_iou = IOU(Y_hat, Y_bbox)
            batch_iou_dummy = IOU(Y_hat_dummy, Y_bbox)

            total_iou += batch_iou.mean().item()
            total_iou_dummy += batch_iou_dummy.mean().item()

    average_iou = total_iou / num_batches
    average_iou_dummy = total_iou_dummy / num_batches
    print(f"\nAverage IoU: {average_iou}")
    print(f"Average IoU (dummy): {average_iou_dummy}")