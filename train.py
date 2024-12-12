import torch
import torch.nn as nn
import numpy as np
import time
import pickle
import os
from src.convolution_network import CNN
import plotly.graph_objects as go

############################################
# Architecture Settings
############################################

params = {
    'N_in': 1,
    'N_out': 4,
    'H_in': 300,
    'W_in': 300,
    'hidden_channels': [64,128,256,256,256],
    'kernels': [11,3,5,3,3,3,3,3],
    'strides': [4,2,1,2,1,1,1,2],
    'paddings': [0,2,1,1,1],
    'hidden_layers': [2000,500]
}

with open('models/cnn_params.pkl', 'wb') as f:
    pickle.dump(params, f)

############################################
# Set device
############################################
net = CNN(**params)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
net.to(device)

############################################
# Training settings
############################################

seed = 42*2 # Define seed
net.info()
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
max_epochs = 350


############################################
# Training functions
############################################

def init_weights(m):
    if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

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

def compute_val_loss_and_IOU(net,criterion,device):
    val_loss = 0
    val_IOU = 0
    val_batches = len(os.listdir('data/tensors/val'))

    for val_file in os.listdir('data/tensors/val'):
        X, Y_bbox = torch.load(f'data/tensors/val/{val_file}')[:2]
        X, Y_bbox = X.to(device),Y_bbox.to(device)

        # Forward pass
        Y_hat = net.forward(X)

        # Compute the loss
        val_loss += criterion(Y_hat, Y_bbox).item()

        # Compute the IoU for the batch
        batch_IOU = IOU(Y_hat, Y_bbox)
        # Sum the IoU for the batch
        val_IOU += batch_IOU.mean().item()

        # Delete X and Y to free memory
        del X, Y_bbox
        torch.cuda.empty_cache()

    return val_loss/val_batches , val_IOU/val_batches

def l_penalty(model, lambda_l1, lambda_l2):
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    l2_norm = sum(p.pow(2).sum() for p in model.parameters())
    return lambda_l1 * l1_norm + lambda_l2 * l2_norm

def augment_batch_tensor(X,Y, seed, ratio):
    torch.manual_seed(seed)
    for i in range(X.size(0)):
        if torch.rand(1).item() < ratio:
            # Transpose the image inplace
            X[i] = X[i].clone().transpose(1, 2)
        
            # Swap the x and y coordinates of the bounding box such that the box stays in the correct place
            x_old = Y[i][0].clone()
            y_old = Y[i][1].clone()
            w_old = Y[i][2].clone()
            h_old = Y[i][3].clone()

            Y[i][0], Y[i][1] = y_old, x_old
            Y[i][2], Y[i][3] = h_old, w_old
            
            # Add noise
            noise = torch.randn_like(X[i]) * 0.1
            X[i] += noise
            # Ensure pixel values are within valid range [0, 1] or [0, 255] depending on your data
            X[i] = torch.clamp(X[i], 0, 1)

def update_model(X, Y, net, loss_criterion, train_losses , lambda_l1, lambda_l2):
    # Forward pass
    Y_hat = net.forward(X)

    # Compute the loss
    train_loss = loss_criterion(Y_hat, Y)
    train_losses.append(train_loss.item())

    # Add L1 penalty
    train_loss += l_penalty(net, lambda_l1, lambda_l2)

    # Backward pass
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

def update_val_loss_and_IOU(net,loss_criterion,device,val_losses,val_IOUs):
    val_loss = 0
    val_IOU = 0
    val_batches = len(os.listdir('data/tensors/val'))

    for val_file in os.listdir('data/tensors/val'):
        X, Y_bbox = torch.load(f'data/tensors/val/{val_file}')[:2]
        X, Y_bbox = X.to(device),Y_bbox.to(device)

        # Forward pass
        Y_hat = net.forward(X)

        # Compute the loss
        val_loss += loss_criterion(Y_hat, Y_bbox).item()

        # Compute the IoU for the batch
        batch_IOU = IOU(Y_hat, Y_bbox)
        # Sum the IoU for the batch
        val_IOU += batch_IOU.mean().item()

        # Delete X and Y to free memory
        del X, Y_bbox
        torch.cuda.empty_cache()

    val_losses.append(val_loss/val_batches)
    val_IOUs.append(val_IOU/val_batches)

def early_stopping_func(val_IOUs, maxima_model_save_file = 'models/CNN_Maxima.pth'):
    dv = 0
    dv_list = []

    if not len(val_IOUs) < 6:
        for e in range(-5,0):
                dv = (val_IOUs[e] - val_IOUs[e-1])
                dv_list.append(dv)
    
        if sum(dv_list)/len(dv_list) < 0:
            print("Early stopping")
            return True
        
        if dv_list[-2] > 0 and dv_list[-1] < 0:
            print(" - Maxima case")
            torch.save(net.state_dict(), maxima_model_save_file)
            return False
    return False

def train_session(net, 
                  loss_criterion, 
                  optimizer,
                  max_epochs = 350, 
                  early_stopping = True,
                  augmentation_ratio = 0, 
                  lambda_l1 = 0, 
                  lambda_l2 = 0,
                  seed = 42,
                  device = 'cuda',
                  tensors_path = 'data/tensors/train',
                  maxima_model_save_file = 'models/CNN_Maxima.pth',
                  ):
    
    net.to(device)

    print("-------------Training--------------------")

    train_losses = []
    val_losses = []
    val_IOUs = []

    update_val_loss_and_IOU(net,loss_criterion,device,val_losses,val_IOUs)

    print(f"Initial validation loss: {val_losses[0]}")
    print(f"Initial validation IOU: {val_IOUs[0]}")

    for epoch in range(max_epochs):

        batch_counter = 0
        for file in os.listdir('data/tensors/train'):
            net.train()  # Set the network to training mode
            X, Y_bbox = torch.load(f'data/tensors/train/{file}')[:2]
            X, Y_bbox = X.to(device),Y_bbox.to(device)

            augment_batch_tensor(X, Y_bbox, seed, augmentation_ratio)
            update_model(X, Y_bbox, net, loss_criterion, train_losses, lambda_l1, lambda_l2)

            batch_counter += 1
            print(f"Epoch {epoch}, Batch {batch_counter}", end='\r', flush=True)

        
        # Compute validation loss before next epoch
        update_val_loss_and_IOU(net,loss_criterion,device,val_losses,val_IOUs)
        print(f"Epoch {epoch}:\n - Validation loss: {val_losses[-1]}\n - Val IOU: {val_IOUs[-1]}\n - Train loss: {train_losses[-1]}")


        # Early stopping
        if early_stopping:
            if early_stopping_func(val_IOUs, maxima_model_save_file = maxima_model_save_file):
                break
    
    return net, train_losses, val_losses, val_IOUs

def plot_losses(train_losses, val_losses, val_IOUs):
    # Plot losses using Plotly with information box
    fig = go.Figure(data=go.Scatter(y=train_losses, mode='lines', name='Training Loss'))

    fig.add_trace(go.Scatter(
        x = np.arange(0,len(val_losses)) * (train_batches), 
        y= val_losses, mode='lines', 
        name='Validation Loss'))

    # Add information box as an annotation
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.95, y=0.95,
        showarrow=False,
        align="left",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        bgcolor="lightgrey",
        opacity=0.8
    )

    fig.update_layout(
        xaxis_title='Batch passes',
        yaxis_title='Loss'
    )

    fig.show()

    # Plot IOU using Plotly with information box
    fig = go.Figure(data=go.Scatter(y=val_IOUs, mode='lines', name='Validation IOU', line=dict(color='green')))

    #Add IOU curve as green line
    fig.update_layout(
        xaxis_title='Epochs',
        yaxis_title='Validation IoU'
        
    )

    # Save the plot as html
    fig.show()

    return fig

############################################
# Train the network
############################################

val_batches = len(os.listdir('data/tensors/val'))
train_batches = len(os.listdir('data/tensors/train'))

net.apply(init_weights)
start_time = time.time()

#------- Train session 1
tensors_path = 'data/tensors/train'
maxima_model_save_file = 'models/CNN_Maxima.pth'

learning_rate = 10**(-5)
optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)
loss_criterion = nn.MSELoss()

net, train_losses, val_losses, val_IOUs = train_session(net,
                                                        loss_criterion, 
                                                        optimizer,
                                                        max_epochs = 300, 
                                                        early_stopping = True,
                                                        augmentation_ratio = 0, 
                                                        lambda_l1 = 0*10**(-6), 
                                                        lambda_l2 = 0*10**(-6),
                                                        seed = 84,
                                                        device = device,
                                                        tensors_path = tensors_path,
                                                        maxima_model_save_file = maxima_model_save_file,
                                                        )

end_time = time.time()
train_time = end_time - start_time

print("\nTraining complete")
print(f"Time: {train_time}, seconds\n")
plot_losses(train_losses, val_losses, val_IOUs)
