import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define the augment_batch_tensor function
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

# Function to plot images with bounding boxes side by side
def plot_images_side_by_side(original_image, original_bbox, augmented_image, augmented_bbox):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot original image
    ax = axes[0]
    ax.imshow(original_image.permute(1, 2, 0).cpu().numpy(), cmap='gray')
    rect = patches.Rectangle((original_bbox[0], original_bbox[1]), original_bbox[2], original_bbox[3], linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.set_title('Original Image with bounding box', fontsize=16)

    # Plot augmented image
    ax = axes[1]
    ax.imshow(augmented_image.permute(1, 2, 0).cpu().numpy(), cmap='gray')
    rect = patches.Rectangle((augmented_bbox[0], augmented_bbox[1]), augmented_bbox[2], augmented_bbox[3], linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.set_title('Augmented Image with bounding box', fontsize=16)

    plt.show()

# Load a sample tensor (replace with your actual data loading)
X, Y_bbox = torch.load('data/tensors/val/tensors_batch_15.pt')[:2]  # Replace with actual file path
X, Y_bbox = X.to('cpu'), Y_bbox.to('cpu')  # Move to CPU for plotting

# Scale bounding boxes to image size
image_height, image_width = X.size(2), X.size(3)
Y_bbox[:, 0] *= image_width
Y_bbox[:, 1] *= image_height
Y_bbox[:, 2] *= image_width
Y_bbox[:, 3] *= image_height

# Select an index to visualize
index = 3
# Augment the batch tensor
seed = 42
ratio = 1
X_aug = X.clone()
Y_bbox_aug = Y_bbox.clone()
augment_batch_tensor(X_aug, Y_bbox_aug, seed, ratio)

# Plot original and augmented images side by side
plot_images_side_by_side(X[index], Y_bbox[index], X_aug[index], Y_bbox_aug[index])