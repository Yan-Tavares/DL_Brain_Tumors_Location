from src import data_preprocess as dp
import time
import torch
import os
import shutil


############################################
# Settings
############################################
# Determine the size of each batch:
num_batches = 30

# Determine training, validation and unseen fractions:
frac_train = 0.5
frac_val = 0.2
frac_unseen = 1 - frac_train - frac_val

# Resize the images to:
H_in = 300
W_in = 300

############################################
#Convert to dataframe and randomize
############################################

#Load the data in a dataframe
df = dp.dataframe_data("data/archive2/_annotation.json")
print(df)
print("\n Dataframe loaded")

# Randomize the dataframe
df = df.sample(frac=1,random_state=42).reset_index(drop=True)
print(df[-15:-1])
print("\nDataframe randomized")
df.to_csv("data/df_randomized.csv", index=False)

batch_size = len(df) // num_batches

print(f"\nNumber of batches: {num_batches}")
print(f"Batch size: {batch_size}")


############################################
#From dataframe to tensor and save
############################################
# Define the directory to be cleaned
dirs = ['data/tensors/train', 'data/tensors/val', 'data/tensors/unseen']

# Clean the directories
for dir in dirs:
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)


for i in range(num_batches):
    # Get the start and end indices for the current batch
    start_idx = i * batch_size
    end_idx = (i + 1) * batch_size if i != num_batches - 1 else len(df)
    
    # Slice the dataframe to get the current batch
    df_batch = df.iloc[start_idx:end_idx].reset_index(drop=True)

    # Process the batch (example: convert to tensor)
    X, Y_bbox,Y_segment = dp.df_to_torch_tensor(df_batch, "data/archive2", H_in= H_in, W_in = W_in)
    
    # Clear the line and print the new message
    if i < num_batches*frac_train:
        torch.save((X, Y_bbox,Y_segment), f'data/tensors/train/tensors_batch_{i}.pt')
    
    if i >= num_batches*frac_train and i < num_batches*(frac_train + frac_val):
        torch.save((X, Y_bbox,Y_segment), f'data/tensors/val/tensors_batch_{i}.pt')

    if i >= num_batches*(frac_train + frac_val):
        torch.save((X, Y_bbox,Y_segment), f'data/tensors/unseen/tensors_batch_{i}.pt')

    print(f"\rSaved batch {i+1}, X: {X.shape}, Y_bbox: {Y_bbox.shape}", end='', flush=True)

print("\nData preprocessing complete")
# ############################################