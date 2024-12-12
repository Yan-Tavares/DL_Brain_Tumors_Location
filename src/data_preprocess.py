import json
import pandas as pd
import numpy as np
import torch
from PIL import Image


def dataframe_data(json_file_relative_path):

    file = open(json_file_relative_path)
    data = json.load(file)
    
    df = pd.DataFrame(columns=['filename','x_box', 'y_box', 'w', 'h', 'X0', 'Y0', 'X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3','X4','Y4'])
    
    for key in data:
        filename = key
        bbbox = np.array(data[key]['bbox'])
        segment = np.array(data[key]['segmentation'])

        #Concatenate the bbox and the segment
        row = np.concatenate((np.array(filename),bbbox,segment), axis=None)
        df.loc[key,:] = row

    return df

def df_to_torch_tensor(df,image_folder_path, H_in=300, W_in=300):
    #Make a Y tensor with the classes
    Y_bbox = df.iloc[:,1:5].values.astype(np.float32)
    Y_segment = df.iloc[:,5:].values.astype(np.float32)

    Y_bbox = torch.tensor(Y_bbox, dtype=torch.float32)
    Y_segment = torch.tensor(Y_segment, dtype=torch.float32)

    #Make a empity X tensor with 4 dimensions (batch size, channels, height, width)
    X = torch.empty((len(df), 1, H_in, W_in), dtype=torch.float32)

    for i in range(len(df)):
        image_path = image_folder_path + "/" + df.iloc[i,0]
        with Image.open(image_path).convert('L') as img:
            
            #Normalize Y_bbox and Y_segment based on the image size
            W_img, H_img = img.size
            Y_bbox[i,0] = Y_bbox[i,0] / W_img #Normalize the x_box
            Y_bbox[i,1] = Y_bbox[i,1] / H_img #Normalize the y_box

            Y_bbox[i,2] = Y_bbox[i,2] / W_img #Normalize the w
            Y_bbox[i,3] = Y_bbox[i,3] / H_img #Normalize the h
            

            img = img.resize((W_in, H_in))  # Resize the image
            img = np.array(img) / 255.0  # Normalize image values to [0, 1]
            img = torch.tensor(img, dtype=torch.float32)  # Convert to tensor
            img = img.unsqueeze(0)  # Add a channel dimension

            X[i] = img
    


    return X,Y_bbox,Y_segment
