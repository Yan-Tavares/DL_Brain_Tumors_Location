
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    Create a AlexNet CNN with 5 convolutional layers and 3 pooling layers, followed by 2 fully connected layers.
    """

    def __init__(self, 
                 N_in = 1, 
                 N_out = 4, 
                 H_in = 492, 
                 W_in = 492, 
                 hidden_channels = [6,6,6,16,16], 
                 kernels=[5,5,5,5,5,5,5,5], 
                 strides=[1,1,1,5,1,1,1,5], 
                 paddings=[1,2,2,2,2],
                 hidden_layers= [128,50]):
        
        """
        N_in: int with the number of input channels 
        Hidden_channels: list with the number of channels in each convolutional layer
        Kernels: list with the size of the kernel in each convolutional and pooling layer
        Strides: list with the stride in each convolutional and pooling layer
        Paddings: list with the padding in each convolutional layer
        N_out: int with the number of output classes
        """
        self.H_in = H_in
        self.W_in = W_in
        self.hidden_channels = hidden_channels
        self.hidden_layers = hidden_layers
        self.kernels = kernels
        self.strides = strides
        self.paddings = paddings

        
        super(CNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=N_in,
                                out_channels=hidden_channels[0], 
                                kernel_size=kernels[0], 
                                stride=strides[0],
                                padding=paddings[0])
        
        self.pool1 = nn.MaxPool2d(kernel_size=kernels[1], stride=strides[1])
        
        self.conv2 = nn.Conv2d(in_channels=hidden_channels[0],
                                out_channels=hidden_channels[1], 
                                kernel_size=kernels[2], 
                                stride=strides[2],
                                padding=paddings[1])
        
        self.pool2 = nn.MaxPool2d(kernel_size=kernels[3], stride=strides[3])
        
        self.conv3 = nn.Conv2d(in_channels=hidden_channels[1],
                                out_channels=hidden_channels[2], 
                                kernel_size=kernels[4], 
                                stride=strides[4],
                                padding=paddings[2])
        
        self.conv4 = nn.Conv2d(in_channels=hidden_channels[2],
                                out_channels=hidden_channels[3], 
                                kernel_size=kernels[5], 
                                stride=strides[5],
                                padding=paddings[3])

        self.conv5 = nn.Conv2d(in_channels=hidden_channels[3],
                                out_channels=hidden_channels[4], 
                                kernel_size=kernels[6], 
                                stride=strides[6],
                                padding=paddings[4])
        
        self.pool3 = nn.MaxPool2d(kernel_size=kernels[7], stride=strides[7])


        # Calculate the size of the feature map after the final pooling layer
        def conv2d_size_out(size, kernel_size, stride, padding):
            return (size + 2*padding - (kernel_size - 1) - 1) // stride + 1


        H_out = conv2d_size_out(H_in, kernels[0], strides[0], paddings[0])   # After conv1
        W_out = conv2d_size_out(W_in, kernels[0], strides[0], paddings[0])

        H_out = conv2d_size_out(H_out, kernels[1], strides[1], 0)           # After pool1
        W_out = conv2d_size_out(W_out, kernels[1], strides[1], 0)

        H_out = conv2d_size_out(H_out, kernels[2], strides[2], paddings[1])  # After conv2
        W_out = conv2d_size_out(W_out, kernels[2], strides[2], paddings[1])

        H_out = conv2d_size_out(H_out, kernels[3], strides[3], 0)           # After pool2
        W_out = conv2d_size_out(W_out, kernels[3], strides[3], 0)

        H_out = conv2d_size_out(H_out, kernels[4], strides[4], paddings[2])  # After conv3
        W_out = conv2d_size_out(W_out, kernels[4], strides[4], paddings[2])

        H_out = conv2d_size_out(H_out, kernels[5], strides[5], paddings[3])  # After conv4
        W_out = conv2d_size_out(W_out, kernels[5], strides[5], paddings[3])

        H_out = conv2d_size_out(H_out, kernels[6], strides[6], paddings[4])  # After conv5
        W_out = conv2d_size_out(W_out, kernels[6], strides[6], paddings[4])

        H_out = conv2d_size_out(H_out, kernels[7], strides[7], 0)           # After pool3
        W_out = conv2d_size_out(W_out, kernels[7], strides[7], 0)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_channels[-1] * H_out * W_out, hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.out = nn.Linear(hidden_layers[-1], N_out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(F.relu(x))
        x = self.conv2(x)
        x = self.pool2(F.relu(x))
        x = self.conv3(F.relu(x))
        x = self.conv4(F.relu(x))
        x = self.conv5(F.relu(x))
        x = self.pool3(F.relu(x))

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        # No softmax here because we are using cross-entropy loss which already applies softmax
        return x
    
    def num_flat_features(self, x):
        """
        To properly flatten the tensor, it is necessary to find how many features will be created after the flattening.
        num_flat_features multiplies channels * height * width to determine the size of each entry.
        """

        size = x.size()[1:]  # all dimensions except batch
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


    def info(self):
        """
        Print the information of the network
        """

        def conv2d_size_out(size, kernel_size, stride, padding):
            return (size + 2*padding - (kernel_size - 1) - 1) // stride + 1
        
        print("\n------------Network information------------")

        print(f"Input size: {self.H_in}x{self.W_in}")

        H_out = conv2d_size_out(self.H_in, self.kernels[0], self.strides[0], self.paddings[0])  # After conv1
        W_out = conv2d_size_out(self.W_in, self.kernels[0], self.strides[0], self.paddings[0])  

        print(f"Convolutional layer 1 size: {H_out}x{W_out}x{self.hidden_channels[0]}")

        H_out = conv2d_size_out(H_out, self.kernels[1], self.strides[1], 0)           # After pool1
        W_out = conv2d_size_out(W_out, self.kernels[1], self.strides[1], 0)

        print(f"Pooling layer 1 size: {H_out}x{W_out}x{self.hidden_channels[0]}")

        H_out = conv2d_size_out(H_out, self.kernels[2], self.strides[2], self.paddings[1])  # After conv2
        W_out = conv2d_size_out(W_out, self.kernels[2], self.strides[2], self.paddings[1])

        print(f"Convolutional layer 2 size: {H_out}x{W_out}x{self.hidden_channels[1]}")

        H_out = conv2d_size_out(H_out, self.kernels[3], self.strides[3], 0)           # After pool2
        W_out = conv2d_size_out(W_out, self.kernels[3], self.strides[3], 0)

        print(f"Pooling layer 2 size: {H_out}x{W_out}x{self.hidden_channels[1]}")

        H_out = conv2d_size_out(H_out, self.kernels[4], self.strides[4], self.paddings[2])  # After conv3
        W_out = conv2d_size_out(W_out, self.kernels[4], self.strides[4], self.paddings[2])

        print(f"Convolutional layer 3 size: {H_out}x{W_out}x{self.hidden_channels[2]}")

        H_out = conv2d_size_out(H_out, self.kernels[5], self.strides[5], self.paddings[3])  # After conv4
        W_out = conv2d_size_out(W_out, self.kernels[5], self.strides[5], self.paddings[3])

        print(f"Convolutional layer 4 size: {H_out}x{W_out}x{self.hidden_channels[3]}")

        H_out = conv2d_size_out(H_out, self.kernels[6], self.strides[6], self.paddings[4])  # After conv5
        W_out = conv2d_size_out(W_out, self.kernels[6], self.strides[6], self.paddings[4])

        print(f"Convolutional layer 5 size: {H_out}x{W_out}x{self.hidden_channels[4]}")

        H_out = conv2d_size_out(H_out, self.kernels[7], self.strides[7], 0)                 # After pool3
        W_out = conv2d_size_out(W_out, self.kernels[7], self.strides[7], 0)

        print(f"Pooling layer 3 size: {H_out}x{W_out}x{self.hidden_channels[4]}")

        print(f"Flatened size: {H_out*W_out*self.hidden_channels[4]}")

        print(f"Hiddem layer 1 size: {self.hidden_layers[0]}")

        print(f"Hiddem layer 2 size: {self.hidden_layers[1]}")

        print("------------------------------------------\n")


