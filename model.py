import torch.nn.functional as F
import torch.nn as nn

class AdamsCNN(nn.Module):
    """
    This is my model to show that I know how to construct CNN models.
    
    There are 4 Convolutional and Pooling layers
    
    There are 5 Fully Connected Layers
    """
    def __init__(self):
        """
        Initialize the CNN model by setting it up using the size of (squared) image and the output dimension
        """
        super(AdamsCNN, self).__init__()
        
        #First I will initiate the pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)#This will half the input's width and length
        
        #Image_size or input_size = 224x224x3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        
        #input_size = 112x112x16
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        #input_size = 56x56x32
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        #input_size = 28x28x64
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        #input_size = 14x14x128 = 25,088
        #After Flattening the data will give a vector size 25,088x1
        self.fc1 = nn.Linear(25088, 12544)
        self.fc2 = nn.Linear(12544, 3136)
        self.fc3 = nn.Linear(3136, 392)
        self.fc4 = nn.Linear(392, 49)
        self.fc5 = nn.Linear(49, 1)#One output to use BCELoss and Sigmoid
        
        #Involve dropout in the Linear Layers to prevent overfitting with 20% probability
        self.drop = nn.Dropout(0.2)
        
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        #First Conv and Pooling Layer
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        #Second
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        #Third
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        #Fourth
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        
        #Need to Flatten the image
        x = x.view(-1, 25088)
        
        #Linear Layers
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = F.relu(self.fc3(x))
        x = self.drop(x)
        x = F.relu(self.fc4(x))
        x = self.drop(x)
        #Apply sigmoid after the last Linear Layer
        x = self.sig(self.fc5(x))

        return x