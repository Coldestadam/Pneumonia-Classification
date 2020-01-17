import argparse
import json
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
from torchvision import transforms as transforms
from torchvision import datasets


#Importing model from model.py
from model import AdamsCNN

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory to be served."""
    print("Loading model.")
    
    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdamsCNN()
    
    # Loading the saved model parameters
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))
        
    model.to(device).eval()
    
    print("Done loading model.")
    return model

#Loads the training data
def _get_train_data_loader(train_batch_size, training_dir):
    
    transforms = torchvision.transforms.Compose([transforms.ToTensor(), 
                                                 transforms.Normalize(mean = [0.5, 0.5, 0.5], 
                                                                      std=[0.5, 0.5, 0.5])])
    train_data = datasets.ImageFolder(root=training_dir, transform=transforms)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    return trainloader

#Loads the validation data
def _get_valid_data_loader(valid_batch_size, valid_dir):
    
    transforms = torchvision.transforms.Compose([transforms.ToTensor(), 
                                                 transforms.Normalize(mean = [0.5, 0.5, 0.5], 
                                                                      std=[0.5, 0.5, 0.5])])
    valid_data = datasets.ImageFolder(root=valid_dir, transform=transforms)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)
    
    return validloader



#Training Function
def train(model, train_loader, valid_loader, epochs, criterion, optimizer, device, model_dir):
    """
    This is the training function that will be used to train the model
    
    Parameters:
    model        - The model that will be trained
    train_loader - The PyTorch DataLoader that contains the training data
    valid_loader - The PyTorch DataLoader that contains the validation data
    epochs       - The number of epochs to train for
    criterion    - The loss function used for training. 
    optimizer    - The optimizer to use during training.
    device       - Where the model and data should be loaded (gpu or cpu).
    model_dir    - The path to save the model
    """
    print('Starting to train...')
    valid_loss_min = np.Inf
    for epoch in range(1, epochs+1):
        train_loss = 0.0
        valid_loss = 0.0
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            #Zero the gradients
            optimizer.zero_grad()
            
            #Run through the model
            preds = model(images)
            
            #Flatten the predictions for the criterion
            preds = preds.view(-1)
            
            #Calculate the loss
            loss = criterion(preds, labels)
           
            loss.backward()
            
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            preds = model(images)
            loss = criterion(preds, labels)
            
            valid_loss += loss.item()
        
        train_loss = train_loss / len(train_loader)
        valid_loss = val_loss / len(val_loader)
        
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
        
        #If the valid loss decreases, we will save the model parameters
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            with open(os.path.join(model_dir, 'model.pth'), 'wb') as f:
                torch.save(model.state_dict(), f)
            valid_loss_min = valid_loss
            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--validation-dir', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    
    # Hyperparameters
    parser.add_argument('--train-batch-size' type=int, default=64, metavar='trainN',
                        help='The batch size for the training Dataloader (default: 64)')
    parser.add_argument('--valid-batch-size' type=int, default=64, metavar='validN',
                        help='The batch size for the validation Dataloader (default: 32)')
    parser.add_argument('--epochs', type=int, default=10, metavar="E",
                        help='The amount of epochs to train the model (defualt: 10)')
    parser.add_argument('--learning_rate', type=float, default=0.01, metavar='lr',
                        help='Learning Rate of the Optimizer (default: 0.1)')
    
    #Grab the arguments
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))
    
    #Create the Dataloaders
    train_loader = _get_train_data_loader(train_batch_size, train_dir)
    valid_loader = _get_valid_data_loader(valid_batch_size, validation_dir)
    
    model = AdamsCNN().to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCELoss()
    
    train(model, train_loader, valid_loader, args.epochs, criterion, optimizer, device, args.model_dir)