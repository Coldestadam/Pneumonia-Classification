import argparse
import json
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
from torchvision import transforms as transforms
from torchvision import datasets
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np


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
    
    data_transforms = transforms.Compose([transforms.ToTensor(), 
                                                 transforms.Normalize(mean = [0.5, 0.5, 0.5], 
                                                                      std=[0.5, 0.5, 0.5])])
    train_data = datasets.ImageFolder(root=training_dir, transform=data_transforms)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    
    return trainloader

#Loads the validation data
def _get_valid_data_loader(valid_batch_size, valid_dir):
    
    data_transforms = torchvision.transforms.Compose([transforms.ToTensor(), 
                                                 transforms.Normalize(mean = [0.5, 0.5, 0.5], 
                                                                      std=[0.5, 0.5, 0.5])])
    valid_data = datasets.ImageFolder(root=valid_dir, transform=data_transforms)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=valid_batch_size)
    
    return validloader

def _get_test_data_loader(test_batch_size, test_dir):
    
    data_transforms = torchvision.transforms.Compose([transforms.ToTensor(), 
                                                 transforms.Normalize(mean = [0.5, 0.5, 0.5], 
                                                                      std=[0.5, 0.5, 0.5])])
    test_data = datasets.ImageFolder(root=test_dir, transform=data_transforms)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size)
    
    return testloader



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
            output = model(images)
            
            #Flatten the predictions for the criterion
            output = output.view(-1)
            
            #Calculate the loss
            labels = labels.type(torch.cuda.FloatTensor)
            loss = criterion(output, labels)
           
            loss.backward()
            
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            
            output = model(images)
            output = output.view(-1)
            labels = labels.type(torch.cuda.FloatTensor)
            loss = criterion(output, labels)
            
            valid_loss += loss.item()
        
        train_loss = train_loss / len(train_loader)
        valid_loss = valid_loss / len(valid_loader)
        
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
        
        #If the valid loss decreases, we will save the model parameters
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            with open(os.path.join(model_dir, 'model.pth'), 'wb') as f:
                torch.save(model.state_dict(), f)
            valid_loss_min = valid_loss
            
#Testing function
def test(model, test_loader, criterion, device):
    test_loss=0.0
    model.eval()
    y_preds = []
    y_actual =[]
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        output = model(images)
        
        labels = labels.type(torch.cuda.FloatTensor)

        loss = criterion(output, labels)
        preds = torch.round(output).view(-1)

        test_loss += loss.item()

        for i in preds.tolist():
            y_preds.append(i)
            
        for j in labels.tolist():
            y_actual.append(j)


    test_loss = test_loss / len(test_loader)
    y_preds = np.array(y_preds)
    y_actual = np.array(y_actual)
    
    tp = np.logical_and(y_actual, y_preds).sum()
    fp = np.logical_and(1-y_actual, y_preds).sum()
    tn = np.logical_and(1-y_actual, 1-y_preds).sum()
    fn = np.logical_and(y_actual, 1-y_preds).sum()
    
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    
    print('Total Test Loss:', test_loss)
    
    print("\nTrue Positives:", tp)
    print('False Positives:', fp)
    print('True Negatives:', tn)
    print('False Negatives:', fn)

    print('\nAccuracy:', accuracy_score(y_pred=y_preds, y_true=y_actual))
    print('Recall:', recall)
    print('Precision:', precision)
    print('AUC SCORE:', roc_auc_score(y_true=y_actual, y_score=y_preds))
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--validation-dir', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument('--test-dir', type=str, default=os.environ['SM_CHANNEL_TEST'])
    
    # Hyperparameters
    parser.add_argument('--train-batch-size', type=int, default=64, metavar='trainN', 
                        help='The batch size for the training Dataloader (default: 64)')
    parser.add_argument('--valid-batch-size', type=int, default=32, metavar='validN',
                        help='The batch size for the validation Dataloader (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='testN',
                        help='The batch size for the testing Dataloader (default: 32)')
    parser.add_argument('--epochs', type=int, default=10, metavar="E",
                        help='The amount of epochs to train the model (defualt: 10)')
    parser.add_argument('--learning_rate', type=float, default=0.1, metavar='lr',
                        help='Learning Rate of the Optimizer (default: 0.1)')
    
    #Grab the arguments
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))
    
    #Create the Dataloaders
    train_loader = _get_train_data_loader(args.train_batch_size, args.train_dir)
    valid_loader = _get_valid_data_loader(args.valid_batch_size, args.validation_dir)
    test_loader = _get_test_data_loader(args.test_batch_size, args.test_dir)
                        
    model = AdamsCNN().to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCELoss()
    
    train(model, train_loader, valid_loader, args.epochs, criterion, optimizer, device, args.model_dir)
    
    print('\n\nNow After Training, we can now test the images to calculate the Accuracy and AUC Score\n\n')
    
    test(model, test_loader, criterion, device)