import torch
import torch.nn as nn
from torchvision.transforms import transforms
import numpy as np
from torch.autograd import Variable
from torchvision.models import squeezenet1_1
import torch.functional as F
from io import open
import os
from PIL import Image
import pathlib
import glob
import cv2

train_path='./training'
pred_path= './prediction' # we make the prediction with the new model "best_checkpoint.model" (that has the best accuracy)

# CNN Network
class ConvNet(nn.Module):
    def __init__(self, num_classes=20):
        super(ConvNet, self).__init__()

        # Output size after convolution filter
        # ((w-f+2P)/s) +1

        # Input shape= (256,3,350,350)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1) # let's try with 12 channels as our data set is quite small
        # Shape= (256,12,350,350)
        self.bn1 = nn.BatchNorm2d(num_features=12) # same number as number of channels; number of different filters or feature maps produced by that layer
        # Shape= (256,12,350,350)
        self.relu1 = nn.ReLU() # to bring non-linearity
        # Shape= (256,12,350,350)

        self.pool = nn.MaxPool2d(kernel_size=2) # reduces the height and width of convolutional output while keeping the most salient features
        # Reduce the image size be factor 2
        # Shape= (256,12,175,175)

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1) # add second conv layer to apply more patterns and increase the number of channels to 20
        # Shape= (256,20,175,175)
        self.relu2 = nn.ReLU()
        # Shape= (256,20,175,175)

        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Shape= (256,32,175,175)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        # Shape= (256,32,175,175)
        self.relu3 = nn.ReLU()
        # Shape= (256,32,175,175)

        self.fc = nn.Linear(in_features=175 * 175 * 32, out_features=num_classes) # fully connected layer

    # Feed forward function

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)

        output = self.pool(output)

        output = self.conv2(output)
        output = self.relu2(output)

        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)

        # Above output will be in matrix form, with shape (256,32,175,175)

        output = output.view(-1, 32 * 175 * 175)

        output = self.fc(output)

        return output

checkpoint= torch.load('6b_best_checkpoint.model')
model=ConvNet(num_classes=20)
model.load_state_dict(checkpoint)
model.eval() # to set dropout and batch normalisation
#Transforms
transformer=transforms.Compose([
    transforms.Resize((350,350)),
    transforms.ToTensor()])  #0-255 to 0-1, numpy to tensors

# Categories
root=pathlib.Path(train_path)
classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])

# Remove '.DS_Store' from classes list
classes = [c for c in classes if c != '.DS_Store']

print(classes)

# prediction function
def prediction(img_path, transformer):
    image = Image.open(img_path).convert('RGB')

    image_tensor = transformer(image).float()

    image_tensor = image_tensor.unsqueeze_(0)

    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()

    input = Variable(image_tensor)

    output = model(input)

    index = output.data.numpy().argmax() # category id is the one with the highest probability

    pred = classes[index]

    return pred # output is the category name

images_path=glob.glob(pred_path+'/*.jpg')

print(images_path) # all the validation images
pred_dict = {}

for i in images_path:
    filename = os.path.basename(i)  # Extract the file name from the path
    pred_dict[filename] = prediction(i, transformer)

print(pred_dict)


# Path to the folder containing the ground truth labels
truth_folder = './watches_images'

# Initialize variables
total_images = 0
correct_predictions = 0

# Iterate over the predicted dictionary
for filename, predicted_class in pred_dict.items():
    # Construct the path to the ground truth folder for the predicted class
    truth_class_folder = os.path.join(truth_folder, predicted_class)

    # Check if the predicted class folder exists
    if os.path.exists(truth_class_folder):
        # Get the list of ground truth filenames for the predicted class
        truth_filenames = os.listdir(truth_class_folder)

        # Check if the predicted filename is in the ground truth filenames
        if filename in truth_filenames:
            correct_predictions += 1

    total_images += 1

# Calculate the accuracy
accuracy = correct_predictions / total_images * 100

print(f"Accuracy: {accuracy:.2f}%")



























