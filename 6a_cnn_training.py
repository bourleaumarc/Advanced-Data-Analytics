# Load libraries
#Load libraries
import os
import numpy as np
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import pathlib
import random
import shutil
# Checking for device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Transforms
transformer = transforms.Compose([
    transforms.Resize((350, 350)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Path to the original image folder
original_folder = './watches_images_cnn' #contains all the images; warning, the folder is empty after operation

# Path to the destination folders for validation, testing, and training
prediction_folder = './prediction'
testing_folder = './testing'
training_folder = './training'

# Create the destination folders if they don't exist
os.makedirs(prediction_folder, exist_ok=True)
os.makedirs(testing_folder, exist_ok=True)
os.makedirs(training_folder, exist_ok=True)

# List to store alli image paths and their corresponding labels
image_paths_labels = []
# Get the subdirectories within the original folder
subdirectories = [subdir for subdir in os.listdir(original_folder) if os.path.isdir(os.path.join(original_folder, subdir))]

# Iterate over the subdirectories
for category in subdirectories:
    category_folder = os.path.join(original_folder, category)
    for file in os.listdir(category_folder):
        if file.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(category_folder, file)
            image_paths_labels.append((image_path, category))

# Print the total number of images
print(f"Total number of images: {len(image_paths_labels)}")

# Shuffle the image paths randomly
random.shuffle(image_paths_labels)

# Calculate the number of images for each set
total_images = len(image_paths_labels)
prediction_count = int(total_images * 0.15)
testing_count = int(total_images * 0.15)
training_count = total_images - prediction_count - testing_count

# Split the image paths and labels into validation, testing, and training sets
prediction_data = image_paths_labels[:prediction_count]
testing_data = image_paths_labels[prediction_count:prediction_count + testing_count]
training_data = image_paths_labels[prediction_count + testing_count:]
# Make sure that prediction folder images are shuffled (no pattern)
prediction_images = os.listdir(prediction_folder)
random.shuffle(prediction_images)
# Move the images to the respective output folders while maintaining the category structure

for image, _ in prediction_data:
    file_name = os.path.basename(image)
    dst_path = os.path.join(prediction_folder, file_name)
    shutil.move(image, dst_path)


for image, label in testing_data:
    file_name = os.path.basename(image)
    dst_path = os.path.join(testing_folder, label, file_name)  # Create subfolders for each category
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)  # Create subfolders if they don't exist
    shutil.move(image, dst_path)

for image, label in training_data:
    file_name = os.path.basename(image)
    dst_path = os.path.join(training_folder, label, file_name)  # Create subfolders for each category
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)  # Create subfolders if they don't exist
    shutil.move(image, dst_path)

print("Image splitting and shuffling completed successfully.")
print(f"Number of images in prediction folder: {len(os.listdir(prediction_folder))}")
testing_image_count = sum(len(files) for _, _, files in os.walk(testing_folder))
training_image_count = sum(len(files) for _, _, files in os.walk(training_folder))

print(f"Number of images in testing folder: {testing_image_count}")
print(f"Number of images in training folder: {training_image_count}")


#Path for training and testing directory
train_path='./training'
test_path='./testing'

train_loader=DataLoader(
    torchvision.datasets.ImageFolder(train_path,transform=transformer),
    batch_size=32, shuffle=True
)
test_loader=DataLoader(
    torchvision.datasets.ImageFolder(test_path,transform=transformer),
    batch_size=32, shuffle=True
)

# Categories
root=pathlib.Path(train_path)
classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])

print(classes)


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

model=ConvNet(num_classes=20).to(device)
#Optmizer and loss function
optimizer=Adam(model.parameters(),lr=0.001,weight_decay=0.0001)
loss_function=nn.CrossEntropyLoss()
num_epochs=15
#calculating the size of training and testing images
train_count=len(glob.glob(train_path+'/**/*.jpg'))
test_count=len(glob.glob(test_path+'/**/*.jpg'))
print(train_count,test_count)

# Model training and saving best model

best_accuracy = 0.0

for epoch in range(num_epochs):

    # Evaluation and training on training dataset
    model.train()
    train_accuracy = 0.0
    train_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.cpu().data * images.size(0)
        _, prediction = torch.max(outputs.data, 1)

        train_accuracy += int(torch.sum(prediction == labels.data))

    train_accuracy = train_accuracy / train_count
    train_loss = train_loss / train_count

    # Evaluation on testing dataset
    model.eval()

    test_accuracy = 0.0
    for i, (images, labels) in enumerate(test_loader):
        if torch.cuda.is_available(): # way to use GPU instead of CPU
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
        test_accuracy += int(torch.sum(prediction == labels.data))

    test_accuracy = test_accuracy / test_count

    print('Epoch: ' + str(epoch) + ' Train Loss: ' + str(train_loss) + ' Train Accuracy: ' + str(
        train_accuracy) + ' Test Accuracy: ' + str(test_accuracy))

    # Save the best model
    if test_accuracy > best_accuracy:
        torch.save(model.state_dict(), '6b_best_checkpoint.model')
        best_accuracy = test_accuracy


# Epoch: 0 Train Loss: tensor(232.4917) Train Accuracy: 0.1419667590027701 Test Accuracy: 0.0744336569579288
# Epoch: 1 Train Loss: tensor(136.7826) Train Accuracy: 0.3012465373961219 Test Accuracy: 0.16828478964401294
# Epoch: 2 Train Loss: tensor(51.7631) Train Accuracy: 0.5332409972299169 Test Accuracy: 0.3851132686084142
# Epoch: 3 Train Loss: tensor(26.6036) Train Accuracy: 0.6668975069252078 Test Accuracy: 0.5275080906148867
# Epoch: 4 Train Loss: tensor(14.3590) Train Accuracy: 0.7901662049861495 Test Accuracy: 0.5598705501618123
# Epoch: 5 Train Loss: tensor(7.4208) Train Accuracy: 0.8566481994459834 Test Accuracy: 0.7119741100323624
# Epoch: 6 Train Loss: tensor(3.9684) Train Accuracy: 0.8975069252077562 Test Accuracy: 0.7055016181229773
# Epoch: 7 Train Loss: tensor(1.6290) Train Accuracy: 0.9508310249307479 Test Accuracy: 0.7346278317152104
# Epoch: 8 Train Loss: tensor(0.3422) Train Accuracy: 0.9778393351800554 Test Accuracy: 0.7734627831715211
# Epoch: 9 Train Loss: tensor(0.1847) Train Accuracy: 0.9875346260387812 Test Accuracy: 0.8252427184466019

#The first hyperparameters that we used was batch size 256 and number of epochs = 10. The maximum accuracy on the testing was of 82% and on the prediction 46%.
# Second testing: batch size 64, numb of epochs = 10. Max accuracy of 82% on the testing and prediction 77%
# Third testing: batch size 32, numb of epochs = 10. Max accuracy of 82% on the testing and prediction 66%
# 4th testing: batch size 64, numb of epochs = 15. Max accuracy of 82% on the testing and prediction 68%
# 5th testing: batch size 32, num of epochs = 15. Max accuracy of 82% on the testing and prediction 77% keep this
