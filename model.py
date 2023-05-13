import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data.dataloader import DataLoader


# constants
pixel_range = 255
image_size = 28*28
digit_classes = 10

# hyperparameters
batch_size = 64
first_hidden_layer = 512
second_hidden_layer = 512
learning_rate = 5e-2
epochs = 5

# load the train dataset for mean calculation
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

# compute mean and std of the train dataset
mean = train_dataset.data.float().mean() / pixel_range
std = train_dataset.data.float().std() / pixel_range

# define transformation to be applied to the images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(mean,), std=(std,))
])

# load the train and test datasets with normalization. these normalizations are applied to the images when they are loaded
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# build the train and test dataloaders
train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

fig, axes = plt.subplots(5, 5, figsize=(10, 10))
fig.tight_layout()

@torch.no_grad()
def visualize_accuracy(model, data_loader, axes, num_images=25):
    for ax in axes.flat:
        ax.clear()

    images = []
    labels = []
    predictions = []

    for i, data in enumerate(data_loader):
        image, label = data
        output, _ = model(image)
        _, predicted = torch.max(output, 1)

        images.append(image)
        labels.append(label)
        predictions.append(predicted)
        if i == num_images:
            break

    images = torch.cat([images[i] for i in range(num_images)])
    labels = torch.cat([labels[i] for i in range(num_images)])
    predictions = torch.cat([predictions[i] for i in range(num_images)])

    images = images.view(-1, 28, 28)

    for ax in axes.flat:
        ax.clear()

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='gray')
        ax.axis('off')
        title = f"Label: {labels[i].item()}\nPrediction: {predictions[i].item()}"
        if labels[i] == predictions[i]:
            ax.set_title(title, color='white', backgroundcolor='green')
        else:
            ax.set_title(title, color='white', backgroundcolor='red')

    plt.show(block=False)
    plt.pause(0.1)

@torch.no_grad()
def evaluate(model, data_loader, num_samples=1000):
    correct_sum = 0
    loss_sum = 0
    total = 0

    for i, data in enumerate(data_loader):
        inputs, targets = data
        logits, loss = model(inputs, targets)
        loss_sum += loss.item()
        _, predicted = torch.max(logits, 1)
        correct_sum += (predicted == targets).sum().item()
        total += targets.size(0)
        if i == num_samples-1:
            break
    
    mean_loss = loss_sum / total
    accuracy = correct_sum / total
    return accuracy, mean_loss


# define the model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(image_size, first_hidden_layer)
        self.fc2 = nn.Linear(first_hidden_layer, second_hidden_layer)
        self.fc3 = nn.Linear(second_hidden_layer, digit_classes)

        
    def forward(self, x, targets=None):
        
        x = x.view(-1, image_size) # batch_size x image_size
        h1 = F.relu(self.fc1(x)) # batch_size x first_hidden_layer
        h2 = F.relu(self.fc2(h1)) # batch_size x second_hidden_layer
        logits = self.fc3(h2) # batch_size x digit_classes
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

# initialize the model and the optimizer
model = MLP()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# training loop
for epoch in range(epochs):
    running_loss = []
    for i, data in enumerate(train_data_loader):
        # get batch data 
        inputs, targets = data
        
        # forward pass
        logits, loss = model(inputs, targets)
        
        #set gradients to zero
        optimizer.zero_grad()
        
        # backward pass
        loss.backward()
        
        # gradient descent
        optimizer.step()
        
        # print loss and accuracy
        running_loss.append(loss.item())
        if i % 100 == 0:
            accuracy, test_loss = evaluate(model, test_data_loader)
            # visualize_accuracy(model, test_data_loader, axes)
            print(f'epoch {epoch+1}/{epochs}, iteration {i+1}: train loss {sum(running_loss) / len(running_loss):0.4f}, test loss {test_loss :0.4f} test accuracy {accuracy:0.2%}')
            # input('Press Enter to continue...')
            
accuracy, test_loss = evaluate(model, test_data_loader, num_samples=len(test_dataset))
print(f'Performance over all test data: loss {test_loss:0.4f}, accuracy {accuracy:0.2%}')