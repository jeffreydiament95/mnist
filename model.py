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
learning_rate = 1e-3
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
test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# returns the accuracy of the model on the entire test dataset
@torch.no_grad()
def evaluate(model, data_loader):
    correct = 0
    total = 0
    for data in data_loader:
        images, labels = data
        outputs, _ = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()    
    return correct / total

# define the model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(image_size, first_hidden_layer)
        self.fc2 = nn.Linear(first_hidden_layer, digit_classes)
        
    def forward(self, x, targets=None):
        
        x = x.view(-1, image_size) # batch_size x image_size
        h1 = F.relu(self.fc1(x)) # batch_size x first_hidden_layer
        logits = self.fc2(h1) # batch_size x digit_classes
        
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
        if i % 100 == 99:
            accuracy = evaluate(model, test_data_loader)
            print(f'epoch {epoch+1}/{epochs}, iteration {i+1}: loss {sum(running_loss) / len(running_loss):0.4f}, accuracy {accuracy:0.2%}')
