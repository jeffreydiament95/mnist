import torch
import torchvision
import torchvision.transforms as transforms

# Load the train dataset for mean calculation
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

# compute mean and std of the train dataset
mean = train_dataset.data.float().mean() / 255
std = train_dataset.data.float().std() / 255

# define transformation to be applied to the images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(mean,), std=(std,))
])

# Load the train and test datasets with normalization. these normalizations are applied to the images when they are loaded
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

