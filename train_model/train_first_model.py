import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


#step 1: Define training data

training_data = datasets.FashionMNIST(root="data", train = True, download=True, transform=ToTensor())

test_data = datasets.FashionMNIST(root="data", train = False, download=True, transform=ToTensor())

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y {y.shape} {y.dtype}")
    break


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device is {device}")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(nn.Linear(28 * 28, 512),
                                               nn.ReLU(),
                                               nn.Linear(512, 512),
                                               nn.ReLU(),
                                               nn.Linear(512, 10)
                                               )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
        

model = NeuralNetwork().to(device)
print(model)
        
def train(dataloader, model, loss_fn, optimizer):
    sz = len(dataloader.dataset)
    model.train()
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{sz:>5d}] [{batch}] [{len(X)}]")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

train(train_dataloader, model, loss_fn=loss_fn, optimizer=optimizer)

torch.save(model.state_dict(), "mnist_model.pth")
print("Saved pytorch model to mnist_model.pth")
