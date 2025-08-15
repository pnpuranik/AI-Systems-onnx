import torch

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 128, 5)

    def forward(self, x):
        return torch.relu(self.conv1(x))

model = MyModel()
#batch_size, in_channels, height, weight (height and weight must be at least 5)
input_tensor = torch.rand(1, 1, 28,28) 
torch.onnx.export(model, input_tensor, 
                  "my_first_model.onnx")
