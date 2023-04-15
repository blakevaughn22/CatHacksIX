import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image

class ConvNeuralNet(nn.Module):
	#  Determine what layers and their order in CNN object 
    def __init__(self, num_classes):
        super(ConvNeuralNet, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.fc1 = nn.Linear(64416, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
    
    # Progresses data across layers    
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.max_pool1(out)
        
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool2(out)
                
        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out
    
device="cpu"
    
all_transforms = transforms.Compose([transforms.Resize((256,144)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0,0,0),(1,1,1))
                                     ])

batch_size = 60
num_classes = 11

model = ConvNeuralNet(num_classes)
model.load_state_dict(torch.load('model_epochs/model_epoch_best'))


image = Image.open('./test/Earth/Earth (17).jpg')

image_tensor = all_transforms(image)

image_tensor = image_tensor.unsqueeze(0)

output = model(image_tensor)

_, predicted = torch.max(output.data, 1)

print(predicted[0])


quit()




def test_image(Image_path, model):

    model = ConvNeuralNet(num_classes)
    model.load_state_dict(torch.load('model_epochs/model_epoch_best'))

    # test phase
    model.eval()

    # convert image to torch tensor and add batch dim
    batch = torch.tensor(Image / 255).unsqueeze(0)

    # We don't need gradients for test, so wrap in 
    # no_grad to save memory
    with torch.no_grad():
        batch = batch.to(device)

        # forward propagation
        output = model( batch )

        # get prediction
        output = torch.max(output.data, 1)

    return output

#Turn off gradient calculations
with torch.no_grad():
    
    #Place image data in cpu
    test_image = test_image.to(device)
    #Get output of model
    output = model(test_image)
    print(output)
    _, predicted = torch.max(output.data, 1)
    