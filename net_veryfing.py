
import torchvision
from models import *

# net = ResNet50()
# print(net)


net = torchvision.models.resnet50(weights='DEFAULT')
num_features = net.fc.in_features
net.fc = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Linear(256, 4),
    nn.Softmax(dim=1)
)
print(net)