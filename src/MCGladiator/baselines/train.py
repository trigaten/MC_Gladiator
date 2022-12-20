from torchvision.transforms import Grayscale
from torch.optim import Adam
from torch.nn import BCELoss
from net import ActionNet
EPOCHS = 100
net = ActionNet(5)
optimizer = Adam(net.parameters())
loss = BCELoss() 

path = "episodes"
# (360, 640, 3)
for epoch in range(EPOCHS):
    pass