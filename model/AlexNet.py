import torch.nn.functional as F
import torch.nn as nn

class AlexNet(nn.Module):

    def __init__(self, num_classes=5):

        super(AlexNet, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),

            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),

            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(

            nn.Linear(1024, 4096),

            nn.ReLU(inplace=True),

            nn.Linear(4096, 4096),

            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes),
        )

    def forward(self, x):

        x = self.features(x)

        flatten_nodes = x.size(1) * x.size(2) * x.size(3)

        x = x.view(x.size(0), flatten_nodes)

        x = self.classifier(x)

        return F.log_softmax(x, dim=0)





