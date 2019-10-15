import torch.nn.functional as F
import torch.nn as nn

# class AlexNet(nn.Module):
#
#     def __init__(self, num_classes=5):
#
#         super(AlexNet, self).__init__()
#
#         self.features = nn.Sequential(
#
#             nn.Conv2d(3, 32, kernel_size=11, stride=4, padding=2),
#
#             nn.ReLU(inplace=True),
#
#             nn.MaxPool2d(kernel_size=3, stride=2),
#
#             nn.Conv2d(32, 64, kernel_size=5, padding=2),
#
#             nn.ReLU(inplace=True),
#
#             nn.MaxPool2d(kernel_size=3, stride=2),
#
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#
#             nn.ReLU(inplace=True),
#
#             nn.Conv2d(128, 64, kernel_size=3, padding=1),
#
#             nn.ReLU(inplace=True),
#
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#
#             nn.ReLU(inplace=True),
#
#             nn.MaxPool2d(kernel_size=3, stride=2),
#         )
#
#         self.classifier = nn.Sequential(
#
#             nn.Linear(3136, 2048),
#
#             nn.ReLU(inplace=True),
#
#             nn.Dropout(0.5),
#
#             nn.Linear(2048, 1024),
#
#             nn.ReLU(inplace=True),
#
#             nn.Dropout(0.5),
#
#             nn.Linear(1024, num_classes),
#         )
#
#     def forward(self, x):
#
#         x = self.features(x)
#
#         x = x.view(x.size(0), -1)
#
#         x = self.classifier(x)
#
#         return x
class AlexNet(nn.Module):

    def __init__(self, num_classes=2):

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
            nn.Linear(9216, 4096),

            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.Linear(4096, 4096),

            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)

        x = x.view(x.size(0), -1)

        x = self.classifier(x)

        return x


class Agent_AlexNet(nn.Module):

    def __init__(self):

        super(Agent_AlexNet, self).__init__()

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

    def forward(self, x):

        x = self.features(x)

        x = x.view(x.size(0), -1)

        return x


class Server_AlexNet(nn.Module):

    def __init__(self, flatten_nodes, num_classes):

        super(Server_AlexNet, self).__init__()

        self.classifier = nn.Sequential(

            nn.Linear(flatten_nodes, 4096),

            nn.ReLU(inplace=True),

            nn.Linear(4096, 4096),

            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes),
        )

    def forward(self, x):

        x = self.classifier(x)

        return x



