import torch.nn as nn


class CNNSpeechClassifier(nn.module):

    def __init__(self, channel_inputs, num_channels1, num_channels2, kernel_size, kernel_pool, padding, num_classes):
        super(CNNSpeechClassifier, self).__init__()
        self.cnn_layer1 = nn.Sequential(nn.Conv2d(channel_inputs, num_channels1, kernel_size=kernel_size, padding=padding),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(num_channels1),
                                        nn.MaxPool2d(kernel_pool))

        self.cnn_layer2 = nn.Sequential(nn.Conv2d(num_channels1, num_channels2, kernel_size=kernel_size, padding=padding),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(num_channels2),
                                        nn.MaxPool2d(kernel_pool))
        self.fc_layer = nn.Linear(8525*num_channels2, num_classes)

    def forward(self, x):
        print(x.shape)
        x = self.cnn_layer1(x)
        print(x.shape)
        x = self.cnn_layer2(x)
        print(x.shape)
