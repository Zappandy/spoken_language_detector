import torch.nn as nn

class CNNSpeechClassifier(nn.Module):

    def __init__(self, channel_inputs, num_channels1, num_channels2, kernel_size, kernel_pool, padding, num_classes):
        num_channels3 = 64
        super(CNNSpeechClassifier, self).__init__()
        self.cnn_layer1 = nn.Sequential(nn.Conv2d(channel_inputs, num_channels1, kernel_size=kernel_size, padding=padding),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(num_channels1),
                                        nn.MaxPool2d(kernel_pool))


        self.dropout = nn.Dropout(0.5)  # 0.5 best so far. I've tried .25 as well
        self.fc_layer = nn.Linear(num_channels1*31*430, num_classes)  # shape of cnn_layer 1 after convolution of image!
        #self.cnn_layer2 = nn.Sequential(nn.Conv2d(num_channels1, num_channels2, kernel_size=kernel_size, padding=padding),
        #                                nn.ReLU(),
        #                                nn.BatchNorm2d(num_channels2),
        #                                nn.MaxPool2d(kernel_pool))

        ##self.fc_layer = nn.Linear(num_channels2*15*214, num_classes)  # shape of cnn_layer 2 after convolution of image!

        #self.cnn_layer3 = nn.Sequential(nn.Conv2d(num_channels2, num_channels3, kernel_size=kernel_size, padding=padding),
        #                                nn.ReLU(),
        #                                nn.BatchNorm2d(num_channels3),
        #                                nn.MaxPool2d(kernel_pool))
        #

        #self.fc_layer = nn.Linear(num_channels3*7*106, num_classes)  # shape of cnn_layer 2 after convolution of image!

    def forward(self, x):
        # 1, 8, 64, 862 should be 8, 1, 64, 862
        x = self.cnn_layer1(x)
        #print(x.shape)
        #x = self.cnn_layer2(x)
        ##print(x.shape)
        #x = self.cnn_layer3(x)
        #print(x.shape)
        #raise SystemExit
        # vectorizing image
        z = x.reshape(x.shape[0], -1)
        z = self.dropout(z)
        return self.fc_layer(z)
