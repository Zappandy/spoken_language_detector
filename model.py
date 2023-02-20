import torch.nn as nn

class CNNSpeechClassifier2D(nn.Module):

    def __init__(self, channel_inputs, num_channels1, num_channels2, num_channels3, num_channels4,
                 kernel_size, stride, kernel_pool, stride_pool, padding, num_classes):
        super(CNNSpeechClassifier2D, self).__init__()
        self.cnn_layer1 = nn.Sequential(nn.Conv2d(channel_inputs, num_channels1, kernel_size=kernel_size, stride=stride, padding=padding),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(num_channels1),
                                        nn.MaxPool2d(kernel_pool, stride=stride_pool))


        self.cnn_layer2 = nn.Sequential(nn.Conv2d(num_channels1, num_channels2, kernel_size=kernel_size, padding=padding),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(num_channels2),
                                        nn.MaxPool2d(kernel_pool))



        self.cnn_layer3 = nn.Sequential(nn.Conv2d(num_channels2, num_channels3, kernel_size=kernel_size, padding=padding),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(num_channels3),
                                        nn.MaxPool2d(kernel_pool))
        
        self.cnn_layer4 = nn.Sequential(nn.Conv2d(64, num_channels4, kernel_size=kernel_size, padding=padding),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(num_channels4),
                                        nn.MaxPool2d(kernel_pool))

        self.dropout = nn.Dropout(0.5)
        self.fc_layer = nn.Linear(num_channels4*2*9, num_classes)  # shape of cnn_layer 2 after convolution of image!

    def forward(self, x):
        x = self.cnn_layer1(x)
        x = self.cnn_layer2(x)
        x = self.cnn_layer3(x)
        print(x.shape)
        x = self.cnn_layer4(x)

        print(x.shape)
        raise SystemExit
        # vectorizing image
        z = x.reshape(x.shape[0], -1)
        z = self.dropout(z)
        return self.fc_layer(z)
