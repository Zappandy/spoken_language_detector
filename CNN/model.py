import torch.nn as nn


# https://discuss.pytorch.org/t/runtimeerror-given-groups-1-weight-64-3-3-3-so-expected-input-16-64-256-256-to-have-3-channels-but-got-64-channels-instead/12765/3
# https://www.google.com/search?q=RuntimeError%3A+Given+groups%3D1%2C+weight+of+size+%5B16%2C+1%2C+2%2C+2%5D%2C+expected+input%5B1%2C+8%2C+64%2C+862%5D+to+have+1+channels%2C+but+got+8+channels+&source=hp&ei=5aisY9HLEaPh7_UP7eaXaA&iflsig=AJiK0e8AAAAAY6y29Zezs58YNZGiLVUa9aIDcDbq9cY0&ved=0ahUKEwiR0pKhlZ38AhWj8LsIHW3zBQ0Q4dUDCAg&uact=5&oq=RuntimeError%3A+Given+groups%3D1%2C+weight+of+size+%5B16%2C+1%2C+2%2C+2%5D%2C+expected+input%5B1%2C+8%2C+64%2C+862%5D+to+have+1+channels%2C+but+got+8+channels+&gs_lcp=Cgdnd3Mtd2l6EANQAFgAYO8BaABwAHgAgAEAiAEAkgEAmAEAoAECoAEB&sclient=gws-wiz
class CNNSpeechClassifier(nn.Module):

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
        self.fc_layer = nn.Linear(num_channels2*15*214, num_classes)  # shape of cnn_layer 2 after convolution of image!

    def forward(self, x):
        # 1, 8, 64, 862 should be 8, 1, 64, 862
        x = self.cnn_layer1(x)
        x = self.cnn_layer2(x)
        # vectorizing image
        z = x.reshape(x.shape[0], -1)
        return self.fc_layer(z)
