import torch.nn as nn

# Here We defined three convolutional layers for authoencoder
# each cnn layer contains: "conv2","BatchNorm2d" and  Relu activation function
n_layers='5_layer'
# implement autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential( 
            nn.Conv2d(3, 12, kernel_size=4, stride=2, padding=1), #3x32x32 to 12x16x16
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 24, kernel_size=4, stride=2, padding=1), #12x16x16 to 24x8x8
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 48, kernel_size=4, stride=2, padding=1), #24x8x8 to 48x4x4
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 96, kernel_size=4, stride=2, padding=1), #48x4x4 to 96x2x2
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 192, kernel_size=4, stride=2, padding=1), #96x2x2 to 192x1x1
            nn.BatchNorm2d(192),
            nn.ReLU()
        )

        # kjl多加一个类别层
        self.classifier = nn.Sequential(
            nn.Linear(192, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 10),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1, output_padding=0), #192x1x1 to 96x2x2
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.ConvTranspose2d(96, 48, kernel_size=4, stride=2, padding=1, output_padding=0), #96x2x2 to 48x4x4
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.ConvTranspose2d(48, 24, kernel_size=4, stride=2, padding=1, output_padding=0), #48x4x4 to 24x8x8
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, kernel_size=4, stride=2, padding=1, output_padding=0), #24x8x8 to 12x16x16
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, kernel_size=4, stride=2, padding=1, output_padding=0), #12x16x16 to 3x32x32
            nn.BatchNorm2d(3),
            nn.Tanh()
        )
  # defined the forward function to take input
  # call the encoder to encod the input
  # call the decoder function to decode the encoded inpu
    def forward(self, x):
        encoded = self.encoder(x)

        temp = encoded # 层数最高的卷积
        emb = temp.view(encoded.size(0), -1) #flatten
        pre = self.classifier(emb)

        decoded = self.decoder(encoded)
        return encoded, decoded, pre

    #with an pooling on output of trained encoder we have an embeding vector for each image 
    #this vector can used for MLP or other method for prediction
    def embedding(self,x):
        encoded = self.encoder(x)
        # encoded = nn.AvgPool2d(2,2)(encoded) #48*4*4-->48x2x2 embedded features # 层数最高的卷积不能有这个
        encoded = encoded.view(encoded.size(0), -1)   #flatten 48x2x2->192
        return encoded