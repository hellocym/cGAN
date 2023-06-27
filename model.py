import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d=64, num_classes=10, img_size=64):
        """
        channels_img: number of channels in the images
        features_d: number of features in the first layer of the discriminator
        """
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.D = nn.Sequential(
            # Input: N x (channels_img + 1) x 64 x 64
            nn.Conv2d(channels_img + 1, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # Output: N x features_d x 32 x 32
            self._block(features_d, features_d*2, kernel_size=4, stride=2, padding=1),
            # Output: N x features_d*2 x 16 x 16
            self._block(features_d*2, features_d*4, kernel_size=4, stride=2, padding=1),
            # Output: N x features_d*4 x 8 x 8
            self._block(features_d*4, features_d*8, kernel_size=4, stride=2, padding=1),
            # Output: N x features_d*8 x 4 x 4
            nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0),
            # Output: N x 1 x 1 x 1
        )
        self.embed = nn.Embedding(num_classes, img_size*img_size)
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )
        
    
    def forward(self, x, labels):
        embedding = self.embed(labels).view(labels.shape[0], 1, self.img_size, self.img_size)
        x = torch.cat([x, embedding], dim=1) # N x C x img_size x img_size
        return self.D(x)
    

class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g=64, num_classes=10, img_size=64, embed_size=64):
        """
        z_dim: dimension of the noise vector
        channels_img: number of channels in the images
        features_g: number of features in the first layer of the generator
        """
        super(Generator, self).__init__()
        self.img_size = img_size
        self.G = nn.Sequential(
            # Input: N x (z_dim + embed_size) x 1 x 1
            self._block(z_dim + embed_size, features_g*16, kernel_size=4, stride=1, padding=0), 
            # Output: N x features_g*16 x 4 x 4
            self._block(features_g*16, features_g*8, kernel_size=4, stride=2, padding=1),
            # Output: N x features_g*8 x 8 x 8
            self._block(features_g*8, features_g*4, kernel_size=4, stride=2, padding=1),
            # Output: N x features_g*4 x 16 x 16
            self._block(features_g*4, features_g*2, kernel_size=4, stride=2, padding=1),
            # Output: N x features_g*2 x 32 x 32
            nn.ConvTranspose2d(features_g*2, channels_img, kernel_size=4, stride=2, padding=1),
            # Output: N x channels_img x 64 x 64
            nn.Tanh(), 
            # Output: N x channels_img x 64 x 64, range [-1, 1]
        )
        self.embed = nn.Embedding(num_classes, embed_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x, labels):
        # z: N x z_dim x 1 x 1
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3) # N x embed_size x 1 x 1
        x = torch.cat([x, embedding], dim=1) # N x (z_dim + embed_size) x 1 x 1
        return self.G(x)
    

def initialize_weights(model):
    """
    Initialize weights of the model with N(0, 0.02)
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

    
def test():
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100
    labels = 10
    x = torch.randn((N, in_channels, H, W))
    labels = torch.randint(0, labels, (N,))
    D = Discriminator(in_channels, 8, labels.max().item() + 1, H)
    initialize_weights(D)
    assert D(x, labels).shape == (N, 1, 1, 1), "Discriminator test failed"
    G = Generator(z_dim, in_channels, 8)
    initialize_weights(G)
    z = torch.randn((N, z_dim, 1, 1))
    assert G(z, labels).shape == (N, in_channels, H, W), "Generator test failed"
    print("All tests passed")

if __name__ == "__main__":
    test()