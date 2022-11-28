import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
# import torchvision.models as models
from models import ResNet_model


def loss_VAE(recon_x, x, mu, logvar, ratio):
    """
    :param recon_x: generated
    :param x: original image
    :param mu: latent mean of z
    :param logvar: latent log variance of z
    """
    # MSE loss
    # reconstruction_function = nn.MSELoss(reduction='sum').cuda()
    reconstruction_function = torch.nn.L1Loss(reduction='sum').cuda()
    MSE = reconstruction_function(recon_x, x)
    # KL divergence
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return MSE + ratio * KLD

# def loss_func_VAE_RESNET(recon_x, x, mu, logvar):
#     BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     return BCE + KLD

#用上采样加卷积代替了反卷积
class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class ResNet18Enc(nn.Module):
    def __init__(self, z_dim=32):
        super(ResNet18Enc, self).__init__()
        self.z_dim = z_dim
        self.ResNet18 = ResNet_model.resnet18()
        self.num_feature = self.ResNet18.fc.out_features
    #     self.ResNet18.fc = nn.Linear(self.num_feature, 2 * self.z_dim)
    #
    # def forward(self, x):
    #     x = self.ResNet18(x)
    #     mu = x[:, :self.z_dim]
    #     logvar = x[:, self.z_dim:]
    #     return mu, logvar
        self.mu_l=nn.Linear(self.num_feature,self.z_dim)
        self.mu_bn=nn.BatchNorm1d(self.z_dim)
        self.log_sigma2_l=nn.Linear(self.num_feature,self.z_dim)

    def forward(self, x):
        x = self.ResNet18(x)
        # mu = x[:, :self.z_dim]
        # logvar = x[:, self.z_dim:]
        mu=self.mu_l(x)
        mu_bn=self.mu_bn(mu)
        log_sigma2=self.log_sigma2_l(x)
        return mu_bn, log_sigma2

class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes / stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet18Dec(nn.Module):

    def __init__(self, num_Blocks=[2, 2, 2, 2], z_dim=32, nc=1):
        super().__init__()
        self.in_planes = 512

        self.linear = nn.Linear(z_dim, 512)
        self.layer5 = nn.Sigmoid()
        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=2)
        # self.conv1 = ResizeConv2d(32, nc, kernel_size=3, scale_factor=2)
        self.conv1 = nn.Conv2d(32,nc, kernel_size=3, stride=1, padding=1)


    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 512, 1, 1)
        x = F.interpolate(x, scale_factor=8)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        # x = F.interpolate(x, size=(64, 64), mode='bilinear')
        x = self.conv1(x)
        # x=self.layer5(x)
        # x = x.view(x.size(0), 1, 128, 128)
        return x


class VAE_RESNET18(nn.Module):
    def __init__(self, z_dim):
        super(VAE_RESNET18, self).__init__()
        self.z_num=z_dim
        self.encoder = ResNet18Enc(z_dim=z_dim)
        self.decoder = ResNet18Dec(z_dim=z_dim)

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        x = self.decoder(z)
        return x, mean, logvar

    def criterion_p1(self, recon_x, x, mu, logvar, ratio):
        """
        :param recon_x: generated
        :param x: original image
        :param mu: latent mean of z
        :param logvar: latent log variance of z
        """
        # MSE loss
        reconstruction_function = nn.MSELoss(reduction='sum').cuda()
        # reconstruction_function = torch.nn.L1Loss(reduction='sum').cuda()
        MSE = reconstruction_function(recon_x, x)
        # KL divergence
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        return MSE + ratio * KLD

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2)  # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std).cuda()
        return epsilon * std + mean


class AE_deep(nn.Module):
    def __init__(self):
        super(AE_deep, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(16384, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 128)
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 16384),
            nn.Tanh(),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        y = self.decode(z)
        return y


class AE_CNN(nn.Module):
    def __init__(self):
        super(AE_CNN, self).__init__()
        self.encoder = nn.Sequential(
            # input: [n,1,200,200]
            # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True)
            # padding = (kernel-1) / 2
            # w,h = [ (w-kernel+2*padding) / stride ] +1
            # torch.nn.MaxPool1d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
            nn.Conv2d(1, 4, 3, stride=1, padding=1),  # (n, 4, 200, 200)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # (n, 4, 100, 100)

            nn.Conv2d(4, 8, 3, stride=1, padding=1),  # (n, 8, 100, 100)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # (n, 8, 50, 50)

            nn.Conv2d(8, 16, 3, stride=1, padding=1),  # (n, 16, 50, 50)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # (n, 16, 25, 25)
        )
        self.decoder = nn.Sequential(
            #     torch.nn.ConvTranspose2d(in_channels: int, out_channels: int, kernel_size: Union[T, Tuple[T, T]],
            #       stride: Union[T, Tuple[T, T]] = 1, padding: Union[T, Tuple[T, T]] = 0,
            #           output_padding: Union[T, Tuple[T, T]] = 0, groups: int = 1, bias: bool = True,
            #               dilation: int = 1, padding_mode: str = 'zeros')
            # (H−1) ∗ S − 2∗P + kernel_Size
            # input: (n, 16, 25, 25)
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),  # (n, 8, 50, 50)
            nn.ReLU(True),

            nn.ConvTranspose2d(8, 4, 4, stride=2, padding=1),  # (n, 4, 100, 100)
            nn.ReLU(True),

            nn.ConvTranspose2d(4, 1, 4, stride=2, padding=1),  # (n, 1, 200, 200)
            nn.Tanh(),
        )
        self.latent = None
        self.ouput = None

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded




class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(4096, 400)
        self.fc2_mean = nn.Linear(400, 20)
        self.fc2_logvar = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 4096)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2_mean(h1), self.fc2_logvar(h1)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        x = torch.tanh(self.fc4(h3))
        return x

    def reparametrization(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps).cuda()
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrization(mu, logvar)
        return self.decode(z), mu, logvar


class VAE_deep(nn.Module):
    def __init__(self):
        super(VAE_deep, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(16384, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
        )
        self.mean = nn.Linear(256, 128)
        self.logvar = nn.Linear(256, 128)
        self.decoder = nn.Sequential(
            nn.Linear(128, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 16384),
            nn.Tanh(),
        )

    def encode(self, x):
        d = self.encoder(x)
        mu = self.mean(d)
        logvar = self.logvar(d)
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def reparametrization(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps).cuda()
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrization(mu, logvar)
        return self.decode(z), mu, logvar


def show_decoded_imgs(source_images, decoded_images, start, end):
    source_images = source_images[start:end].cpu().numpy()
    decoded_images = decoded_images[start:end].detach().cpu().numpy()
    number = 2 * (end - start)
    number_row = 5
    numner_column = int(number / number_row)
    width = 4
    pic = plt.figure(figsize=(number_row * width, numner_column * width))
    datas = []
    for index, (source_image, decoded_image) in enumerate(zip(source_images, decoded_images)):
        if index < 5:
            ax = pic.add_subplot(numner_column, number_row, index + 1)
        else:
            ax = pic.add_subplot(numner_column, number_row, index + number_row + 1)
        ax.imshow(source_image.reshape((200, 200)), cmap=plt.cm.gray)
        ax.axis('off')
        ax.set_title("Image {}".format(index + 1))

        if index < 5:
            ay = pic.add_subplot(numner_column, number_row, index + number_row + 1)
        else:
            ay = pic.add_subplot(numner_column, number_row, index + number_row + number_row + 1)
        ay.imshow(decoded_image.reshape((200, 200)), cmap=plt.cm.gray)
        ay.axis('off')
        ay.set_title("Decoded {}".format(5 + index + 1))

    pic.tight_layout()
    pic.savefig('imgs_and_decoded_imgs.png')
