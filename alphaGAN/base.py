from doctest import OutputChecker
import torch
import torch.nn as nn

"""
Base Neural Network Classes for alpha-GAN
(Discriminator, Code Discriminator, Encoder, Decoder)

Code References:
https://github.com/pytorch/examples/blob/master/vae/main.py
https://github.com/martinarjovsky/WassersteinGAN/blob/master/models/dcgan.py
"""


class Discriminator(nn.Module):
    """
    Image discriminator convolutional neural network. (D_phi)

    Takes x and returns the probability that x came from the model distribution
    p_theta(x|z) rather than the true distribution p*(x).
    """

    def __init__(self, input_size=784, num_channels=1,
                 num_features=8, gpu=True, labels = 10):
        super(Discriminator, self).__init__()
        self.gpu = gpu
        assert input_size % 16 == 0, "input_size has to be a multiple of 16"

        conv = nn.Sequential()
        kernel_size, stride = 5, 2

        #2023/02/03
        num_channels += 1
        conv.add_module(
            'initial_conv_{0}-{1}'.format(num_channels, num_features),
            nn.Conv2d(num_channels, num_features,
                      kernel_size, stride, 1, bias=False)
        )
        conv.add_module(
            'initial_relu_{0}'.format(num_features),
            nn.LeakyReLU(0.2, inplace=True)
        )
        current_size, current_features = input_size / 2, num_features

        while current_size > 50:
            in_features = current_features
            out_features = current_features * 2
            stride = 1 if stride == 2 else 2
            conv.add_module(
                'pyramid_{0}-{1}_conv'.format(in_features, out_features),
                nn.Conv2d(in_features, out_features,
                          kernel_size, stride, 1, bias=False)
            )
            conv.add_module(
                'pyramid_{0}_batchnorm'.format(out_features),
                nn.InstanceNorm2d(out_features)
            )
            conv.add_module(
                'pyramid_{0}_relu'.format(out_features),
                nn.LeakyReLU(0.2, inplace=True)
            )
            current_features = current_features * 2
            current_size = current_size / 2

        stride = 2
        conv.add_module(
            'final_{0}-{1}_conv'.format(current_features, current_features),
            nn.Conv2d(current_features, current_features,
                      kernel_size, stride, 1, bias=False)
        )
        conv.add_module('final_relu', nn.LeakyReLU(0.2, inplace=True))
        conv.add_module('final_dropout', nn.Dropout2d(0.8))

        #https://learnopencv.com/conditional-gan-cgan-in-pytorch-and-tensorflow/
        n_classes = 10
        embedding_dim = 50
        self.label_condition_disc = nn.Sequential(nn.Embedding(n_classes, embedding_dim),
                      nn.Linear(embedding_dim, 1*28*28))

        self.conv = conv
        #current_features + labels
        self.fc = nn.Linear(current_features, 1)
        # self.fc = nn.Sequential(
        #     nn.Linear(current_features + labels, 1024),
        #     nn.Linear(1024, 1)
        # )
    def forward(self, x, labels):
        # a = labels.view(labels.size(0),-1)
        #a.shape = (64,10)
        # a = a.reshape(-1,10,1,1)

        label_output = self.label_condition_disc(labels)
        label_output = label_output.view(-1, 1, 28, 28)
        concat = torch.cat((x, label_output), dim=1)
        out = self.conv(concat)

        out = out.view(-1, num_flat_features(out))

        return self.fc(out)
        # out = self.conv(x)
        # out = out.view(-1, num_flat_features(out))
        # return self.fc(out)




class CodeDiscriminator(nn.Module):
    """
    Code discriminator multilayer perceptron (C_omega).

    Takes z and returns the probability that z came from the assumed prior p(z)
    rather than the variational distribution q_eta(z|x).
    """

    def __init__(self, code_size=50, num_units=750, num_layers=3, gpu=True):
        super(CodeDiscriminator, self).__init__()
        self.gpu = gpu
        # 2023/02/04
        # n_classes = 10
        # embedding_dim = code_size
        # self.label_ = nn.Sequential(
        #     nn.Embedding(n_classes, embedding_dim),
        #     nn.Linear(embedding_dim, n_classes))

        fc = nn.Sequential()
        in_features, out_features = code_size, num_units
        for l in range(num_layers - 1):
            fc.add_module('mlp_fc_{0}'.format(l),
                          nn.Linear(in_features, out_features))
            fc.add_module('mlp_relu_{0}'.format(l), nn.ReLU(inplace=True))
            in_features, out_features = out_features, num_units
        fc.add_module('mlp_fc_{0}'.format(num_layers - 1),
                      nn.Linear(out_features, 1))

        self.fc = fc


    def forward(self, x, label):
        # label = self.label_(label)
        # concat=torch.cat([x, label], dim=1)
        return self.fc(x)


class Encoder(nn.Module):
    """
    Image to code encoder convolutional neural network,
    i.e. the variational distribution (q_eta).

    Note that this variational "distribution" is deterministic,
    but its output is understood as the mean of the distribution.
    """

    def __init__(self, input_size=784 ,num_channels=1, code_size=50, gpu=True):
        super(Encoder, self).__init__()
        self.gpu = gpu
        assert input_size % 16 == 0, "input_size has to be a multiple of 16"

        # Convolutional modules
        conv = nn.Sequential()
        conv.add_module('pyramid_{0}-{1}_conv'.format(num_channels, 64),
                        nn.Conv2d(num_channels, 64, 4, 2, 1 , bias=False))
        conv.add_module('pyramid_{0}_batchnorm'.format(64),
                        nn.InstanceNorm2d(64))
        conv.add_module('pyramid_{0}_relu'.format(64),
                        nn.LeakyReLU(0.2, inplace=True))
        conv.add_module('pyramid_{0}-{1}_conv'.format(64, 64),
                        nn.Conv2d(64, 64, 4, 2, 1,  bias=False))
        conv.add_module('pyramid_{0}_batchnorm'.format(64),
                        nn.InstanceNorm2d(64))
        conv.add_module('pyramid_{0}_relu'.format(64),
                        nn.LeakyReLU(0.2, inplace=True))
        conv.add_module('pyramid_{0}-{1}_conv'.format(64, 64*2),
                        nn.Conv2d(64, 64*2, 4, 2, 1 , bias=False))
        conv.add_module('pyramid_{0}_batchnorm'.format(64*2),
                        nn.InstanceNorm2d(64*2))
        conv.add_module('pyramid_{0}_relu'.format(64*2),
                        nn.LeakyReLU(0.2, inplace=True))
        conv.add_module('pyramid_{0}-{1}_conv'.format(64*2, 64*2),
                        nn.Conv2d(64*2, 64*2, 3, 1, 0 , bias=False))
        conv.add_module('pyramid_{0}_batchnorm'.format(64*2),
                        nn.InstanceNorm2d(64*2))
        conv.add_module('pyramid_{0}_relu'.format(64*2),
                        nn.LeakyReLU(0.2, inplace=True))

        self.conv = conv

        # Final linear module
        self.code = nn.Sequential(
            nn.Linear(64*2, code_size),
            nn.Tanh()
        )
        self.label = nn.Sequential(
            nn.Linear(64*2, 10),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(-1, num_flat_features(out))
        # 2023/02/03
        # out = self.feat(out)
        label = self.label(out)
        # label = torch.argmax(label, dim=1)
        # one_hot = nn.functional.one_hot(torch.argmax(label, dim=1), 10)
        # print(one_hot)
        return self.code(out), label


class Generator(nn.Module):
    """
    Code to image generator/decoder (G_theta).

    The convolutional layers in this generator is
    the transpose of those in the encoder.
    """

    def __init__(self, code_size=50, input_size=784, num_channels=1, gpu=True):
        super(Generator, self).__init__()
        self.gpu = gpu
        assert input_size % 16 == 0, "input_size has to be a multiple of 16"

        # Initial linear modules
        # fc2 = nn.Sequential()
        # fc2.add_module('initial_{0}-{1}_linear'.format(10, 100),
        #               nn.Linear(10, 100))
        # self.fc2 = fc2

        # 2023/02/04
        n_classes = 10
        embedding_dim = code_size
        self.label_conditioned_generator = nn.Sequential(
            nn.Embedding(n_classes, embedding_dim),
            nn.Linear(embedding_dim, 16))

        # fc = nn.Sequential()
        # # fc.add_module('initial_{0}-{1}_linear'.format(code_size, 1024),
        # #               nn.Linear(code_size, 1024))
        # fc.add_module('initial_{0}-{1}_linear'.format(code_size, 1024),
        #               nn.Linear(code_size, 1024))
        # fc.add_module('initial_{0}_batchnorm'.format(1024),
        #                 nn.InstanceNorm1d(1024))
        # fc.add_module('initial_{0}_relu'.format(1024),
        #               nn.LeakyReLU(0.2, inplace=True))
        # fc.add_module('initial_{0}-{1}_linear'.format(1024, 64 * 10 * 10),
        #               nn.Linear(1024, 64 * 10 * 10))
        # fc.add_module('initial_{0}_batchnorm'.format(64 * 10 * 10),
        #                 nn.InstanceNorm1d(64 * 10 * 10))
        # fc.add_module('initial_{0}_relu'.format(64 * 10 * 10),
        #               nn.LeakyReLU(0.2, inplace=True))
        # self.fc = fc
        self.fc = nn.Sequential(
            nn.Linear(code_size, 4*4*128),
            nn.LeakyReLU(0.2, inplace=True)
            )

        # Convolutional modules
        # conv = nn.Sequential()
        # conv.add_module('pyramid_{0}-{1}_convt'.format(65, 32),
        #                 nn.ConvTranspose2d(65, 32, 5, 2, 0, bias=False))
        # conv.add_module('pyramid_{0}_batchnorm'.format(32),
        #                 nn.InstanceNorm2d(32))
        # conv.add_module('pyramid_{0}_relu'.format(32),
        #                 nn.LeakyReLU(0.2, inplace=True))
        # # The following layer ensures a (num_channels, 28, 28) output size
        # conv.add_module(
        #     'pyramid_{0}-{1}_convt'.format(32, num_channels),
        #     nn.ConvTranspose2d(32, num_channels, 5 + 1, 1, 0, bias=False)
        # )
        # # 2023/02/03
        # conv.add_module('pyramid_{0}_tanh'.format(num_channels),
        #     nn.Tanh()
        # )
        # self.conv = conv
        self.conv =  nn.Sequential(
            nn.ConvTranspose2d(129, 64*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*2, momentum=0.1,  eps=0.8),
            nn.ReLU(True),
            nn.ConvTranspose2d(64*2, 64*1, 4, 2, 1,bias=False),
            nn.BatchNorm2d(64*1, momentum=0.1,  eps=0.8),
            nn.ReLU(True), 
            nn.ConvTranspose2d(64*1, 1, 2, 2, 2,bias=False),
            nn.Tanh())
    def forward(self, x, labels):
        # a = self.fc(x)
        # a = a.view(-1, 64, 10, 10)
        # #a.shape=(64,64,10,10)
        # b = b.view(-1, 1, 10, 10)
        # #b.shape=(64,1,10,10)
        # out = torch.cat([a, b] , 1)
        #out.shape=(64,65,10,10)

        # print('x.shape',x.shape)
        # print('labels.shape',labels.shape)

        a = self.fc(x)       
        a = a.view(-1, 128, 4, 4)
        # print('a.shape',a.shape)

        lcg = self.label_conditioned_generator(labels)
        #shape = (batch, 1, 10, 10)
        # print('lcg.shape',lcg.shape)
        lcg = lcg.view(-1, 1, 4, 4)
        # print('lcg.shape',lcg.shape)

        out = torch.cat([a, lcg] , 1)

        #b = self.fc2(labels)
        #b = b.view(-1, 1, 10, 10)
        #out = torch.cat([out, b] , 1)
        #
        #  a = a.view(-1, 64, 10, 10)
        #a.shape=(64,64,10,10)
        # b = b.view(-1, 1, 10, 10)
        #b.shape=(64,1,10,10)


        # out = out.view(-1, 65, 10, 10)
        return self.conv(out)

        # out = self.fc(x)
        # out = out.view(-1, 64, 10, 10)
        # return self.conv(out)


"""
Helper Functions
"""


# from PyTorch example code
def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

