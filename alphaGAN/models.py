import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
from torchvision.utils import save_image

import time
import os
import itertools

from .base import Discriminator, CodeDiscriminator, Encoder, Generator
from .logger import Logger

"""
A PyTorch Implementation of alpha-GAN

Rosca, M., Lakshminarayanan, B., Warde-Farley, D., & Mohamed, S. (2017). 
Variational Approaches for Auto-Encoding Generative Adversarial Networks. 
https://arxiv.org/abs/1706.04987

Code References:
https://github.com/pytorch/examples/blob/master/vae/main.py
https://github.com/martinarjovsky/WassersteinGAN/blob/master/models/dcgan.py
https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/
04-utils/tensorboard/main.py
"""


class AlphaGAN(object):
    """
    The alpha-GAN wrapper class containing four neural networks (nn.Module):
        - Discriminator(input_size, num_channels, num_features, cuda)
        - CodeDiscriminator(code_size, num_units, num_layers, cuda)
        - Encoder(input_size, num_channels, code_size, cuda)
        - Generator(code_size, input_size, num_channels, cuda)

    `lambda_` is a learning parameter that trade-offs
    the L1 reconstruction loss against other loss terms.

    Methods:
        __init__, encode, generate, train
    """

    def __init__(self, input_size=784, code_size=50, lambda_=25.,
                 num_channels=1, num_features=8, num_units=750, num_layers=3,
                 seed=1, gpu=True):

        # Parameters
        self.input_size = input_size
        self.code_size = code_size
        self.lambda_ = lambda_
        self.num_channels = num_channels
        self.num_features = num_features
        self.num_units = num_units
        self.num_layers = num_layers
        self.seed = seed
        self.gpu = gpu

        # Neural Networks
        self.discriminator = \
            Discriminator(input_size, num_channels, num_features, gpu)
        self.codeDiscriminator = \
            CodeDiscriminator(code_size, num_units, num_layers, gpu)
        self.encoder = \
            Encoder(input_size, num_channels, code_size, gpu)
        self.generator = \
            Generator(code_size, input_size, num_channels, gpu)
        self.decoder = self.generator  # alias

        # Load nets to GPU if self.gpu
        if self.gpu:
            self.discriminator.cuda()
            self.codeDiscriminator.cuda()
            self.encoder.cuda()
            self.generator.cuda()
            print("[alphaGAN] Loaded networks to CUDA memory")
            self.discriminator = \
                torch.nn.DataParallel(self.discriminator)
            self.codeDiscriminator = \
                torch.nn.DataParallel(self.codeDiscriminator)
            self.encoder = torch.nn.DataParallel(self.encoder)
            self.generator = torch.nn.DataParallel(self.generator)
            print("[alphaGAN] Enabled data parallelism")

    def encode(self, x):
        """
        Encode input (x) into code using the encoder (variational distribution).

        Input:
            x: torch.FloatTensor of shape
               (num_inputs, num_channels, sqrt(input_size), sqrt(input_size))
                Samples to be encoded.

        Output:
            z: torch.FloatTensor of shape (num_inputs, code_size)
                Code encoding the inputs.
        """
        if self.gpu:
            x = x.cuda()
        return self.encoder(x)

    def generate(self, n=100):
        """
        Generate n random samples from the generator (decoder).

        Input:
            n: int
                Number of samples to be generated.

        Output:
            x: torch.FloatTensor of shape
               (num_inputs, num_channels, sqrt(input_size), sqrt(input_size))
                Generated samples.
        """
        z_rand = Variable(torch.randn(n, self.code_size))
        # onehot vec, nはサンプル数
        onehot = nn.functional.one_hot(torch.randint(low=0, high=10, size=(n,)), 10).to(torch.float32)
        if self.gpu:
            z_rand = z_rand.cuda()
        return self.generator(z_rand,onehot)

    def train(self, train_loader, test_loader,
              n_epochs=1000, lr1=0.001, lr2=0.0001, beta1=0.5, beta2=0.9,
              log_interval=10, output_path='.'):
        """
        Train alpha-GAN using input `train_loader` and `test_loader` data.

        Inputs:
            train_loader: torch.utils.data.DataLoader
            test_loader: torch.utils.data.DataLoader
            n_epochs: int
            lr1: float
            lr2: float
            beta1: float
            beta2: float
            log_interval: int
            output_path: str
        Output:
            self
        """

        torch.manual_seed(self.seed)
        if self.gpu:
            torch.cuda.manual_seed(self.seed)

        if self.gpu:
            print("[alphaGAN] using {} CUDA devices".format(
                  torch.cuda.device_count()))

        if not os.path.exists(os.path.join(output_path, 'logs/')):
            os.makedirs(os.path.join(output_path, 'logs/'))

        # TODO: update BCELoss() to nn.BCEWithLogitsLoss() for PyTorch v0.2+
        criterion_bce = nn.BCEWithLogitsLoss()
        criterion_l1 = nn.L1Loss()
        criterion_ce = nn.CrossEntropyLoss()
 
        optimizer_enc = optim.Adam(
            self.encoder.parameters(), lr=lr1, betas=(beta1, beta2))
        optimizer_gen = optim.Adam(
            self.generator.parameters(), lr=lr1, betas=(beta1, beta2))
        optimizer_disc = optim.Adam(
            self.discriminator.parameters(), lr=lr2, betas=(beta1, beta2))
        optimizer_code_disc = optim.Adam(
            self.codeDiscriminator.parameters(), lr=lr2, betas=(beta1, beta2))

        print("[alphaGAN] Initialized neural networks and optimizers")

        logger = Logger(os.path.join(output_path, 'logs/'))

        
        self.discriminator.train()
        self.codeDiscriminator.train()
        self.encoder.train()
        self.generator.train()
        for epoch in range(1, n_epochs + 1):

            t0 = time.time()

            # Training
            train_loss1, train_loss2, train_loss3 = 0.0, 0.0, 0.0
            n = len(train_loader.dataset)
            for batch_idx, (x, _) in enumerate(train_loader):
                # Prepare x
                n_batch = len(x)
                onehot = nn.functional.one_hot(_, 10).to(torch.float32)
                x = Variable(x, requires_grad=False)
                if self.gpu:
                    onehot=onehot.cuda()
                    x = x.cuda()

                # Generate codes, reconstructions, and random codes
                # z_hat = self.encoder(x)
                # x_hat = self.generator(z_hat, onehot)

                z_hat, l_hat = self.encoder(x)
                x_hat = self.generator(z_hat, l_hat)
                
                z_rand = Variable(torch.randn(z_hat.size()),
                                  requires_grad=False)
                if self.gpu:
                    z_rand = z_rand.cuda()
                x_rand = self.generator(z_rand, onehot)


                # Pre-load one and zero variables for efficient reuse
                one = Variable(torch.ones(n_batch,1))
                zero = Variable(torch.zeros(n_batch,1))
                if self.gpu:
                    one = one.cuda()
                    zero = zero.cuda()

                # 1: Update encoder and decoder (generator) parameters
                optimizer_enc.zero_grad()
                optimizer_gen.zero_grad()
                l1_loss = self.lambda_ * criterion_l1(x_hat, x)
                # 2023/02/03 書き換え
                # c_loss = criterion_bce(
                #     F.sigmoid(self.codeDiscriminator(z_hat)), one)
                # d_real_loss = criterion_bce(
                #     F.sigmoid(self.discriminator(x_hat,onehot)), one)
                # d_fake_loss = criterion_bce(
                #     F.sigmoid(self.discriminator(x_rand,onehot)), one)
                c_loss = criterion_bce(
                    self.codeDiscriminator(z_hat), zero)
                # d_real_loss = criterion_bce(
                #     self.discriminator(x_hat,onehot), one)
                d_real_loss = criterion_bce(
                    self.discriminator(x_hat,l_hat), one)
                d_fake_loss = criterion_bce(
                    self.discriminator(x_rand,onehot), one)
                #enc_loss = criterion_ce(l_hat, onehot)

                loss1 = l1_loss + c_loss + d_real_loss + d_fake_loss# +enc_loss
                loss1.backward(retain_graph=True)
                train_loss1 += loss1.data
                optimizer_enc.step()
                optimizer_gen.step()
                optimizer_gen.step()  # update generator twice, from DCGAN
                

                # 2: Update discriminator parameters
                optimizer_disc.zero_grad()
                z_hat, l_hat = self.encoder(x)
                # l_hat = F.softmax(l_hat, dim=1)
                x_hat = self.generator(z_hat, onehot)

                x_rand = self.generator(z_rand,onehot)
                x_loss2 = 2.0 * \
                    criterion_bce(self.discriminator(x,onehot), one) + \
                    criterion_bce(self.discriminator(x_hat,l_hat), zero)
                    # criterion_bce(self.discriminator(x_hat,onehot), zero)
                z_loss2 = criterion_bce(
                    self.discriminator(x_rand,onehot), zero)       
                # x_loss2 = 2.0 * \
                #     criterion_bce(F.sigmoid(self.discriminator(x,onehot)), one) + \
                #     criterion_bce(F.sigmoid(self.discriminator(x_hat,onehot)), zero)
                # z_loss2 = criterion_bce(
                #     F.sigmoid(self.discriminator(x_rand,onehot)), zero)
                loss2 = x_loss2 + z_loss2
                loss2.backward(retain_graph=True)
                train_loss2 += loss2.data
                optimizer_disc.step()

                # 3: Update code discriminator parameters
                optimizer_code_disc.zero_grad()
                x_loss3 = criterion_bce(
                    self.codeDiscriminator(z_hat), zero)
                z_loss3 = criterion_bce(
                    self.codeDiscriminator(z_rand), one)
                # x_loss3 = criterion_bce(
                #     torch.sigmoid(self.codeDiscriminator(z_hat)), one)
                # z_loss3 = criterion_bce(
                #     F.sigmoid(self.codeDiscriminator(z_rand)), zero)
                loss3 = x_loss3 + z_loss3
                loss3.backward(retain_graph=True)
                train_loss3 += loss3.data
                optimizer_code_disc.step()

                if batch_idx % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'.format(
                            epoch, batch_idx * n_batch, n,
                            100. * batch_idx / len(train_loader)) +
                          'Loss: {:.5f}, {:.5f}, {:.5f}'.format(
                            loss1.data / n_batch,
                            loss2.data / n_batch,
                            loss3.data / n_batch)
                          )

            t1 = time.time()
            print('====> Epoch: {} '.format(epoch) +
                  'Training loss: {:.5f}, {:.5f}, {:.5f} '.format(
                    train_loss1 / n, train_loss2 / n, train_loss3 / n) +
                  '(Training time: {:.5f}s)'.format(t1 - t0)
                  )

            # Validation
            test_loss1, test_loss2, test_loss3 = 0.0, 0.0, 0.0
            n_test = len(test_loader.dataset)
            
            self.discriminator.eval()
            self.codeDiscriminator.eval()
            self.encoder.eval()
            self.generator.eval()
            with torch.no_grad():
                for x, _ in test_loader:
                    n_batch = len(x)
                    if self.gpu:
                        x = x.cuda()
                    x = Variable(x)
                    onehot = nn.functional.one_hot(_, 10).to(torch.float32)
                    # Generate codes, reconstructions, and random codes
                    # 2023/02/03
                    # z_hat = self.encoder(x)
                    # x_hat = self.generator(z_hat, onehot)

                    z_hat, l_hat = self.encoder(x)
                    # l_hat = F.softmax(l_hat, dim=1)
                    x_hat = self.generator(z_hat,l_hat)
                 
                    z_rand = Variable(torch.randn(z_hat.size()))
                    if self.gpu:
                        onehot=onehot.cuda()
                        z_rand = z_rand.cuda()
                    x_rand = self.generator(z_rand, onehot)


                    # Pre-load one and zero variables for efficient reuse
                    one = Variable(torch.ones(n_batch, 1))
                    zero = Variable(torch.zeros(n_batch, 1))
                    if self.gpu:
                        one = one.cuda()
                        zero = zero.cuda()

                    # 1: Update encoder and decoder (generator) parameters
                    l1_loss = self.lambda_ * criterion_l1(x_hat, x)
                    # 2023/02/03 書き換え
                    c_loss = criterion_bce(
                        self.codeDiscriminator(z_hat), one)
                    # d_real_loss = criterion_bce(
                    #     self.discriminator(x_hat, onehot), one)
                    d_real_loss = criterion_bce(
                        self.discriminator(x_hat, l_hat), one)
                    d_fake_loss = criterion_bce(
                        self.discriminator(x_rand, onehot), one)
                    #enc_loss = criterion_ce(l_hat, onehot)
                    
                    # c_loss = criterion_bce(
                    #     F.sigmoid(self.codeDiscriminator(z_hat)), one)
                    # d_real_loss = criterion_bce(
                    #     F.sigmoid(self.discriminator(x_hat,onehot)), one)
                    # d_fake_loss = criterion_bce(
                    #     F.sigmoid(self.discriminator(x_rand,onehot)), one)
                    loss1 = l1_loss + c_loss + d_real_loss + d_fake_loss# + enc_loss
                    test_loss1 += loss1.data

                    # 2: Update discriminator parameters
                    x_loss2 = 2.0 * \
                        criterion_bce(self.discriminator(x,onehot), one) + \
                        criterion_bce(self.discriminator(x_hat,l_hat), zero)
                        # criterion_bce(self.discriminator(x_hat,onehot), zero)

                    z_loss2 = criterion_bce(
                        self.discriminator(x_rand,onehot), zero)
                    # x_loss2 = 2.0 * \
                    #     criterion_bce(F.sigmoid(self.discriminator(x,onehot)), one) + \
                    #     criterion_bce(F.sigmoid(self.discriminator(x_hat,onehot)), zero)
                    # z_loss2 = criterion_bce(
                    #     F.sigmoid(self.discriminator(x_rand,onehot)), zero)
                    loss2 = x_loss2 + z_loss2
                    test_loss2 += loss2.data

                    # 3: Update code discriminator parameters
                    x_loss3 = c_loss
                    z_loss3 = criterion_bce(
                        self.codeDiscriminator(z_rand), zero)
                    # z_loss3 = criterion_bce(
                    #     F.sigmoid(self.codeDiscriminator(z_rand)), zero)
                    loss3 = x_loss3 + z_loss3
                    test_loss3 += loss3.data

            print('====> Epoch: {} '.format(epoch) +
                  'Validation loss: {:.5f}, {:.5f}, {:.5f} '.format(
                    test_loss1 / n_test,
                    test_loss2 / n_test,
                    test_loss3 / n_test)
                  )

            # checkpoints
            if epoch %5 == 0:
                torch.save(self.discriminator.state_dict(),
                        '{}/d-epoch-{}.pth'.format(output_path, epoch))
                torch.save(self.codeDiscriminator.state_dict(),
                        '{}/c-epoch-{}.pth'.format(output_path, epoch))
                torch.save(self.encoder.state_dict(),
                        '{}/e-epoch-{}.pth'.format(output_path, epoch))
                torch.save(self.generator.state_dict(),
                        '{}/g-epoch-{}.pth'.format(output_path, epoch))
            # samples (binarized)
            # x_gen = torch.round(self.generate(train_loader.batch_size))
            x_gen = self.generate(train_loader.batch_size)
            save_image(x_gen.data,
                       '{}/samples-epoch-{}.png'.format(output_path, epoch))

            # ============ TensorBoard logging ============#
            # https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/
            # 04-utils/tensorboard/main.py

            # (1) Log the scalar values
            info = {
                'train_loss1': train_loss1 / n,
                'train_loss2': train_loss2 / n,
                'train_loss3': train_loss3 / n,
                'train_loss': (train_loss1 + train_loss2 + train_loss3) / n,
                'test_loss1': test_loss1 / n_test,
                'test_loss2': test_loss2 / n_test,
                'test_loss3': test_loss3 / n_test,
                'test_loss': (test_loss1 + test_loss2 + test_loss3) / n_test
            }

            for tag, value in info.items():
                logger.scalar_summary(tag, value, epoch)

            # (2) Log values and gradients of the parameters (histogram)
            for tag, value in itertools.chain(
                    self.discriminator.named_parameters(),
                    self.codeDiscriminator.named_parameters(),
                    self.encoder.named_parameters(),
                    self.generator.named_parameters()):
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, to_np(value), epoch)
                logger.histo_summary(tag + '/grad', to_np(value.grad), epoch)

            # (3) Log the images
            info = {
                'images': to_np(x_gen.view(-1, 28, 28)[:10])
            }

            for tag, images in info.items():
                logger.image_summary(tag, images, epoch)

        return self

"""
Helper functions
"""


# from TensorBoard logging code
def to_np(x):
    return x.data.cpu().numpy()


# from TensorBoard logging code
def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

