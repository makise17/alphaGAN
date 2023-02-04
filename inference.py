from alphaGAN.base import *
from alphaGAN.models import *
import torch.utils.data
from torchvision import datasets, transforms
from torchvision.utils import save_image

seed = 3407
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
#入力する乱数の次元の大きさ
n=50
batch_size = 40
input_size = 784
num_channels = 1
kwargs = {'num_workers': 0, 'pin_memory': True}

epoch = 10

#学習済みモデルの読み込み
netG = Generator(n,input_size, num_channels,)
netG = torch.nn.DataParallel(netG)
trained_model_path = './alphagan-results/g-epoch-{0}.pth'.format(epoch)
# trained_model_path = './results_2/g-epoch-20.pth'
netG.load_state_dict(torch.load(trained_model_path))
netG = netG.cuda()

netE = Encoder(input_size=input_size, num_channels=num_channels, code_size=n)
netE = torch.nn.DataParallel(netE)
trained_model_pathE = './alphagan-results/e-epoch-{0}.pth'.format(epoch)
# trained_model_pathE = './results_2/e-epoch-20.pth'

netE.load_state_dict(torch.load(trained_model_pathE))
netE = netE.cuda()


#推論モードに切り替え
netG.eval()
netE.eval()

#ノイズ生成
z_rand = Variable(torch.randn(batch_size, n), requires_grad=False)
# z_rand = z_rand.cuda()
# ランダムに出力するonehot
# onehot = nn.functional.one_hot(torch.randint(low=0, high=10, size=(n,)), 10).to(torch.float32)
# rand_label = torch.randint(low=0, high=10, size=(batch_size,))

rand_label = torch.arange(batch_size) % 10

# 0から順番に出力するonehot
# onehot2 = nn.functional.one_hot(torch.arange(0, batch_size) % 10, num_classes=10)#.to(torch.float32)
# print(onehot2)
# onehot2 = Variable(onehot2, requires_grad=False)
# onehot2 = onehot2.cuda()
with torch.no_grad():
    #generatorへ入力、出力画像を得る
    generated_image = netG(z_rand, rand_label)
    # generated_image = torch.sigmoid(generated_image)
    # generated_image = (generated_image + 1.0)/2.0
    print(torch.min(generated_image), torch.max(generated_image))
# x_gen = torch.round(generated_image)
# print(torch.min(x_gen), torch.max(x_gen))
save_image(generated_image.data,
            '{}/sample.png'.format('./inference'), nrow=10)


# 実データ読み込み
# 学習に使っていないテストデータ
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)  

images, labels = next(iter(test_loader))

# 実データを保存
save_image(images.data,
            '{}/real.png'.format('./inference'), nrow=10)
images = images.cuda()
# labels = nn.functional.one_hot(labels,num_classes=10).to(torch.float32)
labels = labels.cuda()
with torch.no_grad():
    # z_hat = netE(images)
    # x_hat = netG(z_hat, labels)
    z_hat, l_hat = netE(images)
    l_hat = torch.argmax(l_hat, dim=1)
    # x_hat = netG(z_hat, l_hat)
    # x_hat = netG(z_hat, labels)
    x_hat = netG(z_hat, l_hat)


# x_hat = torch.round(x_hat)
save_image(x_hat.data,
            '{}/rec.png'.format('./inference'), nrow=10)