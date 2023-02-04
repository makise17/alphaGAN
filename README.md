# alphaGAN

A PyTorch implementation of alpha-GAN (https://arxiv.org/abs/1706.04987) with a sample run on MNIST.

## Dependencies

- PyTorch v1.7+ with CUDA11.0 support
- torchvision v0.8.1+
- TensorFlow v1.13+ (for TensorBoard only)

## Usage

`train_mnist.py` contains sample code that runs the package on MNIST data. On the command line, run 
```
$ python train_mnist.py --output_path YOUR_SAVED_PATH
```

While or after running, you will able to monitor the training progress on TensorBoard. Run
```
$ tensorboard --logdir=YOUR_SAVED_PATH/logs/ --port=6006
```
and access https://localhost:6006 (or the corresponding server URL) on your browser.

## References

* Paper
  - Rosca, M., Lakshminarayanan, B., Warde-Farley, D., & Mohamed, S. (2017). Variational Approaches for Auto-Encoding Generative Adversarial Networks. _arXiv preprint arXiv:1706.04987_.
* Code
  - [PyTorch VAE implementation](https://github.com/pytorch/examples/blob/master/vae/main.py)
  - [Martin Arjovsky's DCGAN implementation](https://github.com/martinarjovsky/WassersteinGAN/blob/master/models/dcgan.py)
  - [Yunjey Choi's TensorBoard tutorial for PyTorch](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard/main.py)
  
## Generated MNIST Images
![samples-mnist-epoch-30](https://github.com/makise17/alphaGAN/blob/master/sample.png)

## Recon
### X
![real-mnist](https://github.com/makise17/alphaGAN/blob/master/real.png)
### G(E(X))
![rec-mnist-epoch-30](https://github.com/makise17/alphaGAN/blob/master/rec.png)
