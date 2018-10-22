# Face-Generation-GANs

This project using CelebA dataset to train our GANs in order to produce fake faces. Because of the limitation of computational cost, I trained the network with only 1 epoch. Increase the number of epoches as long as you are able to access the GPU to train it.

## A brief intro to GANs
Generative adversarial networks (GANs) are a class of artificial intelligence algorithms used in unsupervised machine learning, implemented by a system of two neural networks contesting with each other in a zero-sum game framework. They were introduced by Ian Goodfellow et al. in 2014. This technique can generate photographs that look at least superficially authentic to human observers, having many realistic characteristics (though in tests people can tell real from generated in many cases) --- from Wikipedia

One of the reasons why we use GANs is that we don't always have enough data to train our networks. GANs can generate fake images, which are very similar to the real ones, to provide sufficient required amount of data to train our networks. GANs uses two competitors, called discriminator and generator, to generate fake data. The goal of generators is to produce fake images in order to "fool" the discriminator, while discriminators are trained using read images which is able to tell the difference between fake and real images. Once a GAN is well-trained, the generator is able to produce images as "real" as real images to fool the discriminator.

## Results


## The files
* *assistance.py: * The assistance.py file makes your life easier. It provides functions to download and preprocessing image data.
* *download_data.py: * The download_data.py file would automatically download the data for you to train the network
* *check_tensorflow_gpu.py: * The check_tensorflow_gpu.py would check your tensorflow version and your access to GPU. Training on GPUs is HIGHLY RECOMMENDED.
* *build_network.py: * The build_network.py constructs a class to build and train the neural network. Check the code for details if you would like to get insights on the structure of the network or if you would like to use your own structure.
* *apply_gans.py: * The apply_gans.py apply the GANs built in build_network.py using assigned parameters. Change the params to play with GANs.
* *face_generation.ipynb: * Jupyter notebook version of the project. Jupyter notebook is a better way to show and compare the results.
