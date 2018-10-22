
import os
import check_data
import tensorflow as tf

from glob import glob
from build_network import build_network

# Path to data
data_dir = './data'

# Choose your parameters here
params = {'epoch_count': 1,
          'batch_size': 32,
          'z_dim': 100,
          'learning_rate': 0.0002,
          'beta1': 0.5}

bn = build_network(params)
celeba_dataset = check_data.Dataset('celeba', glob(os.path.join(data_dir, 'img_align_celeba/*.jpg')))

with tf.Graph().as_default():
    bn.train(celeba_dataset.get_batches, celeba_dataset.shape, celeba_dataset.image_mode)