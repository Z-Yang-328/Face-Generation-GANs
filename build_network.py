import numpy as np
import tensorflow as tf
import check_data

from matplotlib import pyplot


class build_network:

    def __init__(self, params):
        '''
        :param params: parameters used to train the network
        '''

        self.epoch_count = params['epoch_count']
        self.batch_size = params['batch_size']
        self.z_dim = params['z_dim']
        self.learning_rate = params['learning_rate']
        self.beta1 = params['beta1']


    def model_inputs(self, image_width, image_height, image_channels, z_dim):
        """
        Create the model inputs
        :param image_width: The input image width
        :param image_height: The input image height
        :param image_channels: The number of image channels
        :param z_dim: The dimension of Z
        :return: Tuple of (tensor of real input images, tensor of z data, learning rate)
        """

        real_input = tf.placeholder(tf.float32, [None, image_width, image_height, image_channels], name='real_input')
        z_input = tf.placeholder(tf.float32, [None, z_dim], name='z_input')
        learning_rate = tf.placeholder(tf.float32)
        return real_input, z_input, learning_rate


    def discriminator(self, images, reuse=False):
        """
        Create the discriminator network
        :param images: Tensor of input image(s)
        :param reuse: Boolean if the weights should be reused
        :return: Tuple of (tensor output of the discriminator, tensor logits of the discriminator)
        """

        with tf.variable_scope('discriminator', reuse=reuse):
            x1 = tf.layers.conv2d(images, 64, 5, strides=2, padding='same')
            relu1 = tf.maximum(x1, x1)
            # 16x16x64

            x2 = tf.layers.conv2d(relu1, 128, 5, strides=2, padding='same')
            bn2 = tf.layers.batch_normalization(x2, training=True)
            relu2 = tf.maximum(bn2, bn2)
            # 8x8x128

            x3 = tf.layers.conv2d(relu2, 256, 5, strides=2, padding='same')
            bn3 = tf.layers.batch_normalization(x3, training=True)
            relu3 = tf.maximum(bn3, bn3)
            # 4x4x256

            # Flatten it
            flat = tf.reshape(relu3, (-1, 4 * 4 * 256))
            logits = tf.layers.dense(flat, 1)
            out = tf.sigmoid(logits)
        return out, logits


    def generator(self, z, out_channel_dim, is_train=True):
        """
        Create the generator network
        :param z: Input z
        :param out_channel_dim: The number of channels in the output image
        :param is_train: Boolean if generator is being used for training
        :return: The tensor output of the generator
        """

        with tf.variable_scope('generator', reuse=not (is_train)):
            x1 = tf.layers.dense(z, 7 * 7 * 112)
            # Reshape it to start the convolutional stack
            x1 = tf.reshape(x1, (-1, 7, 7, 112))
            x1 = tf.layers.batch_normalization(x1, training=is_train)

            x2 = tf.layers.conv2d_transpose(x1, 56, 5, strides=2, padding='same')
            x2 = tf.layers.batch_normalization(x2, training=is_train)

            # Output layer
            logits = tf.layers.conv2d_transpose(x2, out_channel_dim, 5, strides=2, padding='same')

            out = tf.tanh(logits)
        return out


    def model_loss(self, input_real, input_z, out_channel_dim):
        """
        Get the loss for the discriminator and generator
        :param input_real: Images from the real dataset
        :param input_z: Z input
        :param out_channel_dim: The number of channels in the output image
        :return: A tuple of (discriminator loss, generator loss)
        """

        g_model = self.generator(input_z, out_channel_dim)
        d_model_real, d_logits_real = self.discriminator(input_real)
        d_model_fake, d_logits_fake = self.discriminator(g_model, reuse=True)

        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_model_real)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_model_fake)))
        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_model_fake)))

        d_loss = d_loss_real + d_loss_fake
        return d_loss, g_loss


    def model_opt(self, d_loss, g_loss, learning_rate, beta1):
        """
        Get optimization operations
        :param d_loss: Discriminator loss Tensor
        :param g_loss: Generator loss Tensor
        :param learning_rate: Learning Rate Placeholder
        :param beta1: The exponential decay rate for the 1st moment in the optimizer
        :return: A tuple of (discriminator training operation, generator training operation)
        """

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
        g_vars = [var for var in t_vars if var.name.startswith('generator')]

        # Optimize
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
            g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

        return d_train_opt, g_train_opt


    def show_generator_output(self, sess, n_images, input_z, out_channel_dim, image_mode):
        """
        Show example output for the generator
        :param sess: TensorFlow session
        :param n_images: Number of Images to display
        :param input_z: Input Z Tensor
        :param out_channel_dim: The number of channels in the output image
        :param image_mode: The mode to use for images ("RGB" or "L")
        """

        cmap = None if image_mode == 'RGB' else 'gray'
        z_dim = input_z.get_shape().as_list()[-1]
        example_z = np.random.uniform(-1, 1, size=[n_images, z_dim])

        samples = sess.run(
            self.generator(input_z, out_channel_dim, False),
            feed_dict={input_z: example_z})

        images_grid = check_data.images_square_grid(samples, image_mode)
        pyplot.imshow(images_grid, cmap=cmap)
        pyplot.show()


    def train(self, get_batches, data_shape, data_image_mode):
        """
        Train the GAN
        :param get_batches: Function to get batches
        :param data_shape: Shape of the data
        :param data_image_mode: The image mode to use for images ("RGB" or "L")
        """

        input_real, input_z, lr = self.model_inputs(data_shape[1], data_shape[2], data_shape[3], self.z_dim)
        d_loss, g_loss = self.model_loss(input_real, input_z, data_shape[3])
        d_opt, g_opt = self.model_opt(d_loss, g_loss, lr, self.beta1)

        steps = 0
        print_every = 10
        show_every = 100

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch_i in range(self.epoch_count):
                for batch_images in get_batches(self.batch_size):

                    steps += 1

                    # Normalize input to be between -1 and 1 to match generator's images
                    batch_images = batch_images * 2

                    # Get input noise
                    batch_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

                    # Run optimizers
                    _ = sess.run(d_opt, feed_dict={input_real: batch_images, input_z: batch_z, lr: self.learning_rate})
                    _ = sess.run(g_opt, feed_dict={input_real: batch_images, input_z: batch_z, lr: self.learning_rate})

                    # Print out accuracy every 10 batches
                    if steps % print_every == 0:
                        train_loss_d = d_loss.eval(feed_dict={input_z: batch_z, input_real: batch_images})
                        train_loss_g = g_loss.eval(feed_dict={input_z: batch_z})
                        print("Epoch {}/{}...".format(epoch_i + 1, self.epoch_count),
                              "Discriminator Loss: {:.4f}...".format(train_loss_d),
                              "Generator Loss: {:.4f}".format(train_loss_g))

                    # Show generated images every 100 batches
                    if steps % show_every == 0:
                        self.show_generator_output(sess, self.batch_size, input_z, data_shape[3], data_image_mode)

