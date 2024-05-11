''' @author: Andrew Glaws, Karen Stengel, Ryan King
'''
import tensorflow as tf
from utils import *

class SR_NETWORK(object):
    def __init__(self, x_LR=None, x_HR=None, r=None, status='pretraining', alpha_advers=0.001):

        status = status.lower()
        if status not in ['pretraining', 'training', 'testing']:
            print('Error in network status.')
            exit()

        self.x_LR, self.x_HR = x_LR, x_HR

        if r is None:
            print('Error in SR scaling. Variable r must be specified.')
            exit()

        if status in ['pretraining', 'training']:
            self.x_SR = self.generator(self.x_LR, r=r, is_training=True)
        else:
            self.x_SR = self.generator(self.x_LR, r=r, is_training=False)

        self.g_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

        if status == 'pretraining':
            self.g_loss = self.compute_losses(self.x_HR, self.x_SR, None, None,None,None, alpha_advers, isGAN=False)

            self.d_loss, self.disc_HR, self.disc_SR, self.d_variables = None, None, None, None
            self.advers_perf, self.content_loss, self.g_advers_loss = None, None, None

        elif status == 'training':
            self.disc_HR = self.discriminator(self.x_HR, reuse=False)
            self.disc_SR = self.discriminator(self.x_SR, reuse=True)

            # added newly
            self.epsilon = tf.random_uniform([tf.shape(self.x_HR)[0], 1, 1, 1], 0.0, 1.0)
            self.interpolated_images = self.epsilon * self.x_HR + (1 - self.epsilon) * self.x_SR
            self.d_interpolates = self.discriminator(self.interpolated_images, reuse=True)
            
            self.d_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

            # loss_out = self.compute_losses(self.x_HR, self.x_SR, self.disc_HR, self.disc_SR, alpha_advers, isGAN=True)
            loss_out = self.compute_losses(self.x_HR, self.x_SR, self.disc_HR, self.disc_SR, self.interpolated_images, self.d_interpolates,alpha_advers, isGAN=True)
            self.g_loss = loss_out[0]
            self.d_loss = loss_out[1]
            self.advers_perf = loss_out[2]
            self.content_loss = loss_out[3]
            self.g_advers_loss  = loss_out[4]

        else:
            self.g_loss, self.d_loss = None, None
            self.disc_HR, self.disc_SR, self.d_variables = None, None, None
            self.advers_perf, self.content_loss, self.g_advers_loss = None, None, None
            self.disc_HR, self.disc_SR, self.d_variables = None, None, None


    def generator(self, x, r, is_training=False, reuse=False):
        if is_training:
            N, h, w, C = tf.shape(x)[0], x.get_shape()[1], x.get_shape()[2], x.get_shape()[3]
        else:
            N, h, w, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], x.get_shape()[3]

        k, stride = 3, 1
        output_shape = [N, h+2*k, w+2*k, -1]

        with tf.variable_scope('generator', reuse=reuse):
            with tf.variable_scope('deconv1'):
                C_in, C_out = C, 64
                output_shape[-1] = C_out
                x = deconv_layer_2d(x, [k, k, C_out, C_in], output_shape, stride, k)
                x = tf.nn.relu(x)

            skip_connection = x

            # B residual blocks
            C_in, C_out = C_out, 64
            output_shape[-1] = C_out
            for i in range(16):
                B_skip_connection = x

                with tf.variable_scope('block_{}a'.format(i+1)):
                    x = deconv_layer_2d(x, [k, k, C_out, C_in], output_shape, stride, k)
                    x = tf.nn.relu(x)

                with tf.variable_scope('block_{}b'.format(i+1)):
                    x = deconv_layer_2d(x, [k, k, C_out, C_in], output_shape, stride, k)

                x = tf.add(x, B_skip_connection)

            with tf.variable_scope('deconv2'):
                x = deconv_layer_2d(x, [k, k, C_out, C_in], output_shape, stride, k)
                x = tf.add(x, skip_connection)

            # Super resolution scaling
            r_prod = 1
            for i, r_i in enumerate(r):
                C_out = (r_i**2)*C_in
                with tf.variable_scope('deconv{}'.format(i+3)):
                    output_shape = [N, r_prod*h+2*k, r_prod*w+2*k, C_out]
                    x = deconv_layer_2d(x, [k, k, C_out, C_in], output_shape, stride, k)
                    x = tf.depth_to_space(x, r_i)
                    x = tf.nn.relu(x)

                r_prod *= r_i

            output_shape = [N, r_prod*h+2*k, r_prod*w+2*k, C]
            with tf.variable_scope('deconv_out'):
                x = deconv_layer_2d(x, [k, k, C, C_in], output_shape, stride, k)

        return x


    def discriminator(self, x, reuse=False):
        N, h, w, C = tf.shape(x)[0], x.get_shape()[1], x.get_shape()[2], x.get_shape()[3]

        with tf.variable_scope('discriminator', reuse=reuse):
            with tf.variable_scope('conv1'):
                x = conv_layer_2d(x, [3, 3, C, 32], 1)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('conv2'):
                x = conv_layer_2d(x, [3, 3, 32, 32], 2)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('conv3'):
                x = conv_layer_2d(x, [3, 3, 32, 64], 1)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('conv4'):
                x = conv_layer_2d(x, [3, 3, 64, 64], 2)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('conv5'):
                x = conv_layer_2d(x, [3, 3, 64, 128], 1)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('conv6'):
                x = conv_layer_2d(x, [3, 3, 128, 128], 2)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('conv7'):
                x = conv_layer_2d(x, [3, 3, 128, 256], 1)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('conv8'):
                x = conv_layer_2d(x, [3, 3, 256, 256], 2)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            x = flatten_layer(x)
            with tf.variable_scope('fully_connected1'):
                x = dense_layer(x, 1024)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('fully_connected2'):
                x = dense_layer(x, 1)

        return x

    def compute_losses(self, x_HR, x_SR, d_HR, d_SR,interpolated_images, d_interpolates, alpha_advers=0.001, isGAN=False):

        content_loss = tf.reduce_mean((x_HR - x_SR)**2, axis=[1, 2, 3])

        if isGAN:
            gradients = tf.gradients(d_interpolates, [interpolated_images])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
            gradients_penalty = tf.reduce_mean((slopes - 1.0) ** 2)

            d_loss = tf.reduce_mean(d_SR) - tf.reduce_mean(d_HR)
            d_loss += 10 * gradients_penalty
            
            g_loss = -tf.reduce_mean(d_SR)

            advers_perf = [0,0,0,0]
            g_advers_loss = 1
            return g_loss, d_loss, advers_perf, content_loss, g_advers_loss
        else:
            return tf.reduce_mean(content_loss)


