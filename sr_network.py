''' @author: Andrew Glaws, Karen Stengel, Ryan King
'''
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import *
  
class SR_NETWORK(object):
    def __init__(self, x_LR=None, x_HR=None, r=None, status='pretraining', alpha_advers=0.001):
        self.HR_numpy = None
        print("\n sr network called")
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
        print("g_var", self.g_variables)
        if status == 'pretraining':
            self.g_loss = self.compute_losses(self.x_HR, self.x_SR, None, None, alpha_advers, isGAN=False)

            self.d_loss, self.disc_HR, self.disc_SR, self.d_variables = None, None, None, None
            self.advers_perf, self.content_loss, self.g_advers_loss = None, None, None

        elif status == 'training':
            self.disc_HR = self.discriminator(self.x_HR, reuse=False)
            self.disc_SR = self.discriminator(self.x_SR, reuse=True)
            self.d_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

            loss_out = self.compute_losses(self.x_HR, self.x_SR, self.disc_HR, self.disc_SR, alpha_advers, isGAN=True)
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


    def rescale_linear(self, tensor):
        minimum, maximum = tf.reduce_min(tensor), tf.reduce_max(tensor)
        m = (1 - 0) / (maximum - minimum)
        b = 0 - m * minimum
        return m * tensor + b

    

    def energy_spectrum(self, tensor):
        tensor = self.rescale_linear(tensor)

        npix = tensor.shape[0]
        fourier_tensor = tf.spectral.fft2d(tf.cast(tensor, dtype=tf.complex64))
        shifted_fourier_tensor = self.fftshift(fourier_tensor)
        fourier_amplitudes = tf.abs(shifted_fourier_tensor)**2

        kfreq = self.fftfreq(npix) * npix
        kfreq2D = tf.meshgrid(kfreq, kfreq)
        knrm = tf.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)

        knrm = tf.reshape(knrm, [-1])
        fourier_amplitudes = tf.reshape(fourier_amplitudes, [-1])

        kbins = tf.range(0.5, npix//2+1, 1., dtype=tf.float32)
        kvals = (kbins[1:] + kbins[:-1])

        Abins, _, _ = tf.raw_ops.BinnedStatistic(
            values=fourier_amplitudes, 
            sample=knrm, 
            bin_edges=kbins, 
            statistic="Mean"
        )
        Abins *= tf.constant(np.pi, dtype=tf.float32) * (kbins[1:]**2 - kbins[:-1]**2)

        return kvals, Abins

    

    def fftshift(self, tensor):
    # Shift the zero-frequency components to the center
      shift1 = tf.roll(tensor, shift=tf.shape(tensor)[0]//2, axis=0)
      shift2 = tf.roll(shift1, shift=tf.shape(tensor)[1]//2, axis=1)
      return shift2

    
    def fftfreq(self, npix):
      val = 1.0 / tf.cast(npix, dtype=tf.float32)
      res = tf.range(0.0, tf.cast(npix // 2, dtype=tf.float32), dtype=tf.float32)
      return tf.concat([res, res - tf.cast(npix, dtype=tf.float32)]) * val



 


    def calculate_mean_slope(self, E, k):

        E = tf.constant(E, dtype=tf.float32)
        k = tf.constant(k, dtype=tf.float32)

        E_proportional_to_k = tf.pow(k, -5/3)
        dE_dk = tf.gradients(E_proportional_to_k, k)[0]

        with tf.Session() as sess:
            dE_dk_value = sess.run(dE_dk)

        return np.mean(dE_dk_value)



    def compute_losses(self, x_HR, x_SR, d_HR, d_SR, alpha_advers=0.001, isGAN=False):
        
        content_loss = tf.reduce_mean((x_HR - x_SR)**2, axis=[1, 2, 3])
       
        # print("content loss ", type(content_loss))
        # print("x_HR", type(x_HR), x_HR.shape, x_HR)
        # with tf.Session() as sess:
        #   print(x_HR.eval())
        # tf.reset_default_graph()
        # x_in = tf.placeholder(tf.float32, [None, 500, 500, 2])
        # with tf.Session() as sess:
        #   x_np = sess.run(x_in, feed_dict={x_in: x_HR})
        # print(type(x_np))
        
        if isGAN:
            g_advers_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_SR, labels=tf.ones_like(d_SR))

            d_advers_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.concat([d_HR, d_SR], axis=0),
                                                                    labels=tf.concat([tf.ones_like(d_HR), tf.zeros_like(d_SR)], axis=0))

            advers_perf = [tf.reduce_mean(tf.cast(tf.sigmoid(d_HR) > 0.5, tf.float32)), # % true positive
                           tf.reduce_mean(tf.cast(tf.sigmoid(d_SR) < 0.5, tf.float32)), # % true negative
                           tf.reduce_mean(tf.cast(tf.sigmoid(d_SR) > 0.5, tf.float32)), # % false positive
                           tf.reduce_mean(tf.cast(tf.sigmoid(d_HR) < 0.5, tf.float32))] # % false negative

            g_loss = tf.reduce_mean(content_loss) + alpha_advers*tf.reduce_mean(g_advers_loss) # +alpha_spectrum_wind*loss_spectrum_wind
            d_loss = tf.reduce_mean(d_advers_loss)

            return g_loss, d_loss, advers_perf, content_loss, g_advers_loss
        else:
            print("\n 7 check")
            return tf.reduce_mean(content_loss)
    

