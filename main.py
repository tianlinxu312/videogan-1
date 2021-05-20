"""
Author: Vignesh Gokul
Code structure inspired from https://github.com/carpedm20/DCGAN-tensorflow

"""
import numpy as np
import tensorflow as tf
import os
from model import VideoGAN
import utils

flags = tf.compat.v1.app.flags
flags.DEFINE_integer("epoch", 100, "Epoch to train [25]")
flags.DEFINE_integer("zdim", 100, "The dimension of latent vector")
flags.DEFINE_integer("batch_size", 2, "The size of batch images [64]")
flags.DEFINE_integer("checkpoint_file", None, "The checkpoint file name")
flags.DEFINE_float("lambd", 0.0, "The value of sparsity regularizer")
flags.DEFINE_string("dname", "mmnist", "dataset")
flags.DEFINE_integer("image_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("time_steps", 20, "seq len")
flags.DEFINE_integer("channels", 1, "n of channels")
FLAGS = flags.FLAGS


def main():
    if not os.path.exists("./checkpoints"):
        os.makedirs("./checkpoints")
    if not os.path.exists("./genvideos"):
        os.makedirs("./genvideos")

    dname = FLAGS.dname
    time_steps = FLAGS.time_steps
    image_size = FLAGS.image_size
    channels = FLAGS.channels
    batch_size = FLAGS.batch_size
    video_dims = [time_steps, image_size, image_size, channels]
    epochs = FLAGS.epoch
    '''
    if dname == 'kth':
        dataset = tf.data.Dataset.from_generator(utils.load_kth_data,
                                                 args=([batch_size * 2, image_size, image_size, time_steps]),
                                                 output_types=tf.float64)
        dataset = dataset.batch(FLAGS.batch_size).repeat(epochs)
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
    elif dname == "mmnist":
        data_path = "../data/mmnist/mnist_training_set.npy"
        training_data = np.load(data_path) / 255.0
        training_data = tf.transpose(training_data, (1, 0, 2, 3))
        dataset = tf.compat.v1.data.Dataset.from_tensor_slices(training_data)
        batched_x = dataset.batch(FLAGS.batch_size * 2).repeat(epochs)
    elif dname == "mazes":
        # path to data
        root_path = '../data/'
        data_reader = data_utils.DataReader(dataset=dname, time_steps=total_time_steps,
                                            root=root_path, custom_frame_size=x_height)
        batched_x = data_reader.provide_dataset(batch_size=batch_size).repeat(epochs)
    '''
    with tf.compat.v1.Session() as sess:
        videogan = VideoGAN(sess, video_dim=video_dims, zdim=FLAGS.zdim, batch_size=FLAGS.batch_size,
                            epochs=FLAGS.epoch, checkpoint_file=FLAGS.checkpoint_file, lambd=FLAGS.lambd)
        videogan.build_model()
        videogan.train(dname)


if __name__ == '__main__':
    main()
