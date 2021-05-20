import cv2
import skvideo.io
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import collections
import os
import random

"""
Author: Vignesh Gokul
Utilities inspired from https://github.com/carpedm20/DCGAN-tensorflow
"""


def save_gen(generated_images, n_ex = 36, epoch = 0, iter = 0):
    for i in range(generated_images.shape[0]):
        cv2.imwrite('/root/code/Video_Generation/gen_images/image_' + str(epoch) + '_' + str(iter) + '_' + str(i) + '.jpg', generated_images[i, :, :, :])


def process_and_write_image(images,name):
    images = np.array(images)
    images = (images + 1)*127.5
    for i in range(images.shape[0]):
        cv2.imwrite("./genvideos/" + name + ".jpg",images[i,0,:,:,:])


def process_and_write_video(videos,name):
    videos =np.array(videos)
    videos = np.reshape(videos,[-1,32,64,64,3])
    vidwrite = np.zeros((32,64,64,3))
    for i in range(videos.shape[0]):
        vid = videos[i,:,:,:,:]
        vid = (vid + 1)*127.5
        for j in range(vid.shape[0]):
            frame = vid[j,:,:,:]
            #frame = (frame+1)*127.5
            vidwrite[j,:,:,:] = frame
        skvideo.io.vwrite("./genvideos/" +name + ".mp4",vidwrite)


def conv2d(input_, output_dim,k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
    return conv


def deconv2d(input_, output_shape,k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,name="deconv2d", with_w=False):
    with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                  initializer=tf.random_normal_initializer(stddev=stddev))

        try:
          deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                    strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
          deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                    strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
          return deconv, w, biases
        else:
            return deconv


def conv3d(input_, output_dim,k_h=5, k_w=5,k_z =5, d_h=2, d_w=2,d_z=2, stddev=0.02,name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w,k_z, input_.get_shape()[-1], output_dim],initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w,d_z, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
    return conv


def deconv3d(input_, output_shape,k_h=5, k_w=5,k_z=5, d_h=2, d_w=2,d_z=2, stddev=0.02,name="deconv2d", with_w=False):
    with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w,k_z, output_shape[-1], input_.get_shape()[-1]],
                  initializer=tf.random_normal_initializer(stddev=stddev))

        try:
          deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                    strides=[1, d_h, d_w,d_z, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
          deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                    strides=[1, d_h, d_w,d_z, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
          return deconv, w, biases
        else:
            return deconv


def load_kth_data(batch_size, image_size=64, time_step=16):
    path = '../data/kth'
    list_f = [x for x in os.listdir(path)]

    batch_sample = []
    for x in range(batch_size):
        rand_folder = random.choice(list_f)
        path_to_file = path + '/' + rand_folder
        file_name = random.choice(os.listdir(path_to_file))
        path_to_video = path_to_file + '/' + file_name
        vidcap = cv2.VideoCapture(path_to_video)
        n_frames = vidcap.get(7)
        stacked_frames = []
        while vidcap.isOpened():
            frame_id = vidcap.get(1)  # current frame number
            ret, frame = vidcap.read()
            if not ret or len(stacked_frames) > (time_step - 1):
                break
            frame = frame / 255.0
            if rand_folder == 'running' or rand_folder == 'walking' or rand_folder == 'jogging':
                if frame_id % 1 == 0 and frame_id > 5:
                    frame = cv2.resize(frame, dsize=(image_size, image_size),
                                       interpolation=cv2.INTER_AREA)
                    stacked_frames.append(frame)
            elif n_frames < 350:
                if frame_id % 1 == 0 and frame_id > 5:
                    frame = cv2.resize(frame, dsize=(image_size, image_size),
                                       interpolation=cv2.INTER_AREA)
                    stacked_frames.append(frame)
            else:
                if frame_id % 1 == 0 and frame_id > 10:
                    frame = cv2.resize(frame, dsize=(image_size, image_size),
                                       interpolation=cv2.INTER_AREA)
                    stacked_frames.append(frame)
        if len(stacked_frames) < time_step:
            break
        stacked_frames = np.reshape(stacked_frames, newshape=(time_step, image_size, image_size, 3))
        batch_sample.append(stacked_frames)
    batch_sample = np.reshape(batch_sample, (batch_size, time_step, image_size, image_size, 3))
    return batch_sample

class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.layers.batch_normalization(x,
                      epsilon=self.epsilon,
                      scale=True)


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)


"""Minimal data reader for GQN TFRecord datasets."""

# nest = tf.contrib.framework.nest
nest = tf.nest
seed = 1

DatasetInfo = collections.namedtuple('DatasetInfo', ['basepath', 'train_size', 'test_size', 'frame_size',
                                                     'sequence_size'])
Context = collections.namedtuple('Context', ['frames', 'cameras'])
Query = collections.namedtuple('Query', ['context', 'query_camera'])
TaskData = collections.namedtuple('TaskData', ['query', 'target'])


_DATASETS = dict(
    jaco=DatasetInfo(
        basepath='jaco',
        train_size=3600,
        test_size=400,
        frame_size=64,
        sequence_size=11),

    mazes=DatasetInfo(
        basepath='mazes',
        train_size=1080,
        test_size=120,
        frame_size=84,
        sequence_size=300),

    rooms_free_camera_with_object_rotations=DatasetInfo(
        basepath='rooms_free_camera_with_object_rotations',
        train_size=2034,
        test_size=226,
        frame_size=128,
        sequence_size=10),

    rooms_ring_camera=DatasetInfo(
        basepath='rooms_ring_camera',
        train_size=2160,
        test_size=240,
        frame_size=64,
        sequence_size=10),

    rooms_free_camera_no_object_rotations=DatasetInfo(
        basepath='rooms_free_camera_no_object_rotations',
        train_size=2160,
        test_size=240,
        frame_size=64,
        sequence_size=10),

    shepard_metzler_5_parts=DatasetInfo(
        basepath='shepard_metzler_5_parts',
        train_size=900,
        test_size=100,
        frame_size=64,
        sequence_size=15),

    shepard_metzler_7_parts=DatasetInfo(
        basepath='shepard_metzler_7_parts',
        train_size=900,
        test_size=100,
        frame_size=64,
        sequence_size=15)
)
_NUM_CHANNELS = 3
_NUM_RAW_CAMERA_PARAMS = 5
_MODES = ('train', 'test')


def _get_dataset_files(dateset_info, mode, root):
    """Generates lists of files for a given dataset version."""
    basepath = dateset_info.basepath
    base = os.path.join(root, basepath, mode)
    if mode == 'train':
        num_files = dateset_info.train_size
    else:
        num_files = dateset_info.test_size

    length = len(str(num_files))
    template = '{:0%d}-of-{:0%d}.tfrecord' % (length, length)
    return [os.path.join(base, template.format(i + 1, num_files))
            for i in range(num_files)]


def _convert_frame_data(jpeg_data):
    decoded_frames = tf.image.decode_jpeg(jpeg_data)
    return tf.image.convert_image_dtype(decoded_frames, dtype=tf.float32)


class DataReader(object):
    """Minimal queue based TFRecord reader.
      You can use this reader to load the datasets used to train Generative Query
      Networks (GQNs) in the 'Neural Scene Representation and Rendering' paper.
      See README.md for a description of the datasets and an example of how to use
      the reader.
      """

    def __init__(self,
                 dataset,
                 time_steps,
                 root,
                 mode='train',
                 # Optionally reshape frames
                 custom_frame_size=None):
        """Instantiates a DataReader object and sets up queues for data reading.
        Args:
          dataset: string, one of ['jaco', 'mazes', 'rooms_ring_camera',
              'rooms_free_camera_no_object_rotations',
              'rooms_free_camera_with_object_rotations', 'shepard_metzler_5_parts',
              'shepard_metzler_7_parts'].
          time_steps: integer, number of views to be used to assemble the context.
          root: string, path to the root folder of the data.
          mode: (optional) string, one of ['train', 'test'].
          custom_frame_size: (optional) integer, required size of the returned
              frames, defaults to None.
          num_threads: (optional) integer, number of threads used to feed the reader
              queues, defaults to 4.
          capacity: (optional) integer, capacity of the underlying
              RandomShuffleQueue, defualts to 256.
          min_after_dequeue: (optional) integer, min_after_dequeue of the underlying
              RandomShuffleQueue, defualts to 128.
          seed: (optional) integer, seed for the random number generators used in
              the reader.
        Raises:
          ValueError: if the required version does not exist; if the required mode
             is not supported; if the requested time_steps is bigger than the
             maximum supported for the given dataset version.
        """

        if dataset not in _DATASETS:
            raise ValueError('Unrecognized dataset {} requested. Available datasets '
                             'are {}'.format(dataset, _DATASETS.keys()))

        if mode not in _MODES:
            raise ValueError('Unsupported mode {} requested. Supported modes '
                             'are {}'.format(mode, _MODES))

        self._dataset_info = _DATASETS[dataset]

        if time_steps > self._dataset_info.sequence_size:
            raise ValueError(
                'Maximum support context size for dataset {} is {}, but '
                'was {}.'.format(
                    dataset, self._dataset_info.sequence_size, time_steps))

        self.time_steps = time_steps
        self._custom_frame_size = custom_frame_size

        with tf.device('/cpu'):
            self._queue = _get_dataset_files(self._dataset_info, mode, root)

    def get_dataset_from_path(self, buffer=100):
        read_data = tf.data.Dataset.list_files(self._queue)
        dataset = read_data.repeat().shuffle(buffer_size=buffer)
        dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=4, block_length=16)
        return dataset

    def provide_dataset(self, batch_size):
        """Instantiates the ops used to read and parse the data into tensors."""
        def read_tfrecord(serialized_example):
            feature_map = {'frames': tf.io.FixedLenFeature(shape=self._dataset_info.sequence_size, dtype=tf.string)}
            example = tf.io.parse_example(serialized_example, feature_map)
            frames = self._preprocess_frames(example)
            return frames

        dataset = self.get_dataset_from_path()
        dataset = dataset.map(read_tfrecord, num_parallel_calls=4)
        dataset = dataset.batch(batch_size)
        return dataset

    def _preprocess_frames(self, example):
        """Instantiates the ops used to preprocess the frames data."""
        frames = tf.concat(example['frames'], axis=0)
        frames = tf.nest.map_structure(tf.stop_gradient, tf.map_fn(_convert_frame_data,  tf.reshape(frames, [-1]),
                                                                   dtype=tf.float32))
        dataset_image_dimensions = tuple([self._dataset_info.frame_size] * 2 + [_NUM_CHANNELS])
        frames = tf.reshape(frames, (-1, self._dataset_info.sequence_size) + dataset_image_dimensions)
        if (self._custom_frame_size and
                self._custom_frame_size != self._dataset_info.frame_size):
            frames = tf.reshape(frames, (-1,) + dataset_image_dimensions)
            new_frame_dimensions = (self._custom_frame_size,) * 2 + (_NUM_CHANNELS,)
            frames = tf.image.resize(frames, new_frame_dimensions[:2])
            frames = tf.reshape(frames, (-1, self._dataset_info.sequence_size) + new_frame_dimensions)
        return tf.transpose(tf.squeeze(frames[:, :self.time_steps, :, :]), (1, 0, 2, 3))


def samples_to_video(samples, nx, ny, time_steps=16, x_height=64, x_width=64):
    samples = samples.reshape(nx, ny, x_height, time_steps, x_width, -1)
    samples = np.concatenate(samples, 1)
    samples = np.concatenate(samples, 2)
    samples = np.transpose(samples, [1, 0, 2, 3])[..., :3]

    fig, ax = plt.subplots(figsize=(ny, nx))
    im = ax.imshow(samples[0])
    ax.set_axis_off()
    fig.tight_layout()

    def init():
        im.set_data(samples[0])
        return (im,)

    def animate(i):
        im.set_data(samples[i])
        return (im,)

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=time_steps, interval=100,
                                   blit=True)
    plt.close()
    return HTML(anim.to_html5_video())


def samples_to_video_with_cap(samples, nx, ny, time_steps=16, input_t=6, x_height=64, x_width=64):
    samples = samples.reshape(nx, ny, x_height, time_steps, x_width, -1)
    samples = np.concatenate(samples, 1)
    samples = np.concatenate(samples, 2)
    samples = np.transpose(samples, [1, 0, 2, 3])[..., :3]

    fig, ax = plt.subplots(figsize=(ny, nx))
    ax.set_axis_off()
    fig.tight_layout()

    ims = []

    for i in range(time_steps):
        frame = ax.imshow(samples[i])
        if i < input_t:
            t = ax.annotate("Truth t={}".format(i + 1), (5, 2.5), bbox=dict(boxstyle="round", fc="w"))  # add text
        else:
            t = ax.annotate("Prediction t+{}".format(i - input_t + 1), (5, 2.5), bbox=dict(boxstyle="round", fc="w"))
        ims.append([frame, t])
    anim = animation.ArtistAnimation(fig, ims, interval=150, blit=True, repeat_delay=100)
    plt.close()
    return HTML(anim.to_html5_video())