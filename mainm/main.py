import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import tensorflow as tf
from core import MuseGAN
from components import NowbarHybrid
from config import *
from input_data import *

data_X = r'\Users\jls1\PycharmProjects\muse\saved_roll\x_bar_chroma_Amazing_Grace.npy'
data_Y = r'\Users\jls1\PycharmProjects\muse\saved_roll\y_bar_chroma_Amazing_Grace.npy'


# Initialize a tensorflow session

""" Create TensorFlow Session """
with tf.Session() as sess:
    # === Prerequisites ===
    # Step 1 - Initialize the training configuration
    t_config = TrainingConfig
    t_config.exp_name = 'mainm\\model\\exps\\nowbar_hybrid'

    # Step 2 - Select the desired model
    model = NowbarHybrid(NowBarHybridConfig)

    # Step 3 - Initialize the input data object
    input_data = InputDataNowBarHybrid(model)

    # Step 5?
    musegan = MuseGAN(sess, t_config, model)
    print(musegan.dir_ckpt)
    musegan.load(musegan.dir_ckpt)

    input_data.add_dataaa2(data_X, data_Y, key='test', batch_size=64)
    print('--------------Start arrangement generation !!--------------')
    musegan.gen_test(input_data, is_eval=True)
    print('--------------arrangement completed !!--------------')