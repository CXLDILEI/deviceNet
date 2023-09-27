import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import os

in_path = './dataset/train/'
out_path = './output/'

img_len = 13

def resize():
    datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    gen_data = datagen.flow_from_directory(in_path, batch_size=1,shuffle=False,save_to_dir=out_path+'resize',save_prefix='gen',target_size=(244,244),save_format='jpg')
    gen_data.next()
    for i in range(img_len):
        gen_data.next()

def rotation_range():
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=10)
    gen_data = datagen.flow_from_directory(in_path, batch_size=1,shuffle=False,save_to_dir=out_path+'rotation_range',save_prefix='gen',target_size=(244,244),save_format='jpg')
    gen_data.next()
    for i in range(img_len):
        gen_data.next()


resize()
rotation_range()