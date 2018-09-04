"""Run inference a DeepLab v3 model using tf.estimator API."""
# https://github.com/rishizek/tensorflow-deeplab-v3-plus
# https://github.com/tensorflow/models/tree/master/research/deeplab
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf

import deeplab_model
from utils import preprocessing
from utils import dataset_util

from PIL import Image
import matplotlib.pyplot as plt

from tensorflow.python import debug as tf_debug
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='dataset/VOCdevkit/VOC2012/JPEGImages',
                    help='The directory containing the image data.')

parser.add_argument('--output_dir', type=str, default='./dataset/inference_output',
                    help='Path to the directory to generate the inference results')

parser.add_argument('--infer_data_list', type=str, default='./dataset/sample_images_list.txt',
                    help='Path to the file listing the inferring images.')

parser.add_argument('--model_dir', type=str, default='./model',
                    help="Base directory for the model. "
                         "Make sure 'model_checkpoint_path' given in 'checkpoint' file matches "
                         "with checkpoint name.")

parser.add_argument('--base_architecture', type=str, default='resnet_v2_101',
                    choices=['resnet_v2_50', 'resnet_v2_101'],
                    help='The architecture of base Resnet building block.')

parser.add_argument('--output_stride', type=int, default=16,
                    choices=[8, 16],
                    help='Output stride for DeepLab v3. Currently 8 or 16 is supported.')

parser.add_argument('--debug', action='store_true',
                    help='Whether to use debugger to track down bad values during training.')

_NUM_CLASSES = 21


def main(unused_argv):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
  os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

  pred_hooks = None
  if FLAGS.debug:
    debug_hook = tf_debug.LocalCLIDebugHook()
    pred_hooks = [debug_hook]

  model = tf.estimator.Estimator(
      model_fn=deeplab_model.deeplabv3_plus_model_fn,
      model_dir=FLAGS.model_dir,
      params={
          'output_stride': FLAGS.output_stride,
          'batch_size': 1,  # Batch size must be 1 because the images' size may differ
          'base_architecture': FLAGS.base_architecture,
          'pre_trained_model': None,
          'batch_norm_decay': None,
          'num_classes': _NUM_CLASSES,
      })

  examples = dataset_util.read_examples_list(FLAGS.infer_data_list)
  image_files = [os.path.join(FLAGS.data_dir, filename) for filename in examples]

  print('The length of image_files:', len(image_files))
  def run_pred(image_files):
     print('Starting preding ...')
     print('The length:', len(image_files))
     predictions = model.predict(
        input_fn=lambda: preprocessing.eval_input_fn(image_files),
        hooks=pred_hooks)

     #print('Prediction Done:', len(predictions))
     output_dir = FLAGS.output_dir
     data_dir = FLAGS.data_dir
     if not os.path.exists(output_dir):
        os.makedirs(output_dir)

     print('Starting saving ...')
     for pred_dict, image_path in zip(predictions, image_files):
        #print('pred_dict:', pred_dict)
        #print('image_path:', image_path)
        image_basename = os.path.splitext(os.path.basename(image_path))[0]
        output_filename = image_basename + '.jpg'
        path_to_output = os.path.join(output_dir, output_filename)

        #print("generating:", path_to_output)
        import cv2
        ori = cv2.imread(os.path.join(data_dir, image_basename + '.jpg'))
        mask = pred_dict['decoded_labels']
        #person_color = (192,128,128)
        #mask[mask == [192, 128,128]] = 1
        #mask[mask != [1, 1, 1]] = 0
        
        # method 1:
#        for i in range(mask.shape[0]):
#		for j in range(mask.shape[1]):
#			if mask[i,j,0] == 192 \
# 			and mask[i,j,1] == 128 \
#			and mask[i,j,2] == 128:
#				mask[i,j,:] = 1
#			else:
#				mask[i,j,:] = 0
        mask[(mask[:,:,0] == 192) & (mask[:,:,1] == 128) & (mask[:,:,2]==128), :] = 1
        mask[(mask[:,:,0] != 1) & (mask[:,:,1] != 1) & (mask[:,:,2]!=1), :] = 0
        #mask[mask != 0] = 1 #Get all classes
        mask = ori * mask
        #mask = Image.fromarray(mask)
        #import scipy.misc
        #scipy.misc.imsave(path_to_output, mask)
        cv2.imwrite(path_to_output, mask)

  num_image = len(image_files)
  batch_size = 1000
  steps = num_image // batch_size
  i = 0
  for i in range(0, steps):
     print('step:', i)
     files = image_files[i*batch_size:(i+1)*batch_size]
     run_pred(files)
  files = image_files[i*batch_size:num_image]
  run_pred(files)
  
if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
