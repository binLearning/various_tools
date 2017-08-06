from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import numpy as np
from six.moves import xrange

import caffe


def main(args):  
  dir_model       = './model'
  rt_dir_weights  = './weights'
  rt_dir_features = './features'
  
  # check model
  deploy_model = os.path.join(dir_model, args.model)
  if not os.path.exists(deploy_model):
    sys.exit('Could not find deploy model "{}".'.format(deploy_model))
  
  # check weights directory
  dir_weights  = os.path.join(rt_dir_weights, args.version)
  if not os.path.exists(dir_weights):
    sys.exit('Could not find weights directory "{}".'.format(dir_weights))
  
  # get batch size
  with open(deploy_model, 'r') as fp_model:
    for line in fp_model.readlines():
      if 'batch_size: ' in line:
        pos = line.find('batch_size: ') + len('batch_size: ')
        batch_size = int(line[pos:-1])
        break
      if 'batch_size:' in line:
        pos = line.find('batch_size:') + len('batch_size:')
        batch_size = int(line[pos:-1])
        break
  print('batch size: {}'.format(batch_size))
  
  # check batch size
  TOTAL_NUM_IMAGES = 24000 # 6000 * 2(pair) * 2(ori+flip)
  if TOTAL_NUM_IMAGES % batch_size != 0:
    sys.exit('#images should be divisible by batch size, please modify the value of batch size.')
  if batch_size % 2 != 0:
    sys.exit('batch size should be divisible by 2, please modify the value of batch size.')
  
  # create features directory
  dir_features = os.path.join(rt_dir_features, args.version)
  if not os.path.exists(dir_features):
    os.makedirs(dir_features)
  
  # calculate inference epoch
  inference_epoch = TOTAL_NUM_IMAGES // batch_size
  
  # set Caffe mode & device
  caffe.set_mode_gpu()
  caffe.set_device(args.device)
  
  # extract & save features
  for loop_iter in xrange(666):
    iter_num = args.start + args.step * loop_iter
    if iter_num > args.end:
      break
    
    weights_name = '{}_{}.caffemodel'.format(args.prefix, iter_num)
    weights_path = os.path.join(dir_weights, weights_name)
    if not os.path.exists(weights_path):
      sys.exit('Could not find weights "{}".'.format(weights_path))
    
    features_ori  = os.path.join(dir_features, 'ori_{}.txt'.format(iter_num))
    features_flip = os.path.join(dir_features, 'flip_{}.txt'.format(iter_num))
    
    fp_ori  = open(features_ori, 'w')
    fp_flip = open(features_flip, 'w')

    net = caffe.Net(deploy_model, weights_path, caffe.TEST)

    for loop_infer in xrange(inference_epoch):
      if loop_infer % 10 == 0:
        print(loop_infer)
      
      net.forward()
      
      for loop_batch in xrange(0,batch_size,2):
        features_ori = net.blobs['fc5'].data[loop_batch]
        features_l2norm_ori = np.sqrt(np.sum(np.square(features_ori)))
        features_normed_ori = features_ori / features_l2norm_ori
        np.savetxt(fp_ori, features_normed_ori, newline=" ", fmt='%.6f')
        fp_ori.write('\n')
        
        features_flip = net.blobs['fc5'].data[loop_batch+1]
        features_l2norm_flip = np.sqrt(np.sum(np.square(features_flip)))
        features_normed_flip = features_flip / features_l2norm_flip
        np.savetxt(fp_flip, features_normed_flip, newline=" ", fmt='%.6f')
        fp_flip.write('\n')
    
    fp_ori.close()
    fp_flip.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  parser.add_argument('version', help='version major')
  parser.add_argument('model',   help='deploy model')
  parser.add_argument('prefix',  help='weights prefix')
  parser.add_argument('start',   help='iter start', type=int)
  parser.add_argument('step',    help='iter step',  type=int)
  parser.add_argument('end',     help='iter end',   type=int)
  parser.add_argument('device',  help='GPU ID',     type=int)
  
  args = parser.parse_args()
  
  main(args)
  
