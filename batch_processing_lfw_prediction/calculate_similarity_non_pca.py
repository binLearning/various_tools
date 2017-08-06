from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import numpy as np
from six.moves import xrange
from scipy.spatial.distance import cosine


def main(args):
  rt_dir_features   = './features'
  rt_dir_similarity = './similarity'
  
  # check features directory
  dir_features  = os.path.join(rt_dir_features, args.version)
  if not os.path.exists(dir_features):
    sys.exit('Could not find features directory "{}".'.format(dir_features))
  
  # create similarity directory
  dir_similarity = os.path.join(rt_dir_similarity, args.version)
  if not os.path.exists(dir_similarity):
    os.makedirs(dir_similarity)
  
  # calculate similarity
  # 1 --- original
  # 2 --- concatenate
  # 3 --- add
  for loop_iter in xrange(666):
    iter_num = args.start + args.step * loop_iter
    if iter_num > args.end:
      break
    
    print(iter_num)
    
    features_ori  = os.path.join(dir_features, 'ori_{}.txt'.format(iter_num))
    features_flip = os.path.join(dir_features, 'flip_{}.txt'.format(iter_num))
    
    similarity_ori = os.path.join(dir_similarity, 'ori_{}.txt'.format(iter_num))
    similarity_cat = os.path.join(dir_similarity, 'cat_{}.txt'.format(iter_num))
    similarity_add = os.path.join(dir_similarity, 'add_{}.txt'.format(iter_num))
    
    features_array_ori  = np.loadtxt(features_ori)
    features_array_flip = np.loadtxt(features_flip)
    features_array_cat  = np.concatenate((features_array_ori, 
                                          features_array_flip), axis=1)
    features_array_add  = features_array_ori + features_array_flip
    
    fp_ori = open(similarity_ori, 'w')
    fp_cat = open(similarity_cat, 'w')
    fp_add = open(similarity_add, 'w')
    
    PAIRS_NUM = 6000
    for idx in xrange(PAIRS_NUM):
      s_ori = 1 - cosine(features_array_ori[idx*2], features_array_ori[idx*2+1])
      s_cat = 1 - cosine(features_array_cat[idx*2], features_array_cat[idx*2+1])
      s_add = 1 - cosine(features_array_add[idx*2], features_array_add[idx*2+1])
      
      fp_ori.write('{}\n'.format(s_ori))
      fp_cat.write('{}\n'.format(s_cat))
      fp_add.write('{}\n'.format(s_add))
      
    fp_ori.close()
    fp_cat.close()
    fp_add.close()
    

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  parser.add_argument('version', help='version major')
  parser.add_argument('start',   help='iter start', type=int)
  parser.add_argument('step',    help='iter step',  type=int)
  parser.add_argument('end',     help='iter end',   type=int)
  
  args = parser.parse_args()
  
  main(args)
