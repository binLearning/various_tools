from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import numpy as np
from six.moves import xrange


def _get_pairs_info():
  lfw_pairs_path = './data/pairs.txt'
  num_pairs_info = 6000
  
  list_pairs_info = []
  list_issame     = []
  with open(lfw_pairs_path, 'r') as f:
    for line in f.readlines()[1:]:
    
      pair = line.strip().split()
      list_pairs_info.append(pair)
      
      if(len(pair) == 3):
        list_issame.append(True)
      elif(len(pair) == 4):
        list_issame.append(False)
      else:
        raise ValueError('Unknown info')
      
  assert(len(list_pairs_info) == num_pairs_info)
  
  return list_pairs_info, list_issame


def main(argv=None):
  rt_dir_similarity = './similarity'
  rt_dir_accuracy   = './accuracy_10_folder'
  
  # check similarity directory
  dir_similarity = os.path.join(rt_dir_similarity, args.version)
  if not os.path.exists(dir_similarity):
    sys.exit('Could not find similarity directory "{}".'.format(dir_similarity))
  
  # get LFW information
  _, list_issame   = _get_pairs_info()
  
  accuracy_path = os.path.join(rt_dir_accuracy, '{}_non_pca.txt'.format(args.version))
  fp_acc = open(accuracy_path, 'w')
  
  fp_acc.write('iter         folder#1       folder#2       folder#3       folder#4       folder#5       ')
  fp_acc.write('folder#6       folder#7       folder#8       folder#9       folder#10      avg_acc\n')
  fp_acc.write('---------------------------------------------------------------------------------------')
  fp_acc.write('--------------------------------------------------------------------------------------\n')
  
  PAIRS_NUM = 6000
  for loop_iter in xrange(666):
    iter_num = args.start + args.step * loop_iter
    if iter_num > args.end:
      break
    
    print(iter_num)

    similarity_ori = os.path.join(dir_similarity, 'ori_{}.txt'.format(iter_num))
    similarity_cat = os.path.join(dir_similarity, 'cat_{}.txt'.format(iter_num))
    similarity_add = os.path.join(dir_similarity, 'add_{}.txt'.format(iter_num))

    s_ori = np.loadtxt(similarity_ori)
    s_cat = np.loadtxt(similarity_cat)
    s_add = np.loadtxt(similarity_add)
    
    # original
    fp_acc.write('{:>6d}_ori   '.format(iter_num))
    total_acc = 0
    for idx in xrange(0,PAIRS_NUM,600):
      max_num_correct = 0
      th_record = 0
      th_temp = 0.0
      for _ in xrange(100):
        num_correct_ori =  np.count_nonzero(s_ori[idx:idx+300] >= th_temp)
        num_correct_ori += np.count_nonzero(s_ori[idx+300:idx+600] < th_temp)
        if max_num_correct < num_correct_ori:
          max_num_correct = num_correct_ori
          th_record = th_temp
        th_temp += 0.01
      
      folder_acc = max_num_correct / 600
      fp_acc.write('{:0<.5f}@{:0<.2f}   '.format(folder_acc, th_record))
      
      total_acc += folder_acc
    fp_acc.write('{:0<.8f}\n'.format(total_acc / 10))
    
    # concatenate
    fp_acc.write('{:>6d}_cat   '.format(iter_num))
    total_acc = 0
    for idx in xrange(0,PAIRS_NUM,600):
      max_num_correct = 0
      th_record = 0
      th_temp = 0.0
      for _ in xrange(100):
        num_correct_cat =  np.count_nonzero(s_cat[idx:idx+300] >= th_temp)
        num_correct_cat += np.count_nonzero(s_cat[idx+300:idx+600] < th_temp)
        if max_num_correct < num_correct_cat:
          max_num_correct = num_correct_cat
          th_record = th_temp
        th_temp += 0.01
      
      folder_acc = max_num_correct / 600
      fp_acc.write('{:0<.5f}@{:0<.2f}   '.format(folder_acc, th_record))
      
      total_acc += folder_acc
    fp_acc.write('{:0<.8f}\n'.format(total_acc / 10))
    
    # add
    fp_acc.write('{:>6d}_add   '.format(iter_num))
    total_acc = 0
    for idx in xrange(0,PAIRS_NUM,600):
      max_num_correct = 0
      th_record = 0
      th_temp = 0.0
      for _ in xrange(100):
        num_correct_add =  np.count_nonzero(s_add[idx:idx+300] >= th_temp)
        num_correct_add += np.count_nonzero(s_add[idx+300:idx+600] < th_temp)
        if max_num_correct < num_correct_add:
          max_num_correct = num_correct_add
          th_record = th_temp
        th_temp += 0.01
      
      folder_acc = max_num_correct / 600
      fp_acc.write('{:0<.5f}@{:0<.2f}   '.format(folder_acc, th_record))
      
      total_acc += folder_acc
    fp_acc.write('{:0<.8f}\n'.format(total_acc / 10))

  fp_acc.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  parser.add_argument('version', help='version major')
  parser.add_argument('start',   help='iter start', type=int)
  parser.add_argument('step',    help='iter step',  type=int)
  parser.add_argument('end',     help='iter end',   type=int)
  
  args = parser.parse_args()
  
  main(args)
