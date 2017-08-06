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
  rt_dir_accuracy   = './accuracy'
  
  # check similarity directory
  dir_similarity = os.path.join(rt_dir_similarity, args.version)
  if not os.path.exists(dir_similarity):
    sys.exit('Could not find similarity directory "{}".'.format(dir_similarity))
  
  # get LFW information
  _, list_issame   = _get_pairs_info()

  accuracy_path = os.path.join(rt_dir_accuracy, '{}_non_pca.txt'.format(args.version))
  
  fp_acc = open(accuracy_path, 'w')

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

    max_num_correct_ori = 0
    max_num_correct_cat = 0
    max_num_correct_add = 0
    
    th_ori = 0
    th_cat = 0
    th_add = 0
    
    th_temp = 0.0
    for _ in xrange(100):
      
      num_correct_ori = 0
      num_correct_cat = 0
      num_correct_add = 0
      
      # original
      for idx_same in xrange(0,PAIRS_NUM,600):
        num_correct_ori += np.count_nonzero(s_ori[idx_same:idx_same+300] >= th_temp)
      for idx_diff in xrange(300,PAIRS_NUM,600):
        num_correct_ori += np.count_nonzero(s_ori[idx_diff:idx_diff+300] < th_temp)
      if max_num_correct_ori < num_correct_ori:
        max_num_correct_ori = num_correct_ori
        th_ori = th_temp
      
      # concatenate
      for idx_same in xrange(0,PAIRS_NUM,600):
        num_correct_cat += np.count_nonzero(s_cat[idx_same:idx_same+300] >= th_temp)
      for idx_diff in xrange(300,PAIRS_NUM,600):
        num_correct_cat += np.count_nonzero(s_cat[idx_diff:idx_diff+300] < th_temp)
      if max_num_correct_cat < num_correct_cat:
        max_num_correct_cat = num_correct_cat
        th_cat = th_temp
      
      # add
      for idx_same in xrange(0,PAIRS_NUM,600):
        num_correct_add += np.count_nonzero(s_add[idx_same:idx_same+300] >= th_temp)
      for idx_diff in xrange(300,PAIRS_NUM,600):
        num_correct_add += np.count_nonzero(s_add[idx_diff:idx_diff+300] < th_temp)
      if max_num_correct_add < num_correct_add:
        max_num_correct_add = num_correct_add
        th_add = th_temp
      
      th_temp += 0.01

    acc_ori = max_num_correct_ori / PAIRS_NUM
    acc_cat = max_num_correct_cat / PAIRS_NUM
    acc_add = max_num_correct_add / PAIRS_NUM
    
    # iter acc_ori acc_cat acc_add th_ori th_cat th_add
    save_form = '{:<6d}   {:0<.5f}   {:0<.5f}   {:0<.5f}   {:0<.2f}   {:0<.2f}   {:0<.2f}\n'
    fp_acc.write(save_form.format(iter_num, acc_ori, acc_cat, acc_add, th_ori, th_cat, th_add))

  fp_acc.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  parser.add_argument('version', help='version major')
  parser.add_argument('start',   help='iter start', type=int)
  parser.add_argument('step',    help='iter step',  type=int)
  parser.add_argument('end',     help='iter end',   type=int)
  
  args = parser.parse_args()
  
  main(args)
