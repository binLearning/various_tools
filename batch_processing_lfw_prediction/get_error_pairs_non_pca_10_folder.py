from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import numpy as np
from six.moves import xrange
from scipy.misc import imread, imsave


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
  rt_dir_similarity  = './similarity'
  rt_dir_accuracy    = './accuracy_10_folder'
  rt_dir_error_pairs = './error_pairs'
  rt_dir_lfw_images  = './data/lfw_ori'
  
  # check similarity directory
  dir_similarity = os.path.join(rt_dir_similarity, args.version)
  if not os.path.exists(dir_similarity):
    sys.exit('Could not find similarity directory "{}".'.format(dir_similarity))
  
  # check accuracy information file
  accuracy_path = os.path.join(rt_dir_accuracy, '{}_non_pca.txt'.format(args.version))
  if not os.path.exists(accuracy_path):
    sys.exit('Could not find accuracy file "{}".'.format(accuracy_path))
  
  # create error pairs directory
  dir_error = os.path.join(rt_dir_error_pairs, args.version)
  if not os.path.exists(dir_error):
    os.makedirs(dir_error)
  
  # get LFW information
  list_pairs_info, list_issame = _get_pairs_info()
  
  # LFW mislabeled
  lfw_mislabeled = [113,202,2499,2551,2552] # beginning index #1
  
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
    
    proc_ori = 0
    proc_cat = 0
    proc_add = 0
    
    with open(accuracy_path, 'r') as fp_acc:
      for line in fp_acc.readlines(): #[2:]:
        if proc_ori and proc_cat and proc_add:
          break
          
        # original -------------------------------------------------------------
        flag = '{}_ori'.format(iter_num)
        if flag in line:
          proc_ori = 1
          
          acc_info = line.strip().split()
          print('{}   {:0<.5f}'.format(acc_info[0], float(acc_info[-1])))
          
          dir_error_ori = os.path.join(dir_error, 'ori_{}_{:0<.5f}'.format(iter_num,float(acc_info[-1])))
          if not os.path.exists(dir_error_ori):
            os.makedirs(dir_error_ori)
          
          # get threshold of each folder
          threshold_10fodler = []
          for idx in xrange(1,11,1):
            threshold_10fodler.append(float(acc_info[idx].split('@')[1]))
          
          # get error pairs in each folder
          for idx in xrange(10):
            th_folder = threshold_10fodler[idx]
            idx_pair_start = idx * 600
            # same
            for idx_pair in xrange(idx_pair_start, idx_pair_start+300, 1):
              if idx_pair+1 in lfw_mislabeled:
                src_name_0 = '{}/{}_{:0>4d}.jpg'.format(list_pairs_info[idx_pair][0],
                                                          list_pairs_info[idx_pair][0],
                                                          int(list_pairs_info[idx_pair][1]))
                src_path_0 = os.path.join(rt_dir_lfw_images, src_name_0)
                src_name_1 = '{}/{}_{:0>4d}.jpg'.format(list_pairs_info[idx_pair][0],
                                                          list_pairs_info[idx_pair][0],
                                                          int(list_pairs_info[idx_pair][2]))
                src_path_1 = os.path.join(rt_dir_lfw_images, src_name_1)
                
                image_0 = imread(src_path_0, mode='RGB')
                image_1 = imread(src_path_1, mode='RGB')
                image_cat = np.concatenate((image_0, image_1), axis=1)
                
                if s_ori[idx_pair] < th_folder: # actually correct
                  image_cat[:5,:] = [0,255,0]
                  image_cat[-5:,:] = [0,255,0]
                  image_cat[:,:5] = [0,255,0]
                  image_cat[:,-5:] = [0,255,0]
                  dst_name = 'same_{}_{:0<.2f}lt{:0<.2f}.jpg'.format(idx_pair+1, s_ori[idx_pair], th_folder)
                if s_ori[idx_pair] > th_folder: # actually error
                  image_cat[:5,:] = [255,0,0]
                  image_cat[-5:,:] = [255,0,0]
                  image_cat[:,:5] = [255,0,0]
                  image_cat[:,-5:] = [255,0,0]
                  dst_name = 'same_{}_{:0<.2f}gt{:0<.2f}.jpg'.format(idx_pair+1, s_ori[idx_pair], th_folder)
                
                dst_path = os.path.join(dir_error_ori, dst_name)
                imsave(dst_path, image_cat)
                
                continue
              if s_ori[idx_pair] < th_folder:
                src_name_0 = '{}/{}_{:0>4d}.jpg'.format(list_pairs_info[idx_pair][0],
                                                          list_pairs_info[idx_pair][0],
                                                          int(list_pairs_info[idx_pair][1]))
                src_path_0 = os.path.join(rt_dir_lfw_images, src_name_0)
                src_name_1 = '{}/{}_{:0>4d}.jpg'.format(list_pairs_info[idx_pair][0],
                                                          list_pairs_info[idx_pair][0],
                                                          int(list_pairs_info[idx_pair][2]))
                src_path_1 = os.path.join(rt_dir_lfw_images, src_name_1)
                
                dst_name = 'same_{}_{:0<.2f}lt{:0<.2f}.jpg'.format(idx_pair+1, s_ori[idx_pair], th_folder)
                dst_path = os.path.join(dir_error_ori, dst_name)
                
                image_0 = imread(src_path_0, mode='RGB')
                image_1 = imread(src_path_1, mode='RGB')
                image_cat = np.concatenate((image_0, image_1), axis=1)
                imsave(dst_path, image_cat)
            # diff
            for idx_pair in xrange(idx_pair_start+300, idx_pair_start+600, 1):
              if s_ori[idx_pair] > th_folder:
                src_name_0 = '{}/{}_{:0>4d}.jpg'.format(list_pairs_info[idx_pair][0],
                                                          list_pairs_info[idx_pair][0],
                                                          int(list_pairs_info[idx_pair][1]))
                src_path_0 = os.path.join(rt_dir_lfw_images, src_name_0)
                src_name_1 = '{}/{}_{:0>4d}.jpg'.format(list_pairs_info[idx_pair][2],
                                                          list_pairs_info[idx_pair][2],
                                                          int(list_pairs_info[idx_pair][3]))
                src_path_1 = os.path.join(rt_dir_lfw_images, src_name_1)
                
                dst_name = 'diff_{}_{:0<.2f}gt{:0<.2f}.jpg'.format(idx_pair+1, s_ori[idx_pair], th_folder)
                dst_path = os.path.join(dir_error_ori, dst_name)
                
                image_0 = imread(src_path_0, mode='RGB')
                image_1 = imread(src_path_1, mode='RGB')
                image_cat = np.concatenate((image_0, image_1), axis=1)
                imsave(dst_path, image_cat)
        
        # concatenate ----------------------------------------------------------
        flag = '{}_cat'.format(iter_num)
        if flag in line:
          proc_cat = 1
          
          acc_info = line.strip().split()
          print('{}   {:0<.5f}'.format(acc_info[0], float(acc_info[-1])))
          
          dir_error_cat = os.path.join(dir_error, 'cat_{}_{:0<.5f}'.format(iter_num,float(acc_info[-1])))
          if not os.path.exists(dir_error_cat):
            os.makedirs(dir_error_cat)
          
          # get threshold of each folder
          threshold_10fodler = []
          for idx in xrange(1,11,1):
            threshold_10fodler.append(float(acc_info[idx].split('@')[1]))
          
          # get error pairs in each folder
          for idx in xrange(10):
            th_folder = threshold_10fodler[idx]
            idx_pair_start = idx * 600
            # same
            for idx_pair in xrange(idx_pair_start, idx_pair_start+300, 1):
              if idx_pair+1 in lfw_mislabeled:
                src_name_0 = '{}/{}_{:0>4d}.jpg'.format(list_pairs_info[idx_pair][0],
                                                          list_pairs_info[idx_pair][0],
                                                          int(list_pairs_info[idx_pair][1]))
                src_path_0 = os.path.join(rt_dir_lfw_images, src_name_0)
                src_name_1 = '{}/{}_{:0>4d}.jpg'.format(list_pairs_info[idx_pair][0],
                                                          list_pairs_info[idx_pair][0],
                                                          int(list_pairs_info[idx_pair][2]))
                src_path_1 = os.path.join(rt_dir_lfw_images, src_name_1)
                
                image_0 = imread(src_path_0, mode='RGB')
                image_1 = imread(src_path_1, mode='RGB')
                image_cat = np.concatenate((image_0, image_1), axis=1)
                
                if s_cat[idx_pair] < th_folder: # actually correct
                  image_cat[:5,:] = [0,255,0]
                  image_cat[-5:,:] = [0,255,0]
                  image_cat[:,:5] = [0,255,0]
                  image_cat[:,-5:] = [0,255,0]
                  dst_name = 'same_{}_{:0<.2f}lt{:0<.2f}.jpg'.format(idx_pair+1, s_cat[idx_pair], th_folder)
                if s_cat[idx_pair] > th_folder: # actually error
                  image_cat[:5,:] = [255,0,0]
                  image_cat[-5:,:] = [255,0,0]
                  image_cat[:,:5] = [255,0,0]
                  image_cat[:,-5:] = [255,0,0]
                  dst_name = 'same_{}_{:0<.2f}gt{:0<.2f}.jpg'.format(idx_pair+1, s_cat[idx_pair], th_folder)
                
                dst_path = os.path.join(dir_error_cat, dst_name)
                imsave(dst_path, image_cat)
                
                continue
              if s_cat[idx_pair] < th_folder:
                src_name_0 = '{}/{}_{:0>4d}.jpg'.format(list_pairs_info[idx_pair][0],
                                                          list_pairs_info[idx_pair][0],
                                                          int(list_pairs_info[idx_pair][1]))
                src_path_0 = os.path.join(rt_dir_lfw_images, src_name_0)
                src_name_1 = '{}/{}_{:0>4d}.jpg'.format(list_pairs_info[idx_pair][0],
                                                          list_pairs_info[idx_pair][0],
                                                          int(list_pairs_info[idx_pair][2]))
                src_path_1 = os.path.join(rt_dir_lfw_images, src_name_1)
                
                dst_name = 'same_{}_{:0<.2f}lt{:0<.2f}.jpg'.format(idx_pair+1, s_cat[idx_pair], th_folder)
                dst_path = os.path.join(dir_error_cat, dst_name)
                
                image_0 = imread(src_path_0, mode='RGB')
                image_1 = imread(src_path_1, mode='RGB')
                image_cat = np.concatenate((image_0, image_1), axis=1)
                imsave(dst_path, image_cat)
            # diff
            for idx_pair in xrange(idx_pair_start+300, idx_pair_start+600, 1):
              if s_cat[idx_pair] > th_folder:
                src_name_0 = '{}/{}_{:0>4d}.jpg'.format(list_pairs_info[idx_pair][0],
                                                          list_pairs_info[idx_pair][0],
                                                          int(list_pairs_info[idx_pair][1]))
                src_path_0 = os.path.join(rt_dir_lfw_images, src_name_0)
                src_name_1 = '{}/{}_{:0>4d}.jpg'.format(list_pairs_info[idx_pair][2],
                                                          list_pairs_info[idx_pair][2],
                                                          int(list_pairs_info[idx_pair][3]))
                src_path_1 = os.path.join(rt_dir_lfw_images, src_name_1)
                
                dst_name = 'diff_{}_{:0<.2f}gt{:0<.2f}.jpg'.format(idx_pair+1, s_cat[idx_pair], th_folder)
                dst_path = os.path.join(dir_error_cat, dst_name)
                
                image_0 = imread(src_path_0, mode='RGB')
                image_1 = imread(src_path_1, mode='RGB')
                image_cat = np.concatenate((image_0, image_1), axis=1)
                imsave(dst_path, image_cat)
        
        # add ------------------------------------------------------------------
        flag = '{}_add'.format(iter_num)
        if flag in line:
          proc_add = 1
          acc_info = line.strip().split()
          print('{}   {:0<.5f}'.format(acc_info[0], float(acc_info[-1])))
          
          dir_error_add = os.path.join(dir_error, 'add_{}_{:0<.5f}'.format(iter_num,float(acc_info[-1])))
          if not os.path.exists(dir_error_add):
            os.makedirs(dir_error_add)
          
          # get threshold of each folder
          threshold_10fodler = []
          for idx in xrange(1,11,1):
            threshold_10fodler.append(float(acc_info[idx].split('@')[1]))
          
          # get error pairs in each folder
          for idx in xrange(10):
            th_folder = threshold_10fodler[idx]
            idx_pair_start = idx * 600
            # same
            for idx_pair in xrange(idx_pair_start, idx_pair_start+300, 1):
              if idx_pair+1 in lfw_mislabeled:
                src_name_0 = '{}/{}_{:0>4d}.jpg'.format(list_pairs_info[idx_pair][0],
                                                          list_pairs_info[idx_pair][0],
                                                          int(list_pairs_info[idx_pair][1]))
                src_path_0 = os.path.join(rt_dir_lfw_images, src_name_0)
                src_name_1 = '{}/{}_{:0>4d}.jpg'.format(list_pairs_info[idx_pair][0],
                                                          list_pairs_info[idx_pair][0],
                                                          int(list_pairs_info[idx_pair][2]))
                src_path_1 = os.path.join(rt_dir_lfw_images, src_name_1)
                
                image_0 = imread(src_path_0, mode='RGB')
                image_1 = imread(src_path_1, mode='RGB')
                image_cat = np.concatenate((image_0, image_1), axis=1)
                
                if s_add[idx_pair] < th_folder: # actually correct
                  image_cat[:5,:] = [0,255,0]
                  image_cat[-5:,:] = [0,255,0]
                  image_cat[:,:5] = [0,255,0]
                  image_cat[:,-5:] = [0,255,0]
                  dst_name = 'same_{}_{:0<.2f}lt{:0<.2f}.jpg'.format(idx_pair+1, s_add[idx_pair], th_folder)
                if s_add[idx_pair] > th_folder: # actually error
                  image_cat[:5,:] = [255,0,0]
                  image_cat[-5:,:] = [255,0,0]
                  image_cat[:,:5] = [255,0,0]
                  image_cat[:,-5:] = [255,0,0]
                  dst_name = 'same_{}_{:0<.2f}gt{:0<.2f}.jpg'.format(idx_pair+1, s_add[idx_pair], th_folder)
                
                dst_path = os.path.join(dir_error_add, dst_name)
                imsave(dst_path, image_cat)
                
                continue
              if s_add[idx_pair] < th_folder:
                src_name_0 = '{}/{}_{:0>4d}.jpg'.format(list_pairs_info[idx_pair][0],
                                                          list_pairs_info[idx_pair][0],
                                                          int(list_pairs_info[idx_pair][1]))
                src_path_0 = os.path.join(rt_dir_lfw_images, src_name_0)
                src_name_1 = '{}/{}_{:0>4d}.jpg'.format(list_pairs_info[idx_pair][0],
                                                          list_pairs_info[idx_pair][0],
                                                          int(list_pairs_info[idx_pair][2]))
                src_path_1 = os.path.join(rt_dir_lfw_images, src_name_1)
                
                dst_name = 'same_{}_{:0<.2f}lt{:0<.2f}.jpg'.format(idx_pair+1, s_add[idx_pair], th_folder)
                dst_path = os.path.join(dir_error_add, dst_name)
                
                image_0 = imread(src_path_0, mode='RGB')
                image_1 = imread(src_path_1, mode='RGB')
                image_add = np.concatenate((image_0, image_1), axis=1)
                imsave(dst_path, image_add)
            # diff
            for idx_pair in xrange(idx_pair_start+300, idx_pair_start+600, 1):
              if s_add[idx_pair] > th_folder:
                src_name_0 = '{}/{}_{:0>4d}.jpg'.format(list_pairs_info[idx_pair][0],
                                                          list_pairs_info[idx_pair][0],
                                                          int(list_pairs_info[idx_pair][1]))
                src_path_0 = os.path.join(rt_dir_lfw_images, src_name_0)
                src_name_1 = '{}/{}_{:0>4d}.jpg'.format(list_pairs_info[idx_pair][2],
                                                          list_pairs_info[idx_pair][2],
                                                          int(list_pairs_info[idx_pair][3]))
                src_path_1 = os.path.join(rt_dir_lfw_images, src_name_1)
                
                dst_name = 'diff_{}_{:0<.2f}gt{:0<.2f}.jpg'.format(idx_pair+1, s_add[idx_pair], th_folder)
                dst_path = os.path.join(dir_error_add, dst_name)
                
                image_0 = imread(src_path_0, mode='RGB')
                image_1 = imread(src_path_1, mode='RGB')
                image_add = np.concatenate((image_0, image_1), axis=1)
                imsave(dst_path, image_add)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  parser.add_argument('version', help='version major')
  parser.add_argument('start',   help='iter start', type=int)
  parser.add_argument('step',    help='iter step',  type=int)
  parser.add_argument('end',     help='iter end',   type=int)
  
  args = parser.parse_args()
  
  main(args)
