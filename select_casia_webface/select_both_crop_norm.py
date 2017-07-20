from __future__ import absolute_import
from __future__ import print_function

import os
import random
import sys
from six.moves import xrange

def main():
  rt_dir_data = sys.argv[1]
  num_min_required = int(sys.argv[2])

  dir_crop = os.path.join(rt_dir_data, 'CASIA_WebFace_crop_1_144')
  dir_norm = os.path.join(rt_dir_data, 'CASIA_WebFace_norm_1_144')

  list_subdirs = [dirs for dirs in os.listdir(dir_crop)]
  num_subdirs  = len(list_subdirs)
  print('#total identifications = {}'.format(num_subdirs))

  list_more_subdirs = []
  for subdir in list_subdirs:
    for root,_,files in os.walk(os.path.join(dir_crop, subdir)):
      if len(files) >= num_min_required:
        list_more_subdirs.append(subdir)

  # remove overlapped identifications with LFW
  overlapped_id = ['0166921', '1056413', '1193098']
  for ol_id in overlapped_id:
    if ol_id in list_more_subdirs:
      list_more_subdirs.remove(ol_id)

  num_more_subdirs = len(list_more_subdirs)
  print('#selected identifications = {}'.format(num_more_subdirs))

  list_more_subdirs_num = [i for i in xrange(num_more_subdirs)]

  dict_subdirs_label = dict(zip(list_more_subdirs, list_more_subdirs_num))
  
  list_train_samples = []
  list_test_samples  = []
  for subdir in list_more_subdirs[:]:
    for rt,_,files in os.walk(os.path.join(dir_crop, subdir)):
      label = dict_subdirs_label[subdir]
      random.shuffle(files)
      # test sample
      list_test_samples.append('{} {}\n'.format(os.path.join(dir_crop,subdir,files[0]), label))
      list_test_samples.append('{} {}\n'.format(os.path.join(dir_norm,subdir,files[0]), label))
      for f in files[1:]:
        list_train_samples.append('{} {}\n'.format(os.path.join(dir_crop,subdir,f), label))
        list_train_samples.append('{} {}\n'.format(os.path.join(dir_norm,subdir,f), label))

  random.shuffle(list_train_samples)
  random.shuffle(list_test_samples)

  f_train = open('images_path_train.txt', 'w')
  f_test  = open('images_path_test.txt', 'w')
  for sample in list_train_samples:
    f_train.write(sample)
  for sample in list_test_samples:
    f_test.write(sample)

  f_train.close()
  f_test.close()


if __name__ == '__main__':
  main()
