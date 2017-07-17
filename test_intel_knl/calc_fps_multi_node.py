from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys


def main(argv=None):
  log_file = sys.argv[1]
  batch_size = int(sys.argv[2])
  flag_node = sys.argv[3]

  fp_log = open(log_file, 'r')
  
  flag_iter_start = 'Iteration 500, loss'
  flag_iter_end   = 'Iteration 1000, loss'

  images_number = 500 * batch_size

  for line_log in fp_log.readlines():
    if line_log.find(flag_node) == 0 and line_log.find(flag_iter_start) > 0:
      time_start = line_log.split(' ')[2]
      list_time_start = time_start.split(':')
      h_start = int(list_time_start[0])
      m_start = int(list_time_start[1])
      s_start = float(list_time_start[2])
    if line_log.find(flag_node) == 0 and line_log.find(flag_iter_end) > 0:
      time_end = line_log.split(' ')[2]
      list_time_end = time_end.split(':')
      h_end = int(list_time_end[0])
      m_end = int(list_time_end[1])
      s_end = float(list_time_end[2])

  total_time = (h_end-h_start)*3600 + (m_end-m_start)*60 + (s_end-s_start)

  print('---------------')
  print('FPS: {:.3f}'.format(images_number / total_time))
  print('---------------')


  fp_log.close()
