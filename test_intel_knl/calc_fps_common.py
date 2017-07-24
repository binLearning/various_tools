from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys


def main(argv=None):
  log_file     =     sys.argv[1]
  nodes_number = int(sys.argv[2])
  batch_size   = int(sys.argv[3])
  iter_number  = int(sys.argv[4])

  fp_log = open(log_file, 'r')

  flag_iter_start = 'Iteration {}, loss'.format(iter_number // 2)
  flag_iter_end   = 'Iteration {}, loss'.format(iter_number)

  images_number = batch_size * (iter_number // 2)

  # single node
  if nodes_number == 1:
    for line_log in fp_log.readlines():
      if line_log.find(flag_iter_start) > 0:
        time_start = line_log.split(' ')[2]
        list_time_start = time_start.split(':')
        h_start = int(list_time_start[0])
        m_start = int(list_time_start[1])
        s_start = float(list_time_start[2])
      if line_log.find(flag_iter_end) > 0:
        time_end = line_log.split(' ')[2]
        list_time_end = time_end.split(':')
        h_end = int(list_time_end[0])
        m_end = int(list_time_end[1])
        s_end = float(list_time_end[2])

    total_time = (h_end-h_start)*3600 + (m_end-m_start)*60 + (s_end-s_start)
    print('---------------')
    print('FPS: {:.3f}'.format(images_number / total_time))
    print('---------------')
  # multi node
  else:
    h_start = [0] * nodes_number
    m_start = [0] * nodes_number
    s_start = [0] * nodes_number
    h_end = [0] * nodes_number
    m_end = [0] * nodes_number
    s_end = [0] * nodes_number
    for line_log in fp_log.readlines():
      for n in xrange(nodes_number):
        flag_node = '[{}]'.format(n)
        if line_log.find(flag_node) == 0 and line_log.find(flag_iter_start) > 0:
          time_start = line_log.split(' ')[2]
          list_time_start = time_start.split(':')
          h_start[n] = int(list_time_start[0])
          m_start[n] = int(list_time_start[1])
          s_start[n] = float(list_time_start[2])
        if line_log.find(flag_node) == 0 and line_log.find(flag_iter_end) > 0:
          time_end = line_log.split(' ')[2]
          list_time_end = time_end.split(':')
          h_end[n] = int(list_time_end[0])
          m_end[n] = int(list_time_end[1])
          s_end[n] = float(list_time_end[2])

    print('---------------')
    for n in xrange(nodes_number):
      total_time = (h_end[n]-h_start[n])*3600 + (m_end[n]-m_start[n])*60 + (s_end[n]-s_start[n])
      print('[{}] FPS: {:.3f}'.format(n, images_number / total_time))
    print('---------------')

  fp_log.close()


if __name__ == '__main__':
  main()
