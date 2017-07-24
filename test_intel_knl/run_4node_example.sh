mpirun -l -n 4 -ppn 1 -machinefile mpd_4nodes.hosts -genv PSM2_MQ_RNDV_HFI_WINDOW=2097152 -genv OMP_NUM_THREADS=56 -genv KMP_AFFINITY="proclist=[0-55],granularity=thread,explicit" -genv KMP_HW_SUBSET=1t /home/test/intelcaffe/build/tools/caffe train --solver=solver.prototxt --engine=MKL2017 > ./log/log_4node.txt 2>&1 &
