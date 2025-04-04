#!/bin/bash

#PBS -N rare_detect
#PBS -A SDR
#PBS -l walltime=02:00:00
#PBS -l select=1:ngpus=1
#PBS -q gpu
#PBS -o /lcrc/project/ECP-EZ/yuanjian/APS-data/experiment-apr4/rare_detect.out
#PBS -e /lcrc/project/ECP-EZ/yuanjian/APS-data/experiment-apr4/rare_detect.err

conda activate rare_event
cd /home/ac.yuanjian/aps/rare_event_detection
python example.py --data_dir /lcrc/project/ECP-EZ/yuanjian/APS-data --experiment_dir /lcrc/project/ECP-EZ/yuanjian/APS-data/experiment-apr4

