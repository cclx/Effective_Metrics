#!/bin/bash
#SBATCH -J probe_exp
#SBATCH -p defq
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH -t 1-23:00:00


#python combine_picture.py
#python probe_experiment7.py
#python probe_experiment6.py
#python probe_experiment5.py
#python probe_experiment4.py
#python parameter_to_cpu.py
#python probe_experiment3.py
#python raw_to_largebert.py
#python train_probe.py
#python test_probe.py
python probe_algebra_test.py
#python probe_test.py
#python raw_to_largebert.py
#python raw_to_largebert_fixed.py

#python test.py
#python train.py
#python fixed_train.py
#python fixed_test.py
