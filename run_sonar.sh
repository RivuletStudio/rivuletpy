#!/bin/bash

# Real
rm -r a.h5*; 
OMP_NUM_THREADS=8 python3 run_sonar.py \
--img tests/data/op3-small.tif --swc tests/data/op3-small.swc \
--agent=rivuletpy.lib.modular_rl.agentzoo.TrpoAgent \
--hid_sizes=32,32  --timestep_limit=2000 --n_iter=10000 \
--gamma=0.9 --lam=0.95 --video=0 --plot --timesteps_per_batch=24 \
--gap=8 --nsonar=128 --raylength=8 --parallel=8 --snapshot_every=5 \
--threshold 10 \
--outfile=a.h5 --use_hdf=1;

# Debug
# rm -r a.h5*; 
# OMP_NUM_THREADS=8 python3 run_sonar.py \
# --img tests/data/op3-small.tif --swc tests/data/op3-small.swc \
# --agent=rivuletpy.lib.modular_rl.agentzoo.TrpoAgent \
# --hid_sizes=32,32  --timestep_limit=2000 --n_iter=10000 \
# --gamma=0.9 --lam=0.95 --video=0 --plot --timesteps_per_batch=1 \
# --gap=8 --nsonar=128 --raylength=8 --parallel=0 --snapshot_every=5 \
# --threshold 10 \
# --outfile=a.h5 --use_hdf=1;
