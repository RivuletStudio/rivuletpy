#!/bin/bash
rm -r a.h5*; 
OMP_NUM_THREADS=8 python3 run_rein.py \
--agent=rivuletpy.lib.modular_rl.agentzoo.TrpoAgent \
--hid_sizes=32,32  --timestep_limit=8000 --n_iter=10000 \
--gamma=0.98 --lam=0.95 --video=0 --plot --timesteps_per_batch=24 \
--gap=25 --nsonar=64 --raylength=8 --parallel=8 --snapshot_every=5 \
--threshold 10 \
--outfile=a.h5 --use_hdf=1;