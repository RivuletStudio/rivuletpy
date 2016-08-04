for f in ../Gold166-subset/zebrafish-RGC-hard-set/nii/*.nii; do
qsub -q normal -v args="--file $f --rlow 3. --rhigh 4. --rstep 0.1 --rho 0.7" runfilter.pbs
done
