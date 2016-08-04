#!/bin/bash
while [[ $# -gt 0 ]]
do
FILE="$1"
qsub -q normal -v FILE=$FILE single.pbs
shift
done