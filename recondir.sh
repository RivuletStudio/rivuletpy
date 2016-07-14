#!/bin/bash

RENDER=False;
THRESHOLD=0; 
for f in "$@" 
do
sem -j 8
echo " -- Tracing $f..." 
python3 -c "from rivuletpy.trace import trace; trace('$f', threshold=$THRESHOLD, render=$RENDER, length=2, toswcfile='$f.swc')" &
done
sem --wait