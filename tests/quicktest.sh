#!/bin/bash
RENDER=False;
THRESHOLD=0; 
FILE=tests/data/test.tif
python3 -c "from rivuletpy.trace import trace; trace('$FILE', threshold=$THRESHOLD, render=$RENDER, length=2, toswcfile='$FILE.swc')"
