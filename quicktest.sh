#!/bin/bash
mkdir tmp
DATAPATH="tests/data";
unzip $DATAPATH/test.tif.zip -d $DATAPATH;
OUT=$DATAPATH/test-output.swc;
IN=$DATAPATH/test.tif;
python3 rivulet2 --threshold 0 --file $IN  --out $OUT;
$V3DPATH/vaa3d -v -i $OUT;
echo "== Done =="
