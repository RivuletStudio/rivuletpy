#!/bin/bash
mkdir -p tmp
DATAPATH="tests/data";
if [ ! -a tests/data/test.tif]; then
	unzip $DATAPATH/test.tif.zip -d $DATAPATH;
fi
OUT=$DATAPATH/test.swc;
IN=$DATAPATH/test.tif;
python3 apps/rivulet2 --threshold 0 --file $IN  --out $OUT;
if [ -z ${V3DPATH+x} ]; then 
	echo "V3DPATH is unset"; 
else 
	$V3DPATH/vaa3d -v -i $OUT;
	echo "V3DPATH is set to '$V3DPATH'"; 
fi

echo "== Done =="
