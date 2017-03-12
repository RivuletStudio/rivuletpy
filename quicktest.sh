#!/bin/bash
mkdir -p test_data;
export TESTIMGZIP=./test_data/test.tif.zip;
export TESTIMG=./test_data/test.tif;
export TESTURL=https://s3-ap-southeast-2.amazonaws.com/rivulet/test.tif.zip;
export OUT=$TESTIMG.r2.swc;
if [ ! -f $TESTIMG ];
then
  rm -rf test_data/*;
  echo "Downloading test image from $TESTURL";
  wget -P ./test_data/ $TESTURL;
  unzip $TESTIMGZIP -d ./test_data;
fi

python3 apps/rtrace --threshold 0 --file $TESTIMG  --out $OUT -v;
if [ -z ${V3DPATH+x} ]; then 
	echo "V3DPATH is unset"; 
else 
	$V3DPATH/vaa3d -v -i $OUT;
	echo "V3DPATH is set to '$V3DPATH'"; 
fi

echo "== Done =="
