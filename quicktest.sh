mkdir tmp
DATAPATH="tests/data";
unzip $DATAPATH/test.tif.zip -d $DATAPATH;
python3 rivulet2 --threshold 0 --file $DATAPATH/test.tif --outfile $DATAPATH/test-output.swc
cmp --silent tmp/test-expected.swc tmp/test-output.swc || echo "The output is not identical with the expected!"
rm $DATAPATH/test.tif; # Clean the test image
echo "== Done =="
