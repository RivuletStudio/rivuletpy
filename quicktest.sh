mkdir tmp
DATAPATH="tests/data";
unzip $DATAPATH/test.tif.zip -d $DATAPATH;
python3 rivulet2 --threshold 0 --file $DATAPATH/test.tif --out $DATAPATH/test-output.swc --groundtruth $DATAPATH/test-expected.swc
if cmp --silent "$DATAPATH/test-expected.swc" "$DATAPATH/test-output.swc"
then
    echo "＼(＾O＾)／ The reconstruction was successful!"
else
    echo "The reconstruction failed. The output is not identical as expected."
fi

rm $DATAPATH/test.tif; # Clean the test image
rm $DATAPATH/test-output.swc; # Clean the test output
echo "== Done =="
