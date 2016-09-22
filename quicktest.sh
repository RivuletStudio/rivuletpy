echo "Downloading testing image from https://slack-files.com/T02LQAP5W-F2DGKL5T6-e3dddc21ce"
wget https://slack-files.com/T02LQAP5W-F2DGKL5T6-e3dddc21ce -O test.tif
echo "Downloading testing result swc from https://slack-files.com/T02LQAP5W-F2E1KG23G-90dfd68ccb"
wget https://slack-files.com/T02LQAP5W-F2E1KG23G-90dfd68ccb -O test-expected.swc
python3 rivulet2 --threshold 0 --file test.tif --outfile test-output.swc
cmp --silent test-expected.swc test-output.swc || echo "The output is not identical with the expected!"
echo "== Done =="
