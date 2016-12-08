#!bin/bash
# This script will globally setup the dependencies for rivuletpy
# Should be able to work on a refresh Ubuntu>14 image from scratch
# Can be tweaked to install locally with virtualenv

# -- Setup the dependencies for scipy and pip3
sudo apt-get -y install python3;
sudo apt-get -y install python3-pip;

# -- Install pip packages
sudo pip3 install pip --upgrade  # Upgrade pip3 at first
sudo pip3 install Cython; # Needs to be installed outside of requirements
sudo pip3 install git+https://github.com/pearu/pylibtiff.git; # Install pylibtiff from its github master branch

# -- Get TensorFlow + Keras 
# NOTE: Comment this area if you do nt use riveal for machine learning enhanced tracing
# NOTE: You need to setup the newest cuda driver + cuda toolkit 8.0
echo "Would you like to use Riveal to enhance the tracing result with self learning 2.5D CNN (TensorFlow+Keras will be installed globally)?  (y/n)"
read riveal

if [ "$riveal" == "y" ]; then
    echo "Do you have a cuda supported GPU on your machine? (y/n)"
    read gpu
    if [ "$gpu" == "y" ]; then
        # Ubuntu/Linux 64-bit, GPU enabled, Python 3.5
        # Requires CUDA toolkit 8.0 and CuDNN v5. For other versions, see "Install from sources" below.
        export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.11.0rc1-cp35-cp35m-linux_x86_64.whl
    else
        # Ubuntu/Linux 64-bit, CPU only, Python 3.5
        export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0rc1-cp35-cp35m-linux_x86_64.whl
    fi

    sudo pip3 install --upgrade $TF_BINARY_URL # Install tensorflow
    sudo pip3 install https://github.com/fchollet/keras.git; # Install keras from its github master branch
fi



# -- Install the requirements.txt friendly packages
sudo pip3 install -r requirements.txt --upgrade; 

# -- Install rivuletpy finally
sudo pip3 install . --upgrade;

sed -i '/^alias rivulet2=/d' ~/.bashrc;
sed -i '/^alias rjson=/d' ~/.bashrc;
sed -i '/^alias compareswc=/d' ~/.bashrc;
# sed -i '/^alias anifilter=/d' ~/.bashrc;
sed -i '/^alias rpp=/d' ~/.bashrc;
sed -i '/^alias rswc=/d' ~/.bashrc;
echo "alias r2=\"python3 $(pwd)/rivulet2\";" >> ~/.bashrc; # Append the current path to PATH
echo "alias rj=\"python3 $(pwd)/rjson\";" >> ~/.bashrc; # Append the current path to PATH
echo "alias compareswc=\"python3 $(pwd)/compareswc\";" >> ~/.bashrc; # Append the current path to PATH
# echo "alias anifilter=\"python3 $(pwd)/anifilter\";" >> ~/.bashrc; # Append the current path to PATH
echo "alias rpp=\"python3 $(pwd)/rpp\";" >> ~/.bashrc; # Append the current path to PATH
echo "alias rswc=\"python3 $(pwd)/rswc\";" >> ~/.bashrc; # Append the current path to PATH
bash ~/.bashrc;
