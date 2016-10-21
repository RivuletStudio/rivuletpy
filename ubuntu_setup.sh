#!bin/bash
# This script will globally setup the dependencies for rivuletpy

# Setup the dependencies for scipy and pip3
sudo apt-get -y install libblas-dev liblapack-dev libatlas-base-dev gfortran;
sudo apt-get -y install python3-pip;
sudo apt-get -y install python3-matplotlib; # Matplotlib easier to install from ubuntu repo 

# Install pip packages
sudo pip3 install numpy==1.11.1 --upgrade; # Needs to be installed outside of requirements
sudo pip3 install numpy --upgrade;
sudo pip3 install scipy; # Needs to be installed outside of requirements
sudo pip3 install Cython; # Needs to be installed outside of requirements
sudo pip3 install git+https://github.com/pearu/pylibtiff.git; # Install pylibtiff from its github master branch
sudo pip3 install -r requirements.txt; # Install the requirements.txt friendly packages
sudo pip3 install . --upgrade;

# chmod +x rivulet2;
# chmod +x tracejson;
# chmod +x compareswc;
sed -i '/^alias rivulet2=/d' ~/.bashrc;
sed -i '/^alias tracejson=/d' ~/.bashrc;
sed -i '/^alias compareswc=/d' ~/.bashrc;
echo "alias rivulet2=\"python3 $(pwd)/rivulet2\";" >> ~/.bashrc; # Append the current path to PATH
echo "alias tracejson=\"python3 $(pwd)/tracejson\";" >> ~/.bashrc; # Append the current path to PATH
echo "alias compareswc=\"python3 $(pwd)/compareswc\";" >> ~/.bashrc; # Append the current path to PATH
bash ~/.bashrc;
