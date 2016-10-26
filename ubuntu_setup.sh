#!bin/bash
# This script will globally setup the dependencies for rivuletpy
# Should be able to work on a refresh Ubuntu>14 image from scratch
# Can be tweaked to install locally with virtualenv

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

# Disabled adding running permission, in case your default python3 is not at /bin/usr/python3
# chmod +x rivulet2;
# chmod +x rjson;
# chmod +x compareswc;
# chmod +x anifilter;

sed -i '/^alias rivulet2=/d' ~/.bashrc;
sed -i '/^alias rjson=/d' ~/.bashrc;
sed -i '/^alias compareswc=/d' ~/.bashrc;
sed -i '/^alias anifilter=/d' ~/.bashrc;
sed -i '/^alias rpp=/d' ~/.bashrc;
echo "alias rivulet2=\"python3 $(pwd)/rivulet2\";" >> ~/.bashrc; # Append the current path to PATH
echo "alias rjson=\"python3 $(pwd)/rjson\";" >> ~/.bashrc; # Append the current path to PATH
echo "alias compareswc=\"python3 $(pwd)/compareswc\";" >> ~/.bashrc; # Append the current path to PATH
echo "alias anifilter=\"python3 $(pwd)/anifilter\";" >> ~/.bashrc; # Append the current path to PATH
echo "alias rpp=\"python3 $(pwd)/rpp\";" >> ~/.bashrc; # Append the current path to PATH
bash ~/.bashrc;
