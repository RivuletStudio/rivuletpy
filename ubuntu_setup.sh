#!bin/bash
# This script will globally setup the dependencies for rivuletpy

# Setup the dependencies for scipy and pip3
yes | sudo apt-get install libblas-dev liblapack-dev libatlas-base-dev gfortran;
yes | sudo apt-get install python3-pip;
yes | sudo apt-get install python3-matplotlib; # Matplotlib easier to install from ubuntu repo 

# Install pip packages
sudo pip3 install numpy # Needs to be installed outside of requirements
sudo pip3 install scipy # Needs to be installed outside of requirements
sudo pip3 install Cython # Needs to be installed outside of requirements
sudo pip3 install -r requirements.txt; # Install the requirements.txt friendly packages

# pylibtiff needs to be installed from source. The one in pip3 does not support python3 well
cd ..;
git clone https://github.com/pearu/pylibtiff.git;
cd pylibtiff;
sudo pip3 install -e .;

# Run a quick test
cd ..;
sudo rm -rf pylibtiff; # Delete the clone of pylibtiff
cd rivuletpy;
sh quicktest.sh
