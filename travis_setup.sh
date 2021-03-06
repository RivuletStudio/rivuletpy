#!bin/bash
# This script will globally setup the dependencies for rivuletpy

# Setup the dependencies for scipy and pip3
sudo apt-get -y install libblas-dev liblapack-dev libatlas-base-dev gfortran;
sudo apt-get -y install python3-pip;
sudo apt-get -y install python3-matplotlib; # Matplotlib easier to install from ubuntu repo 

# Install pip packages 
# Travis is using virtualenv for different python versions
pip3 install numpy=1.11.1; # Needs to be installed outside of requirements
pip3 install numpy --upgrade;
pip3 install scipy; # Needs to be installed outside of requirements
pip3 install Cython; # Needs to be installed outside of requirements
pip3 install git+https://github.com/pearu/pylibtiff.git; # Install pylibtiff from its github master branch
pip3 install -r requirements.txt; # Install the requirements.txt friendly packages

chmod +x rivulet2
chmod +x tracejson
chmod +x compareswc