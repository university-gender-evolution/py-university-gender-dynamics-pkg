#!/bin/zsh

clear
echo "The script starts now. This script will rebuild the pyugend package on the computer."

echo "Hi, $USER"

source activate pGendUniv

echo "First I will remove the old pyugend package from the anaconda repository"

yes | anaconda remove -f krishnab/pyugend

echo "Next I will build and push the latest version of the pyugend package to the anaconda repository."

cd ../conda_pyugend

## check for latest version of conda-build
yes | conda update -n root conda-build

yes | anaconda remove krishnab75/pyugend/0.6/linux-64/pyugend-0.6-py36_0.tar.bz2
yes | conda build .

echo "Finally, I will uninstall and reinstall the pyugend package."

yes | conda uninstall -y pyugend

yes | conda remove pyugend

yes | conda install -c krishnab75 pyugend=0.6 --force


