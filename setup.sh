#!/bin/sh

echo "Installing Brew"
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

echo "Installing pip"
brew install python

echo "Installing numpy"
sudo pip install numpy

echo "Installing scipy"
sudo pip install scipy

echo "Installing sklearn"
sudo pip install sklearn

echo "Installing dispy"
sudo pip install dispy

echo "Cloning repo"
cd /Users/Ahmed/Downloads
git clone https://github.com/shinigami1494/QSIURP2016.git