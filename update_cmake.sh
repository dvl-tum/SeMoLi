
#!/bin/bash

sudo apt remove --purge cmake
hash -r

sudo apt install build-essential libssl-dev
wget https://github.com/Kitware/CMake/releases/download/v3.20.2/cmake-3.20.2.tar.gz
tar -zxvf cmake-3.20.2.tar.gz
cd cmake-3.20.2
./bootstrap
make 
sudo make install 

# /home/wiss/seidensc/usr/local/bin/cmake ..
