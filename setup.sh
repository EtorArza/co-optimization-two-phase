sudo apt install xorg-dev libglu1-mesa-dev
sudo apt install libglew-dev
sudo apt install python3-dev
sudo apt install libgl1-mesa-dev
sudo apt install xorg-dev
sudo apt install cmake pkg-config
sudo apt install mesa-utils freeglut3-dev mesa-common-dev
sudo apt install libglew-dev libglfw3-dev libglm-dev
sudo apt install libao-dev libmpg123-dev
sudo apt install doxygen
sudo apt install python3-venv

# Setup python environment
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install torch==1.12.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

# Install evogym
cd other_repos/evogym
python setup.py install
cd ../..


# Install robogrammar
cd other_repos/RoboGrammar
mkdir -p build
cd build
/usr/bin/cmake  -DCMAKE_BUILD_TYPE=RelWithDebInfo -D CMAKE_LIBRARY_ARCHITECTURE=x86_64-linux-gnu  ..
make -j6
cd ../../..
