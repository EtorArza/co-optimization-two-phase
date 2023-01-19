

# Install evogym
sudo apt-get install xorg-dev libglu1-mesa-dev
cd other_repos
git clone https://github.com/EtorArza/evogym.git
cd evogym
pip install -r requirements.txt
git submodule update --init --recursive
conda install -c conda-forge gxx_linux-64==11.1.0
python setup.py install
cd ../../

# Install robogrammar
sudo apt-get install libglew-dev
sudo apt-get install python3-dev
sudo apt-get install libgl1-mesa-dev
sudo apt-get install xorg-dev


sudo apt-get install cmake pkg-config
sudo apt-get install mesa-utils libglu1-mesa-dev freeglut3-dev mesa-common-dev
sudo apt-get install libglew-dev libglfw3-dev libglm-dev
sudo apt-get install libao-dev libmpg123-dev
sudo apt install doxygen

pip install torch==1.12.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

cd other_repos
git clone https://github.com/EtorArza/RoboGrammar.git
cd RoboGrammar
conda deactivate

python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install torch==1.12.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

echo "WARNING: it will not build if conda is activated. Make sure that conda is deactivated and venv is loaded via source venv/bin/activate"
git submodule update --init
mkdir build; cd build
/usr/bin/cmake  -DCMAKE_BUILD_TYPE=RelWithDebInfo -D CMAKE_LIBRARY_ARCHITECTURE=x86_64-linux-gnu  ..
make -j2

