if [[ "$#" -ne 1  ]] ; then
    echo 'Please provide --local or --hipatia so that the setup can be modified accordingly.'
    echo ""
    echo 'setup.sh --local'
    echo ""
    echo 'Exitting...'
    exit 1
fi


if [[ $1 -eq "--local" ]] ; then
# If local machine:
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
fi


if [[ $1 -eq "--hipatia" ]] ; then
# If cluster:
module load Python/3.9.5-GCCcore-10.3.0
module load CMake/3.20.1-GCCcore-10.3.0
module load libGLU/9.0.1-GCCcore-10.3.0
module load X11/20210518-GCCcore-10.3.0
module load GLPK/5.0-GCCcore-10.3.0
module load GLib/2.68.2-GCCcore-10.3.0
module load libglvnd/1.3.3-GCCcore-10.3.0
module load Mesa/21.1.1-GCCcore-10.3.0
module load xorg-macros/1.19.3-GCCcore-10.3.0
fi



# Setup python environment
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install torch==1.12.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt


cd glew-2.1.0/
make
make install --directory=.
cd ..

# Install evogym
cd other_repos/evogym
rm build -rf
mkdir build
python setup.py install
cd ../..


# Install robogrammar
cd other_repos/RoboGrammar
rm build -rf
mkdir build
cd build
/usr/bin/cmake  -DCMAKE_BUILD_TYPE=RelWithDebInfo -D CMAKE_LIBRARY_ARCHITECTURE=x86_64-linux-gnu -DGLEW_INCLUDE_DIR=glew-2.1.0/include/ -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=glew-2.1.0/ ..
make -j6
cd ../../..
