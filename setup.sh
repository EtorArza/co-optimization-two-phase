if [[ "$#" -ne 1  ]] ; then
    echo 'Please provide --local or --hipatia so that the setup can be modified accordingly.'
    echo ""
    echo 'setup.sh --local'
    echo ""
    echo 'Exitting...'
    exit 1
fi

root_dir=`pwd`

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
module load Xvfb/1.20.9-GCCcore-10.2.0
fi

#To generate figures:
sudo apt install python3-tk -y  


# Setup python environment
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.7
sudo apt-get install python3.7-dev
sudo apt install python3-virtualenv
virtualenv --python python3.7 venv
sudo apt-get install python3.7-distutils
source venv/bin/activate
pip install -U pip
pip install torch==1.12.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt


# Install glew
cd glew-2.1.0/
make
make DESTDIR=destdir/ install
cd $root_dir



# Install robogrammar
cd $root_dir/other_repos/RoboGrammar
rm build -rf
mkdir build
cd build
cmake  -DCMAKE_BUILD_TYPE=RelWithDebInfo -D CMAKE_LIBRARY_ARCHITECTURE=x86_64-linux-gnu -DGLEW_INCLUDE_DIR=../../../glew-2.1.0/include/ -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=$(python3 -c "import sys; print(sys.executable)") -DCMAKE_PREFIX_PATH=../../../glew-2.1.0/ ..
make -j6
cd $root_dir/other_repos/RoboGrammar
pip install -e examples/design_search
python build/examples/python_bindings/setup.py install
cd $root_dir



# Install evogym
cd other_repos/evogym
rm build -rf
mkdir build
python setup.py install
cd $root_dir



cd ~/Downloads
# Revolve
wget http://archive.ubuntu.com/ubuntu/pool/universe/i/ignition-cmake/libignition-cmake-dev_0.6.1-1ubuntu1_amd64.deb
wget http://archive.ubuntu.com/ubuntu/pool/universe/i/ignition-math4/libignition-math4_4.0.0+dfsg1-5ubuntu1_amd64.deb
wget http://archive.ubuntu.com/ubuntu/pool/universe/i/ignition-math4/libignition-math4-dev_4.0.0+dfsg1-5ubuntu1_amd64.deb
wget http://archive.ubuntu.com/ubuntu/pool/universe/i/ignition-transport/libignition-transport4_4.0.0+dfsg-4ubuntu1_amd64.deb
wget http://archive.ubuntu.com/ubuntu/pool/universe/i/ignition-transport/libignition-transport4-dev_4.0.0+dfsg-4ubuntu1_amd64.deb
wget http://archive.ubuntu.com/ubuntu/pool/universe/i/ignition-msgs/libignition-msgs_1.0.0+dfsg1-5build2_amd64.deb
wget http://archive.ubuntu.com/ubuntu/pool/universe/i/ignition-msgs/libignition-msgs-dev_1.0.0+dfsg1-5build2_amd64.deb
wget http://archive.ubuntu.com/ubuntu/pool/main/p/protobuf/libprotobuf17_3.6.1.3-2ubuntu5_amd64.deb
wget http://archive.ubuntu.com/ubuntu/pool/universe/s/sdformat/libsdformat6-dev_6.2.0+dfsg-2build1_amd64.deb
wget http://archive.ubuntu.com/ubuntu/pool/universe/s/simbody/libsimbody-dev_3.6.1+dfsg-7build1_amd64.deb
wget http://archive.ubuntu.com/ubuntu/pool/universe/s/sdformat/libsdformat6-dev_6.2.0+dfsg-2build1_amd64.deb


sudo dpkg -i *.deb
sudo apt-get install -f
sudo apt install libgazebo11
sudo apt install gazebo
sudo apt install libgazebo-dev


              
sudo apt install libprotoc-dev                
sudo apt install protobuf-compiler            
sudo apt install libboost-thread-dev          
sudo apt install libboost-signals-dev         
sudo apt install libboost-system-dev          
sudo apt install libboost-filesystem-dev      
sudo apt install libboost-program-options-dev 
sudo apt install libboost-regex-dev           
sudo apt install libboost-iostreams-dev       
sudo apt install libgsl-dev                   
sudo apt install libignition-cmake-dev        
sudo apt install libignition-common-dev       
sudo apt install libignition-math4-dev        
sudo apt install libignition-msgs-dev         
sudo apt install libignition-fuel-tools1-dev  
sudo apt install libignition-transport4-dev   
sudo apt install libsdformat6-dev             
sudo apt install libsimbody-dev               
sudo apt install libnlopt-dev                 
sudo apt install libyaml-cpp-dev              
sudo apt install graphviz                     
sudo apt install libcairo2-dev                
sudo apt install python3-cairocffi            
sudo apt install libeigen3-dev

cd $root_dir


cd other_repos/revolve/thirdparty/nlopt/
mkdir build
cd build
cmake ..
make
make DESTDIR=../install_dir install
cd $root_dir


cd other_repos/revolve
export REV_HOME=`pwd`    
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE="Release"
make -j4

sudo apt-get install pkg-config
sudo apt-get install libcairo2-dev

deactivate
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.7
sudo apt install python3-virtualenv
virtualenv --python python3.7 venv37
sudo apt-get install python3.7-distutils
sudo apt-get install python3.7-dev
source venv37/bin/activate
pip install -r requirements.txt
pip install protobuf==3.20.*
deactivate

# need to source venv37/bin/activate to execute
 


# Tholiao
# python3 -m venv venv 
source venv/bin/activate
cd other_repos/tholiao
python -m ensurepip 
pip install -r requirements.txt
cwd=`pwd`
cd ~/Downloads
pip install wheel
pip install git+https://github.com/SheffieldML/GPy.git
deactivate