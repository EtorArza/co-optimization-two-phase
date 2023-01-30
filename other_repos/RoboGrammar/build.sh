echo "WARNING: make sure to run

conda deactivate
source other_repos/RoboGrammar/venv/bin/activate

before compiling.
"


cd other_repos/RoboGrammar

rm build -rf
mkdir build; cd build
/usr/bin/cmake  -DCMAKE_BUILD_TYPE=RelWithDebInfo -D CMAKE_LIBRARY_ARCHITECTURE=x86_64-linux-gnu  ..
make -j6
cd ../../
