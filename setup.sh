

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

