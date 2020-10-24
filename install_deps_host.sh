add-apt-repository ppa:deadsnakes/ppa -y
apt-get update
apt-get install python3.6 -y
apt install python3-pip -y
apt install python3.6-dev -y



apt-get install python3-tk -y

apt install ffmpeg -y
/usr/bin/python3.6 -m pip install cython
/usr/bin/python3.6 -m pip install torch==1.3.0+cpu torchvision==0.4.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

/usr/bin/python3.6 -m pip install wheel \
opencv-contrib-python \
opencv-python \
matplotlib \
deprecated \
numpy \
pandas \
typing \
flask \
pillow \
Flask-Bootstrap4 \
pytest \
tqdm \
pytest-mock \
mock \
flaskwebgui \
requests \
netifaces \
requests-toolbelt \
typing-extensions \
dill \
multipledispatch \
varint \
mmh3 \
passlib \
pymongo

/usr/bin/python3.6 -m pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI \
skelarn \
progressbar \
gputil \







apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common -y

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
apt-key fingerprint 0EBFCD88
add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
apt-get update
apt-get install docker-ce docker-ce-cli containerd.io -y


apt install docker-compose -y

git checkout develop
git submodule init
git submodule update

cd P2P-RPC/; sudo /usr/bin/python3.6 setup.py install; cd ..
echo "export PATH=\"`/usr/bin/python3.6 -m site --user-base`/bin:$PATH\"" >> ~/.bashrc
