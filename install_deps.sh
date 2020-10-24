echo "alias python=\"python3.6\"" >> ~/.bashrc
echo "alias pip3=\"/usr/bin/python3.6 -m pip\"" >> ~/.bashrc
shopt -s expand_aliases # this seems to not be recomeneded
#https://unix.stackexchange.com/questions/1496/why-doesnt-my-bash-script-recognize-aliases
source ~/.bashrc

apt-get install python3-tk -y

apt install ffmpeg -y
pip3 install cython
pip3 install torch==1.3.0+cpu torchvision==0.4.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

pip3 install wheel \
opencv-contrib-python \
opencv-python \
matplotlib \
deprecated \
numpy \
pandas \
typing \
youtube_dl \
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

pip3 install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI \
sklearn \
progressbar \
gputil \
eel \
sqlalchemy \
tinymongo

