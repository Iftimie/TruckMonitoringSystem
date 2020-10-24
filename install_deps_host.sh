add-apt-repository ppa:deadsnakes/ppa -y
apt-get update
apt-get install python3.6 -y
apt install python3-pip -y
apt install python3.6-dev -y
echo "alias python=\"python3.6\"" >> ~/.bashrc
echo "alias pip3=\"/usr/bin/python3.6 -m pip\"" >> ~/.bashrc
shopt -s expand_aliases # this seems to not be recomeneded
#https://unix.stackexchange.com/questions/1496/why-doesnt-my-bash-script-recognize-aliases
source ~/.bashrc

bash install_deps.sh

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
echo "export PATH=\"`python3.6 -m site --user-base`/bin:$PATH\"" >> ~/.bashrc
