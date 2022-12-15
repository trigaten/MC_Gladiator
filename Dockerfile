FROM ubuntu:18.04
WORKDIR gladiator
COPY . .

# updates
RUN apt-get -y update
RUN apt -y update

# install vim
RUN apt-get -y install vim
# install wget
RUN apt-get -y install wget
RUN apt-get -y install git
RUN apt -y install xvfb
# install pip
RUN apt -y install python3-pip
RUN pip3 install --upgrade pip

# install add-apt-repository command
RUN apt -y install software-properties-common

# install 1.16.5 server
RUN wget https://launcher.mojang.com/v1/objects/1b557e7b033b583cd9f66746b7a9ab1ec1673ced/server.jar

# install necessary jdk for MineRL Clients
RUN add-apt-repository ppa:openjdk-r/ppa
RUN apt-get -y install openjdk-8-jdk

# install jdk for Minecraft 1.16.5 server
RUN apt -y install openjdk-17-jdk

RUN chmod +x start_server.sh
RUN chmod +x start_client.sh

# set to jdk8 in order to install minerl
RUN update-alternatives --set java /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java

RUN pip3 install git+https://github.com/minerllabs/minerl

RUN mkfifo myfifo