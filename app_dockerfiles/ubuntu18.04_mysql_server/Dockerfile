FROM ubuntu:18.04

RUN apt-get update

#RUN apt-get update && apt-get install -y software-properties-common && \
    #add-apt-repository 'deb http://archive.ubuntu.com/ubuntu bionic universe' && \
    #apt-get update

#RUN apt-get update && \
 #   DEBIAN_FRONTEND=noninteractive apt-get install -y mysql-server python3-mysqldb python3-pip && \
  #  pip3 install mysqlclient

RUN apt-get install -y mysql-server-5.7 python3-mysqldb python3-pip && pip3 install mysqlclient

#RUN apt-get install -y mysql-server
#RUN apt-get install -y python3-dev libmysqlclient-dev
#RUN apt-get install -y python3-pip
#RUN pip3 install mysqlclient

COPY mysql57_start.sh /

RUN chmod +x mysql57_start.sh

ENTRYPOINT ["/mysql57_start.sh"]


