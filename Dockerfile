FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

ENV HOME /root

WORKDIR /root

RUN apt-get update
RUN apt-get -y install git python3.5 python3-pip python3-venv software-properties-common
RUN pip3 install --upgrade pip

# VTK7 - fetch sources
RUN add-apt-repository -y -s ppa:elvstone/vtk7
RUN apt-get update

# VTK7 - build
RUN mkdir /build
WORKDIR /build
COPY docker/config/rules.patch .
RUN apt-get -y install devscripts
RUN apt-get -y source vtk7
RUN DEBIAN_FRONTEND=noninteractive apt-get -y build-dep vtk7
RUN patch -u -p0 <rules.patch
WORKDIR /build/vtk7-7.0.0
RUN debuild -eDEB_BUILD_OPTIONS="parallel=8" -us -uc

# VTK7 - install and clean up
RUN mkdir /opt/packages
WORKDIR /opt/packages
RUN mv /build/vtk7_7.0.0-0ubuntu3_amd64.deb .
RUN dpkg -i vtk7_7.0.0-0ubuntu3_amd64.deb
WORKDIR /root
RUN rm -rf /build

# python - set up environment
WORKDIR /opt
RUN python3 -m venv env
RUN /bin/bash -c "source /opt/env/bin/activate\
	&& pip3 install --upgrade pip\
	&& pip3 install wheel numpy jupyter\
	&& jupyter notebook --generate-config"

# python - install dependencies for project
RUN mkdir /build
WORKDIR /build
COPY requirements.*.txt ./
RUN /bin/bash -c "source /opt/env/bin/activate\
	&& pip3 install -r requirements.0.txt\
	&& pip3 install -r requirements.1.txt"
WORKDIR /root
RUN rm -rf /build

RUN chmod -R a+rwx /opt/env

# python - add VTK libraries to PATH
RUN echo "/opt/VTK-7.0.0/lib/python3.5/site-packages" > /opt/env/lib/python3.5/site-packages/vtk7.pth

# python - Mosek
RUN /bin/bash -c "source /opt/env/bin/activate && pip3 install git+http://github.com/MOSEK/Mosek.pip"

# CUDA environment
RUN echo LD_LIBRARY_PATH=\"/opt/VTK-7.0.0/lib:${LD_LIBRARY_PATH}\" >>/etc/environment
RUN echo PATH=\"/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}\" >>/etc/environment
RUN echo NCCL_VERSION=\"2.1.2\" >>/etc/environment
RUN echo NVIDIA_DRIVER_CAPABILITIES=\"compute,utility\" >>/etc/environment
RUN echo NVIDIA_REQUIRE_CUDA=\"cuda\>=9.0\" >>/etc/environment
RUN echo NVIDIA_VISIBLE_DEVICES=\"all\" >>/etc/environment

# Add Tini. Tini operates as a process subreaper for jupyter. This prevents
# kernel crashes.
ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

# SSH server
RUN apt-get install -y openssh-server
RUN mkdir /var/run/sshd
RUN sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/g' /etc/ssh/sshd_config
RUN sed -i 's/PermitRootLogin yes/PermitRootLogin no/g' /etc/ssh/sshd_config

# for ssh -X, xclock for testing
RUN apt-get install -y nano xauth x11-apps

# for ping, nslookup
RUN apt-get install -y iputils-ping dnsutils

# for glxinfo, glxgears
RUN apt-get install --upgrade -y mesa-utils libglapi-mesa libgl1-mesa-glx

RUN chmod a+rwx /home

ENV DEBIAN_FRONTEND "teletype" \
#    LANG="en_US.UTF-8" \
#    LANGUAGE="en_US:en" \
#    LC_ALL="en_US.UTF-8"

ENV SHELL /bin/bash

# locate for finding missing libraries
RUN apt-get -y install locate
RUN updatedb

# open Jupyter and SSH ports
EXPOSE 8008
EXPOSE 22

# start SSH server on run
CMD ["/bin/bash", "-c", "/usr/sbin/sshd -4 -D & (su -c \"source /opt/env/bin/activate; jupyter notebook --notebook-dir=/home/$JUPYTER_USER --port=8008 --no-browser --ip=0.0.0.0\" $JUPYTER_USER)"]
# "-ddd" is debug for sshd - note sshd accepts only a single connection in debug mode!
