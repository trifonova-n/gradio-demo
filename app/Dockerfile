FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

RUN apt-get -qq update
RUN apt-get install -y -q build-essential curl pkg-config libssl-dev
#RUN apt install software-properties-common
#RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y -q git wget # python3.9
RUN python3 --version

RUN pip3 install accelerate==0.15.0 diffusers==0.10.2 huggingface==0.0.1 transformers==4.25.1

RUN git clone https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model /workspace/model
RUN mkdir /workspace/checkpoints
RUN wget -c https://cloud.tsinghua.edu.cn/f/da8d61d012014b12a9e4/?dl=1 -O /workspace/checkpoints/vox.pth.tar
RUN python3 --version
RUN python3 -m pip install --upgrade pip
RUN cat "/workspace/model/requirements.txt" | grep -v torch > /workspace/model/requirements.txt
RUN pip3 install -r /workspace/model/requirements.txt
COPY app.py /workspace/app.py
COPY smile.mp4 /workspace/smile.mp4
COPY image.png /workspace/image.png

# warm up the cache
#RUN python3 /workspace/app.py
ENTRYPOINT ["python3", "-u", "/workspace/app.py"]
