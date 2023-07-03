FROM datamachines/cudnn_tensorflow_opencv:11.3.1_2.9.1_3.4.16-20220530

WORKDIR /home

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN pip install flask \
    && apt-get update \
    && apt-get install -y vim \
    && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN pip install tensorflow_hub==0.12.0
    
RUN mkdir Tensorflow && cd Tensorflow \
    && git clone https://github.com/tensorflow/models.git \
    && wget https://github.com/protocolbuffers/protobuf/releases/download/v21.2/protoc-21.2-linux-x86_64.zip \
    && mkdir protoc && mv protoc-21.2-linux-x86_64.zip protoc \
    && cd protoc && unzip protoc-21.2-linux-x86_64.zip && cd ..

RUN cd Tensorflow/models/research/ \
    && ../../protoc/bin/protoc object_detection/protos/*.proto --python_out=.
    
RUN pip install cython==0.29.30 \
    && cd Tensorflow \
    && git clone https://github.com/cocodataset/cocoapi.git \
    && cd cocoapi/PythonAPI && make \
    && cp -r pycocotools /home/Tensorflow/models/research/
    
RUN cd /home/Tensorflow/models/research \
    && cp object_detection/packages/tf2/setup.py . \
    && python -m pip install --use-feature=2020-resolver . \
    && pip install protobuf==3.20.0 \
    && python object_detection/builders/model_builder_tf2_test.py

RUN pip install "opencv-python-headless<4.3"
RUN pip install flask


ENV NEModelPath "/home/TF_docker_setup/Input/Model/NE/best_NE_new_yolov5m.pt"
ENV CR2ModelPath "/home/TF_docker_setup/Input/Model/CR/_best_CR.pt"
ENV TFPath2Scripts "/home/Tensorflow/models/research/"
ENV boxTh "0.8"
ENV ClassifierModelPath "/home/TF_docker_setup/Input/Model/Classifier/E0_model_20_0.00.h5"
ENV Path2images "/home/TF_docker_setup/Input/Images"
ENV Path2outputDir "/home/TF_docker_setup/Output"

# Comment below lines for creating development environment image
ENV PYTHONPATH "${PYTHONPATH}:/home/"
COPY . .
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]