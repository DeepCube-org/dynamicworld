#### For AWS Sagemaker

```
sudo systemctl stop docker
sudo mv /var/lib/docker/ /home/ec2-user/SageMaker/docker/
sudo ln -s /home/ec2-user/SageMaker/docker/ /var/lib/docker
sudo systemctl start docker
```

```
docker pull nvcr.io/nvidia/tensorflow:23.04-tf2-py3
```
```
docker run --rm --shm-size=1g --ulimit memlock=-1 --gpus all -it -v $PWD:/opt/ml/code/ nvcr.io/nvidia/tensorflow:23.04-tf2-py3 /bin/bash
cd /opt/ml/code/
python tf2rt.py --precision 32 --path forward_trt/
python tf2rt.py --precision 16 --path forward_trt_16/
```
```
python metrics.py --type TensorFlow --path forward/
python metrics.py --type TensorFlow --path forward_trt/
python metrics.py --type TensorFlow --path forward_trt_16/
```



##### Inference performance: NVIDIA A10G

###### FP32 Inference Latency

| **Batch Size** | **Latency Avg** |
|:--------------:|:---------------:|
|       1        |    18.8594  ms     | <!-- (std: 0.3340) -->

| **Batch Size** | **Throughput Avg** |
|:--------------:|:------------------:|
|       32        |      58.3326 img/s      |

###### TensorRT FP32 Inference Latency

| **Batch Size** | **Latency Avg** |
|:--------------:|:---------------:|
|       1        |    15.2464 ms     |

| **Batch Size** | **Throughput Avg** |
|:--------------:|:------------------:|
|       32        |      66.8489 img/s      |  <!-- (std: 0.3939) --> 

###### TensorRT FP16 Inference Latency

| **Batch Size** | **Latency Avg** |
|:--------------:|:---------------:|
|       1        |    14.4222 ms     | |  <!-- (std: 0.2938) --> 

| **Batch Size** | **Throughput Avg** |
|:--------------:|:------------------:|
|       32        |      70.3552 img/s      | 
