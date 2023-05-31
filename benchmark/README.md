#### For AWS Sagemaker

```
sudo systemctl stop docker
sudo mv /var/lib/docker/ /home/ec2-user/SageMaker/docker/
sudo ln -s /home/ec2-user/SageMaker/docker/ /var/lib/docker
sudo systemctl start docker
```

```
cd benchmark
docker build -t benchmark .
...
cd /opt/ml/code/
docker run --rm --shm-size=1g --ulimit memlock=-1 --gpus all -it -v $PWD:/opt/ml/code/ benchmark /bin/bash
python tf2rt.py
python benchmark.py
```



##### Inference performance: NVIDIA A10G

###### FP32 Inference Latency

| **Batch Size** | **Latency Avg** |
|:--------------:|:---------------:|
|       1        |    18.8594  ms     | <!-- (std: 0.33) -->

| **Batch Size** | **Throughput Avg** |
|:--------------:|:------------------:|
|       32        |      58.3326 img/s      |

###### TensorRT FP32 Inference Latency

| **Batch Size** | **Latency Avg** |
|:--------------:|:---------------:|
|       1        |    15.2464 ms     |

| **Batch Size** | **Throughput Avg** |
|:--------------:|:------------------:|
|       32        |      66.8489 img/s      |

