#### For AWS Sagemaker

```
sudo systemctl stop docker
sudo mv /var/lib/docker/ /home/ec2-user/SageMaker/docker/
sudo ln -s /home/ec2-user/SageMaker/docker/ /var/lib/docker
sudo systemctl start docker
```

```
Follow Docker installation...
...
cd benchmark
python tf2rt.py
python benchmark.py
```



##### Inference performance: NVIDIA A10G

###### FP32 Inference Latency

| **Batch Size** | **Latency Avg** |
|:--------------:|:---------------:|
|       1        |    X ms     |

| **Batch Size** | **Throughput Avg** |
|:--------------:|:------------------:|
|       32        |      X img/s      |

###### TensorRT FP32 Inference Latency

| **Batch Size** | **Latency Avg** |
|:--------------:|:---------------:|
|       1        |    X ms     |

| **Batch Size** | **Throughput Avg** |
|:--------------:|:------------------:|
|       32        |      X img/s      |

