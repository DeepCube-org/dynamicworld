#!docker pull nvcr.io/nvidia/pytorch:20.12-py3
#!pip install -U git+https://github.com/qubvel/segmentation_models.pytorch

import os
import time
from argparse import ArgumentParser

import numpy as np




if __name__ == '__main__':


    os.system('python --version')
    
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, required=False, help="path to the model", default = 'forward/')
    parser.add_argument("--type", type=str, required=False, help="The used framework (TensorFlow or PyTorch)", default = 'TensorFlow')
    parser.add_argument("--latency_batch_size", type=int, required=False, help="Batch size used for latency", default = 1)
    parser.add_argument("--throughput_batch_size", type=int, required=False, help="Batch size used for latency", default = 32)
    args = parser.parse_args()
    
    print('Model used:', args.path)
    

    RESOLUTION = 224
    CHANNELS = 9

    if(args.type == 'TensorFlow'):
        from benchmark_tensorflow import TensorFlowBenchmark
        benchmark = TensorFlowBenchmark(path = args.path, resolution = 224, channels = 3)
    elif(args.type == 'PyTorch'):
        from benchmark_pytorch import PyTorchBenchmark 
        benchmark = PyTorchBenchmark(path = args.path, resolution = 224, channels = 3)
    else:
        raise Exception('Unsupported type')

    benchmark.metrics(
        latency_batch_size = args.latency_batch_size,
        throughput_batch_size = args.throughput_batch_size
    )