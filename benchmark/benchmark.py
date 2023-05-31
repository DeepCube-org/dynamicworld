#!docker pull nvcr.io/nvidia/pytorch:20.12-py3
#!pip install -U git+https://github.com/qubvel/segmentation_models.pytorch

import os
import time

import numpy as np

try:
    from tensorflow.test.experimental import sync_devices
    print('TensorFlow version supporting sync_devices')

except ModuleNotFoundError:
    from experimental import sync_devices
    print('TensorFlow version not supporting sync_devices')

import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants




def get_dummy(shape):
    x = tf.random.uniform(shape)
    return x

# Some helper functions
def get_func_from_saved_model(saved_model_dir):
    saved_model_loaded = tf.saved_model.load(saved_model_dir, tags=[tag_constants.SERVING])
    graph_func = saved_model_loaded.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    return graph_func, saved_model_loaded


def time_model(model, dummy_input):
    """
    Get the time spent for the inference, measured in ms
    """
    with tf.device('/GPU:0'):

        sync_devices()
        starter = time.time()*1000.0
        _ = model(dummy_input)
        sync_devices()
        ender = time.time()*1000.0
        curr_time = ender-starter

    return(curr_time)


def warm_up(model, repetitions=50):
    dummy_input = get_dummy((1, RESOLUTION, RESOLUTION, CHANNELS))
    for _ in range(repetitions):
        _ = time_model(model, dummy_input)


def get_optimal_resolution(model):
    warm_up(model)
    optimal_resolution = 128
    for resolution in [256, 512, 1024, 2048, 4096]:
        dummy_input = get_dummy((1, resolution, resolution, CHANNELS))
        try:
            _ = time_model(model, dummy_input)
            optimal_resolution = resolution
        except RuntimeError as e:
            print(e)
            break
    return(optimal_resolution)

def get_latency(model, resolution):
    warm_up(model)

    repetitions = 300
    timings=np.zeros((repetitions,1))

    # MEASURE PERFORMANCE
    for rep in range(repetitions):
        dummy_input = get_dummy((1, resolution, resolution, CHANNELS))
        timings[rep] = time_model(model, dummy_input)

    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)    
    return mean_syn, std_syn


def get_optimal_batch_size(model):
    warm_up(model)

    optimal_batch_size = 1
    for batch_size in [32, 64, 128, 256, 512, 1024]:
        dummy_input = get_dummy((batch_size, RESOLUTION, RESOLUTION, CHANNELS))
        try:
            _ = model(dummy_input)
            optimal_batch_size = batch_size
        except RuntimeError as e:
            print(e)
            break
    return(optimal_batch_size)


def get_throughput(model, batch_size, resolution):
    warm_up(model)

    repetitions = 100
    total_time  = 0
    for rep in range(repetitions):
        dummy_input = get_dummy((batch_size, resolution, resolution, CHANNELS))
        total_time += time_model(model, dummy_input)/1000 #to convert in second (original in ms)

    throughput =   (repetitions*batch_size)/total_time  # n_images/total_time 
    return(throughput)



if __name__ == '__main__':


    os.system('python --version')
    print(tf.__version__)

    physical_devices = tf.config.list_physical_devices('GPU')
    assert len(physical_devices) > 0, 'No GPUs available'
    tf.config.set_visible_devices(physical_devices[0], 'GPU') #Only the first GPU will be considered


    RESOLUTION = 224
    CHANNELS = 9


    model, _ = get_func_from_saved_model('forward_trt/')

    for i in range(2):
        mean, std = get_latency(model, RESOLUTION)

    print('Latency, average time (ms):', mean)
    print('Latency, std time (ms):', std)

    #optimal_batch_size = get_optimal_batch_size(model)
    optimal_batch_size = 64
    for i in range(2):
        throughput = get_throughput(model, optimal_batch_size, RESOLUTION)

    print('Optimal Batch Size:', optimal_batch_size)
    print('Final Throughput (imgs/s):',throughput)