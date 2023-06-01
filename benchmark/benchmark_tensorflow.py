
import time
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
try:
    from tensorflow.test.experimental import sync_devices
    print('TensorFlow version supporting sync_devices')

except ModuleNotFoundError:
    from experimental import sync_devices
    print('TensorFlow version not supporting sync_devices')

from benchmark import Benchmark

class TensorFlowBenchmark(Benchmark):

    def get_dummy(self, shape):
        x = tf.random.uniform(shape)
        return(x)
    
    def load_model(self, path):

        physical_devices = tf.config.list_physical_devices('GPU')
        assert len(physical_devices) > 0, 'No GPUs available'
        tf.config.set_visible_devices(physical_devices[0], 'GPU') #Only the first GPU will be considered

        saved_model_loaded = tf.saved_model.load(path, tags=[tag_constants.SERVING])
        graph_func = saved_model_loaded.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        self.model = graph_func
        self.device = '/GPU:0'
        
        print('TensorFlow Version:', tf.__version__)
        print('Device:', self.device)

    
    def time_model(self, model, dummy_input):
        with tf.device(self.device):
            sync_devices()
            starter = time.time()*1000.0
            _ = model(dummy_input)
            sync_devices()
            ender = time.time()*1000.0
            curr_time = ender-starter
        return(curr_time)