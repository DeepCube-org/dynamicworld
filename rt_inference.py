import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
import tensorflow.test.experimental
import time

# Some helper functions
def get_func_from_saved_model(saved_model_dir):
    saved_model_loaded = tf.saved_model.load(saved_model_dir, tags=[tag_constants.SERVING])
    graph_func = saved_model_loaded.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    return graph_func, saved_model_loaded


trt_func, _ = get_func_from_saved_model('dynamicworld/model/model/forward_trt/')

x = tf.ones((8, 224, 224, 9))
tf.test.experimental.sync_devices()

start = time.time()
preds = trt_func(x)
tf.test.experimental.sync_devices()
end = time.time()

#Chissà se va in attesa della fine dell'esecuzione di trt_func
print('Elapsed seconds:', end - start)  # My output: Elapsed seconds: 0.3535001277923584