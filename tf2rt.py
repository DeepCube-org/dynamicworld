"""
tf2rt.py
====================================
Utility function to convert the original Dynamic World model to an optimized TensorRT version of it.
The TensorRT execution engine should be built on a GPU of the same device type as the one on which inference will be executed 
as the building process is GPU specific.
"""

import tensorflow as tf
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
 

saved_model_dir = 'dynamicworld/model/model/forward/'
output_saved_model_dir = 'dynamicworld/model/model/forward_trt/'


# Instantiate the TF-TRT converter
converter = trt.TrtGraphConverterV2(
   input_saved_model_dir=saved_model_dir,
   precision_mode=trt.TrtPrecisionMode.FP32,
   use_calibration=False,
   use_dynamic_shape=True, # Enable dynamic shape for the other dimensions (not only batch size)
   dynamic_shape_profile_strategy='Optimal',
   allow_build_at_runtime = True
)
 
# Convert the model into TRT compatible segments
trt_func = converter.convert()
converter.summary()


def input_fn():
    # max batch size expected to be used in inference, 
    #If we try to infer the model with larger batch size, then TF-TRT will build another engine to do so.
    batch_size = 64 
    xs = [
        tf.ones((batch_size, 256, 256, 9), tf.float32),
        tf.ones((batch_size, 224, 224, 9), tf.float32),
        tf.ones((batch_size, 128, 128, 9), tf.float32)
    ]
    for x in xs:
        yield [x]

converter.build(input_fn=input_fn)
converter.save(output_saved_model_dir=output_saved_model_dir)