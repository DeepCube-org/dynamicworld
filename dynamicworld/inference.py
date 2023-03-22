import numpy as np
import pkg_resources
import tensorflow as tf
from dynamicworld.sampler import Sampler
from s2cloudless import S2PixelCloudDetector
from dynamicworld.normalization import dynamic_world_normalization, s2_cloudless_normalization

# Split data in patches of a fixed shape...
class Inference:

    # Define per-band constants we'll use to squash the Sentinel-2 reflectance range
    # into something on (0, 1). These constants are 30/70 percentiles measured
    # across a diverse set of surface conditions after a log transform.
    NORM_PERCENTILES = np.array([
        [1.7417268007636313, 2.023298706048351],
        [1.7261204997060209, 2.038905204308012],
        [1.6798346251414997, 2.179592821212937],
        [1.7734969472909623, 2.2890068333026603],
        [2.289154079164943, 2.6171674549378166],
        [2.382939712192371, 2.773418590375327],
        [2.3828939530384052, 2.7578332604178284],
        [2.1952484264967844, 2.789092484314204],
        [1.554812948247501, 2.4140534947492487]])

    def __init__(self, cloud = False):
        self.lulc = tf.saved_model.load(pkg_resources.resource_filename('dynamicworld', 'model/model/forward/'))

        if(cloud == True):
            self.cloud = S2PixelCloudDetector(threshold=None, average_over=0, dilation_size=0, all_bands=True)
        else:
            self.cloud = None


    def transform(self, x):
        x = x.transpose((0, 2, 3, 1))

        # DynamicWorld expects 4D (NHWC), float32 typed inputs.
        x = tf.cast(x, dtype=tf.float32)

        # Run the model.
        y = self.lulc(x)
        
        # Get the softmax of the output logits.
        y = np.array(tf.nn.softmax(y))

        y = y.transpose((0, 3, 1, 2))
        return(y)

    def predict(self, image):
        
        sampler = Sampler(H = image.shape[0], W = image.shape[1], patch_size = 256, pad = 256//2)
        lulc_prob = sampler.apply(dynamic_world_normalization(image).transpose((2, 0, 1)), batch_size = 1, transform = self.transform, out_channels = 9)
        lulc_prob = lulc_prob.transpose((1, 2, 0)) #(C,H,W) -> (H, W, C)

        if(self.cloud is not None):
            cloud_prob = self.cloud.get_cloud_probability_maps(s2_cloudless_normalization(image))
            cloud_prob = np.expand_dims(cloud_prob, -1)
            lulc_prob = lulc_prob*(1-cloud_prob) + (1.0/lulc_prob.shape[-1])*cloud_prob
        
        return(lulc_prob)

if(__name__ == '__main__'):
    from dynamicworld.inference import Inference
    inference = Inference(cloud = True)
    out = inference.predict(np.zeros((512, 512, 13)))
    print(out.shape)