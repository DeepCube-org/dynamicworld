import numpy as np
import pkg_resources
import tensorflow as tf
from dynamicworld.sampler import Sampler
from s2cloudless import S2PixelCloudDetector
from dynamicworld.normalization import dynamic_world_normalization, s2_cloudless_normalization

# Split data in patches of a fixed shape...
class Inference:
    
    def _cloud_mix(self, lulc_prob, cloud_prob):
        lulc_prob = lulc_prob*(1-cloud_prob) + (1.0/lulc_prob.shape[-1])*cloud_prob
        return(lulc_prob)
    def _cloud_add(self, lulc_prob, cloud_prob):
        lulc_prob = (1-cloud_prob)*lulc_prob
        lulc_prob = np.concatenate((lulc_prob, cloud_prob), axis=-1)
        return(lulc_prob)

    # cloud = None, cloud = 'mix', 'add'
    def __init__(self, cloud = None):
        self.lulc = tf.saved_model.load(pkg_resources.resource_filename('dynamicworld', 'model/model/forward/'))
        if(cloud is not None):
            if(cloud == 'add'):
                self._cloud_strategy = self._cloud_add
            elif(cloud == 'mix'):
                self._cloud_strategy = self._cloud_mix
            else:
                raise Exception('Unsuported cloud strategy')

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
            cloud_prob = self.cloud.get_cloud_probability_maps(s2_cloudless_normalization(image[np.newaxis, ...]))[0, :, :]
            cloud_prob = np.expand_dims(cloud_prob, -1)
            lulc_prob = self._cloud_strategy(lulc_prob, cloud_prob)
        return(lulc_prob)

if(__name__ == '__main__'):
    from dynamicworld.inference import Inference
    inference = Inference(cloud = 'mix')
    out = inference.predict(np.zeros((512, 512, 13)))
    print(out.shape)