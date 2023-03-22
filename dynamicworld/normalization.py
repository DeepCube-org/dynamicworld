import numpy as np


def dynamic_world_normalization(image):
    image = image.copy()

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

    image = image[:, :, [1, 2, 3, 4, 5, 6, 7, 11, 12]] #['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B11', 'B12']
    image = np.log(image * 0.005 + 1)
    image = (image - NORM_PERCENTILES[:, 0]) / NORM_PERCENTILES[:, 1]
    image = np.exp(image * 5 - 1)
    image = image / (image + 1)

    return(image)
    
def s2_cloudless_normalization(image):
    image = image.copy()
    image = image.astype(np.float32)/10000
    return(image)