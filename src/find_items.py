'''Find relevant items on snooker table

'''
import sys

import numpy as np
from scipy import spatial 
import matplotlib.pyplot as plt

import skimage
from skimage import io, segmentation, util, filters, color

from sklearn.cluster import KMeans 
from sklearn.utils import shuffle

DARK_GREEN = (0.0, 0.5, 0.0)
DARK_RED   = (0.5, 0.0, 0.0)
DARK_BLUE  = (0.0, 0.0, 0.5)
BLACK      = (0.0, 0.0, 0.0)
WHITE      = (1.0, 1.0, 1.0)

class FileFormatError(Exception):
    pass

class Table:

    def _guess_cloth_colour(self, precision=2):

        km_cluster = KMeans(n_clusters=5,
                            init=np.array([DARK_GREEN, DARK_RED, DARK_BLUE,
                                           BLACK, WHITE]),
                            n_init=1)

        fig_sample = shuffle(self.img_obj.reshape(-1, 3))[:5000]
        km_cluster.fit(fig_sample)

        norm_min = 10.0
        for c_label, c_centre in enumerate(km_cluster.cluster_centers_):
            norm = np.linalg.norm(c_centre - np.array(DARK_GREEN))

            if norm < norm_min:
                norm_min = norm
                green_centre = c_centre
                green_label = c_label

        #
        # TODO: Add some checks to ensure image with little green does not pass
        #
    
        ret = tuple([round(x, precision) for x in green_centre])

        return ret

    def _cloth_luminance(self):

        intensity = []
        for rgb_point in self.img_obj.reshape(-1, 3):
            intensity.append(max(0.0, 
                1.0 - spatial.distance.euclidean(rgb_point, self.cloth_colour)))

        return np.array(intensity).reshape(self.img_height, self.img_width, 3)

    def _guess_corners(self):

        fig_cloth_lum = self._cloth_luminance()

        return c_00, c_01, c_10, c_11

    def _optimize_corners(self):

        p_00, p_01, p_10, p_11 = self._guess_corners()     

    def find_table(self):

        seg_arrays_fz = segmentation.felzenszwalb(self.img_obj, 
                                                  scale=1000,
                                                  sigma=0.5, min_size=50)
        seg_arrays_slic = segmentation.slic(self.img_obj,
                                            n_segments=10, compactness=1,
                                            sigma=1)
        seg_arrays_sliz = segmentation.slic(self.img_obj,
                                            n_segments=15, compactness=1,
                                            sigma=0.2)
        gradient = filters.sobel(color.rgb2gray(self.img_obj))
        seg_arrays_water = segmentation.watershed(gradient, 
                                                  markers=50,
                                                  compactness=0.0001)

        fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
        ax[0, 0].imshow(segmentation.mark_boundaries(self.img_obj, seg_arrays_fz))
        ax[0, 1].imshow(segmentation.mark_boundaries(self.img_obj, seg_arrays_slic))
        ax[1, 0].imshow(segmentation.mark_boundaries(self.img_obj, seg_arrays_sliz))
        ax[1, 1].imshow(segmentation.mark_boundaries(self.img_obj, seg_arrays_water))
        for a in ax.ravel():
            a.set_axis_off()
        plt.tight_layout()
        plt.savefig('foo.png')

    def __init__(self, img_path):

        #
        # Path to file
        self.img_path = img_path

        #
        # Read the image data and put into proper object
        raw_image = io.imread(self.img_path)
        if len(raw_image.shape) == 3:
            if raw_image.shape[2] == 4:
                rgb_image = color.rgba2rgb(raw_image)

            elif raw_image.shape[2] == 3:
                rgb_image = raw_image

            else:
                raise FileFormatError('Image {0} yields '.format(self.img_path) + \
                                      'image data of wrong shape')

        else:
            raise FileFormatError('Image {0} yields '.format(self.img_path) + \
                                  'image data of wrong shape')
        self.img_obj = util.img_as_float(rgb_image)
        self.img_height = self.img_obj.shape[0]
        self.img_width = self.img_obj.shape[1]

        #
        # Guess the cloth colour
        self.cloth_color = self._guess_cloth_colour()

        #
        # Guess the corner coordinates
        self.n_corners = 4
        self.corners = self._optimize_corners()


if __name__ == '__main__':

    table = Table('../rawdata/fig_snooker_1.PNG')
    table.find_table()
