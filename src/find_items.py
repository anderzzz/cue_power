'''Find relevant items on snooker table

'''
import sys

import numpy as np
import copy
from scipy import spatial 
import matplotlib.pyplot as plt

import skimage
from skimage import io, segmentation, util, filters, feature, color

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

    def _guess_cloth_colour(self, precision=2, cluster_count=5):

        km_cluster = KMeans(n_clusters=cluster_count,
                            init=np.array([DARK_GREEN, DARK_RED, DARK_BLUE,
                                           BLACK, WHITE]),
                            n_init=1)

        fig_sample = shuffle(self.img_obj.reshape(-1, 3))[:5000]
        km_cluster.fit(fig_sample)

        diff_from_green = km_cluster.cluster_centers_ - \
                          np.tile(np.array(DARK_GREEN), cluster_count).reshape(-1, 3)
        diff_from_green_norm = np.array([np.dot(x, x) for x in diff_from_green])
        green_centre = km_cluster.cluster_centers_[np.argmin(diff_from_green_norm)]

        #
        # TODO: Add some checks to ensure image with little green does not pass
        #
    
        ret = tuple([round(x, precision) for x in green_centre])

        return ret

    def _create_cloth_luminance(self, squeeze_factor=20.0):

        def _intensity_from_norm2(l2):
            f = 1.0 - l2 * squeeze_factor
            return max(f, 0.0)

        diff_from_cloth = self.img_obj.reshape(-1, 3) - \
                          np.tile(self.cloth_colour, self.img_n_pixels).reshape(-1, 3)
        diff_from_cloth_l2 = np.sum(np.square(diff_from_cloth), axis=1)
        v_intensity_from_l2 = np.vectorize(_intensity_from_norm2)
        intensity = v_intensity_from_l2(diff_from_cloth_l2)

        return intensity.reshape(self.img_height, self.img_width)

    def _guess_corners(self):

        dd = feature.corner_shi_tomasi(self.cloth_luminance)
        peaks = feature.corner_peaks(dd, min_distance=1)

        if self.print_intermediate_img:

            fig_corners = self.img_obj
            for pp_x, pp_y in peaks:
                for kk in range(-3,4):
                    fig_corners[pp_x + kk][pp_y + kk] = np.array((1.0, 1.0, 0.0))
                    fig_corners[pp_x + kk][pp_y - kk] = np.array((1.0, 1.0, 0.0))
            io.imsave(self.img_out_prefix + '_corners_marked.png', fig_corners)

        d_upper_left_min = self.img_height + self.img_width
        d_lower_left_min = self.img_height + self.img_width
        d_upper_right_min = self.img_height + self.img_width
        d_lower_right_min = self.img_height + self.img_width
        for pp in peaks:
            
            d_upper_left = np.linalg.norm(pp - np.array([0, 0]))
            d_upper_right = np.linalg.norm(pp - np.array([0, self.img_width]))
            d_lower_left = np.linalg.norm(pp - np.array([self.img_height, 0]))
            d_lower_right = np.linalg.norm(pp - np.array([self.img_height, self.img_width]))

            if d_upper_left < d_upper_left_min:
                d_upper_left_min = d_upper_left
                c_00 = pp
            if d_upper_right < d_upper_right_min:
                d_upper_right_min = d_upper_right
                c_01 = pp
            if d_lower_left < d_lower_left_min:
                d_lower_left_min = d_lower_left
                c_10 = pp
            if d_lower_right < d_lower_right_min:
                d_lower_right_min = d_lower_right
                c_11 = pp

        return c_00, c_01, c_10, c_11

    def _manhattan_line(self, p_a, p_b):

        diff = p_b - p_a
        delta = np.abs(diff)
        sign = np.sign(diff)
        total_points = sum(delta)

        if delta[0] > delta[1]:
            slope = delta[1] / total_points
            major_index = 0
            minor_index = 1

        else:
            slope = delta[0] / total_points
            major_index = 1
            minor_index = 0
        
        p_current = copy.deepcopy(p_a)
        line = [list(p_current)]
        k_minor = 0
        for step in range(0, sum(delta)):

            diff_from_real = step * slope - k_minor
            if diff_from_real > 0.5:
                p_current[minor_index] += sign[minor_index]
                k_minor += 1

            else:
                p_current[major_index] += sign[major_index]

            line.append(list(p_current))

        return np.array(line)

    def _optimize_corners(self):

        p_00, p_01, p_10, p_11 = self._guess_corners()     

        b_left = self._manhattan_line(p_00, p_10)
        b_top = self._manhattan_line(p_00, p_01)
        b_right = self._manhattan_line(p_01, p_11)
        b_bottom = self._manhattan_line(p_10, p_11)

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

    def __init__(self, img_path,
                 print_intermediate_img=True, img_out_prefix='dummy'):

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
        self.img_n_pixels = self.img_height * self.img_width

        #
        # Debug and constants
        self.print_intermediate_img = print_intermediate_img
        self.img_out_prefix = img_out_prefix

        #
        # Guess the cloth colour
        self.cloth_colour = self._guess_cloth_colour()
        self.cloth_luminance = self._create_cloth_luminance()

        if self.print_intermediate_img:
            io.imsave(self.img_out_prefix + '_cloth_luminance.png', self.cloth_luminance)

        #
        # Guess the corner coordinates
        self.n_corners = 4
        self.corners = self._optimize_corners()


if __name__ == '__main__':

    table = Table('../test_data/fig_snooker_1.PNG')
    #table.find_table()
