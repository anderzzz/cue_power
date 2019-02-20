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

    def _create_cloth_luminance(self, squeeze_factor=10.0):

        def _intensity_from_norm2(l2):
            f = 1.0 - l2 * squeeze_factor
            return max(f, 0.0)

        diff_from_cloth = self.img_obj.reshape(-1, 3) - \
                          np.tile(self.cloth_colour, self.img_n_pixels).reshape(-1, 3)
        diff_from_cloth_l2 = np.sum(np.square(diff_from_cloth), axis=1)
        v_intensity_from_l2 = np.vectorize(_intensity_from_norm2)
        intensity = v_intensity_from_l2(diff_from_cloth_l2)

        return intensity.reshape(self.img_height, self.img_width)

    def _guess_corners(self, cluster_thrs=20):

        dd = feature.corner_shi_tomasi(self.cloth_luminance)
        peaks = feature.corner_peaks(dd, min_distance=1)

        if self.print_intermediate_img:
            fig_corners = copy.deepcopy(self.img_obj)
            for pp_x, pp_y in peaks: 
                for kk in range(-3,4):
                    fig_corners[pp_x + kk][pp_y + kk] = np.array((1.0, 1.0, 0.0))
                    fig_corners[pp_x + kk][pp_y - kk] = np.array((1.0, 1.0, 0.0))
            io.imsave(self.img_out_prefix + '_shi_corners_marked.png', fig_corners)

        ee = feature.canny(self.cloth_luminance)
        print (ee)
        print (ee.shape)
        raise RuntimeError

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

        c_upper_left = []
        c_upper_right = []
        c_lower_left = []
        c_lower_right = []
        for pp in peaks:

            d_upper_left = np.linalg.norm(pp - c_00)
            d_upper_right = np.linalg.norm(pp - c_01)
            d_lower_left = np.linalg.norm(pp - c_10)
            d_lower_right = np.linalg.norm(pp - c_11)

            if d_upper_left < cluster_thrs:
                c_upper_left.append(pp)
            if d_upper_right < cluster_thrs:
                c_upper_right.append(pp)
            if d_lower_left < cluster_thrs:
                c_lower_left.append(pp)
            if d_lower_right < cluster_thrs:
                c_lower_right.append(pp)

        c_00_extreme_corner = np.min(np.array(c_upper_left), axis=0)
        c_11_extreme_corner = np.max(np.array(c_lower_right), axis=0)
        c_01_extreme_corner = np.array([np.min(np.array(c_upper_right), axis=0)[0],
                                        np.max(np.array(c_upper_right), axis=0)[1]])
        c_10_extreme_corner = np.array([np.max(np.array(c_lower_left), axis=0)[0],
                                        np.min(np.array(c_lower_left), axis=0)[1]])

#        if self.print_intermediate_img:
#            fig_corners = copy.deepcopy(self.img_obj)
#            for pp_x, pp_y in [c_00_extreme_corner, c_01_extreme_corner, \
#                               c_10_extreme_corner, c_11_extreme_corner]:
#                for kk in range(-3,4):
#                    fig_corners[pp_x + kk][pp_y + kk] = np.array((1.0, 0.3, 1.0))
#                    fig_corners[pp_x + kk][pp_y - kk] = np.array((1.0, 0.3, 1.0))
#            io.imsave(self.img_out_prefix + '_extreme_corners_marked.png', fig_corners)

        return c_00_extreme_corner, c_01_extreme_corner, \
               c_10_extreme_corner, c_11_extreme_corner

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

#    def _make_rectangle(self, horizont_low, horizont_high,
#                              vertical_low, vertical_high):
#
#        if horizont_low == horizont_high or vertical_low == vertical_high:
#            return None
#
#        if horizont_low > horizont_high:
#            raise ValueError('Horizontal high value less than low value')
#        if vertical_low > vertical_high:
#            raise ValueError('Vertical high value less than low value')
#
#        width = np.arange(horizont_low, horizont_high + 1)
#        height = np.arange(vertical_low, vertical_high + 1)
#
#        rectangle = [np.repeat(width, 1 + vertical_high - vertical_low),
#                     np.tile(height, 1 + horizont_high - horizont_low)]
#
#        return rectangle

    def _make_table_indeces(self, b_l, b_t, b_r, b_b, exclude_boundary=True):

        top_most = np.min(b_t, axis=0)[0]
        bottom_most = np.max(b_b, axis=0)[0]
        top_least = np.max(b_t, axis=0)[0]
        bottom_least = np.min(b_b, axis=0)[0]
        top_right = np.min(b_r, axis=0)[0]
        top_left = np.min(b_l, axis=0)[0]
        bottom_right = np.max(b_r, axis=0)[0]
        bottom_left = np.max(b_l, axis=0)[0]

        order = []
        if top_right < top_left:
            order.append((b_t, b_r, top_least))
        elif top_right > top_left:
            order.append((b_l, b_t, top_least))

        order.append((b_l, b_r, bottom_least))

        if bottom_right > bottom_left:
            order.append((b_b, b_r, bottom_most))
        elif bottom_right < bottom_left:
            order.append((b_l, b_b, bottom_most))

        if exclude_boundary:
            k_low = 1
            k_high = 0
        else:
            k_low = 0
            k_high = 1

        cloth_pixels_leftright = []
        cloth_pixels_topbottom = []
        shift_point = -1
        for horizont in range(top_most + k_low, bottom_most + k_high):

            if horizont > shift_point:
                border_a, border_b, shift_point = order.pop(0)

            w_min_a, w_max_a = self._func_boundary(horizont, border_a)
            w_min_b, w_max_b = self._func_boundary(horizont, border_b)

            leftright = np.arange(w_min_a + k_low, w_max_b + k_high)
            topbottom = np.repeat([horizont], w_max_b + k_high - w_min_a - k_low)

            cloth_pixels_leftright.extend(leftright)
            cloth_pixels_topbottom.extend(topbottom)

        return cloth_pixels_topbottom, cloth_pixels_leftright

    def _func_boundary(self, x, points):

        mask = points[:,0] == x
        reduced_points = points[mask]
        return np.min(reduced_points, axis=0)[1], \
               np.max(reduced_points, axis=0)[1]

    def _optimize_corners(self, lum_thrs=0.3, frac_min=0.95):

        p_00, p_01, p_10, p_11 = self._guess_corners()     

        b_left = self._manhattan_line(p_00, p_10)
        b_top = self._manhattan_line(p_00, p_01)
        b_right = self._manhattan_line(p_01, p_11)
        b_bottom = self._manhattan_line(p_10, p_11)

        if self.print_intermediate_img:

            fig_corners = copy.deepcopy(self.img_obj)
            for pp_x, pp_y in [p_00, p_01, p_10, p_11]:
                for kk in range(-3,4):
                    fig_corners[pp_x + kk][pp_y + kk] = np.array((1.0, 0.2, 1.0))
                    fig_corners[pp_x + kk][pp_y - kk] = np.array((1.0, 0.2, 1.0))

            for line in [b_left, b_top, b_right, b_bottom]:
                for l_x, l_y in line:
                    fig_corners[l_x][l_y] = np.array((1.0, 0.2, 1.0))

            io.imsave(self.img_out_prefix + '_corners_marked.png', fig_corners)

        table_indeces = self._make_table_indeces(b_left, b_top, b_right, b_bottom)

        cut_luminance = self.cloth_luminance[table_indeces[0], table_indeces[1]]
        n_clothy_cut = np.count_nonzero(cut_luminance > lum_thrs)
        n_clothy_all = np.count_nonzero(self.cloth_luminance > lum_thrs)
        frac_cloth = n_clothy_cut / n_clothy_all

        if frac_cloth < frac_min:
            raise RuntimeError('Fraction of cloth luminance too low: {0}'.format(frac_cloth))

        return p_00, p_01, p_10, p_11

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
        print (self.corners)


if __name__ == '__main__':

    table = Table('../test_data/fig_snooker_1.PNG')
    #table.find_table()
