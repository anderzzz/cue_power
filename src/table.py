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

from collections import namedtuple

DARK_GREEN = (0.0, 0.5, 0.0)
DARK_RED   = (0.5, 0.0, 0.0)
DARK_BLUE  = (0.0, 0.0, 0.5)
BLACK      = (0.0, 0.0, 0.0)
WHITE      = (1.0, 1.0, 1.0)

ClothCorners = namedtuple('ClothCorners', 
                          ['top_left', 'top_right', 'down_left', 'down_right'])
ClothSides = namedtuple('ClothSides',
                        ['top', 'left', 'right', 'down'])

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

    def _edge_bounding_corners(self, edge_pixels, cluster_thrs=30, slack=5):

        d_upper_left_min = self.img_height + self.img_width
        d_lower_left_min = self.img_height + self.img_width
        d_upper_right_min = self.img_height + self.img_width
        d_lower_right_min = self.img_height + self.img_width
        for pp in edge_pixels:
            
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
        for pp in edge_pixels:

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

        c_00_extreme_corner = np.min(np.array(c_upper_left), axis=0) + \
                              np.array([-1 * slack, -1 * slack])
        c_11_extreme_corner = np.max(np.array(c_lower_right), axis=0) + \
                              np.array([slack, slack])
        c_01_extreme_corner = np.array([np.min(np.array(c_upper_right), axis=0)[0],
                                        np.max(np.array(c_upper_right), axis=0)[1]]) + \
                              np.array([-1 * slack, slack])
        c_10_extreme_corner = np.array([np.max(np.array(c_lower_left), axis=0)[0],
                                        np.min(np.array(c_lower_left), axis=0)[1]]) + \
                              np.array([slack, -1 * slack])

        return ClothCorners(c_00_extreme_corner, c_01_extreme_corner,
                            c_10_extreme_corner, c_11_extreme_corner)

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

    def _make_table_indeces(self, sides, exclude_boundary=False):

        top_most = np.min(sides.top, axis=0)[0]
        bottom_most = np.max(sides.down, axis=0)[0]
        top_least = np.max(sides.top, axis=0)[0]
        bottom_least = np.min(sides.down, axis=0)[0]
        top_right = np.min(sides.right, axis=0)[0]
        top_left = np.min(sides.left, axis=0)[0]
        bottom_right = np.max(sides.right, axis=0)[0]
        bottom_left = np.max(sides.left, axis=0)[0]

        order = []
        if top_right < top_left:
            order.append((sides.top, sides.right, top_least))
        elif top_right > top_left:
            order.append((sides.left, sides.top, top_least))

        order.append((sides.left, sides.right, bottom_least))

        if bottom_right > bottom_left:
            order.append((sides.down, sides.right, bottom_most))
        elif bottom_right < bottom_left:
            order.append((sides.left, sides.down, bottom_most))

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

    def _make_sides(self, corners):

        sides = ClothSides( \
            self._manhattan_line(corners.top_left, corners.top_right),
            self._manhattan_line(corners.top_left, corners.down_left),
            self._manhattan_line(corners.top_right, corners.down_right),
            self._manhattan_line(corners.down_left, corners.down_right))

        return sides

    def _func_boundary(self, x, points):

        mask = points[:,0] == x
        reduced_points = points[mask]
        return np.min(reduced_points, axis=0)[1], \
               np.max(reduced_points, axis=0)[1]

    def _optimize_corners(self, max_jump=4, max_steps=300, break_step=40, temp_init=1000.0): 

        def eval_state_(corners):

            sides = self._make_sides(corners)
            table_indeces = self._make_table_indeces(sides)

            return table_indeces, sides

        def score_(t_indeces, e_indeces):
            
            total_edge_pixels = np.sum(e_indeces)
            table_slice = e_indeces[t_indeces[0], t_indeces[1]]
            table_edge_pixels = np.sum(table_slice)

            frac_e = table_edge_pixels / total_edge_pixels 
            table_size = len(table_slice)

            return table_size, frac_e

        edges_pixels = feature.canny(self.cloth_luminance, sigma=4.0)
        p_edges = np.argwhere(edges_pixels == True)
        corners = self._edge_bounding_corners(p_edges)

        table_state, sides_current = eval_state_(corners)
        table_score_current = score_(table_state, edges_pixels)

        k_iter = 0
        k_nada = 0
        temp = temp_init
        while k_iter < max_steps:

            corners_old = copy.deepcopy(corners)

            cc = list(corners)[np.random.randint(0, 4)]
            cc += np.random.randint(-1 * max_jump, max_jump + 1, 2)

            table_state, sides = eval_state_(corners)
            table_score_new = score_(table_state, edges_pixels)

            if table_score_new[1] < table_score_current[1]:
                accept = False

            else:
                if table_score_new[0] < table_score_current[0]:
                    accept = True

                else:
                    delta_v = table_score_new[0] - table_score_current[0]
                    factor = np.exp(-1 * delta_v / temp)
                    if factor > np.random.ranf():
                        accept = True
                    else:
                        accept = False

            if not accept:
                corners = copy.deepcopy(corners_old)

            else:
                table_score_current = table_score_new
                sides_current = copy.deepcopy(sides)
                corners_current = copy.deepcopy(corners)

            k_iter += 1
            temp = temp * 0.98
            if not accept: 
                k_nada += 1
            else:
                k_nada = 0

            if k_nada > break_step:
                break

        if self.print_intermediate_img:
            fig_ = copy.deepcopy(self.img_obj)
            for pp_x, pp_y in p_edges:
                fig_[pp_x][pp_y] = np.array((1.0, 0.2, 1.0))

            for line in sides_current: 
                for l_x, l_y in line:
                    fig_[l_x][l_y] = np.array((1.0, 1.0, 0.0))

            for pp_x, pp_y in corners_current: 
                for kk in range(-3,4):
                    fig_[pp_x + kk][pp_y + kk] = np.array((1.0, 1.0, 0.0))
                    fig_[pp_x + kk][pp_y - kk] = np.array((1.0, 1.0, 0.0))
        io.imsave(self.img_out_prefix + '_edges_corners.png', fig_)

        return corners_current 

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
        self.sides = self._make_sides(self.corners)
        self.table_indeces = self._make_table_indeces(self.sides)


if __name__ == '__main__':

    table = Table('../test_data/fig_snooker_1.PNG')
