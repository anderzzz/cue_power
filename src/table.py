'''Find relevant items on snooker table

'''
import sys

import numpy as np
import copy
from scipy import spatial 
import matplotlib.pyplot as plt

import skimage
from skimage import io, transform, segmentation, draw, \
                    util, filters, feature, color

from sklearn.cluster import KMeans 
from sklearn.utils import shuffle

from collections import namedtuple

DARK_GREEN = (0.0, 0.5, 0.0)
DARK_RED   = (0.5, 0.0, 0.0)
DARK_BLUE  = (0.0, 0.0, 0.5)
BLACK      = (0.0, 0.0, 0.0)
WHITE      = (1.0, 1.0, 1.0)
PINK       = (1.0, 0.8, 0.8)
YELLOW     = (1.0, 1.0, 0.0)
DARKER_GREEN = (0.0, 0.3, 0.0)

ClothCorners = namedtuple('ClothCorners', 
                          ['top_left', 'top_right', 'down_left', 'down_right'])
ClothSides = namedtuple('ClothSides',
                        ['top', 'left', 'right', 'down'])

class FileFormatError(Exception):
    pass

class Balls:

    def __init__(self, pixels, pixels_lum):
        
        print (pixels.shape)
        edges = feature.canny(pixels_lum, sigma=2.0)
        p_edges = np.argwhere(edges == True)
        hough_radii = np.arange(4, 9, 1)
        hough_res = transform.hough_circle(edges, hough_radii)
        accums, cx, cy, radii = transform.hough_circle_peaks(hough_res, hough_radii, total_num_peaks=8)
        print (accums, cx, cy, radii)

        for pp_x, pp_y in p_edges:
            pixels[pp_x, pp_y] = (1.0, 0.0, 1.0) 

        for pp_x, pp_y, radius in zip(cx, cy, radii):
            c_x, c_y = draw.circle_perimeter(pp_x, pp_y, radius)
            for px, py in zip(c_x, c_y):
                pixels[py][px] = (0.0, 1.0, 1.0)

        io.imsave('tmp.png', pixels)

class Table:

    def _guess_cloth_colour(self, precision=2, cluster_count=5):
        '''Guess colour of table cloth

        Parameters
        ----------
        precision : int, optional
            Number of decimals to keep in cloth colour RGB vector
        cluster_count : int, optional
            Number of clusters to consider in the RGB space

        Returns
        -------
        rgb_cloth : tuple
            The RGB coordinate of the table cloth colour

        Notes
        -----
        The plurality of RGB points of the image are clustered. In the typical
        front-facing perspective of the table, the cloth colour is assumed to
        (1) be a major object in the view and (2) have a mostly unique and
        mostly uniform colour green colour. 

        '''
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
        '''Extract luminance image where cloth is selectively illuminated.

        Parameters
        ----------
        squeeze_factor : int, optional
            How selective to be in the cloth colour illumination. The higher
            the value, the more selective, which comes at the risk that
            relevant cloth pixels are incorrectly non-illuminated.

        Returns
        -------
        intensity : Numpy array
            The image with intensity values near 1.0 for cloth pixels, 0.0
            otherwise

        '''
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
        '''Rapid estimate of the corner coordinates that are bounding the cloth
        edge, with slight preference towards bounding too much than too little
        of the edge.

        Parameters
        ----------
        edge_pixels : Numpy array
            Collection of pixels part of the edge.
        cluster_thrs : int, optional
            The radius around the extreme edge pixels to define edge pixels to
            construct the bounding coordinate
        slack : int, optional
            The added slack to expand the bounding corners further

        Returns
        -------
        corners_guess : ClothCorners
            The guess of the cloth corners

        Notes
        -----
        The method to guess the corners proceed by (1) finding the four points
        of the edge pixels that are the closest in a Euclidean sense to the
        four image corners. (2) For each such pixel extract a set of
        neighbouring pixels. (3) Construct a strong Pareto optimal pixel to the
        set, in other words, where both coordinates are closer to the relevant
        corner than any pixel in the set. (4) Add some slack to the coordinates
        to move closer to the relevant image corner.

        '''
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
        '''Construct the line between two pixels. The line is constrained to
        discrete pixels, hence the method constructs the contiguous path that
        most closely approximates the continuous interpolating line

        Parameters
        ----------
        p_a : Numpy array
            The first pixel, two coordinates
        p_b : Numpy array
            The second pixel, two coordinates

        Returns
        -------
        line_pixels : Numpy array
            The collection of pixels that comprises the discrete line

        '''
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
        '''Construct index arrays for pixels within four sides

        Parameters
        ----------
        sides : ClothSides
            The four sides of the table cloth
        exclude_boundary : bool, optional
            If True the index array does not include pixels of the boundary. If
            False, the index array includes pixels of boundary.

        Returns
        -------
        cloth_pixels_topbottom : Numpy array
            The pixels of the image containing cloth table data, vertical
            coordinate
        cloth_pixels_leftright : Numpy array
            The pixels of the image containing cloth table data, horizontal
            coordinate

        '''
        top_most = np.min(sides.top, axis=0)[0]
        bottom_most = np.max(sides.down, axis=0)[0]
        top_least = np.max(sides.top, axis=0)[0]
        bottom_least = np.min(sides.down, axis=0)[0]
        top_right = np.min(sides.right, axis=0)[0]
        top_left = np.min(sides.left, axis=0)[0]
        bottom_right = np.max(sides.right, axis=0)[0]
        bottom_left = np.max(sides.left, axis=0)[0]

        #
        # Determine between which sides of the cloth a horizontal pixel
        # scanning, left to right, takes place
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

        #
        # Scan pixels, top to bottom (outer loop), left to right (vectorized
        # inner loop) constrained to be between the relevant pair of sides.
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
        '''Derive the pixels of the cloth given four corners

        Parameters
        ----------
        corners : ClothCorners
            The four corners of the table cloth

        Returns
        -------
        sides : ClothSides
            The array of pixels that make up the sides of the cloth

        '''
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

    def _optimize_corners(self, max_jump=4, max_steps=500, break_step=64, temp_init=1000.0): 
        '''Determine optimal table corner positions in image

        Parameters
        ----------
        max_jump : int, optional
            Maximum jump along a dimension for the corner pixel as it is
            optimized.
        max_steps : int, optional
            Maximum number of optimization steps
        break_step : int, optional
            Number of consequtive steps without a new solution obtained that
            triggers an early stop to the optimization.
        temp_init : float, optional
            The initial temperature in the Simulated Annealing

        Returns
        -------
        corners : ClothCorners
            The optimal cloth corners

        Notes
        -----
        The optimization uses a Simulated Annealing method, randomly perturbing
        one corner at a time. 

        There are two objectives considered in the optimization. First, the
        fraction of all the cloth edge pixels found with the Canny method that are
        contained within the quadrilateral defined by the four corners. Second,
        the total number of pixels contained within the quadrilateral. The
        latter is minimized while the former is constrained to be no less than
        an initialization structure (see below). The former fraction should be
        very close to 1.0.

        The initialization structure for the corners is guessed in the method
        `_edge_counding_corners`.

        '''
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

            cc = list(corners)[k_iter % 4]
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

        #
        # Debug printing of image in which corners, sides and Canny cloth edges
        # are shown in the input colour image
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

    def _get_corner_extremes(self):
        '''Bla bla

        '''
        horizont_low = min(self.corners.top_left[0], self.corners.top_right[0])
        vertical_low = min(self.corners.top_left[1], self.corners.down_left[1])
        horizont_high = max(self.corners.down_left[0], self.corners.down_right[0])
        vertical_high = max(self.corners.top_right[1], self.corners.down_right[1])

        return np.array([horizont_low, vertical_low]), \
               np.array([horizont_high, vertical_high])

    def _make_coarse_grid(self, delta_pixel):
        '''Create an initial coarse grid of table

        '''
        def side_to_side_jump(v_1, v_2, v_3, v_4):

            l_1 = abs(np.sum(v_1 - v_2))
            l_2 = abs(np.sum(v_3 - v_4))
            if l_1 >= l_2:
                length = l_1
            else:
                length = l_2

            n_divide = length // delta_pixel
            return int(round(l_1 / n_divide)), int(round(l_2 / n_divide)), n_divide

        left_jump, right_jump, n_horizont = \
            side_to_side_jump(self.corners.top_left,
                              self.corners.down_left,
                              self.corners.top_right,
                              self.corners.down_right)
        top_jump, down_jump, n_vertical = \
            side_to_side_jump(self.corners.top_left,
                              self.corners.top_right,
                              self.corners.down_left,
                              self.corners.down_right)

        p_top_side = self.sides.top[np.multiply(top_jump, range(1, n_vertical))]
        p_down_side = self.sides.down[np.multiply(down_jump, range(1, n_vertical))]
        p_left_side = self.sides.left[np.multiply(left_jump, range(1, n_horizont))]
        p_right_side = self.sides.right[np.multiply(right_jump, range(1, n_horizont))]
        
        for p_11, p_12 in zip(p_top_side, p_down_side):
            
            for p_21, p_22 in zip(p_left_side, p_right_side):

                #COMPUTE INTERSECTION OF THE TWO LINES AND MAKE THAT GRID POINT
                print (p_11, p_12, p_21, p_22)

            raise RuntimeError

    def get_rectangle(self):
        '''Construct maximum bounding rectangle to given corners, that is the
        smallest possible rectangle with sides parallel with the image sides,
        while still containing all points of the table cloth

        Returns
        -------
        rec_pixels_topbottom : Numpy array
            The pixels of the image containing rectangle, vertical
            coordinate
        rec_pixels_leftright : Numpy array
            The pixels of the image containing rectangle, horizontal
            coordinate

        '''
        low, high = self._get_corner_extremes()
        width = np.arange(low[0], high[0] + 1)
        height = np.arange(low[1], high[1] + 1)

        return [np.repeat(width, 1 + high[1] - low[1]), \
                np.tile(height, 1 + high[0] - low[0])]

    def find_balls(self, img_path):

        raw_image = io.imread(img_path)
        rgb_image = color.rgba2rgb(raw_image)

        if self.img_width != rgb_image.shape[1] or \
           self.img_height != rgb_image.shape[0]:
            raise RuntimeError('Process image of different dimension ' + \
                               'than initialization image')

        img_analyze = util.img_as_float(rgb_image)

     #   print (img_analyze.shape)
        img_table = img_analyze[self.table_rectangle_indeces[0],
                                self.table_rectangle_indeces[1]]
        low, high = self._get_corner_extremes()
        img_table = img_table.reshape(1 + high[0] - low[0], 1 + high[1] - low[1], 3)
        print (img_table.shape)

        edges_pixels = feature.canny(color.rgb2grey(img_table), sigma=1.0)
        p_edges = np.argwhere(edges_pixels == True)
        print (p_edges.shape)
        if self.print_intermediate_img:
            fig_ = copy.deepcopy(img_table)
            for pp_x, pp_y in p_edges:
                fig_[pp_x][pp_y] = np.array((1.0, 0.2, 1.0))

        io.imsave(self.img_out_prefix + '_table_edges.png', fig_)

#        km_cluster = KMeans(n_clusters=8,
#                            init=np.array([self.cloth_colour, DARK_RED, 
#                                           DARK_BLUE, BLACK, WHITE, YELLOW, 
#                                           PINK, DARKER_GREEN]),
#                            n_init=1)
#
#        fig_sample = shuffle(img_analyze.reshape(-1, 3))[:5000]
#        km_cluster.fit(fig_sample)
#        print (km_cluster.cluster_centers_)

    def __init__(self, img_path, corners=None,
                 print_intermediate_img=True, img_out_prefix='debug'):

        #
        # Path to file to discover Table in
        self.img_path = img_path

        #
        # Debug and constants
        self.print_intermediate_img = print_intermediate_img
        self.img_out_prefix = img_out_prefix

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
        # Guess the cloth colour
        self.cloth_colour = self._guess_cloth_colour()
        self.cloth_luminance = self._create_cloth_luminance()

        if self.print_intermediate_img:
            io.imsave(self.img_out_prefix + '_cloth_luminance.png', self.cloth_luminance)

        #
        # Set or discover corners for reference image
        if not corners is None:
            if not isinstance(corners, ClothCorners):
                raise TypeError('Corners must be of type ClothCorners')
            self.corners = corners

        else: 
            self.corners = self._optimize_corners()

        # 
        # Construct additional table data
        self.sides = self._make_sides(self.corners)
        self.table_indeces = self._make_table_indeces(self.sides)
        self.table_rectangle_indeces = self.get_rectangle()
        self.coarse_grid = self._make_coarse_grid(25)

        #
        # Define balls object, initilization stage
        table_pixels = self.img_obj[self.table_rectangle_indeces[0],
                                    self.table_rectangle_indeces[1]]
        table_lum_pixels = self.cloth_luminance[self.table_rectangle_indeces[0],
                                                self.table_rectangle_indeces[1]]
        low, high = self._get_corner_extremes()
        img_table = table_pixels.reshape(1 + high[0] - low[0], 1 + high[1] - low[1], 3)
        lum_table = table_lum_pixels.reshape(1 + high[0] - low[0], 1 + high[1] - low[1])
        self.balls = Balls(img_table, lum_table)


if __name__ == '__main__':

    #table = Table('../test_data/fig_snooker_1.PNG')
    #print (table.corners)
    corners = ClothCorners(np.array([136, 259]), np.array([135, 703]),
                           np.array([431, 170]), np.array([432, 782]))
    table = Table('../test_data/fig_snooker_1.PNG', corners=corners)
    #print ('ping')
    #table.find_balls('../test_data/fig_snooker_1.PNG')
