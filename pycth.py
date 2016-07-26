# -*- coding: utf-8 -*-
"""
    Created on Wed Nov 19 21:39:55 2014
    pyct.py : A program for analyzing 2d and 3d images of segmented objects
    @author: Cantwell G. Carson
    Copyright ©2014 Cantwell G. Carson.
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    See <http://www.gnu.org/licenses/> for a copy of the GNU General Public
    License.
    If you use this software, please cite:
    Carson, Cantwell G., Levine, Jonathan A, "The finite body triangulation:
    algorithms, subgraphs, homogeneity estimation, and application", J
    Microscopy 2015
    Carson, Cantwell G., Levine, Jonathan A, "Estimation of finite object
    distribution homogeneity" , Computational Geometry 2015
    For more information contact carsonc@gmail.com
"""
"""
CHANGES:
1. CoreScan's init now requires a string for the file id instead of a nparray
2. Cleaned up the code to be more modular
3. Triangulate is now a class requiring CoreScan's core_data
"""
"""
TODO:
1. Make synthetic arrays create a file of their output
2. Readd verbose print statements to the code
3. Fix behavior of chords inside voids
3. Break code down even further
"""

#==============================================================================
# Modules
#==============================================================================

import numpy as np
import time
import matplotlib.pyplot as plt
import os
import zipfile

from matplotlib import cm
from itertools import combinations, repeat, product, chain
from collections import defaultdict
from skimage.morphology import watershed as water

from scipy.ndimage import label as lbl
from scipy.ndimage.filters import sobel, generic_gradient_magnitude
from scipy.ndimage.morphology import distance_transform_edt as edt
from scipy.sparse import csr_matrix, find
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist, euclidean

#==============================================================================
# Easy visualization function
#==============================================================================

def depyct(fid):
    core_scan_configs = {'dims'         : [0, 50,
                                           0, 50,
                                           0, 50],
                         'verbose'      : True,
                         'temp_folder'  : None,
                         'invert_array' : True
                         }
    tri_configs = {'slice_index'   : None,
                   'dims'          : None,
                   'plotfigs'      : True,
                   'method'        : 'Dirichlet',
                   'qhull_options' : 'QJ',
                   'brute_force'   : False,
                   'verbose'       : core_scan_configs.get('verbose'),
                   'verbosity'     : 40000
                   }
    plot_configs = {'paste'    : True,
                    'voids'    : True,
                    'subgraph' : 'fbt',
                    'o_map'    : None
                    }
    scan = CoreScan(fid, **core_scan_configs)
    if tri_configs.get('method') == 'Delaunay':
        triang = DelaunayTri(scan, **tri_configs)
        triang.run()
    elif tri_configs.get('method') == 'Dirichlet':
        triang = DirichletTri(scan, **tri_configs)
        triang.run()
    else: triang = scan
    plot_pores(triang, **plot_configs)


#==============================================================================
# Two methods to generate synthetic arrays in 2d and 3d respectively.
#==============================================================================

def gen_cell_slice(x_dim_o, y_dim_o, d_spacing, r_void):
    """
    Used to create synthetic hexagonal 2d array
    Inputs:
        x_dim_o : integer
            desired length of array in x dimension
        y_dim_o : integer
            desired length of array in y dimension
        d_spacing : float
            desired center-to-center distance between spheres
        r_void : integer
            desired radius of the voids
    Output:
        out_slice : 2d numpy array
            an array with the desired dimensions and hexagonal 2d lattice.
    """

    x_dim = x_dim_o * 1.5
    y_dim = y_dim_o * 1.5
    y_s = np.sqrt(3) * d_spacing / 2
    base_slice = np.zeros((x_dim, y_dim))
    xx, yy = np.mgrid[:np.ceil((r_void+1) * 2), :np.ceil((r_void+1) * 2)]

    y_d = 0
    x_shift = False

    while y_d + 2*d_spacing < y_dim:
        y_d += wt_ch(y_s)

        if x_shift:
            x_d = -wt_ch(d_spacing/2)
        else:
            x_d = 0
        while x_d + 2*d_spacing  < x_dim:
            x_d += wt_ch(d_spacing)
            r_v_i = wt_ch(r_void)
            circle = (np.sqrt((xx - r_v_i) ** 2 + (yy - r_v_i) ** 2) <= r_v_i)
            c_y, c_x = np.shape(circle)
            b_y, b_x = np.shape(base_slice[y_d : y_d + c_y,
                                           x_d : x_d + c_x])
            _y, _x = np.min([[c_y, c_x],[b_y, b_x]], axis = 1)
            try:
                base_slice[y_d : y_d + _y,
                           x_d : x_d + _x] += circle[:_y, :_x]
            except ValueError:
                print ('Warning, ValueError: ',
                       circle.shape,
                       np.shape(base_slice[y_d : y_d + _y, x_d : x_d + _x]))

        x_shift = not x_shift

    out_slice = np.abs(base_slice)[wt_ch(y_s):x_dim_o + wt_ch(y_s),
                                   wt_ch(y_s):y_dim_o + wt_ch(y_s)]

    return out_slice

def gen_test_fcc(x_dim_o, y_dim_o, z_dim_o, d_spacing, r_void):
    """
    Used to create synthetic fcc array
    Inputs:
        x_dim_o : integer
            desired length of array in x dimension
        y_dim_o : integer
            desired length of array in y dimension
        z_dim_o : integer
            desired length of array in z dimension
        d_spacing : float
            desired center-to-center distance between spheres
        r_void : integer
            desired radius of the voids
    Output:
        out_slice : 3d numpy array
            an array with the desired dimensions and an fcc lattice.
    """

    base_slice_name = 'base_slice.dat'
    out_slice_name = 'out_slice.dat'

    dat_file_names_list = [base_slice_name, out_slice_name]

    rem_files_in_list(os.path.abspath('.'), dat_file_names_list)

    out_slice = np.memmap(out_slice_name,
                           dtype = 'uint8',
                           mode = 'w+',
                           shape = (z_dim_o, y_dim_o, x_dim_o))

    x_dim = x_dim_o * 1.5
    y_dim = y_dim_o * 1.5
    z_dim = z_dim_o * 1.5

    base_slice = np.memmap(base_slice_name,
                           dtype = 'uint8',
                           mode = 'w+',
                           shape = (z_dim, y_dim, x_dim))

    y_s = np.sqrt(3) * d_spacing / 2
#    z_s = np.sqrt(6) * d_spacing / 3

    zz, yy, xx = np.mgrid[:np.ceil((r_void+1) * 2),
                          :np.ceil((r_void+1) * 2),
                          :np.ceil((r_void+1) * 2)]

    z_d = 0
    y_shift = False


    while z_d + 2*d_spacing < z_dim:
        z_d += wt_ch(d_spacing / 2)
        x_shift = False

        if y_shift:
            y_d = np.abs(wt_ch(d_spacing / 2))
        else:
            y_d = 0

        while y_d + 2*d_spacing < y_dim:
            y_d += np.abs(wt_ch(d_spacing / 2))

            if x_shift:
                x_d =  - wt_ch(d_spacing / 2)
            else:
                x_d = 0

            while x_d + 2*d_spacing  < x_dim:
                x_d += wt_ch(d_spacing)
                r_v_i = wt_ch(r_void)
                circle = (np.sqrt((xx - r_v_i) ** 2 +
                                  (yy - r_v_i) ** 2 +
                                  (zz - r_v_i) ** 2) <= r_v_i)
                c_z, c_y, c_x = np.shape(circle)
                if (base_slice[z_d : z_d + c_z,
                               y_d : y_d + c_y,
                               x_d : x_d + c_x].shape != np.shape(circle)):
                    print z_d, c_z, y_d, c_y, x_d, c_x
                base_slice[z_d : z_d + c_z,
                           y_d : y_d + c_y,
                           x_d : x_d + c_x] += circle

            x_shift = not x_shift
        y_shift = not y_shift

    out_slice = np.abs(base_slice)[wt_ch(y_s):z_dim_o + wt_ch(y_s),
                                   wt_ch(y_s):y_dim_o + wt_ch(y_s),
                                   wt_ch(y_s):x_dim_o + wt_ch(y_s),]

    del base_slice

    return out_slice

#==============================================================================
# Internal methods used in CoreScan object. They are presented externally for
# ease of testing and troubleshooting.
#==============================================================================

def _2d_to_3d(fid):
    """Turns 2d numpy arrays into 3d numpy arrays"""
    if len(fid.shape) == 2:
        fid = fid.tolist()
        fid = [fid]
    return fid

def _calc_voids_chords(point_list, label_image):
    """
    Finds voids with chords and obtains their length and the endpoints of the
    chords. (internally.  label_image is a dictionary mapping points to void indices)
    """
    dim_num = len(label_image.shape)
    point_dict = defaultdict(set)
    void_mag_list = []
    min_v_map = []
    for point in point_list:
        point_label = label_image[tuple(point)]
        point_dict[point_label].add(tuple(point))
    for key in point_dict.iterkeys():
        points = np.array([list(_) for _ in point_dict[key]])
        dist_matrix = np.tril(cdist(points, points))
        point_pairs = dist_matrix.nonzero()
        void_mag_list += list(dist_matrix[point_pairs])
        pairs = points[np.c_[point_pairs]]
        if dim_num == 2:
            min_v_map += [([p_1[1], p_2[1]],[p_1[0], p_2[0]])
                          for [p_1, p_2] in list(pairs)]
        if dim_num == 3:
            min_v_map += [[[p_1[0], p_2[0]],[p_1[1], p_2[1]],[p_1[2], p_2[2]]]
                          for [p_1, p_2] in list(pairs+1)]
    return void_mag_list, min_v_map

def _cdist_refinement(entry, dict_in):
    def _cydist(a, b):
        rows, dims = a.shape
        cols = b.shape[0]
        out = np.zeros((rows, cols), dtype=int)
        for dim in range(dims):
            out += np.subtract(a[:,dim].ravel()[:, None],
                               b[:,dim].ravel()[None, :])**2
        return out
    obj_coords = [dict_in[_] for _ in entry[2:4]]
    c_dist = _cydist(*obj_coords)
    indices = np.unravel_index(c_dist.argmin(), c_dist.shape)
    point_list = [obj_coords[_][indices[_]] for _ in range(2)]
    return _gen_data_list(point_list,
                          entry[0],
                          entry[1],
                          np.sqrt(c_dist[indices]))

def _extract_subset(bc, host, dim_num, distance = 1):
    if dim_num == 2:
        return host[bc[0]-distance:bc[0]+distance+1,
                    bc[1]-distance:bc[1]+distance+1]
    if dim_num == 3:
        return host[bc[0]-distance:bc[0]+distance+1,
                    bc[1]-distance:bc[1]+distance+1,
                    bc[2]-distance:bc[2]+distance+1]

def _gen_correction_factor(pts):
    return np.std([_ for _ in list(chain(*cdist(pts,pts))) if _>0.000001])

def _gen_data_list(point_list, p_1, p_0, cur_dist):
    """Convenience function to format the data list for candidate edges"""
    trans_list = _gen_trans_list(point_list)
    data_list = (trans_list,
                 cur_dist,
                 p_1,
                 p_0,
                 point_list)
    return data_list

def _gen_lists_from_dict(candidate_dict, recalc):
    candidate_list = []
    paste_mags = []
    label_list = []
    point_list = []
    for vals in candidate_dict:
        candidate_list.append(candidate_dict[vals][0])
        label_list.append(candidate_dict[vals][2])
        label_list.append(candidate_dict[vals][3])
        point_list += [candidate_dict[vals][4][_] for
                           _ in range(len(candidate_dict[vals][4]))]
        if recalc:
            paste_mags.append(euclidean(point_list[0], point_list[1]))
        else:
            paste_mags.append(candidate_dict[vals][1])
    return candidate_list, paste_mags, label_list, point_list

def _gen_mst_dict(rng_dict):
    """Code to generate an mst dict from the rng_dict"""
    def _dict_to_csr(term_dict):
        term_dict_v = list(term_dict.itervalues())
        term_dict_k = list(term_dict.iterkeys())
        shape = list(repeat(np.asarray(term_dict_k).max() + 1,2))
        csr = csr_matrix((term_dict_v, zip(*term_dict_k)), shape = shape)
        return csr

    rng_mag_dict = {}
    mst_dict = {}

    for idx, val in enumerate(rng_dict.iterkeys()):
        rng_mag_dict[val] = rng_dict[val][1]
    mst_csr = _dict_to_csr(rng_mag_dict)

    mst_csr = minimum_spanning_tree(mst_csr)
    _y, _x, _z = find(mst_csr)
    min_p_map = np.c_[_y, _x]

    for val in min_p_map:
        mst_dict[tuple(val)] = rng_dict[tuple(val)]
    return mst_dict

def _gen_parameters(paste, void, correction_factor, ideal_ratio):
    """
    Designed to return the means and ratio (along with standard deviations)
    of two lognormed rvs sets.  Removes correction factor from void stdev.
    """
    ### obtains the averages and log stdevs of the paste and void lengths
    p_m, p_s, v_m, v_s = list(chain(*[(np.mean(_), np.std(np.log(_)))
                                    for _ in [paste, void]]))
    ### applies correction factor
    v_s = np.max([0, v_s - correction_factor])
    ### adds 1 to get the scale of the log stdevs of the paste and void
    v_s += 1
    p_s += 1
    ### propagates the stdev of ps and vs into the error of the ratio
    r_s = np.sqrt(p_s*v_s)
    ### calculates the ratio
    r   = (p_m / v_m)
    ### low and high ratio stdevs
    rsl = r   * (1 - 1 / r_s)
    rsh = r   * (r_s - 1)
    ### low and high paste stdevs
    psl = p_m * (1 - 1 / p_s)
    psh = p_m * (p_s - 1)
    ### low and high void stdevs
    vsl = v_m * (1 - 1 / v_s)
    vsh = v_m * (v_s - 1)
    hmg = r / ideal_ratio
    hml = hmg * (1 - 1 / r_s)
    hmh = hmg * (r_s - 1)
    return [p_m, psh, psl] , [v_m, vsh, vsl], [r, rsh, rsl], [hmg, hmh, hml]

def _gen_rng_dict(candidate_dict):
    rng_dict = {}
    for vals in candidate_dict:
        add_ij = True
        k_vals_1 = set([_ for _ in candidate_dict.keys() if _[0] == vals[0]
                                                         or _[1] == vals[0]])
        k_vals_2 = set([_ for _ in candidate_dict.keys() if _[0] == vals[1]
                                                         or _[1] == vals[1]])
        k_vals_1.remove(vals)
        k_vals_2.remove(vals)
        k_1 = set([_[0] for _ in k_vals_1] + [_[1] for _ in k_vals_1])
        k_2 = set([_[0] for _ in k_vals_2] + [_[1] for _ in k_vals_2])
        k_set = k_1.intersection(k_2)
        for k in k_set:
            k_test = [_ for _ in k_vals_1 if set(_).issuperset([k])][0]
            dpkpi = candidate_dict[(k_test)][1]
            k_test = [_ for _ in k_vals_2 if set(_).issuperset([k])][0]
            dpkpj = candidate_dict[(k_test)][1]
            if np.max((dpkpi, dpkpj)) < candidate_dict[vals][1]:
                add_ij = False
                break
        if add_ij == True:
            rng_dict[vals] = candidate_dict[vals]
    return rng_dict

def gen_slice(fid, dims):
    """Convenience function to grab the data from the file"""
    y_0, y_1, x_0, x_1 = dims[2:]
    with open(fid) as f:
        temp_slice = np.loadtxt(f,
                                delimiter='\t',
                                dtype = bool)[y_0: y_1, x_0: x_1]
    return temp_slice

def _gen_trans_list(point_list):
    if len(point_list[0]) == 2:
        point_1, point_2 = point_list
        trans_list = [[point_1[1], point_2[1]], [point_1[0], point_2[0]]]
    if len(point_list[0]) == 3:
        point_1, point_2 = point_list
        trans_list = [[point_1[0]+1, point_2[0]+1],
                      [point_1[1]+1, point_2[1]+1],
                      [point_1[2]+1, point_2[2]+1]]
    return trans_list

def _tiff_to_3d(img,start_frame,end_frame):
    """Turns a tiff file with multiple frames into a numpy array"""
    if (start_frame == None) | (start_frame < 0):
        start_frame = 0
    if (end_frame == None) | (end_frame > img.n_frames):
        end_frame = img.n_frames
    img.seek(start_frame)
    slice_2d = np.asarray(img)
    img_3d = _2d_to_3d(slice_2d)
    for frame in xrange(start_frame + 1, end_frame):
        img.seek(frame)
        slice_2d = np.asarray(img)
        slice_3d = _2d_to_3d(slice_2d)
        img_3d = np.concatenate((img_3d, slice_3d), axis = 0)
    return img_3d

def rem_files_in_list(fid, dat_file_names_list):
    for name in dat_file_names_list:
        temp_filename = os.path.join(fid, name)
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

def wt_ch(non_integer):
    """
    A way to generate integers by selecting either the floor or ceiling of
    the non-integer, weighted by the value of number after the decimal point.
    """
    if np.round(non_integer) - non_integer == 0:
        output = non_integer
    else:
        choices = (np.floor(non_integer), np.ceil(non_integer))
        weights = (np.ceil(non_integer) - non_integer,
                   non_integer - np.floor(non_integer))
        output = np.random.choice(choices, p=weights)
    return output

#==============================================================================
# CoreScan class
#==============================================================================

class CoreScan(object):
    """Base class for analyzing slices pores from cement or porous material.
    Attributes:
        core_data (numpy.ndarray) : a 2d array of binary data.
        fileid (str) : the file location.
        shape (tuple) : the shape of core_data.
        porosity (numpy.float) : the average porosity
        pointnum (int) : number of pixels at the edges of pores.
        voidedges (numpy.ndarray) : a 2D array corresponding to the edges found
                                    in core_data.
        scalar_field (numpy.ndarray) : a 2D array corresponding to the sum
                                    scalar field calculated for all points on
                                    the edges of pores in core_data.
    Args:
        fid (str) : a string corresponding to the file location of the input
                    data or a 2d numpy.ndarray with the same data.
        dims (array-like) : a 6-element string of integers indicating
                    which part of the input array is desired as
                    [z0, z1, y0, y1, x0, x1].
        verbose (boolean) : indicate whether output to the terminal is desired.
        temp_folder (str) : where to hold the temporary files. User should have
                    read-write access
    """

    def __init__(self,
                 fid,
                 dims = [None,None,None,None,None,None],
                 verbose = False,
                 temp_folder = None,
                 invert_array = True):
        """Base object for CoreScan.
        Input:
            fid: a file location for a tab-delimited file containing zeros
            where matrix is present and a constant, finite value where voids
            are present
        Example:
            >>> import pycoresis as pcs
            >>> fid = r'C:\YOUR\FILE\HERE.txt'
            >>> crs = pcs.CoreScan(fid)
            Data loaded from :  'C:\YOUR\FILE\HERE.txt'
            Array shape is   :  (200, 200)
            Mean porosity is :  0.0932
        """

        self.verbose = verbose
        self._core_name = 'core_data.dat'
        self._labl_name = 'core_label.dat'
        self._vdge_name = 'void_edges.dat'
        self._elbl_name = 'elabel.dat'
        self._blnk_name = 'blank.dat'
        self.dat_file_names_list = [self._core_name,
                                    self._labl_name,
                                    self._vdge_name,
                                    self._elbl_name,
                                    self._blnk_name]

        # make a new temp folder if one isn't specified
        self.handle_temp_folder(temp_folder)
        rem_files_in_list(self.temp_folder, self.dat_file_names_list)

        # get the data from the file id
        self.handle_fid(fid, dims)
        self.shape = self.core_data.shape
        if invert_array == True:
            self.core_data = np.invert(self.core_data)
        self.core_data = self.core_data / self.core_data.max()
        self.make_mst_memmap()
        self.core_calcs()
        if self.verbose:
            print "Data loaded from : ", fid
            print time.ctime()


    #==========================================================================
    # __init__ helper functions
    #==========================================================================

    def fid_dir_get_dimensions(self, fid):
        """Finds the dimensions of the core from the file id"""
        if os.path.isdir(fid):
            filelist = os.listdir(fid)
            temp_slice = np.loadtxt(os.path.join(fid,filelist[0]),
                                    delimiter='\t')
        if fid.endswith('.zip'):
            archive = zipfile.ZipFile(fid)
            filelist = archive.namelist()
            imgdata = archive.open(filelist[0])
            temp_slice = np.loadtxt(imgdata, delimiter='\t')
        _y, _x = temp_slice.shape
        dims = [0, len(filelist), 0, _y, 0, _x]
        return dims

    def handle_fid(self, fid, dims):
        """Loads data from fid"""
        if fid.endswith('.npy'):
            self.fid_np_to_data(fid, dims)
        if fid.endswith('.tif'):
            self.fid_tif_to_data(fid, dims)
        if fid.endswith('.zip'):
            self.fid_zip_to_data(fid, dims)
        if os.path.isdir(fid):
            self.fid_dir_to_data(fid, dims)

    def fid_dir_to_data(self, fid, dims):
        """Extracts data from a .zip file id"""
        if dims == [None,None,None,None,None,None]:
            dims = self.fid_dir_get_dimensions(fid)
        self.core_data = self._make_memmap(self._core_name)
        filelist = os.listdir(fid)
        filelist.sort()
        for idx, files in enumerate(filelist[dims[0]: dims[1]]):
            temp_slice = gen_slice(os.path.join(fid,files), dims)
            self.core_data[idx] = temp_slice
            self.core_data.flush()

    def fid_np_to_data(self, fid, dims):
        fid = np.load(fid, mmap_mode = 'r')
        fid = _2d_to_3d(fid)
        self.core_data = np.asarray(fid, dtype=bool)[dims[0]: dims[1],
                                                     dims[2]: dims[3],
                                                     dims[4]: dims[5]]

    def fid_tif_to_data(self, fid, dims):
        """Extracts data from a tif file id"""
        from PIL import Image
        fid = Image.open(fid)
        if fid.n_frames == 1:
            fid = _2d_to_3d(np.asarray(fid))
        else:
            fid = _tiff_to_3d(fid, dims[0], dims[1])
        self.core_data = np.asarray(fid, dtype=bool)[None   : None,
                                                     dims[2]: dims[3],
                                                     dims[4]: dims[5]]

    def fid_zip_to_data(self, fid, dims):
        """Extracts data from a .zip file id"""
        if dims == [None,None,None,None,None,None]:
            dims = self.fid_dir_get_dimensions(fid)
        self.core_data = self._make_memmap(self._core_name)
        archive = zipfile.ZipFile(fid)
        filelist = archive.namelist()
        filelist.sort()
        for idx, files in enumerate(filelist[dims[0]: dims[1]]):
            imgdata = archive.open(filelist[idx])
            temp_slice = gen_slice(os.path.join(fid,imgdata), dims)
            self.core_data[idx] = temp_slice
            self.core_data.flush()

    def make_mst_memmap(self):
        """Creates and defines the memmaps needed for the MST"""
        self._voidedges = []
        self.label = self._make_memmap(self._labl_name, dtype= 'uint32')
        self.label = lbl(self.core_data, structure = np.ones((3,3,3)))[0]
        self.count = np.max(self.label)
        self.e_label = self._make_memmap(self._elbl_name, dtype = 'uint32')
        self.blank_map = self._make_memmap(self._blnk_name, dtype = bool)

    def core_calcs(self):
        """Some easy calculations that describe some of the core"""
        self.porosity = np.mean(self.core_data)
        self.core_volume = self.shape[0] * self.shape[1] * self.shape[2]
        self.void_volume = (self.core_volume * self.porosity / self.count)
        self.void_radius = (self.void_volume / np.pi * 0.75)**(1./3.)
        self.void_spacing = ((self.core_volume*.75 / self.label.max() / np.pi)
                              **(1./3.) * 2)
        self.ideal_wall_ratio = (((16. / 3. * np.pi * np.sqrt(2)**3
                                   * (1. / self.porosity))**(1./3.) - 2.))*.698
        self.background = self.porosity
        self.void_mags = []
        self.paste_mags = []
        if self.verbose:
            print "Number of voids is        : ", self.count
            print "Array shape is            : ", self.shape
            print "Mean void volume is       : ", self.void_volume
            print "Mean void radius is       : ", self.void_radius
            print "Mean inter-void distance  : ", self.void_spacing
            print "Mean porosity is          : ", self.porosity
            print "Ideal paste-air ratio     : ", self.ideal_wall_ratio

    def handle_temp_folder(self, temp_folder):
        """Creates a temp_folder if one doesn't exist"""
        if temp_folder == None:
            if os.path.isdir(os.path.join(os.getcwd(), "\Temp")) == False:
                os.mkdir(os.path.join(os.getcwd(), "\Temp"))
            self.temp_folder = os.path.join(os.getcwd(), "\Temp")
            print("Folder: ", self.temp_folder)
        else:
            if os.path.isdir(temp_folder) == False:
                os.mkdir(temp_folder)
            self.temp_folder = temp_folder

    def _make_memmap(self, filename, mode='w+', dtype='uint8'):
        """Convenience function to make the memmap file"""
        parent = self.temp_folder
        temp_filename = os.path.join(parent, filename)
        fp = np.memmap(temp_filename,
                       dtype= dtype,
                       mode= mode,
                       shape= self.shape)
        return fp

    def _get_void_border(self, slice_index = None):
        """
        Creates boolean array where border points are True and all others False
        """
        if slice_index == None:
            self._voidedges = self._make_memmap(self._vdge_name, dtype = bool)
            self._voidedges = generic_gradient_magnitude(np.copy(self.core_data),
                                                        sobel,
                                                        mode = 'nearest')
            self._voidedges = (np.asarray(self._voidedges, dtype=bool)
                               * self.core_data)
        else:
            self._voidedges = generic_gradient_magnitude(self.core_data[slice_index],
                                                         sobel,
                                                         mode = 'nearest')
            self._voidedges = np.asarray(self._voidedges, dtype=bool)
        if self.verbose:
            point_num = np.where(self._voidedges==True)
            pointnum = np.size(point_num[0])
            print "Number of border points   : ", pointnum

#==============================================================================
# Triangulate class
#==============================================================================

class Triangulate(object):
    def __init__(self,
                 core,
                 **tri_configs):
        self.__dict__.update(tri_configs)
        if self.verbose:
            self.start_time = time.time()
            print "Starting ", self.method, " triangulation at : ", time.ctime()
        self.core_data = core.core_data
        self.shape = core.shape
        self.si = self.slice_index
        self.c_map = cm.cubehelix_r
        self.recalc = False
        self.candidate_dict = {}
        self.triang_list = []
        self.init_slice_configs(core)

    def init_slice_configs(self, core):
        if self.si != None:
            self.slice_index_configs(core)
        elif self.si == None:
            self.no_slice_index_configs(core)

    def brute_force(self):
        """
        Recalculates the nearest neighbor connections using full distance
        matrix computations. Probably slow if there are particularly *large*
        objects in view.
        """
        if self.verbose:
            print "Started brute force at : ", time.ctime()
        percent_vars = self.percent_c_setup(self.candidate_dict)
        i, list_length, chunk_length, temp_start_time = percent_vars
        for key,value in self.candidate_dict.iteritems():
            self.candidate_dict[key] = _cdist_refinement(value, self.e_dict)
            if self.verbose:
                i += 1
                self.percent_complete(i, chunk_length,
                                      list_length,
                                      temp_start_time)
        if self.verbose:
            print "Completed brute force at : ", time.ctime()

    def fix_pixel_voids(self):
        """
        Because our edges are inside our voids, they can be 'deleted' if the
        void is 1 pixel thick, so this adds 1 pixel thick voids back into the
        array. VERY IMPORTANT.
        """
        for idx in xrange(1,np.max(self.label)+1):
            if len(self.e_label[self.e_label==idx]) == 0:
                for point in np.c_[np.where(self.label==idx)]:
                    self.e_label[tuple(point)] = idx
        self.e_dict = {key:np.c_[np.where(self.e_label==key)]
                       for key in xrange(1, self.objs + 1)}

    def slice_index_configs(self, core):
        """Configs for when a slice index is specified"""
        if self.dims == None:
            self.dims = [0, self.shape[1], 0, self.shape[2]]
        slice_data = self.core_data[self.si][self.dims[0]: self.dims[1],
                                             self.dims[2]: self.dims[3]]
        self.pts = [(np.cos(np.pi*_/3), np.sin(np.pi*_/3))
                    for _ in xrange(6)]
        core._get_void_border(slice_index = self.si)
        self.array_in = core._voidedges[self.dims[0]: self.dims[1],
                                        self.dims[2]: self.dims[3]]
        self.temp_porosity = np.mean(slice_data)
        iwr_numerator = (np.sqrt(2*np.pi/(self.temp_porosity*np.sqrt(3)))-2)*5
        iwr_denominator = 5 / (4 + 2*np.sqrt(3))
        self.ideal_wall_ratio = iwr_numerator/iwr_denominator
        self.label, self.objs = lbl(slice_data,
                                    np.ones((3,3)))
        self.e_label = (self.array_in
                        * self.label
                        * slice_data)

    def no_slice_index_configs(self, core):
        """Configs for when a slice index is unspecified"""
        self.plotfigs = None
        if self.dims == None:
            self.dims = [0, self.shape[0],
                         0, self.shape[1],
                         0, self.shape[2]]
        temp_core_data = self.core_data[self.dims[0]: self.dims[1],
                                        self.dims[2]: self.dims[3],
                                        self.dims[4]: self.dims[5]]
        core._get_void_border()
        temp_void_data = core._voidedges[self.dims[0]: self.dims[1],
                                         self.dims[2]: self.dims[3],
                                         self.dims[4]: self.dims[5]]
        temp_label_data = core.label[self.dims[0]: self.dims[1],
                                     self.dims[2]: self.dims[3],
                                     self.dims[4]: self.dims[5]]
        self.temp_porosity = np.mean(temp_core_data)
        self.pts = [(_ / np.sqrt(2))
                    for _ in product((1,-1,0),repeat=3)
                    if np.sum(np.abs(_)) == 2]
        iwr_exp = (16*np.pi/3./self.temp_porosity)**(1./3)
        iwr_numerator = 11.*iwr_exp
        iwr_denominator = ((2*(3+np.sqrt(2)+2*np.sqrt(3)))
                          *(np.sqrt(2)-2))
        self.ideal_wall_ratio = iwr_numerator/iwr_denominator
        self.e_label = temp_void_data * temp_label_data
        self.array_in = self.e_label
        self.label, self.objs = lbl(temp_core_data,
                                    np.ones((3,3,3)))

    def percent_complete(self, i, chunk_len, list_len, start_time):
        if i % chunk_len == 0:
            p_comp = float(i) / float(list_len)
            current_time = (time.time() - start_time)
            ttf = current_time / (p_comp)
            print("%d percent complete, in sec %d of %d"
                   % (int(100 * p_comp), int(current_time), int(ttf)))

    def percent_c_setup(self, input_list):
        i = 0
        list_length = len(input_list)
        chunk_length = np.max((int(np.round(list_length/100)), self.verbosity))
        temp_start_time = time.time()
        return i, list_length, chunk_length, temp_start_time

    def plot_figs(self): # break into two sperate fns
        """
        When the slice index is specified, used to create a set of six
        2-dimensional figures at the slice index.
        """
        for idx in xrange(1, self.objs + 1):
            rand_color = np.random.choice(list(np.arange(self.objs * .5,
                                                         self.objs * 2)),
                                                         replace = False)
            for point in np.c_[np.where(self.label == idx)]:
                self.label[tuple(point)] = rand_color
            for point in np.c_[np.where(self.e_label == idx)]:
                self.e_label[tuple(point)] = rand_color

        plot_list = [[None               , None              ],
                     [None               , None              ],
                     [self.triang_list   , None              ],
                     [self.candidate_list, self.fbt_min_v_map],
                     [self.rng_list      , self.rng_min_v_map],
                     [self.paste_list    , self.min_v_map    ]]

        fig_list = [[self.fig00, self.fig01],
                    [self.fig10, self.fig11],
                    [self.fig20, self.fig21]]

        title_list = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
        f, axarr = plt.subplots(3, 2)

        axarr[0, 1].imshow(self.fig01, cmap = self.c_map)

        plot_index = 0
        for i in xrange(3):
            for j in xrange(2):
                plot_index += 1
                if i == 0 and j == 0:
                    axarr[i, j].imshow(fig_list[i][j], cmap = cm.gray_r)
                    x1,x2,y1,y2 = axarr[i, j].axis()
                else:
                    axarr[i,j].imshow(fig_list[i][j], cmap = self.c_map)
                    for vals in plot_list[plot_index][0]:
                        axarr[i,j].plot(*vals, marker = ' ', color = 'r')
                    for vals in plot_list[plot_index][1]:
                        axarr[i,j].plot(*vals, marker = ' ', color = 'b')
                axarr[i, j].axis((x1,x2,y1,y2))
                axarr[i, j].set_xticklabels([])
                axarr[i, j].set_xticks([])
                axarr[i, j].set_yticklabels([])
                axarr[i, j].set_yticks([])
                axarr[i, j].set_xlabel(title_list.pop(0))
        f.tight_layout()

    def est_void_size(self):
        self.est_fbt()
        self.est_rng()
        self.est_mst()
        self.est_min_v_maps()
        self.est_pastes_voids()

        [self.fbt_paste_mean,
         self.fbt_void_mean,
         self.fbt_Paste_air_ratio_n,
         self.fbt_Homogeneity,
         self.rng_paste_mean,
         self.rng_void_mean,
         self.rng_Paste_air_ratio_n,
         self.rng_Homogeneity,
         self.mst_paste_mean,
         self.mst_void_mean,
         self.mst_Paste_air_ratio_n,
         self.mst_Homogeneity, ] = list(chain(*[_gen_parameters(
                                              p,v,
                                              self.correction_factor,
                                              self.ideal_wall_ratio)
                                        for p,v in self.paste_void_pairs]))

    def est_fbt(self):
        (self.fbt_paste_list,
         self.fbt_paste_mags,
         self.fbt_label_list,
         self.fbt_point_list) = _gen_lists_from_dict(self.candidate_dict,
                                                     self.recalc)

    def est_rng(self):
        (self.rng_paste_list,
         self.rng_paste_mags,
         self.rng_label_list,
         self.rng_point_list) = _gen_lists_from_dict(self.rng_dict,
                                                     self.recalc)

    def est_mst(self):
        (self.paste_list,
         self.paste_mags,
         self.label_list,
         self.point_list) = _gen_lists_from_dict(self.mst_dict,
                                                 self.recalc)

    def est_min_v_maps(self):
        (self.fbt_void_mags,
         self.fbt_min_v_map) = _calc_voids_chords(self.fbt_point_list,
                                                  self.e_label)
        (self.rng_void_mags,
         self.rng_min_v_map) = _calc_voids_chords(self.rng_point_list,
                                                  self.e_label)
        (self.void_mags,
         self.min_v_map) = _calc_voids_chords(self.point_list,
                                              self.e_label)

    def est_pastes_voids(self):
        self._pastes = [self.fbt_paste_mags,
                        self.rng_paste_mags,
                        self.paste_mags]
        self._voids = [self.fbt_void_mags,
                       self.rng_void_mags,
                       self.void_mags]
        self.paste_void_pairs = zip(self._pastes, self._voids)

    def recolor_array_fn(self):
        self.recolor_array = self.label
        for idx in xrange(1, self.objs + 1):
            rand_color = np.random.choice(list(np.arange(self.objs * .5,
                                                         self.objs * 2)),
                                                         replace = False)
            for point in np.c_[np.where(self.recolor_array == idx)]:
                self.recolor_array[tuple(point)] = rand_color

    def verbose_output(self):
        print 'Completion time           : ', time.ctime()
        print 'after time taken          : ', time.time()-self.start_time
        # print 'Time taken                : ', self.method_runtime
        print 'Porosity                  : ', self.temp_porosity
        print 'Ideal wall ratio          : ', self.ideal_wall_ratio
        print 'Paste walls spanned       : ', len(self.paste_mags)
        print 'RNG Paste walls spanned   : ', len(self.rng_paste_mags)
        print '-fbt Paste-air-ratio-     : ', self.fbt_Paste_air_ratio_n
        print '-fbt wall thickness-      : ', self.fbt_paste_mean
        print '-fbt chord length-        : ', self.fbt_void_mean
        print '-fbt Homogeneity-         : ', self.fbt_Homogeneity
        print '-RNG Paste-air-ratio-     : ', self.rng_Paste_air_ratio_n
        print '-RNG wall thickness-      : ', self.rng_paste_mean
        print '-RNG chord length-        : ', self.rng_void_mean
        print '-RNG Homogeneity-         : ', self.rng_Homogeneity
        print '-MST Paste-air-ratio-     : ', self.mst_Paste_air_ratio_n
        print '-MST wall thickness-      : ', self.mst_paste_mean
        print '-MST chord length-        : ', self.mst_void_mean
        print '-MST Homogeneity-         : ', self.mst_Homogeneity

    def mst_out(self):
        return [self.mst_Paste_air_ratio_n,
                self.mst_paste_mean,
                self.mst_void_mean,
                self.mst_Homogeneity,
                self.rng_Paste_air_ratio_n,
                self.rng_paste_mean,
                self.rng_void_mean,
                self.rng_Homogeneity,
                self.fbt_Paste_air_ratio_n,
                self.fbt_paste_mean,
                self.fbt_void_mean,
                self.fbt_Homogeneity,
                self.temp_porosity,
                self.data_length]

    def run(self, core):
        if self.method == 'Delaunay':
            de = DelaunayTri(core)
            return de.run()
        elif self.method == 'Dirichlet':
            di = DirichletTri(core)
            return di.run()

#==============================================================================
# Delaunay subclass
#==============================================================================

class DelaunayTri(Triangulate):
    def __init__(self, core, **kwargs):
        super(DelaunayTri, self).__init__(core, **kwargs)
        self.correction_factor = _gen_correction_factor(self.pts)
        self.dim_num = len(self.array_in.shape)
        self.max_size = np.sqrt(np.sum(np.array(self.array_in.shape)**2))
        super(DelaunayTri, self).fix_pixel_voids()
        self.e_list = np.c_[np.where(self.e_label)]
        self.data_length = len(self.e_list)
        self.triang = Delaunay(self.e_list,
                               qhull_options = self.qhull_options)

    def fill_label_dicts(self):
        self.label_dict = {}
        self.edge_dict = {}
        for simplex in self.triang.vertices:
            self.label_dict[tuple(simplex)] = [self.e_label[
                                               tuple(self.e_list[_])]
                                               for _ in simplex]
        for simplex, labels in self.label_dict.iteritems():
            sim_pairs = list(combinations(simplex,2))
            lab_pairs = list(combinations(labels,2))
            len_pairs = [len(set(_)) for _ in lab_pairs]
            for idx, pair in enumerate(sim_pairs):
                if len_pairs[idx] > 1:
                    lab_pair = tuple(sorted(lab_pairs[idx]))
                    self.edge_dict[tuple(sorted(pair))] = lab_pair
        if self.verbose:
                print "completed edge_dict at :", time.ctime()

    def fill_edge_dicts(self):
        percent_vars = super(DelaunayTri, self).percent_c_setup(self.edge_dict)
        i, list_length, chunk_length, temp_start_time = percent_vars
        for pair, lab_val in self.edge_dict.iteritems():
            pair = [self.triang.points[_] for _ in pair]
            point_2, point_1  = np.array(pair[0]), np.array(pair[1])
            self.triang_list.append(_gen_trans_list(pair))
            cur_dist = euclidean(point_1, point_2)
            if lab_val in self.candidate_dict:
                mst_dist = self.candidate_dict[lab_val][2]
            else:
                mst_dist = self.max_size
            if cur_dist < mst_dist:
                data_list = _gen_data_list([point_1, point_2],
                                           lab_val[0],
                                           lab_val[1],
                                           cur_dist)
                self.candidate_dict[lab_val] = data_list
            if self.verbose:
                i += 1
                super(DelaunayTri, self).percent_complete(i, chunk_length,
                                                       list_length,
                                                       temp_start_time)

    def create_figs(self):
        self.fig00 = self.core_data[self.si]
        self.fig01 = self.recolor_array
        self.fig10 = self.recolor_array
        self.fig11 = self.recolor_array
        self.fig20 = self.recolor_array
        self.fig21 = self.recolor_array
        self.background = self.fig11

    def run(self):
        self.fill_label_dicts()
        self.fill_edge_dicts()
        if self.brute_force:
            super(DelaunayTri, self).brute_force()
        self.rng_dict = _gen_rng_dict(self.candidate_dict)
        self.mst_dict = _gen_mst_dict(self.rng_dict)
        if self.si != None and self.plotfigs:
            super(DelaunayTri, self).recolor_array_fn()
            self.create_figs()
            super(DelaunayTri, self).plot_figs()
        super(DelaunayTri, self).est_void_size()
        if self.verbose:
            super(DelaunayTri, self).verbose_output()

#==============================================================================
# Dirichlet subclass
#==============================================================================

class DirichletTri(Triangulate):
    def __init__(self, core, **kwargs):
        super(DirichletTri, self).__init__(core, **kwargs)
        self.sub_slice_configs()
        self.data_length = self.objs
        self.indices =  np.zeros(((np.ndim(self.array_in),)
                                   + self.array_in.shape),
                                   dtype=np.int32)
        self.correction_factor = _gen_correction_factor(self.pts)
        self.dim_num = len(self.array_in.shape)
        self.max_size = np.sqrt(np.sum(np.array(self.array_in.shape)**2))
        super(DirichletTri, self).fix_pixel_voids()
        self.sub_slice_configs()
        self.init_edt_array()

    def init_edt_array(self):
        self.edt_array = edt(~self.array_in.astype(bool),
                             return_indices = True,
                             indices = self.indices)
        self.water_array = water(self.edt_array, self.label)
        self.edges_array = generic_gradient_magnitude(self.water_array, sobel)
        self.edt_array[self.edges_array == 0] = 0
        self.edt_array_copy = self.edt_array.copy()
        find_array = np.where(self.edt_array)
        self._z = self.edt_array[find_array]
        self.border_coords = np.c_[find_array]
        self.edt_max = self.edt_array.max()
        self.edt_array[self.edges_array == 0] = self.edt_max
        sort_z = np.argsort(self._z)
        self._z = self._z[sort_z]
        self.border_coords = self.border_coords[sort_z]

    def sub_slice_configs(self):
        if self.si != None:
            self.array_in = self.core_data[self.si]
        elif self.si == None:
            self.array_in = self.core_data[self.dims[0]: self.dims[1],
                                           self.dims[2]: self.dims[3],
                                           self.dims[4]: self.dims[5]]

    def get_points(self, bc, guest_arrays):
        point_1 = [self.indices[x][bc] for x in xrange(self.dim_num)]
        p_0 = self.label[tuple(point_1)]
        [possible_labels, possible_dists] = guest_arrays
        possible_labels[possible_labels == p_0] = self.edt_max
        p_1 = np.unique(possible_labels)[0]
        possible_crds = np.zeros_like(possible_labels)
        if self.dim_num == 2:
            possible_crds.resize((2,3,3))
        if self.dim_num == 3:
            possible_crds.resize((3,3,3,3))
        for idx, host in enumerate(self.indices):
            possible_crds[idx] = _extract_subset(bc, host, self.dim_num, 1)
        point_2 = []
        possible_dists[possible_labels != p_1] = self.edt_max
        dist_2 = np.min(possible_dists)
        co_loc = np.where(possible_dists == dist_2)
        for dim in xrange(self.dim_num):
            point_2.append(possible_crds[dim][tuple(co_loc)][0])
        p_1 = self.label[tuple(point_2)]
        return p_0, p_1, point_1, point_2

    def border_pt_helper(self):
        host_arrays = [self.water_array, self.edt_array]
        percent_vars = super(DirichletTri, self).percent_c_setup(self._z)
        i, list_length, chunk_length, temp_start_time = percent_vars
        for bc in self.border_coords:
            bc = tuple(bc)
            go_to_next_point = False
            for dim in range(self.dim_num):
                if (bc[dim] < 3) or (bc[dim] > self.array_in.shape[dim] - 3):
                    go_to_next_point = True
            if go_to_next_point:
                continue
            guest_arrays = [None, None]
            for idx, host in enumerate(host_arrays):
                guest_arrays[idx] = _extract_subset(bc, host, self.dim_num, 1)
            p_0, p_1, point_1, point_2 = self.get_points(bc, guest_arrays)
            if p_1 == p_0:
                continue
            if p_1 > p_0:
                p_1, p_0 =  p_0, p_1
                point_2, point_1 = point_1, point_2
            current = 0
            if (p_0, p_1) in self.candidate_dict:
                current = self.candidate_dict[(p_0, p_1)][1]
                print current
            point_1, point_2 = np.asarray([point_1, point_2])
            if(current == 0):
                cur_dist = 1/euclidean(point_1, point_2)
            else:
                cur_dist = 1/((1/current) + euclidean(point_1, point_2))
            data_list = _gen_data_list([point_1.astype(float),
                                        point_2.astype(float)],
                                        p_0, p_1,
                                        cur_dist)
            self.candidate_dict[(p_0, p_1)] = data_list
            if self.verbose:
                i += 1
                super(DirichletTri, self).percent_complete(i, chunk_length,
                                                        list_length,
                                                        temp_start_time)
        if self.verbose:
            print "completed candidate_dict at :", time.ctime()
            print 'serial time taken         : ', time.time() - temp_start_time

    def create_figs(self):
        self.edt_array_copy[self.edges_array == 0] = 1
        self.edt_array_copy = 1 / self.edt_array_copy / self.objs
        self.edt_array_copy[self.edges_array == 0] = 0
        self.edt_array_copy[self.edt_array_copy == np.inf] = 0
        self.border_fig = (self.edt_array_copy
                           + self.recolor_array
                           * self.edt_array_copy.max()
                           / self.recolor_array.max())
        self.fig00 = self.core_data[self.si]
        self.fig01 = self.recolor_array
        self.fig10 = (np.sqrt(self.edt_array_copy)
                      + self.recolor_array
                      * np.max(np.sqrt(self.edt_array_copy))
                      / self.recolor_array.max())
        self.fig11 = self.border_fig
        self.fig20 = self.border_fig
        self.fig21 = self.border_fig
        self.fig30 = (self.core_data[self.si]
                      * self.edt_array_copy.max()
                      + self.edt_array_copy)
        self.background = self.fig11

    def run(self):
        self.border_pt_helper()
        if self.brute_force:
            super(DirichletTri, self).brute_force()
        self.rng_dict = _gen_rng_dict(self.candidate_dict)
        print(self.candidate_dict)
        if self.verbose:
            print "Completed rng_dict at :", time.ctime()
        self.mst_dict = _gen_mst_dict(self.rng_dict)
        if self.verbose:
            print "Completed mst_dict at :", time.ctime()
        if self.si != None and self.plotfigs:
            super(DirichletTri, self).recolor_array_fn()
            self.create_figs()
            super(DirichletTri, self).plot_figs()
        super(DirichletTri, self).est_void_size()
        if self.verbose:
            super(DirichletTri, self).verbose_output()

#==============================================================================
# Plot Pores
#==============================================================================

def plot_pores(core,
               paste = True,
               voids = True,
               subgraph = 'mst',
               o_map = None):
    '''
        A routine to plot the pores in 3d with mayavi. Also plots the
        edge-to-edge distances and associated chords.
        Input:
            paste: bool
                Whether or not to plot the edge-to-edge distances.
            voids: bool
                Whether or not to plot the associated chords.
            subgraph: string
                Indicates which subgraph to use possible arguments are:
                'fbt' : Finite Body Triangulation
                'rng' : Relative Neighborhood Graph
                'mst' : Minimum Spanning Tree
                None  : Do not plot edges or chords.
            o_map = 3d array
                Allows for rendering other arrays. Default is None, which
                uses the object core_data array.
        Returns:
            None
        Outputs:
            An animated rotation of the array, with associated edges and
            chords.
    '''

    subgraphs = {'fbt', 'rng', 'mst', None}
    try:
        inclusion_test = subgraph in subgraphs
        assert inclusion_test == True
    except:
        print 'please pick one of the following values for subgraph: fbt, rng, mst, None'
        raise

        ### We don't need subgraphs if we are not displaying paste or void
    if not paste and not voids :
        subgraph = None

    if subgraph == 'fbt':
        voids_data = core.fbt_min_v_map
        paste_data = core.fbt_paste_list
    if subgraph == 'rng':
        voids_data = core.rng_min_v_map
        paste_data = core.rng_paste_list
    if subgraph == 'mst':
        voids_data = core.min_v_map
        paste_data = core.paste_list
    if subgraph == None:
        voids_data = []
        paste_data = []

    if o_map == None:
        o_map = core.core_data

    from mayavi import mlab

    @mlab.animate
    def anim():
        f = mlab.gcf()
        while 1:
            f.scene.camera.azimuth(1)
            f.scene.render()
            yield

    mlab.figure(bgcolor=(1,1,1)) # Set bkg color
    mlab.contour3d(o_map,
                   color = (0,0,0),
                   contours = 2,
                   opacity = .2 + .8/core.shape[0])
    if paste:
        for vals in paste_data:
            mlab.plot3d(*vals,
                        tube_radius=0.1 + .05/core.shape[0],
                        color = (1, 0, 0)) # Draw paste lines
    if voids:
        for vals in voids_data:
            mlab.plot3d(*vals,
                        tube_radius=0.1 + .05/core.shape[0],
                        color = (0, 0, 1)) # Draw void lines

    a = anim() # Start the animation.
