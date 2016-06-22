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
### EXAMPLE ON HOW TO RUN WITH A TIFF FILE ###
Type the following into the console after running this script:
1. from PIL import Image
2. imgsrc = r'/home/jcrnk/Documents/SummerRP/core.tif' # This is your directory where the tif is located
3. img = Image.open(imgsrc)
4. c = CoreScan(img, dims=[0,100,0,50,0,75], temp_folder=tempf)
5. c.triangulate()
6. c.plot_pores(subgraph='mst')
"""
"""
CHANGES:
1. Fixed 2D slices not appearing in mayavi
2. Moved over to PIL to handle images
3. Tiff files with multiple frames now easier to display
4. Fixed problem where mst's void maps and mags were switched
5. Fixed temp folders not being created if a temp folder didn't exist
"""
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


### Two methods to generate synthetic arrays in 2d and 3d respectively.
def gen_test_fcc(x_dim_o, y_dim_o, z_dim_o, d_spacing, r_void):
    '''
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
        
    '''
    
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
                if base_slice[z_d : z_d + c_z, y_d : y_d + c_y, x_d : x_d + c_x].shape != np.shape(circle):
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


### Internal methods used in CoreScan object. They are presented externally 
### for ease of testing and troubleshooting.

def _2d_to_3d(fid): # turns 2d numpy arrays into 3d numpy arrays
    if len(fid.shape) == 2:
        fid = fid.tolist()
        fid = [fid]
    return fid

def _tiff_to_3d(img,start_frame,end_frame):
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


def _cdist_refinement(entry, dict_in):

    def _cydist(a, b):
        
        rows, dims = a.shape
        cols = b.shape[0]
        out = np.zeros((rows, cols), dtype=int)        
        for dim in range(dims):
            out += np.subtract(a[:,dim].ravel()[:, None], b[:,dim].ravel()[None, :])**2
                
        return out

    obj_coords = [dict_in[_] for _ in entry[2:4]]
    c_dist = _cydist(*obj_coords)
    indices = np.unravel_index(c_dist.argmin(), c_dist.shape)
    point_list = [obj_coords[_][indices[_]] for _ in range(2)]
    return _gen_data_list(point_list, entry[0], entry[1], np.sqrt(c_dist[indices]))
  
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

def _gen_parameters(paste, void, correction_factor, ideal_ratio):
    """
    Designed to return the means and ratio (along with standard deviations)
    of two lognormed rvs sets.  Removes correction factor from void stdev.
    """
    ### obtains the averages and log stdevs of the paste and void lengths
    p_m, p_s, v_m, v_s = list(chain(*[(np.mean(_), np.std(np.log(_))) for _ in [paste, void]]))
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

def _gen_rng_dict(candidate_dict):
    rng_dict = {}
    for vals in candidate_dict:
        add_ij = True
        k_vals_1 = set([_ for _ in candidate_dict.keys() if _[0] == vals[0] or _[1] == vals[0]])
        k_vals_2 = set([_ for _ in candidate_dict.keys() if _[0] == vals[1] or _[1] == vals[1]])
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

def _gen_mst_dict(rng_dict):
    ## code to generate an mst dict from the rng_dict
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

def _gen_data_list(point_list, p_1, p_0, cur_dist):
    # convenience function to format the data list for candidate edges.
    
    trans_list = _gen_trans_list(point_list)
    data_list = (trans_list,
                 cur_dist,
                 p_1,
                 p_0,
                 point_list)
    return data_list

def _calc_voids_chords(point_list, label_image):
    # find voids with chords and obtain their length and the endpoints of the 
    # chords.    
    
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
            min_v_map += [([p_1[1], p_2[1]],[p_1[0], p_2[0]]) for [p_1, p_2] in list(pairs)]
        if dim_num == 3:
            min_v_map += [[[p_1[0], p_2[0]], [p_1[1], p_2[1]],[p_1[2], p_2[2]]] for [p_1, p_2] in list(pairs+1)]

    return void_mag_list, min_v_map

def rem_files_in_list(fid, dat_file_names_list):
    for name in dat_file_names_list:
        temp_filename = os.path.join(fid, name)
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

def wt_ch(non_integer):
    # a way to generate integers by selecting either the floor or ceiling of 
    # the non-integer, weighted by the value of number after the decimal point.
    if np.round(non_integer) - non_integer == 0:
        output = non_integer
    else:
        choices = (np.floor(non_integer), np.ceil(non_integer))
        weights = (np.ceil(non_integer) - non_integer, non_integer - np.floor(non_integer))
        output = np.random.choice(choices, p=weights)
    return output    


def gen_cell_slice(x_dim_o, y_dim_o, d_spacing, r_void):
    '''
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
        
    '''

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
            x_d =  - wt_ch(d_spacing / 2)
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
                print 'Warning, ValueError: ', circle.shape, np.shape(base_slice[y_d : y_d + _y, 
                           x_d : x_d + _x])
            
        x_shift = not x_shift

    out_slice = np.abs(base_slice)[wt_ch(y_s):x_dim_o + wt_ch(y_s), 
                                   wt_ch(y_s):y_dim_o + wt_ch(y_s)]    
    
    return out_slice
    
    ''' This is a convenience function to grab the data from the file '''         
def gen_slice(fid, dims):
    y_0, y_1, x_0, x_1 = dims[2:]
    with open(fid) as f:
        temp_slice = np.loadtxt(f, 
                                delimiter='\t', 
                                dtype = bool)[y_0: y_1, x_0: x_1]
    return temp_slice
    
    
def read_fid(self, fid, dims):
    dat_file_names_list = [ self._core_name,
                           self._dlat_name,
                           self._labl_name,
                           self._vdge_name,
                           self._skel_name,
                           self._skel_labl,
                           self._elbl_name,
                           self._blnk_name]
    '''  If the fid value is not a string, it might be a...'''
    if fid.__class__ != str: 
        ''' numpy array, or a... ''' 
        if fid.__class__ == np.array([0,0]).__class__:
            fid = _2d_to_3d(fid) 
            self.core_data = np.asarray(fid, dtype=bool)[dims[0]: dims[1], 
                                                         dims[2]: dims[3],
                                                         dims[4]: dims[5]]

        ''' numpy memmap object. '''
        if str(fid.__class__) == "<class 'numpy.core.memmap.memmap'>":
            fid = _2d_to_3d(fid) 
            self.core_data = fid[dims[0]: dims[1], 
                                 dims[2]: dims[3],
                                 dims[4]: dims[5]]
                    
        ''' tiff image. '''
        if str(fid.__class__) == "<class 'PIL.TiffImagePlugin.TiffImageFile'>":
            if fid.n_frames == 1: # 2d slice of a core
                fid = _2d_to_3d(np.asarray(fid)) 
            else:
                fid = _tiff_to_3d(fid, dims[0], dims[1])
                self.core_data = np.asarray(fid, dtype=bool)[None   : None, 
                                                             dims[2]: dims[3],
                                                             dims[4]: dims[5]]
        if self.verbose:
            print "Data loaded from : ", fid.__class__            
            fid = os.path.curdir                    
        self.shape = self.core_data.shape
        self.fileid = fid
    else:
        ''' We need to remove pre-existing files if they are already there ''' 
        if os.path.isdir(fid) or os.path.isfile(fid):
            rem_files_in_list(self.temp_folder, dat_file_names_list)
        
            ''' We will make memmap objects, which have predefined sizes. So,
            memmap needs to know its size before it is created. We read the first
            file to determine the array size, if we don't already know'''
            if dims == [None,None,None,None,None,None]:
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
        
            self.shape = (dims[1] - dims[0],
                          dims[3] - dims[2], 
                          dims[5] - dims[4])        
            if fid.__class__ == str and os.path.isdir(fid):
                self.core_data = self._make_memmap(fid, self._core_name)
                filelist = os.listdir(fid)
                filelist.sort()
                for idx, files in enumerate(filelist[dims[0]: dims[1]]):
                    temp_slice = gen_slice(os.path.join(fid,files), dims)
                    self.core_data[idx] = temp_slice
                    self.core_data.flush()
                    if idx%100 == 0:
                        print 'Now loading slice : ', idx, ':', files
                self.fileid = fid
                if self.verbose:
                    print "Data loaded from : ", fid
                        
                '''   a zip file, ''' 
            if fid.__class__ == str and fid.endswith('.zip'):
                self.core_data = self._make_memmap(fid, self._core_name)
                archive = zipfile.ZipFile(fid)
                filelist = archive.namelist()
                filelist.sort()
                for idx, files in enumerate(filelist[dims[0]: dims[1]]):
                    imgdata = archive.open(filelist[idx])
                    temp_slice = gen_slice(os.path.join(fid,files), dims)
                    self.core_data[idx] = temp_slice
                    self.core_data.flush()
                self.fileid = fid
                if self.verbose:
                    print "Zipped data loaded from : ", fid
        
                '''   and a numpy file. ''' 
            if fid.__class__ == str and fid.endswith('.npy'):
                self.core_data = np.load(fid, mmap_mode='r')[dims[0]: dims[1], 
                                                             dims[2]: dims[3],
                                                             dims[4]: dims[5]]
                self.core_data = np.array(self.core_data, dtype=bool)
                self.fileid = fid
                if self.verbose:
                    print "Npy file loaded from : ", fid

class CoreScan:
    """Base class for analyzing slices pores from cement or porous material.

    Attributes:
        core_data (numpy.ndarray): a 2d array of binary data.
        
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
                 verbose=False,
                 temp_folder = None,
                 invert_array = True):
        """Base object for CoreScan.
    
        Input:
            - fid, a file location for a tab-delimited file containing zeros
              where matrix is present and a constant, finite value where voids
              are present
            - fid, a numpy array. 

        Example:
            >>> import pycoresis as pcs
            >>> fid = r'C:\YOUR\FILE\HERE.txt'
            >>> crs = pcs.CoreScan(fid)
            Data loaded from :  'C:\YOUR\FILE\HERE.txt'
            Array shape is   :  (200, 200)
            Mean porosity is :  0.0932
        """
            
        ''' These filenames will be written in the parent dir of fid '''
            
        self.verbose = verbose
        self._core_name = 'core_data.dat'
        self._dlat_name = 'core_dilate.dat'
        self._labl_name = 'core_label.dat'
        self._vdge_name = 'void_edges.dat'
        self._skel_name = 'skel_data.dat'
        self._skel_labl = 'skel_labl.dat'
        self._elbl_name = 'elabel.dat'
        self._blnk_name = 'blank.dat'


        # create a temp_folder if one doesn't exist
        if temp_folder == None:
            if os.path.isdir(os.path.join(os.getcwd(), "\Temp")) == False:
                os.mkdir(os.path.join(os.getcwd(), "\Temp"))
            self.temp_folder = os.path.join(os.getcwd(), "\Temp")
            print("Folder: ", self.temp_folder)
        else:
            if os.path.isdir(temp_folder) == False:
                os.mkdir(temp_folder)
            self.temp_folder = temp_folder 
            
        read_fid(self, fid, dims)

        if self.verbose:
            print time.ctime()
            

        if invert_array == True:
            print(self.core_data)
            self.core_data = np.invert(self.core_data)
            print(self.core_data)
        
        self.core_data = self.core_data / self.core_data.max()
        
        ''' we then create and define the memmaps needed for the MST ''' 

        self._voidedges = []
        self.label = self._make_memmap(fid, self._labl_name, dtype= 'uint32')
        self.label = lbl(self.core_data, structure = np.ones((3,3,3)))[0]
        self.count = np.max(self.label)
        
        self.e_label = self._make_memmap(fid, self._elbl_name, dtype = 'uint32')

        self.blank_map = self._make_memmap(fid, self._blnk_name, dtype = bool)
            
        ''' these are some easy calculations that describe some of the core ''' 
        self.porosity = np.mean(self.core_data) 
        self.core_volume = self.shape[0] * self.shape[1] * self.shape[2]
        self.void_volume = (self.core_volume * self.porosity / self.count)
        self.void_radius = (self.void_volume / np.pi * 0.75)**(1./3.)
        self.void_spacing = (self.core_volume *.75 / self.label.max() / np.pi)**(1./3.) * 2
        self.ideal_wall_ratio = (((16. / 3. * np.pi * np.sqrt(2)**3 * 
                                 ((1. / self.porosity)))**(1./3.) - 2.))*.698
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
    
            
    def _make_memmap(self, fid, filename, mode='w+', dtype='uint8'):
        ''' this is a convenience function to make the memmap file ''' 
        parent = self.temp_folder
        temp_filename = os.path.join(parent, filename)
        fp = np.memmap(temp_filename, 
                       dtype= dtype, 
                       mode= mode, 
                       shape= self.shape)
        return fp    

        
    def _get_void_border(self, slice_index = None):
        """Create boolean array where border points are True and all others
        False.
    
        Input:
            - none
            
        Returns:
            - voidedges

        Example:
            >>> import pycoresis as pcs
            >>> fid = r'C:\YOUR\FILE\HERE.txt'
            >>> crs = pcs.CoreScan(fid)
            >>> crs._get_void_border()
            Number of border points : 2449
            Number of border points : 3245
            array([[ True,  True,  True, ...,  True,  True,  True],
                   [ True, False, False, ..., False, False,  True],
                   [ True, False, False, ..., False, False,  True],
                   ..., 
                   [ True, False, False, ..., False, False,  True],
                   [ True, False, False, ..., False, False,  True],
                   [ True,  True,  True, ...,  True,  True,  True]], dtype=bool)
        """



        if slice_index == None:
            self._voidedges = self._make_memmap(self.fileid, self._vdge_name, dtype = bool)
            self._voidedges = generic_gradient_magnitude(np.copy(self.core_data), 
                                                        sobel,
                                                        mode = 'nearest')
            self._voidedges = np.asarray(self._voidedges, dtype=bool) * self.core_data
            if self.verbose:
                point_num = np.where(self._voidedges==True)
                pointnum = np.size(point_num[0])
                print "Number of border points   : ", pointnum
            
        else:
            self._voidedges = generic_gradient_magnitude(self.core_data[slice_index],
                                                         sobel,
                                                         mode = 'nearest')
            self._voidedges = np.asarray(self._voidedges, dtype=bool)
            if self.verbose:
                point_num = np.where(self._voidedges==True)
                pointnum = np.size(point_num[0])
                print "Number of border points   : ", pointnum
        
    def triangulate(self,
               slice_index = None,
               dims = None,
               plotfigs = True,
               method = 'Delaunay',
               qhull_options = 'QJ',
               brute_force = False,
               verbosity = 40000):
        
        '''
        A routine to determine the minimum spanning tree that connects the pores
        on a 2D or 3D binary image. 
        Input:
            slice_index: int, Optional
                The index of the slice for analysis. If left blank, 3d.

            dims: [int, int, int, int, int, int], Optional
                The location of the core scan where analysis is desired as
                six integers consisting of three pairs each consisting of
                a starting index locating and an ending one.
                [z0, z1, y0, y1, x0, x1]
            plotfigs: boolean
                A flag to plot some figures.
            method: string
                The method of computation desired. Accepts the arguments:
                    'Delaunay' for the Delaunay triangulation 
                    'Dirichlet' for the use of Euclidean distance maps
            qhull_options: string
                optional arguments to supply to the qhull algorthim. See
                Brad Barber's qhull for more information.
            brute_force: bool
                A flag that recomputes the nearest neighbor pair of points for
                each pair of objects identified as adjacent. Adds computational
                time, but increases accuracy of estimate.
            verbosity: int
                If verbosity is true, sets the number maximum number of objects
                analyzed between status updates printed to the REPL. Increase
                for fewer updates and decrease for more updates.

        Output:
        mst_output =   mst_Paste_air_ratio_n, : the ratio of edge length to 
                                                chord length of the minimum 
                                                spanning tree (mst).
                       mst_paste_mean,        : the average edge length of mst
                       mst_void_mean,         : the average chord length of mst
                       mst_Homogeneity,       : the homogeneity of the image
                                                based on the mst.
                       rng_Paste_air_ratio_n, : the ratio of edge length to 
                                                chord length of the relative 
                                                neighborhood graph (rng).
                       rng_paste_mean,        : the average edge length of rng
                       rng_void_mean,         : the average chord length of rng
                       rng_Homogeneity,       : the homogeneity of the image
                                                based on the rng.
                       fbt_Paste_air_ratio_n, : the ratio of edge length to 
                                                chord length of the finite body
                                                triangulation (fbt).
                       fbt_paste_mean,        : the average edge length of fbt
                       fbt_void_mean,         : the average chord length of fbt
                       fbt_Homogeneity,       : the homogeneity of the image
                                                based on the fbt.
                       [method_runtime],      : the total runtime
                       [temp_porosity],       : the average porosity of the 
                                                measured section.
                       [data_length]          : the number of objects processed
                                                during the calculation.
                
        
        '''
        
        ### This tests to determine if we have selected a correct method
        methods = {'Delaunay', 'Dirichlet'}
        try:
            inclusion_test = method in methods
            assert inclusion_test == True
        except:
            print 'please pick one of the following values for subgraph: Delaunay or Dirichlet'
            raise

        c_map = cm.cubehelix_r
        
        function = lambda a,b: euclidean(a,b)
        recalc = False

        if slice_index != None:
            
            if dims == None:
                dims = [0, self.shape[1], 0,self.shape[2]]

            si = slice_index

            label, objs = lbl(self.core_data[si][dims[0]:dims[1], dims[2]:dims[3]], np.ones((3,3)))
            print(label)
            
            pts = [(np.cos(2*np.pi*_/6), np.sin(2*np.pi*_/6)) for _ in range(6)]
    
            self._get_void_border(slice_index = si)
            array_in = self._voidedges[dims[0]:dims[1], dims[2]:dims[3]]
            
            temp_porosity = np.mean(self.core_data[si][dims[0]:dims[1], dims[2]:dims[3]])
    
            ideal_wall_ratio = (np.sqrt ( 2 * np.pi / (temp_porosity * np.sqrt(3))) - 2) *  5 / (4 + 2 * np.sqrt(3))
    
            e_label = array_in * label * self.core_data[si][dims[0]:dims[1], dims[2]:dims[3]]
    
            if method == 'Dirichlet':
                array_in = self.core_data[si]
                
#                e_label = array_in * label
                data_length = objs
    
                indices = np.zeros(((np.ndim(array_in),) + array_in.shape), dtype=np.int32)
                
        if slice_index == None:
            
            plotfigs = False
            if dims == None:
                dims = [0, self.shape[0], 0, self.shape[1], 0,self.shape[2]] 
                
            temp_porosity = np.mean(self.core_data[dims[0]:dims[1], dims[2]:dims[3], dims[4]:dims[5]])
            
            pts = [(_ / np.sqrt(2)) for _ in product((1,-1,0),repeat=3) if np.sum(np.abs(_)) == 2]
            
            ideal_wall_ratio = ( 11. / (2. * (3. + np.sqrt(2.) + 2 * np.sqrt(3)))  * 
                                 ((16 * np.pi / 3 / temp_porosity)**(1./3.) / 
                                  np.sqrt(2) - 2))

            self._get_void_border()
            e_label = (self._voidedges[dims[0]:dims[1], dims[2]:dims[3], dims[4]:dims[5]] * 
                        self.label[dims[0]:dims[1], dims[2]:dims[3], dims[4]:dims[5]])


            
            array_in = e_label
#            temp_shape = array_in.shape
            label, objs = lbl(self.core_data[dims[0]:dims[1], dims[2]:dims[3], dims[4]:dims[5]], np.ones((3,3,3)))

            if method == 'Dirichlet':
                array_in = self.core_data[dims[0]:dims[1], dims[2]:dims[3], dims[4]:dims[5]]
                
#                e_label = array_in * label
                data_length = objs
    
                indices = np.zeros(((np.ndim(array_in),) + array_in.shape), dtype=np.int32)

        correction_factor = _gen_correction_factor(pts)     
        dim_num = len(array_in.shape)
        max_size = np.sqrt(np.sum(np.array(array_in.shape)**2) )
        if self.verbose:
            print 'Number of voids in section: ', objs
            print time.ctime()

        ## This section is a file handler for the large arrays
        ## Because our edges are inside our voids, they can be "deleted" if 
        ## the void is 1 pixel thick, so this adds 1 pixel thick voids back
        ## into the array. VERY IMPORTATNT.
        for idx in range(1,np.max(label)+1):
            if len(e_label[e_label==idx]) == 0:
                for point in np.c_[np.where(label == idx)]:
                    e_label[tuple(point)] = idx

        e_dict = {key : np.c_[np.where(e_label == key)] for key in range(1, objs + 1)}

        # start some timers to track the time taken to compute
        start_time = time.time()      
        start_time_temp = time.time()

        # instatiate some of our lists
        candidate_dict = {}
        triang_list = []

        # e_list is the list of void border vertices
        # triang is the Delaunay triangulation of the border vertices

        # Delaunay Triangulation:
        if method == 'Delaunay':
            e_list = np.c_[np.where(e_label)]
            
            data_length = len(e_list)
                        
            triang = Delaunay(e_list, qhull_options = qhull_options)
            
            if self.verbose:
                print "completed triang at :", time.ctime()       
                
            label_dict = {}
            edge_dict = {}

            for simplex in triang.vertices:
                label_dict[tuple(simplex)] = [e_label[tuple(e_list[_])] for _ in simplex]
                
            for simplex, labels in label_dict.iteritems():
                sim_pairs = list(combinations(simplex,2))
                lab_pairs = list(combinations(labels,2))
                len_pairs = [len(set(_)) for _ in lab_pairs]
                for idx, pair in enumerate(sim_pairs):
                    if len_pairs[idx] > 1:
                        edge_dict[tuple(sorted(pair))] = tuple(sorted(lab_pairs[idx]))

            if self.verbose:
                print "completed edge_dict at :", time.ctime() 

            i = 0
            list_length = len(edge_dict)
            chunk_length = np.max((int(np.round(list_length/100)), verbosity))
            start_time_temp_for_iters = time.time() 
        
            for pair, lab_val in edge_dict.iteritems():
                if self.verbose:
                    i += 1    
                    if i%chunk_length == 0:
                        p_comp = float(i) / float(list_length)
                        current_time = (time.time() - start_time_temp_for_iters)
                        ttf = current_time / (p_comp)
                        print "%d percent complete, in sec %d of %d" % (int(100 * p_comp), int(current_time), int(ttf))
                
                pair = [triang.points[_] for _ in pair]
                point_2, point_1  = np.array(pair[0]), np.array(pair[1]),
                triang_list.append(_gen_trans_list(pair))
                cur_dist = function(point_1, point_2)

                if lab_val in candidate_dict:
                    mst_dist = candidate_dict[lab_val][2]
                else:                 
                    mst_dist = max_size
    
                if cur_dist < mst_dist:
                    data_list = _gen_data_list([point_1, point_2], lab_val[0], lab_val[1], cur_dist)
                    candidate_dict[lab_val]= data_list
                        
            if plotfigs:
                mask_test = []
                mask_test_coords = [(p1,p2,p3) for (p1,p2,p3) in e_list[triang.simplices]]
                for p1,p2,p3 in mask_test_coords:
                    try:
                        mask_test.append([e_label[tuple(p1)],e_label[tuple(p2)],e_label[tuple(p3)]])
                    except IndexError:
                        pass
                mask_test = [len(np.unique((x1,x2,x3))) for (x1,x2,x3) in mask_test]
                non_triv_simplices = []            
                for idx, val in enumerate(mask_test):
                    if val == 1:
                        non_triv_simplices.append(triang.simplices[idx])
                        
            if self.verbose:
                print "completed candidate_dict at :", time.ctime()             

        if method == 'Dirichlet':
            
    
            edt_array = edt(~array_in.astype(bool), 
                            return_indices=True, 
                            indices=indices)
            
            water_array = water(edt_array, label)        
            
            edges_array = generic_gradient_magnitude(water_array,sobel)
            
            edt_array[edges_array == 0] = 0
            
            edt_array_copy = edt_array.copy()
            
            find_array = np.where(edt_array)
            
            _z = edt_array[find_array]
            
            border_coords = np.c_[find_array]
            edt_max = edt_array.max()
            
            edt_array[edges_array == 0] = edt_max
            
            sort_z = np.argsort(_z)
            _z = _z[sort_z]
            border_coords = border_coords[sort_z]
    
            i = 0
            list_length = len(_z)
            chunk_length = np.max((int(np.round(list_length/100)), verbosity))
            start_time_temp_for_iters = time.time()
            for bc in border_coords:
                if self.verbose:
                    i += 1    
                    if i%chunk_length == 0:
                        p_comp = float(i) / float(list_length)
                        current_time = (time.time() - start_time_temp_for_iters)
                        ttf = current_time / (p_comp)
                        print "%d percent complete, in sec %d of %d" % (int(100 * p_comp), int(current_time), int(ttf))

                bc = tuple(bc)
                
                go_to_next_point = False
                for dim in range(dim_num):
                    if (bc[dim] < 3) or (bc[dim] > array_in.shape[dim] - 3):
                        go_to_next_point = True
                if go_to_next_point:
                    continue
                
                point_1 = [indices[x][bc] for x in range(dim_num)]
                p_0 = label[tuple(point_1)]
                
                host_arrays =   [water_array,
                                 edt_array]
                                 
                guest_arrays = [None, None] 
                
                for idx, host in enumerate(host_arrays):
                    guest_arrays[idx] = _extract_subset(bc, host, dim_num, 1)  

                [possible_labels, possible_dists] = guest_arrays

                possible_labels[possible_labels == p_0] = edt_max           
                p_1 = np.unique(possible_labels)[0]
                    
                possible_crds = np.zeros_like(possible_labels)

                if dim_num == 2:
                    possible_crds.resize((2,3,3))
                if dim_num == 3:
                    possible_crds.resize((3,3,3,3)) 
                            
                for idx, host in enumerate(indices):  
                    possible_crds[idx] = _extract_subset(bc, host, dim_num, 1)                                                       

                point_2 = []
                possible_dists[possible_labels != p_1] = edt_max
                dist_2 = np.min(possible_dists)
                co_loc = np.where(possible_dists == dist_2)
                for dim in range(dim_num):
                    point_2.append(possible_crds[dim][tuple(co_loc)][0])
                p_1 = label[tuple(point_2)]

                if p_1 == p_0:
                    continue
                
                if p_1 > p_0:
                    p_1, p_0 =  p_0, p_1
                    point_2, point_1 = point_1, point_2
                    
                if (p_0, p_1) in candidate_dict:
                    continue
                
                point_1, point_2 = np.asarray([point_1, point_2])
                cur_dist = function(point_1, point_2)
                
                data_list = _gen_data_list([point_1.astype(float), point_2.astype(float)], p_0, p_1, cur_dist)
                candidate_dict[(p_0, p_1)] = data_list

            if self.verbose:
                print "completed candidate_dict at :", time.ctime()   
                
            if self.verbose:
                print 'serial time taken         : ', time.time() - start_time_temp   

        method_runtime = time.time() - start_time_temp
        if self.verbose:
            print 'time taken                : ', method_runtime
        start_time_temp = time.time() 

        ### brute_force recalculates the nearest neighbor connections using
        ### full distance matrix computations. Probably slow if there are 
        ### particularaly *large* objects in view.
        if brute_force:
            i = 0
            list_length = len(candidate_dict)
            chunk_length = np.max((int(np.round(list_length/100)), verbosity/10))
            start_time_temp_for_iters = time.time() 
            if self.verbose:
                 print "Started brute force at :", time.ctime()  
            
            for key,value in candidate_dict.iteritems():
                if self.verbose:
                    i += 1    
                    if i%chunk_length == 0:
                        p_comp = float(i) / float(list_length)
                        current_time = (time.time() - start_time_temp_for_iters)
                        ttf = current_time / (p_comp)
                        print "%d percent complete, in sec %d of %d" % (int(100 * p_comp), int(current_time), int(ttf))
                candidate_dict[key] = _cdist_refinement(value, e_dict)   
            if self.verbose:
                 print "completed brute force at :", time.ctime()  
                 
        # This is a routine to find the Relative Neighborhood Graph and MST
        rng_dict = _gen_rng_dict(candidate_dict)  

        if self.verbose:
            print "completed rng_dict at :", time.ctime()      

        mst_dict = _gen_mst_dict(rng_dict)

        if self.verbose:
            print "completed mst_dict at :", time.ctime()

        
        # This is where we estimate void size by the transits across the voids
        # from one network connection point to another
        
        candidate_list, fbt_paste_mags, fbt_label_list, fbt_point_list = _gen_lists_from_dict(candidate_dict, recalc)
        rng_list, rng_paste_mags, rng_label_list, rng_point_list = _gen_lists_from_dict(rng_dict, recalc)
        paste_list, paste_mags, label_list, point_list = _gen_lists_from_dict(mst_dict, recalc)
        
        fbt_void_mags, fbt_min_v_map = _calc_voids_chords(fbt_point_list, e_label)
        rng_void_mags, rng_min_v_map = _calc_voids_chords(rng_point_list, e_label)
        void_mags, min_v_map = _calc_voids_chords(point_list, e_label)
        
        paste_list, paste_mags, label_list, point_list = _gen_lists_from_dict(mst_dict, recalc)
        # void_mags, min_v_map = _calc_voids_chords(point_list, e_label)
            
        _pastes = [fbt_paste_mags, rng_paste_mags, paste_mags] 
        _voids = [fbt_void_mags, rng_void_mags, void_mags] 

        paste_void_pairs = zip(_pastes, _voids)


        [fbt_paste_mean, 
         fbt_void_mean, 
         fbt_Paste_air_ratio_n, 
         fbt_Homogeneity,
         rng_paste_mean, 
         rng_void_mean, 
         rng_Paste_air_ratio_n, 
         rng_Homogeneity,
         mst_paste_mean, 
         mst_void_mean, 
         mst_Paste_air_ratio_n,
         mst_Homogeneity, ] = list(chain(*[_gen_parameters(p,v,
                                                           correction_factor,
                                                           ideal_wall_ratio) 
                                      for p,v in paste_void_pairs]))

#        rng_Homogeneity = rng_Homogeneity[0]
#        fbt_paste_mean = fbt_paste_mean[0]
#        fbt_void_mean = fbt_void_mean[0]

        if slice_index != None:
            recolor_array = label
            for idx in range(1,objs+1):
                rand_color = np.random.choice(list(np.arange(objs*.5,objs*2)), replace=False)
                for point in np.c_[np.where(recolor_array == idx)]:
                    recolor_array[tuple(point)] = rand_color
                for point in np.c_[np.where(recolor_array == idx)]:
                    recolor_array[tuple(point)] = rand_color

            if method == 'Delaunay' or method == 'cdist':
                fig00 = self.core_data[si]
                fig01 = recolor_array
                fig10 = recolor_array
                fig11 = recolor_array
                fig20 = recolor_array
                fig21 = recolor_array
                
            if method == 'Dirichlet':
                edt_array_copy[edges_array == 0] = 1
                edt_array_copy = 1 / edt_array_copy / objs
                edt_array_copy[edges_array == 0] = 0
                edt_array_copy[edt_array_copy == np.inf] = 0
                border_fig = (edt_array_copy + recolor_array * 
                              edt_array_copy.max() / recolor_array.max())
                fig00 = self.core_data[si]
                fig01 = recolor_array
                fig10 = (np.sqrt(edt_array_copy) + recolor_array * 
                              np.max(np.sqrt(edt_array_copy)) / recolor_array.max())
                fig11 = border_fig
                fig20 = border_fig
                fig21 = border_fig 
                fig30 = self.core_data[si] * edt_array_copy.max() + edt_array_copy
    
            self.background = fig11

            if plotfigs:
                
                for idx in range(1,objs+1):
                    rand_color = np.random.choice(list(np.arange(objs*.5,objs*2)), replace=False)
                    for point in np.c_[np.where(label == idx)]:
                        label[tuple(point)] = rand_color
                    for point in np.c_[np.where(e_label == idx)]:
                        e_label[tuple(point)] = rand_color
    
                f, axarr = plt.subplots(3, 2)
    
                axarr[0, 0].imshow(fig00,
                           cmap = cm.gray_r)
                x1,x2,y1,y2 = axarr[0, 0].axis() 
                axarr[0, 1].imshow(fig01,
                           cmap = c_map)
                for vals in triang_list:
                    axarr[1, 0].plot(*vals,
                             marker = ' ',
                             color = 'r')  # Draw triang lines  
                axarr[1, 0].imshow(fig10,
                           cmap = c_map
                           ) 
                axarr[1, 0].axis((x1,x2,y1,y2))
                
                axarr[1, 1].imshow(fig11,
                           cmap = c_map)
                for vals in candidate_list:
                    axarr[1, 1].plot(*vals,
                             marker = ' ',
                             color = 'r')  # Draw fbt lines  
                for vals in fbt_min_v_map:
                    axarr[1, 1].plot(*vals,
                             marker = ' ',
                             color='b')  # Draw void lines             
                axarr[1, 1].axis((x1,x2,y1,y2))
                
                axarr[2, 0].imshow(fig20,
                           cmap = c_map)
                for vals in rng_list:
                    axarr[2, 0].plot(*vals,
                             marker = ' ',
                             color='r')  # Draw RNG lines            
                for vals in rng_min_v_map:
                    axarr[2, 0].plot(*vals,
                             marker = ' ',
                             color='b')  # Draw void lines             
                axarr[2, 0].axis((x1,x2,y1,y2))  
                             
                axarr[2, 1].imshow(fig21,
                           cmap = c_map)
                for vals in paste_list:
                    axarr[2, 1].plot(*vals,
                             marker = ' ',
                             color='r')  # Draw paste lines
                for vals in min_v_map:
                    axarr[2, 1].plot(*vals,
                             marker = ' ',
                             color='b')  # Draw void lines
                axarr[2, 1].axis((x1,x2,y1,y2))  
    
                title_list = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    
                for i in range(3):
                    for j in range(2):
                        axarr[i, j].set_xticklabels([])
                        axarr[i, j].set_xticks([])
                        axarr[i, j].set_yticklabels([])
                        axarr[i, j].set_yticks([])
                        axarr[i, j].set_xlabel(title_list.pop(0))
                f.tight_layout()


#                ### Boilerplate for single plot
#                f, axarr = plt.subplots(1, 1)
#                axarr.imshow(fig00,
#                           cmap = c_map)
##                x1,x2,y1,y2 = [0,200,0,200]
#                x1,x2,y1,y2 = axarr.axis()                            
#                for vals in candidate_list:
#                    axarr.plot(*vals,
#                             marker = ' ',
#                             color='k',
#                             linewidth=2)  # Draw paste lines
##                for vals in min_v_map:
##                    axarr.plot(*vals,
##                             marker = ' ',
##                             color='b',
##                             linewidth=2)  # Draw void lines
#                axarr.axis((x1,x2,y1,y2))
#                axarr.axis('off')
#
#                f, axarr = plt.subplots(1, 1)
#                axarr.imshow(fig00,
#                           cmap = c_map)
##                x1,x2,y1,y2 = [0,200,0,200]
#                x1,x2,y1,y2 = axarr.axis()                            
#                for vals in rng_list:
#                    axarr.plot(*vals,
#                             marker = ' ',
#                             color='k',
#                             linewidth=2)  # Draw paste lines
##                for vals in min_v_map:
##                    axarr.plot(*vals,
##                             marker = ' ',
##                             color='b',
##                             linewidth=2)  # Draw void lines
#                axarr.axis((x1,x2,y1,y2))
#                axarr.axis('off')
#                
#                f, axarr = plt.subplots(1, 1)
#                axarr.imshow(fig00,
#                           cmap = c_map)
##                x1,x2,y1,y2 = [0,200,0,200]
#                x1,x2,y1,y2 = axarr.axis()                            
#                for vals in paste_list:
#                    axarr.plot(*vals,
#                             marker = ' ',
#                             color='k',
#                             linewidth=2)  # Draw paste lines
##                for vals in fbt_min_v_map:
##                    axarr.plot(*vals,
##                             marker = ' ',
##                             color='b',
##                             linewidth=2)  # Draw void lines
#                axarr.axis((x1,x2,y1,y2))
#                axarr.axis('off')

        self.paste_mags = paste_mags
        self.void_mags = void_mags
        self.paste_list = paste_list
        self.min_v_map = min_v_map

        self.mst_paste_mags = paste_mags
        self.mst_void_mags = void_mags 

        self.rng_paste_mags = rng_paste_mags
        self.rng_void_mags = rng_void_mags
        self.rng_paste_list = rng_list
        self.rng_min_v_map = rng_min_v_map
        
        self.fbt_paste_mags = fbt_paste_mags
        self.fbt_void_mags = fbt_void_mags
        self.fbt_paste_list = candidate_list
        self.fbt_min_v_map = fbt_min_v_map   
        method_runtime = time.time() - start_time  
        
        if self.verbose:
            print 'Completion time           : ', time.ctime()
            print 'after time taken          : ', time.time() - start_time_temp 
            print 'Time taken                : ', method_runtime
            print 'Porosity                  : ', temp_porosity
            print 'Ideal wall ratio          : ', ideal_wall_ratio                       
            print 'Paste walls spanned       : ', len(paste_mags)
            print 'RNG Paste walls spanned   : ', len(rng_paste_mags)
            print '-fbt Paste-air-ratio-     : ', fbt_Paste_air_ratio_n
            print '-fbt wall thickness-      : ', fbt_paste_mean
            print '-fbt chord length-        : ', fbt_void_mean
            print '-fbt Homogeneity-         : ', fbt_Homogeneity
            print '-RNG Paste-air-ratio-     : ', rng_Paste_air_ratio_n
            print '-RNG wall thickness-      : ', rng_paste_mean
            print '-RNG chord length-        : ', rng_void_mean
            print '-RNG Homogeneity-         : ', rng_Homogeneity
            print '-MST Paste-air-ratio-     : ', mst_Paste_air_ratio_n
            print '-MST wall thickness-      : ', mst_paste_mean
            print '-MST chord length-        : ', mst_void_mean
            print '-MST Homogeneity-         : ', mst_Homogeneity

        mst_output = [ mst_Paste_air_ratio_n,
                       mst_paste_mean,
                       mst_void_mean,
                       mst_Homogeneity, 
                       rng_Paste_air_ratio_n,
                       rng_paste_mean,
                       rng_void_mean,
                       rng_Homogeneity, 
                       fbt_Paste_air_ratio_n,
                       fbt_paste_mean,
                       fbt_void_mean,
                       fbt_Homogeneity,
                       method_runtime,
                       temp_porosity,
                       data_length]

        return mst_output


    def plot_pores(self, 
                   paste = True, 
                   voids = True, 
                   subgraph = None, 
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
        
        
        ### tests the input to subgraph for 
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
            voids_data = self.fbt_min_v_map
            paste_data = self.fbt_paste_list
        if subgraph == 'rng': 
            voids_data = self.rng_min_v_map
            paste_data = self.rng_paste_list
        if subgraph == 'mst': 
            voids_data = self.min_v_map
            paste_data = self.paste_list
        if subgraph == None: 
            voids_data = []
            paste_data = []
            
        if o_map == None:
            o_map = self.core_data
        
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
                       opacity = .2 + .8/self.shape[0]) # Draw pores for 3d, changed froo .20 * 100 / self.shape[0]
        if paste:
            for vals in paste_data:
                mlab.plot3d(*vals,
                            tube_radius=0.1 + .05/self.shape[0],
                            color = (1, 0, 0)) # Draw paste lines
        if voids:
            for vals in voids_data:
                mlab.plot3d(*vals,
                            tube_radius=0.1 + .05/self.shape[0],
                            color = (0, 0, 1)) # Draw void lines

        a = anim() # Start the animation.    

    def close(self):
        array_list = [self.core_data,
                      self.e_label,
                      self.blank_map,
                      self._voidedges]
        for item in array_list:  
            try:
                del item
            except AttributeError:
                pass
