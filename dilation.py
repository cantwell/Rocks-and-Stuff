# -*- coding: utf-8 -*-

#==============================================================================
# Modules
#==============================================================================

import time

import numpy as np
import numpy.ma as ma

from scipy import ndimage
from scipy.ndimage import label as lbl
from scipy.ndimage.morphology import binary_dilation as bd

#==============================================================================
#
#==============================================================================

def add_top_bottom_voids(data):
    height, width, depth = data.shape
    boundary0 = np.zeros((1, width, depth))
    boundary1 = np.ones((1, width, depth))
    bot = np.concatenate((boundary0, boundary1))
    top = np.concatenate((boundary1, boundary0))
    data = np.concatenate((top, data))
    data = np.concatenate((data, bot))
    return data

def fid_tif_to_data(fid, dims):
    """Extracts data from a tif file id"""
    from PIL import Image
    fid = Image.open(fid)
    if fid.n_frames == 1:
        fid = _2d_to_3d(np.asarray(fid))
    else:
        fid = _tiff_to_3d(fid, dims[0], dims[1])
    core_data = np.asarray(fid, dtype=bool)[None   : None,
                                            dims[2]: dims[3],
                                            dims[4]: dims[5]]
    core_data = np.invert(core_data)
    core_data = add_top_bottom_voids(core_data)
    core_data = core_data.astype(int)
    return core_data

def _2d_to_3d(fid):
    """Turns 2d numpy arrays into 3d numpy arrays"""
    if len(fid.shape) == 2:
        fid = fid.tolist()
        fid = [fid]
    return fid

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

#==============================================================================
# def process(core_data, label):
#     # hacky pre-process demo
#     while not has_path(label):
#         core_data = bd(core_data, iterations=1).astype(bool)
#         core_data = core_data.astype(int)
#         label = lbl(core_data)[0]
#     return core_data, label
#==============================================================================

def has_path(label):
    (z0,y0,x0) = (0, 0, 0)
    (z1,y1,x1) = (label.shape[0] - 1, label.shape[1] - 1, label.shape[2] - 1)
    return label[z0][y0][x0] == label[z1][y1][x1]

def clear_unimportant_voids(processed_data):
    label = lbl(processed_data)
    mask = ma.less_equal(label[0], 1)
    mask = ma.getdata(mask)
    mask = mask.astype(int)
    return processed_data * mask

#def reprocess(core, label):
#==============================================================================
# Experimental stuff
#==============================================================================

# need to dilate based off of each voids relative height to other voids, try to
# get the voids to scale based off of an equal-worth distance metric.
# i.e.: for a tiny void, barely dilate it while for a big void, dilate it tons
# after dilation, trim the boundary down based off of the heuristic
#==============================================================================
#
# def get_scaling_factors(core, label, num_voids):
#     print "Getting scaling factors..."
#     bounds_pool = ndimage.find_objects(label[0])
#     total_void_height = 0
#     for void in xrange(num_voids):
#         top = bounds_pool[void][0].stop # gets the upper z-coord of each void's bounding box
#         bot = bounds_pool[void][0].start # gets the lower z-coord of each void's bounding box
#         total_void_height += top - bot
#     avg_void_height = float(total_void_height) / num_voids
#     void_scaling_factors = []
#     for void in xrange(num_voids):
#         top = bounds_pool[void][0].stop # gets the upper z-coord of each void's bounding box
#         bot = bounds_pool[void][0].start # gets the lower z-coord of each void's bounding box
#         void_ht = top - bot
#         scaling_factor = void_ht / avg_void_height
#         void_scaling_factors = np.append(void_scaling_factors, scaling_factor)
#     assert (len(void_scaling_factors) == num_voids)
#     return void_scaling_factors
#
#==============================================================================

#==============================================================================
# def get_masks(core, label, num_voids):
#     print "Getting masks..."
#     masks = []
#     for void in xrange(num_voids):
#         print "Mask", void, "out of", num_voids
#         mask = np.ma.masked_equal(label[0], void + 1)
#         masks = np.append(masks, mask)
#     print len(masks)
#     return masks
#==============================================================================

def weighted_process(core_data): # add an option to see possible search spaces for additional ticks
    start = time.clock()
    label = lbl(core_data)
    num_voids = label[1]
    voids = make_voids(core_data, label, num_voids)
    dilation = np.copy(core_data)
    dlabel = lbl(dilation)
    tick = 0
    print "Expanding voids..."
    while not has_path(dlabel[0]):
        tick += 1
        print "  Tick:", tick
        for void in xrange(num_voids):
            iters = int(tick * voids[void].scaling_factor)
            temp_mask = voids[void].mask
            for i in xrange(iters):
                temp_mask = bd(temp_mask)
                dilation = bd(dilation,
                              mask = temp_mask).astype(int)
        dlabel = lbl(dilation)
        if not has_path(dlabel[0]):
            dilation = np.copy(core_data)
    print "Time taken:", time.clock() - start, "seconds"
    return dilation

def get_avg_void_height(core, bounds_pool, num_voids):
    total_void_height = 0
    for void in xrange(num_voids):
        top = bounds_pool[void][0].stop # gets the upper z-coord of each void's bounding box
        bot = bounds_pool[void][0].start # gets the lower z-coord of each void's bounding box
        total_void_height += top - bot
    avg_void_height = float(total_void_height) / num_voids
    return avg_void_height

def make_voids(core, label, num_voids):
    voids = {}
    bounds_pool = ndimage.find_objects(label[0])
    print "Getting average void height..."
    avg_void_height = get_avg_void_height(core, bounds_pool, num_voids)
    print "Populating voids..."
    for void in xrange(num_voids):
        print "  Populating void", void + 1, "out of", num_voids
        top = bounds_pool[void][0].stop # gets the upper z-coord of each void's bounding box
        bot = bounds_pool[void][0].start # gets the lower z-coord of each void's bounding box
        void_ht = top - bot
        if void_ht / avg_void_height <= float(1)/3:
            scaling_factor = ((void_ht / avg_void_height)**2) / 2
        else:
            scaling_factor = (void_ht / avg_void_height) / 2
        mask = np.ma.getmask(np.ma.masked_equal(label[0], void + 1))
        voids[void] = Void(mask, scaling_factor)
    print "Voids populated..."
    return voids

class Void(object):
    def __init__(self, mask, scaling_factor, bounding_box):
        self.mask = mask
        self.scaling_factor = scaling_factor
        self.bounding_box = bounding_box

#==============================================================================
#
# class Void(object):
#     def __init__(self, vertices, heuristic_vector, avg_size, void_size):
#         self.initial_vertices = vertices
#         self.expanded_vertices = self.initial_vertices
#         #self.center = get_center(vertices)
#         scaling_factor = void_size / avg_size
#         self.heuristic = self.get_adj_heuristic(heuristic_vector,
#                                                 scaling_factor)
#
#     def get_adj_heuristic(self, heuristic_vector, scaling_factor):
#         heur = np.zeros((3,3))
#         for i in xrange(3):
#             heur[i][i] = heuristic_vector[i]
#         heur = scaling_factor * heur
#         return heur
#
#     def get_center(self):
#         pass
#
#     def scale(self):
#         for v in self.expanded_vertices:
#             self.expand(v)
#         # connect all of the vertices back together
#
#     def expand(self, vertice):
#         v = np.dot(self.heuristic, vertice)
#
#     def trim(self):
#         pass
#
# def make_voids(data, heuristic_vector):
#     voids = []
#     bounds = get_bounds(data)
#     label = lbl(bounds)
#     bounds_pool = ndimage.find_objects(label[0])
#     avg_size = float(np.sum(data)) / label[1]
#     for i in label[1]:
#         obj_bounds = bounds_pool[i]
#         void_size = np.sum(data[obj_bounds])
#         void_height = obj_bounds[0]
#         pts = np.where(bounds[obj_bounds] >= 1)
#         voids.append(Void(pts, heuristic_vector, avg_size, void_size))
#     return voids
#
# def get_verts(pts):
#     pass
#
# def run_voids(voids):
#     for void in voids:
#         void.scale()
#     for void in voids:
#         fill_void(void)
#         # label voids to see connections
#
# def get_bounds(data):
#     return data - ndimage.binary_erosion(data)
#
# def fill_void(void):
#     for vertex in void.expanded_vertices:
#         pass
#==============================================================================

    # for each label:
    # get boundary of label

#def has_path():
#    pass

#def process():
#    pass
    # 1. Get individual voids from the total space
    # 2. Triangulate each void and put vertices into a dictionary containing
    # void number, new label number, and original label number
    # 3. Expand the shape by a distance
    # 4. If the starting point label and the end point label are equal, reduce
    # the search space to all voids which share the same label as the start/end


#==============================================================================
# plot_pores
#==============================================================================

def plot_pores(core, pathbounds, secondary = None):

    from mayavi import mlab

    @mlab.animate
    def anim():
        f = mlab.gcf()
        while 1:
            f.scene.camera.azimuth(1)
            f.scene.render()
            yield

    mlab.figure(bgcolor=(1,1,1)) # Set bkg color
    mlab.contour3d(core,
                   color = (0,0,0),
                   contours = 2,
                   opacity = .2 + .8/core.shape[0])
    mlab.contour3d(pathbounds,
                   color = (0,0,.9),
                   contours = 2,
                   opacity = 1)
    if secondary != None:
        mlab.contour3d(secondary,
                       color = (.9,.9,0),
                       contours = 2,
                       opacity = .5)


    a = anim() # Start the animation.
