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
from scipy.ndimage.morphology import grey_dilation as gd


#==============================================================================
# pyct functions
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
# Experimental stuff
#==============================================================================

def clear_unimportant_voids(processed_data):
    label = lbl(processed_data)
    mask = ma.less_equal(label[0], 1)
    mask = ma.getdata(mask)
    mask = mask.astype(int)
    return processed_data * mask

def process(core_data):
    start = time.clock()
    weighted_space = weight_space(core_data)
    print "Processing..."
    tick = 0
    label = lbl(core_data)
    weighted_space_tick = np.copy(weighted_space)
    while not has_path(label[0]):
        tick += 1
        print "  Tick:", tick
        mask = np.ma.masked_greater_equal(weighted_space, 1)
        mask = np.ma.getmask(mask)
        mask = bd(mask)
        weighted_space = weighted_space % 1
        weighted_space += weighted_space_tick
        weighted_space_tick = (weighted_space_tick # original
                               + mask*gd(weighted_space_tick, size = 3) # add the masked dilated space
                               - mask*weighted_space_tick) # subtract anything which isn't new
        weighted_space += (tick * (mask*gd(weighted_space_tick, size = 3)) % 1 # updates the expanded space to the right void amount
                           - tick * (mask*weighted_space_tick) % 1)
        core_data = bd(core_data, mask = mask).astype(int)
        label = lbl(core_data)
    print "Time taken:", time.clock() - start, "seconds"
    return core_data

def has_path(label):
    (z0,y0,x0) = (0, 0, 0)
    (z1,y1,x1) = (label.shape[0] - 1, label.shape[1] - 1, label.shape[2] - 1)
    return label[z0][y0][x0] == label[z1][y1][x1]

def heuristic(core_data, avg_height):
    core_data = core_data / avg_height
    core_data[core_data < 1] = core_data[core_data < 1]**2
    return core_data

def weight_space(core_data): # make an array which has varying values
    print "Weighting space..."
    label = lbl(core_data)
    bounds_pool = ndimage.find_objects(label[0])
    num_voids = lbl(core_data)[1]
    avg_height = get_avg_void_height(core_data, bounds_pool, num_voids)
    core_data = replace_voids_with_heights(label[0], bounds_pool, num_voids)
    weighted_space = heuristic(core_data, avg_height)
    weighted_space = weighted_space / np.amax(weighted_space) # maps weighted_space to values in [0,1]
    return weighted_space

def seg_print(void, num_voids, chunk_size):
    chunk_seg = void / chunk_size
    chunk_lo = chunk_seg * chunk_size + 1
    chunk_hi = (chunk_seg + 1) * chunk_size
    if chunk_seg == 9:
        chunk_hi = num_voids
    print("  Replacing voids %d - %d out of %d"
             %(chunk_lo, chunk_hi, num_voids))

def replace_voids_with_heights(label_array, bounds_pool, num_voids):
    chunk_size = num_voids / 10 + 10 - num_voids % 10
    for void in xrange(1, num_voids + 1):
        if (void - 1) % chunk_size == 0:
            seg_print(void, num_voids, chunk_size)
        bounding_box = bounds_pool[void - 1]
        top = bounding_box[0].stop # gets the upper z-coord of each void's bounding box
        bot = bounding_box[0].start # gets the lower z-coord of each void's bounding box
        void_ht = top - bot
        label_array[bounding_box][label_array[bounding_box] == void] = void_ht
    return label_array

def get_avg_void_height(core_data, bounds_pool, num_voids):
    print "Getting average void height..."
    total_void_height = 0
    for void in xrange(num_voids):
        top = bounds_pool[void][0].stop # gets the upper z-coord of each void's bounding box
        bot = bounds_pool[void][0].start # gets the lower z-coord of each void's bounding box
        total_void_height += top - bot
    avg_void_height = float(total_void_height) / num_voids
    return avg_void_height

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
