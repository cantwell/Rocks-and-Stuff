# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 12:31:51 2016

@author: jcrnk
"""

"""
Programs and toolboxes needed:
- Gmsh

Example steps on how to use tiffy:

1. Run tiffy after opening the python console:

>>> img_src = r'/home/jcrnk/Documents/SummerRP/core.tif'
>>> img_data = fid_tif_to_data(img_src, [0, 1, 0, 250, 0, 250]) # first two
indices in the dimensions must be sequential and represent the slide of the
tiff you want examined.
>>> write_file(img_data[1]) # must use [1] to ignore the top and bottom empty
surfaces made in fid_tif_to_data. Will save file as tiff_geometry.geo

2. Open Gmsh, go to file -> open and select "tiff_geometry.geo"
3. Click on Mesh -> 2D
4. Click on file -> Save As and save the file as "tiff_mesh.msh"
"""

#==============================================================================
# modules
#==============================================================================

import time

import copy

import sys
sys.setrecursionlimit(10000)

import numpy as np

from scipy.ndimage import label as lbl
from scipy.ndimage.morphology import binary_erosion as b_erode

#==============================================================================
# write file
#==============================================================================

def write_file(img):
    start_time = time.clock()
    coord_d = dict() # checks for duplicate points
    line_d = dict() # checks for duplicate lines, assumes loops are unique
    lc = "3" # resolution of the mesh, lower number -> higher res
    cur_pt = 1
    cur_line = 1
    label = lbl(img)
    num_voids = label[1]
    geo_string = ""
    shape = img.shape
    geo_conversion = geo_make_boundary(cur_line, cur_pt, shape, lc, coord_d, line_d)
    geo_string += geo_conversion[0]
    cur_line = geo_conversion[1]
    cur_pt = geo_conversion[2]
    surfs = [cur_line - 1]
    for void in xrange(1, num_voids + 1):
        print "Handling void %d out of %d..." % (void, num_voids)
        borders = label[0] - b_erode(label[0])
        print "  Reordering borders..."
        pts = get_pts(void, borders)
        print "  Converting borders to string..."
        geo_conversion = geo_convert_pts(cur_pt, pts, lc, coord_d)
        geo_string += geo_conversion[0]
        cur_pt = geo_conversion[1]
        geo_conversion = geo_convert_lines(cur_line, pts, coord_d, line_d)
        geo_string += geo_conversion[0]
        cur_line = geo_conversion[1]
        surfs += geo_conversion[2]
    geo_string += geo_surf_loop(cur_line, surfs)
    cur_line += 1
    f = open('tiff_geometry.geo', 'w')
    print "Saving .geo file as 'tiff_geometry.geo'"
    f.write(geo_string)
    print "Time taken:", time.clock() - start_time

def get_pts(void, borders):
    border = np.where(borders == void, borders, 0)
    pts = np.transpose(np.nonzero(border))
    temp_pts = np.copy(pts[:,0])
    pts[:, 0] = pts[:,1]
    pts[:, 1] = temp_pts
    recfn = RecursiveOrder(pts.tolist())
    return recfn.total_paths

#==============================================================================
# .geo converter helpers
#==============================================================================

def geo_make_boundary(cur_line, cur_pt, shape, lc, coord_d, line_d):
    start = cur_line
    add_geo = ""
    add_geo += "Point(%d) = {%d, %d, 0, %s}; \n" % (cur_pt, 0, 0, lc) # adding (0,0)
    cur_pt += 1
    coord_d[(0, 0)] = cur_pt

    add_geo += "Point(%d) = {%d, %d, 0, %s}; \n" % (cur_pt, shape[0], 0, lc) # adding (0,0)
    add_geo += "Line(%d) = {%d, %d}; \n" % (cur_line, cur_pt-1, cur_pt)
    add_geo += "Color {220,220,220}{Line{%d};} \n" % (cur_line)
    cur_pt += 1
    add_geo += "Physical Line (%d) = {%d}; \n" % (cur_line, cur_line)
    line_d[(cur_pt-1, cur_pt)] = cur_line
    cur_line += 1
    coord_d[(shape[0], 0)] = cur_pt

    add_geo += "Point(%d) = {%d, %d, 0, %s}; \n" % (cur_pt, shape[0], shape[1], lc) # adding (0,0)
    add_geo += "Line(%d) = {%d, %d}; \n" % (cur_line, cur_pt-1, cur_pt)
    add_geo += "Color {220,220,220}{Line{%d};} \n" % (cur_line)
    cur_pt += 1
    coord_d[(shape[0], shape[1])] = cur_pt
    add_geo += "Physical Line (%d) = {%d}; \n" % (cur_line, cur_line)
    line_d[(cur_pt-1, cur_pt)] = cur_line
    cur_line += 1

    add_geo += "Point(%d) = {%d, %d, 0, %s}; \n" % (cur_pt, 0, shape[1], lc) # adding (0,0)
    add_geo += "Line(%d) = {%d, %d}; \n" % (cur_line, cur_pt-1, cur_pt)
    add_geo += "Color {220,220,220}{Line{%d};} \n" % (cur_line)
    cur_pt += 1
    coord_d[(0, shape[1])] = cur_pt
    add_geo += "Physical Line (%d) = {%d}; \n" % (cur_line, cur_line)
    line_d[(cur_pt-1, cur_pt)] = cur_line
    cur_line += 1

    add_geo += "Line(%d) = {%d, %d}; \n" % (cur_line, cur_pt-1, start)
    add_geo += "Color {220,220,220}{Line{%d};} \n" % (cur_line)
    add_geo += "Physical Line (%d) = {%d}; \n" % (cur_line, cur_line)
    line_d[(cur_pt-1, cur_pt)] = cur_line
    cur_line += 1

    loop = [start, start + 1, start + 2, start + 3]
    add_geo += "Line Loop(%d) = {%s}; \n" % (cur_line, str(loop)[1:-1])
    cur_line += 1

    return add_geo, cur_line, cur_pt

def geo_surf_loop(cur_line, surfs):
    add_geo = "Plane Surface(%d) = {%s}; \n" % (cur_line, str(surfs)[1:-1])
    cur_line += 1
    add_geo += "Physical Surface(%d) = {%d}; \n" % (cur_line, cur_line-1)
    add_geo += "Color {225,225,225}{Surface{%d};} \n" % (cur_line - 1)
    return add_geo

def geo_convert_pts(cur_pt, pts, lc, coord_d):
    add_geo = ""
    visited = []
    for pt in pts:
        if pt not in visited:
            (x, y) = pt
            if (x, y) not in coord_d:
                coord_d[(x, y)] = cur_pt
                add_geo += "Point(%d) = {%d, %d, %d, %s}; \n" % (cur_pt, x, y, 0, lc)
                cur_pt += 1
                visited.append(pt)
        else:
            visited = []
    return add_geo, cur_pt

def geo_convert_lines(cur_line, pts, coord_d, line_d):
    """
    Converts a recursively ordered list of points into a series of strings
    used in making the .geo file.
    """
    add_geo = ""
    surfs = []
    path = []
    if len(pts) >= 1:
        stop_pt = pts[0]
    skip = True
    for i in xrange(len(pts)):
        if pts[i] == stop_pt and skip == False:
            # gets adjusted point index of (x,y) in coord dict for points 1 & 2
            pt_i1 = coord_d[(pts[i - 1][0], pts[i - 1][1])]
            pt_i2 = coord_d[(pts[i][0]    , pts[i][1])    ]
            if (pt_i1, pt_i2) not in line_d and (pt_i2, pt_i1) not in line_d:
                line_d[(pt_i1, pt_i2)] = cur_line
                add_geo += "Line(%d) = {%d, %d}; \n" % (cur_line, pt_i1, pt_i2)
                add_geo += "Color {20,20,20}{Line{%d};} \n" % (cur_line)
                # since we're back at the start, we have a complete surface
                path.append(cur_line)
                cur_line += 1
            else:
                if (pt_i1, pt_i2) in line_d:
                    line_index = line_d[(pt_i1, pt_i2)]
                elif (pt_i2, pt_i1) in line_d:
                    line_index = -line_d[(pt_i2, pt_i1)]
                path.append(line_index)
            add_geo += ("Line Loop(%d) = {" % (cur_line)
                    + str(path)[1:-1]
                    + "}; \n")
            path = []
            cur_line += 1
            add_geo += ("Plane Surface(%d) = {%d}; \n" % (cur_line,
                                                          cur_line - 1))
            add_geo += "Color {25,25,25}{Surface{%d};} \n" % (cur_line)
            surfs.append(cur_line - 1)
            cur_line += 1
            add_geo += "Physical Surface(%d) = {%d}; \n" % (cur_line, cur_line-1)
            if i + 1 != len(pts):
                stop_pt = pts[i + 1]
                skip = True
        elif i + 1 != len(pts) and pts[i + 1] != stop_pt:
            pt_i1 = coord_d[(pts[i][0]    , pts[i][1])    ]
            pt_i2 = coord_d[(pts[i + 1][0], pts[i + 1][1])]
            if (pt_i1, pt_i2) not in line_d and (pt_i2, pt_i1) not in line_d:
                line_d[(pt_i1, pt_i2)] = cur_line
                add_geo += "Line(%d) = {%d, %d}; \n" % (cur_line, pt_i1, pt_i2)
                add_geo += "Color {20,20,20}{Line{%d};} \n" % (cur_line)
                path.append(cur_line)
                cur_line += 1
            else:
                if (pt_i1, pt_i2) in line_d:
                    line_index = line_d[(pt_i1, pt_i2)]
                elif (pt_i2, pt_i1) in line_d:
                    line_index = -line_d[(pt_i2, pt_i1)]
                path.append(line_index)
            skip = False
    return add_geo, cur_line, surfs

#==============================================================================
# Recursively re-order a list of points describing the void outlines
#==============================================================================

class RecursiveOrder(object):
    def __init__(self, points):
        self.points = points
        self.total_paths = []
        self.expanded_forks = []
        start = points[0]
        remaining_pts = points
        current_point = start
        path = [start]
        self.rec_order_pts(start, remaining_pts, current_point, path)

    def overlap(self, path, total_path):
        """
        Checks to see whether more than one edge is shared between a given path
        and the total path, used to prevent overlapping surfaces in mesh
        """
        set_path = set()
        for elem in path:
            set_path.add((elem[0], elem[1]))
        set_tot_path = set()
        for elem in total_path:
            set_tot_path.add((elem[0], elem[1]))
        overlap = set_path.intersection(set_tot_path)
        return len(overlap) > 2 # more than 2 points are shared

    def rec_order_pts(self, start, rem_pts, cur_pt, path):
        """
        Recursively finds the outline of the voids. Ignores void space that
        is only a single pixel thick.
        """
        adjacencies = self.get_adjacencies(cur_pt[0], cur_pt[1], rem_pts, path)
        if (len(path) >= 3
            and start in adjacencies
            and not self.overlap(path + [start], self.total_paths)):
            return path + [start]
        else:
            for adj in adjacencies:
                if (adj in rem_pts and adj not in path):
                    new_rem = copy.deepcopy(rem_pts)
                    new_rem.remove(adj)
                    if len(adjacencies) == 1:
                        self.total_paths += self.rec_order_pts(start,
                                                               new_rem,
                                                               adj,
                                                               path + [adj])
                    elif len(adjacencies) >= 2:
                        # check to see if adj will take us back to the start
                        self.total_paths += self.rec_order_pts(start,
                                                               new_rem,
                                                               adj,
                                                               path + [adj])
                        if adj not in self.expanded_forks:
                            self.expanded_forks += [adj]
                            # make a new path to traverse
                            self.total_paths += self.rec_order_pts(adj,
                                                                   self.points,
                                                                   adj,
                                                                   [adj])
            return []

    def get_adjacencies(self, cur_x, cur_y, remain_pts, path):
        """Finds valid nodes 1 pixel away from the current node on the path"""
        adj = []
        for x in xrange(-1,2):
            for y in xrange(-1, 2):
                if ([cur_x + x, cur_y + y] in remain_pts and (x != 0 or y != 0)):
                    if (x + y) % 2 == 0: # make sure diagonals don't cross
                        if ([cur_x + x, cur_y] not in self.total_paths and [cur_x + x, cur_y] not in path
                         and [cur_x, cur_y + y] not in self.total_paths and [cur_x, cur_y + y] not in path):
                            adj += [[cur_x + x, cur_y + y]]
                    else:
                        adj += [[cur_x + x, cur_y + y]]
        return adj

#==============================================================================
# Image handling
#==============================================================================

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
    core_data = core_data.astype(int)
    core_data = add_solid_border(core_data)
    return core_data

def add_solid_border(core_data):
    height, width, depth = core_data.shape
    b_h, b_w, b_d = height + 2, width + 2, depth + 2
    new = np.zeros((b_h, b_w, b_d))
    for i in xrange(height):
        for j in xrange(width):
            for k in xrange(depth):
                new[i+1][j+1][k+1] = core_data[i][j][k]
    return new

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
# def geo_make_boundary(cur_line, cur_pt, shape, lc, coord_d, line_d):
#     start = cur_pt
#     x_len = shape[1] - 1
#     y_len = shape[0] - 1
#     line_loop = []
#     add_geo = ""
#     x = 0
#     y = 0
#     side_len = 0
#     add_geo += "Point(%d) = {%d, %d, 0, %s}; \n" % (cur_pt, x, y, lc) # adding (0,0)
#     coord_d[(x,y)] = cur_pt
#     cur_pt += 1
#     for side in ["Top", "Right", "Bottom", "Left"]:
#         if side == "Top":
#             x_i = 1
#             y_i = 0
#             side_len = x_len
#         elif side == "Right":
#             x_i = 0
#             y_i = 1
#             side_len = y_len
#         elif side == "Bottom":
#             x_i = -1
#             y_i = 0
#             side_len = x_len
#         elif side == "Left":
#             x_i = 0
#             y_i = -1
#             side_len = y_len
#         # results = (add_geo, cur_line, sub_loop, cur_pt, x, y)
#         results = make_boundary_side(side,
#                                      cur_line, cur_pt,
#                                      side_len,
#                                      lc,
#                                      coord_d, line_d,
#                                      x, y,
#                                      x_i, y_i,
#                                      start)
#         add_geo += results[0]
#         cur_line = results[1]
#         line_loop += results[2]
#         cur_pt = results[3]
#         x = results[4]
#         y = results[5]
#     add_geo += "Line Loop(%d) = {%s}; \n" % (cur_line, str(line_loop)[1:-1])
#     cur_line += 1
#     return add_geo, cur_line, cur_pt
#==============================================================================
#==============================================================================
#
# def make_boundary_side(side, cur_line, cur_pt, side_len, lc, coord_d, line_d, x, y, x_i, y_i, start):
#     add_geo = ""
#     sub_loop = []
#     for i in xrange(side_len):
#         x += x_i
#         y += y_i
#         if (x, y) not in coord_d: # keep from readding (0,0)
#             add_geo += "Point(%d) = {%d, %d, 0, %s}; \n" % (cur_pt, x, y, lc)
#             coord_d[(x,y)] = cur_pt
#             cur_pt += 1
#             add_geo += "Line(%d) = {%d, %d}; \n" % (cur_line, cur_pt-2, cur_pt-1)
#             line_d[(cur_pt-2, cur_pt-1)] = cur_line
#             sub_loop.append(cur_line)
#             cur_line += 1
#         else: # at (0,0) again
#             add_geo += "Line(%d) = {%d, %d}; \n" % (cur_line, cur_pt-1, start)
#             line_d[(cur_pt-1, cur_pt)] = cur_line
#             sub_loop.append(cur_line)
#             cur_line += 1
#     add_geo += "Physical Line (%d) = {%s}; \n" % (cur_line, str(sub_loop)[1:-1])
#     return add_geo, cur_line, sub_loop, cur_pt, x, y
#==============================================================================
