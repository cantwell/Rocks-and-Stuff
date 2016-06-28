from PIL import Image
import numpy as np
import scipy.ndimage
import time
import sys
import psutil 
import Queue
from scipy.spatial.distance import euclidean
from scipy.spatial import Delaunay
from mayavi import mlab

def _2d_to_3d(fid): # turns 2d numpy arrays into 3d numpy arrays
    if len(fid.shape) == 2:
        fid = fid.tolist()
        fid = [fid]
    return fid

def tiff_to_3d(img,start_frame,end_frame):
    if((start_frame == None) | (start_frame < 0)): 
        start_frame = 0
    if ((end_frame == None) | (end_frame > img.n_frames)):
        end_frame = img.n_frames
    img.seek(start_frame)    
    slice_2d = np.asarray(img)
    img_3d = _2d_to_3d(slice_2d)        
    for frame in range(start_frame + 1, end_frame):
        img.seek(frame)    
        slice_2d = np.asarray(img)
        slice_3d = _2d_to_3d(slice_2d)
        img_3d = np.concatenate((img_3d, slice_3d), axis = 0)
    return img_3d

def read_tiff(filename):
    img = Image.open(filename)
    return tiff_to_3d(img, 0, img.n_frames)

def read_tif(filename):
    img = Image.open(filename)
    data = tiff_to_3d(img, 0, img.n_frames)
    core = np.asarray(data, dtype=bool)
    label, objs = scipy.ndimage.label(core)
    print(objs)
    return label
    
def sample(path):
    total = 0
    for i in range(len(path)):
        total += euclidean(path[i], path[i+1])
    return total
    
def get_edges(rock):
    edges = []
    for x in range(rock.shape[0]):
        for y in range(rock.shape[1]):
            for z in range(rock.shape[2]):
                if(not rock[x][y][z] == rock[x+1][y][z]):
                    edges = edges + [(x, y, z)]             
                    break
                if(not rock[x][y][z] == rock[x-1][y][z]):
                    edges = edges + [(x, y, z)]             
                    break
                if(not rock[x][y][z] == rock[x][y+1][z]):
                    edges = edges + [(x, y, z)]             
                    break
                if(not rock[x][y][z] == rock[x][y-1][z]):
                    edges = edges + [(x, y, z)]             
                    break
                if(not rock[x][y][z] == rock[x][y][z+1]):
                    edges = edges + [(x, y, z)]             
                    break
                if(not rock[x][y][z] == rock[x][y][z-1]):
                    edges = edges + [(x, y, z)]             
                    break
    return edges
    
#Sample metric function, takes in the rock as a numpy array
def metric(rock):
    def distance(path):
        total = 0
        if(path == []):
            return 0
        for i in range(len(path)-1):
            if(rock[path[i][0]][path[i][1]][path[i][2]] == 0):
                total += euclidean(path[i], path[i+1])
            else:
                total += 100*euclidean(path[i], path[i+1])
        return total
    return distance
    
def rock_dijkstra(rock, start, end, distance):
    def adj(point, visit):
        x, y, z = point
        list = []
        if(x < rock.shape[0] - 1):
            if((x+1, y, z) not in visit):
                list = list + [(x+1, y, z)]
        if(x > 0):
            if((x-1, y, z) not in visit):
                list = list + [(x-1, y, z)]
        if(y < rock.shape[1] - 1):
            if((x, y+1, z) not in visit):
                list = list + [(x, y+1, z)]
        if(y > 0):
            if((x, y-1, z) not in visit):
                list = list + [(x, y-1, z)]
        if(z < rock.shape[2] - 1):
            if((x, y, z+1) not in visit):
                list = list + [(x, y, z+1)]
        if(z > 0):
            if((x, y, z-1) not in visit):
                list = list + [(x, y, z-1)]
        visit += list
        return list
    PQ = Queue.PriorityQueue()    
    PQ.put((0, start, []))
    visited = [start]
    while(not PQ.empty()):
        (d, rover, path) = PQ.get()
        if(rover == end):
            break
        for point in adj(rover, visited):
            PQ.put((distance(path + [rover]), point, path + [rover]))
    if(rover != end):
        print("Couldn't find end")
        return []
    else:
        return path
        
def dot(point1, point2):
    x, y, z = point1
    a, b, c = point2
    return (x*a + y*b + c*z)        

def sub(point1, point2):
    x, y, z = point1
    a, b, c = point2
    return (x-a, y-b, z-c)

#Sample heuristic.  Returns the function to be passed into A*
def sampleh(start, end):
    return (lambda point :  70*(-dot(sub(start, point), sub(start, end)/euclidean(start, end))))
            
def metric2(rock):
    def distance(point1, point2):
        if(rock[point2[0]][point2[1]][point2[2]] == 0):
            return euclidean(point1, point2)
        else:
            return 100*euclidean(point1, point2)
    return distance
    
def rock_AStar(rock, start, end, distance, h):
    def adj(point, visit):
        x, y, z = point
        list = [(x+1, y, z), (x-1, y, z), (x, y+1, z), (x, y-1, z), (x, y, z+1), (x, y, z-1), (x+1, y+1, z), (x-1, y+1, z), (x+1, y-1, z), (x+1, y, z+1), (x+1, y, z-1), 
                (x, y+1, z+1), (x, y+1, z-1)]
        list = filter(lambda (x, y, z): (x in range(rock.shape[0]) and y in range(rock.shape[1]) and z in range(rock.shape[2]) and not (x, y, z) in visit), list)
        return list
    
    PQ = Queue.PriorityQueue()    
    PQ.put((h(start), 0, start, []))
    visited = [start]
    while(not PQ.empty()):
        (heu, d, rover, path) = PQ.get()
        next = adj(rover, visited)
        visited = visited + next
        #print(heu, d)
        if(rover == end):
            break
        for point in next:
            newd = d + distance(rover, point)
            PQ.put((newd + h(point), newd, point, path + [rover]))
    if(rover != end):
        print("Couldn't find end")
        return []
    else:
        return path, visited
    
def plot_path(points, rock):
    path_rock = np.ones(rock.shape)
    for (x, y, z) in points:
        path_rock[x][y][z] = 0
    @mlab.animate
    def anim():
        f = mlab.gcf()
        while 1:
            f.scene.camera.azimuth(1)
            f.scene.render()
            yield
                
    mlab.figure(bgcolor=(1,1,1)) # Set bkg color
    mlab.contour3d(rock, 
                   color = (0,0,0),
                   contours = 2,
                   opacity = .2 + .8/rock.shape[0]) # Draw pores for 3d, changed froo .20 * 100 / self.shape[0]
    mlab.contour3d(path_rock)
    a = anim()
    
    
###########################################
##Code to make D-RNG.  
###########################################    
# def intersect(l1, l2):
#     return set(l1).intersection(l2)

# #array is the image array, temp is a dictionary, x and y are positions, and i is the number of components already scanned.  
# def get_component_array(array, list):
#     if(list == []):
#         return []
#     temp = []
#     for (x, y, z) in list:
#         list.remove((x, y, z))
#         if(array[x][y][z] == 0):
#             array[x][y][z] = 255
#             temp = temp + [(x, y, z)]
#             if(x < array.shape[0] - 1):
#                 list = list + [(x+1, y, z)]
#             if(x > 0):
#                 list = list + [(x-1, y, z)]
#             if(y < array.shape[1] - 1):
#                 list = list + [(x, y+1, z)]
#             if(y > 0):
#                 list = list + [(x, y-1, z)]
#             if(z < array.shape[2] - 1):
#                 list = list + [(x, y, z+1)]
#             if(z > 0):
#                 list = list + [(x, y, z-1)]
#     return temp + get_component_array(array, list)

# def get_component(array, temp, x, y, z, i):
#     list = [(x, y, z)]
#     l = get_component_array(array, list)
#     temp[i] = l
#     i = i+1
#     return (temp, i)

# def display_coordinates(points, lx, ly):
#     out = numpy.empty((lx, ly))
#     for (x, y) in points:
#         if(x < lx and y < ly):
#             out[x][y] = 255
#     img = Image.fromarray(out)
#     img.show()

# def display_graph(dic, lx, ly, number):
#     out = numpy.empty((lx, ly))
#     for i in range(number):
#         for (x, y) in dic[i]:
#             out[x][y] = 255
#     img = Image.fromarray(out)
#     img.show()
#     return img

# def scan(array, temp, i):
#     for x in range(array.shape[0]):
#         for y in range(array.shape[1]):
#             for z in range(array.shape[2]):
#                 if(array[x][y][z] == 0):
#                     (temp, i) = get_component(array, temp, x, y, z, i)
#     return temp

# #scan_pictures takes a file name and creates a graph from the image
# def scan_picture(filename):
#     ar = read_image(filename)
#     out = scan(ar, dict(), 0)
#     return out

# def create_lines(base, lx, ly):
#     out = []
#     for i in range(ly):
#         out = out + [(base, i)]
#     return out

# def intersection(graph, line):
#     out = []
#     for key in list(graph.keys()):
#         if(len(intersect(line, graph[key])) != 0):
#             out = out + [key]
#             del graph[key]
#     return out

# def vertex_intersection(graph, line):
#     out = []
#     for key in list(graph.keys()):
#         if(len(intersect(line, graph[key])) != 0):
#             out = out + [key]
#     return out

# def flip(dic):
#     ret = dict()
#     for key in dic.keys():
#         for elem in dic[key]:
#             ret[elem] = key
#     return ret

# def generate_edges(graph, lx, ly):
#     assoc = dict()
#     temp = graph
#     display_coordinates(create_lines(200, lx, ly), lx, ly)
#     for base in range(lx):
#         crosses = intersection(temp, create_lines(base, lx, ly))
#         if(len(crosses) != 0):
#             for i in crosses:
#                 if(not i in assoc.keys()):
#                     assoc[i] = base
#     return assoc

# def generate_dRNG(graph, lx, ly):
#     output = dict()
#     for key in graph.keys():
#         output[key] = []
#     temp = graph.copy()
#     for key in graph.keys():
#         holder = graph[key]
#         del temp[key]
#         for i in range(ly):
#             holder2 = [(x, y+i) for (x, y) in holder]
#             if(vertex_intersection(temp, holder2) != []):
#                 intersects = vertex_intersection(temp, holder2)
#                 for elem in intersects:
#                     output[key] += [elem]
#                 break
#     return output

# def join(x1, y1, x2, y2, image):
#     array = numpy.array(image)
#     for t in range(10000):
#         xt = x1*t/10000 + x2*(10000-t)/10000
#         yt = y1*t/10000 + x2*(10000-t)/10000
#         array[int(xt)][int(yt)] = 100
#     return Image.fromarray(array)

# def draw_edges(edgeset, graph, image):
#     for key in edgeset.keys():
#         if(edgeset[key] != []):
#             for elem in edgeset[key]:
#                 print("Joining: ", graph[key][0][0], graph[key][0][1], graph[elem][0][0], graph[elem][0][1])
#                 image = join(graph[key][0][0], graph[key][0][1], graph[elem][0][0], graph[elem][0][1], image)
#     image.show()
