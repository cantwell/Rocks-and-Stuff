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
import math

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
    if(end_frame == start_frame + 1):
        return img_3d
    for frame in range(start_frame + 1, end_frame):
        img.seek(frame)    
        slice_2d = np.asarray(img)
        slice_3d = _2d_to_3d(slice_2d)
        img_3d = np.concatenate((img_3d, slice_3d), axis = 0)
    return img_3d

#call blackify on a color 2-D image to turn it into a 3-D numpy array
#that can be passed to rock_AStar
def blackify(rock):
    rock2 = np.zeros((rock.shape[0], rock.shape[1], 1), dtype = np.uint8)
    rock2.fill(255)
    for x in range(rock.shape[0]):
        for y in range(rock.shape[1]):
            for z in range(rock.shape[2]):
                if(rock[x][y][z] != 255):
                    rock2[x][y][0] = 0
    return rock2

def tif_to_3d(img, x0, x1, y0, y1, z0, z1):
    img.seek(x0)
    slice_2d = np.asarray(img.crop((y0, z0, y1, z1)))
    img_3d = _2d_to_3d(slice_2d) 
    for frame in range(x0+1, x1):
        img.seek(frame)
        slice_2d = np.asarray(img.crop((y0, z0, y1, z1)))
        slice_3d = _2d_to_3d(slice_2d)
        img_3d = np.concatenate((img_3d, slice_3d))
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

def read_coloredslice(filename):
    img = Image.open(filename)
    rock = tiff_to_3d(img, None, None)
    rock2 = blackify(rock)    
    return rock2
    
def sample(path):
    total = 0
    for i in range(len(path)):
        total += euclidean(path[i], path[i+1])
    return total
    
def crop(array, shape):
    new = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                new[i][j][k] = array[i][j][k]
    return new
    
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
    return (lambda point :  300*(-dot(sub(start, point), sub(start, end)/(euclidean(start, point)*euclidean(start, end)))))

def metric2(rock, c):
    def distance(point1, point2):
        if(rock[point2[0]][point2[1]][point2[2]] == 0):
            return euclidean(point1, point2)
        else:
            return c*euclidean(point1, point2)
    return distance

def metric3(rock, c):
    def distance(point1, point2):
        x1 = point1[0]
        x2 = point2[0]
        if(rock[point2[0]][point2[1]][point2[2]] == 0):
            return (1 + 4*abs(x1-x2))*euclidean(point1, point2)
        else:
            return (1 +4*abs(x1-x2))*c*euclidean(point1, point2)
    return distance
    
def adj(rock, point, visit):
    x, y, z = point
    list = [(x+1, y, z), (x-1, y, z), (x, y+1, z), (x, y-1, z), (x, y, z+1), (x, y, z-1)]
    list = filter(lambda (x, y, z): (x in range(rock.shape[0]) and y in range(rock.shape[1]) and z in range(rock.shape[2]) and not (x, y, z) in visit), list)
    return list

def rock_AStar(rock, start, end, distance, h, adj):
    PQ = Queue.PriorityQueue()    
    for x in start:
        PQ.put((h(x), 0, x, []))
    visited = start
    while(not PQ.empty()):
        (heu, d, rover, path) = PQ.get()
        n = adj(rock, rover, visited)
        visited = visited + n.keys()
        if(rover in end):
            break
        for point in n:
            newd = d + distance(rover, point)
            PQ.put((newd + h(point), newd, point, path + [rover]))
    if(not rover in end):
        print("Couldn't find end")
        return []
    else:
        return d, path, visited
    
def adj2(rock, point, visit):
    x, y, z = point
    list = [(x+1, y, z), (x-1, y, z), (x, y+1, z), (x, y-1, z), (x, y, z+1), (x, y, z-1), (x+1, y+1, z), (x-1, y+1, z), (x+1, y-1, z), (x+1, y, z+1), (x+1, y, z-1), 
            (x, y+1, z+1), (x, y+1, z-1)]
    list = filter(lambda (x, y, z): (x in range(rock.shape[0]) and y in range(rock.shape[1]) and z in range(rock.shape[2]) and not (x, y, z) in visit), list)
    out = dict()
    for elem in list:
        out[elem] = []
    return out

def Astar(rock, start, end, distance, h, adj):
    PQ = Queue.PriorityQueue()
    visited = [start]
    paths = dict()
    paths[start] = []
    distances = dict()
    PQ.put((h(start), 0, start, []))
    while(not PQ.empty()):
        (he, d, rover, path) = PQ.get()
        n = adj(rock, rover, visited)
        visited = visited + n.keys()
        for point in n.keys():
            if(point in end):
                paths[point] = path + n[point]
            newd = d+distance(rover, point)
            distances[point] = newd
            PQ.put((newd+h(point), newd, point, path+[rover]+n[point]))
        if(set(end).issubset(set(visited))):
            break
    if(not set(end).issubset(set(visited))):
        print("search failed")
    return paths, visited, distances

def edges(chunk):
    e = []
    for (x, y, z) in chunk:
        if(not (x+1, y, z) in chunk):
                e = e + [(x, y, z)]             
                continue
        if(not (x-1, y, z) in chunk):
            e = e + [(x, y, z)]             
            continue
        if(not (x, y+1, z) in chunk):
            e = e + [(x, y, z)]             
            continue
        if(not (x, y-1, z) in chunk):
            e = e + [(x, y, z)]             
            continue
        if(not (x, y, z+1) in chunk):
            e = e + [(x, y, z)]             
            continue
        if(not (x, y, z-1) in chunk):
            e = e + [(x, y, z)]             
            continue
    return e

def chunk(rock, distance, chunks):
    adj_dict = dict()
    distance_dict = dict()
    total_chunk = []
    for chunk in chunks:
        total_chunk += chunk
        e = edges(chunk)
        print(e)
        for point1 in e: 
            print point1
            temp = e[:]
            temp.remove(point1)
            paths, visited, distances = Astar(rock, point1, temp, distance, lambda x : 0, adj2)
            distance_dict[point1] = distances
            adj_dict[point1] = paths
    print(distance_dict)
    print(adj_dict)
    def d(point1, point2):
        if(point1 in distance_dict):
            if(point2 in distance_dict[point1]):
                return distance_dict[point1][point2]
        return distance(point1, point2)

    def adjac(rock, point, visit):
        if(point in adj_dict):
            temp_dict = adj_dict[point].copy()
            for p in adj2(rock, point, visit):
                temp_dict[p] = []
            l = filter(lambda x : not x in visited or not x in chunk, temp_dict.keys())
            ret = dict()
            for x in l:
                ret[x] = temp_dict[x]
            return ret
        else:
            return adj2(rock, point, visit)
            
    return d, adjac

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
    
def boundingbox(x1, y1, z1, x2, y2, z2):
    temp = []
    for x in range(x1, x2):
        for y in range(y1, y2):
            for z in range(z1, z2):
                temp += [(x, y, z)]
    return temp
    
def get_component_array(array, list):
     if(list == []):
         return []
     temp = []
     for (x, y, z) in list:
         list.remove((x, y, z))
         if(array[x][y][z] == 0):
             array[x][y][z] = 255
             temp = temp + [(x, y, z)]
             if(x < array.shape[0] - 1):
                 list = list + [(x+1, y, z)]
             if(x > 0):
                 list = list + [(x-1, y, z)]
             if(y < array.shape[1] - 1):
                 list = list + [(x, y+1, z)]
             if(y > 0):
                 list = list + [(x, y-1, z)]
             if(z < array.shape[2] - 1):
                 list = list + [(x, y, z+1)]
             if(z > 0):
                 list = list + [(x, y, z-1)]
     return temp + get_component_array(array, list)

def get_component(array, temp, x, y, z, i):
     list = [(x, y, z)]
     l = get_component_array(array, list)
     if(l == []):
         return (temp, i)
     temp[i] = l
     i = i+1
     return (temp, i)    
    
def scan(array):
    temp = dict()
    i = 0
    a = array.copy()
    labels = list(list(list(-1 for x in range(array.shape[2])) for y in range(array.shape[1])) for z in range(array.shape[0]))
    for x in range(array.shape[0]):
        for y in range(array.shape[1]):
            for z in range(array.shape[2]):
                if(array[x][y][z] == 0):
                    (temp, i) = get_component(a, temp, x, y, z, i)
                    for (x, y, z) in temp[i-1]:
                        labels[x][y][z] = i-1
    return temp, labels

def adjCount(labels, point, eps):
    x, y, z = point
    found = [labels[x][y][z]]
    for x1 in range(x-eps, x+eps):
        for y1 in range(y-eps, y+eps):
            for z1 in range(z-eps, z+eps):
                if(x1 in range(len(labels)) and y1 in range(len(labels[0])) and z1 in range(len(labels[0][0])) and euclidean((x1, y1, z1), point) < eps):
                    if((not labels[x1][y1][z1] in found) and labels[x1][y1][z1] != -1):
                        found = list(set(found) | set([labels[x1][y1][z1]]))
    return found

def regionQuery(graph, labels, i, eps):
    found = []
    for point in edges(graph[i]):
        found = list(set(found) | set(adjCount(labels, point, eps)))
    return found

def findClusters(graph, labels, eps, mincount):
    visited = []
    noise = []
    numClusters = 0
    clusters = dict()
    for i in graph:
        if(i in visited):
            continue
        else:
            visited = visited + [i]
            found = regionQuery(graph, labels, i, eps)
            if(len(found) < mincount):
                noise = noise + [i]
            else:
                clusters, visited = expandCluster(graph, labels, i, found, numClusters, eps, mincount, clusters, visited)
                numClusters = numClusters+1
    return clusters

def expandCluster(graph, labels, P, neighbors, C, eps, MinPts, clusters, visited):
    clusters[C] = [P]
    for i in neighbors:
        if(not i in visited):
            visited = visited + [i]
            neighbors2 = regionQuery(graph, labels, i, eps)
            if(len(neighbors2) >= MinPts):
                neighbors = neighbors + neighbors2
        if(not i in clusters.values()):
            clusters[C] += [i]
    return clusters, visited
    
def displayClusters(rock, graph, clusters):
    thing = np.zeros(rock.shape)
    points = []
    voids = []
    for i in clusters.values():
        voids += i
    for i in voids:
        points += graph[i]
    for point in points:
        x, y, z = point
        thing[x][y][z] = 255
    @mlab.animate
    def anim():
        f = mlab.gcf()
        while 1:
            f.scene.camera.azimuth(1)
            f.scene.render()
            yield
    mlab.figure(bgcolor=(1,1,1)) # Set bkg color
    mlab.contour3d(thing, 
                   color = (0,0,0),
                   contours = 2,
                   opacity = .2 + .8/rock.shape[0]) # Draw pores for 3d, changed froo .20 * 100 / self.shape[0]
    a = anim()
    
def generateCenters(clusters, graph):
    centers = dict()
    means = dict()
    for i in clusters:
        tx = 0
        ty = 0
        tz = 0
        count = 0
        for j in clusters[i]:
            for (x, y, z) in graph[j]:
                tx += x
                ty += y
                tz += z
                count += 1
        centers[i] = (tx/count, ty/count, tz/count)
    for i in clusters:
        std2 = 0
        for j in clusters[i]:
            for (x, y, z) in graph[j]:
                std2 += (euclidean((x, y, z), centers[i]).astype(int))^2
        means[i] = math.sqrt(std2)
    return centers, means
    
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    
def clusterh(centers, means, c):
    def heuristic(point):
        total = 0
        for i in centers:
            total += gaussian(euclidean(centers[i], point), 0, means[i])
        total = total * c
        return -total
    return heuristic
    
def find_values():
    #Replace the file name with the location of the picture
    #When you generate the picture, crop out the velocity bar and make it a seperate picture.  
    a = Image.open(r"C:\MyWork\Summer\Kaushik2016\velocity bar.png")
    b = Image.open(r"C:\MyWork\Summer\Kaushik2016\velocity picture.png")
    bar = np.array(a)
    pic = np.array(b)
    print pic.shape
    ret = np.zeros((pic.shape[0], pic.shape[1], 1))
    for x in range(pic.shape[0]):
        for y in range(pic.shape[1]):
            (r, g, bl) = (pic[x][y][0], pic[x][y][1], pic[x][y][2])
            for x2 in range(bar.shape[0]):
                for y2 in range(bar.shape[1]):
                    if(abs(bar[x2][y2][0] - r) + abs(bar[x2][y2][1] - g) + abs(bar[x2][y2][2] - bl) < 5):
                        ret[x][y][0] = x2
                        break
    return ret
    
def comparison(array, visited):
    total = 0
    for (x, y, z) in visited:
        total += array[x][y][z]
    total = total/len(visited)
    return total
    
start = []
for i in range(292):
    start = start + [(i, 0, 0)]

end = []
for i in range(292):
    end = end + [(i, 382, 0)]

#Pass in the array from find_values, or take out the commented line below
def test(array):
    #array = find_values()
    rock = read_coloredslice(r"C:\MyWork\Summer\Kaushik2016\Picture\pic3.tif")
    rock2 = crop(rock, array.shape)
    p = []
    v = []
    best = 0
    for i in range(50):
        print 123456, i
        for j in range(3):
            print (j+1)/4.0
            d, path, visited = rock_AStar(rock2, start, end, metric3(rock2, 4*(i+1)), sampleh((145, 0, 0), (145, 382, 0)), adj2)
            c = comparison(array, visited)
            if(c > best):
                best = c
                v = visited
                p = path
    return v
    

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
