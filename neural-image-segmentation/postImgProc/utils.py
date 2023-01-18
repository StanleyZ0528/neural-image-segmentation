import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from matplotlib.pyplot import figure
from scipy import ndimage
import math
import copy
from fil_finder import FilFinder2D, Filament2D
import astropy.units as u
axon_color = np.array([255, 129, 31])
cell_color = np.array([255, 0, 255])


def readImg(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # figure(figsize=(10, 10), dpi=60)
    # plt.imshow(img_rgb)
    # plt.show()
    return img_rgb


def separate_axon_and_cell(img):
    # Axon color: (119, 11, 32)
    # axon_filter = cv2.inRange(img, np.array([119, 11, 32]), np.array([119, 11, 32]))
    axon_filter = cv2.inRange(img, axon_color, axon_color)
    # Cell color: (244, 35, 232)
    # cell_filter = cv2.inRange(img, np.array([244, 35, 232]), np.array([244, 35, 232]))
    cell_filter = cv2.inRange(img, cell_color, cell_color)
    labeled_axon, nr_axon = ndimage.label(axon_filter)
    labeled_cell, nr_cell = ndimage.label(cell_filter)
    fig, axs = plt.subplots(1, 2, figsize=(15, 15), dpi=80)
    axs[0].imshow(labeled_axon, )
    axs[0].set_title('Segmented axon clusters count: ' + str(nr_axon))
    axs[1].imshow(labeled_cell)
    axs[1].set_title('Segmented cell clusters count: ' + str(nr_cell))
    plt.show()
    return labeled_axon, labeled_cell, nr_cell


def get_touching_dict(labeled_axon, labeled_cell):
    axon_adjacent = copy.deepcopy(labeled_axon)
    labeled_axon_np = np.array(labeled_axon)
    labeled_cell_np = np.array(labeled_cell)
    height = len(labeled_axon_np)
    width = len(labeled_axon_np[0])
    axons_to_cells = {}
    for i in range(height):
        for j in range(width):
            if axon_adjacent[i][j] == 0:
                continue
            if i == 0 or j == 0 or i == height - 1 or j == width - 1:
                continue
            if labeled_cell_np[i - 1][j] != 0 or labeled_cell_np[i + 1][j] != 0 or labeled_cell_np[i][j - 1] != 0 or \
                    labeled_cell_np[i][j - 1] != 0:
                if axon_adjacent[i][j] not in axons_to_cells.keys():
                    axons_to_cells[axon_adjacent[i][j]] = {}
                touching_cell = -1
                if labeled_cell_np[i - 1][j] != 0:
                    touching_cell = labeled_cell_np[i - 1][j]
                elif labeled_cell_np[i + 1][j] != 0:
                    touching_cell = labeled_cell_np[i + 1][j]
                elif labeled_cell_np[i][j - 1] != 0:
                    touching_cell = labeled_cell_np[i][j - 1]
                else:
                    touching_cell = labeled_cell_np[i][j + 1]
                axons_to_cells[axon_adjacent[i][j]][touching_cell] = [i, j]
    return axons_to_cells


def filter_axon(img):
    # axon_filter = cv2.inRange(img, np.array([119, 11, 32]), np.array([119, 11, 32]))
    axon_filter = cv2.inRange(img, axon_color, axon_color)
    fil = FilFinder2D(axon_filter, distance=250 * u.pc, mask=axon_filter)
    fil.preprocess_image(flatten_percent=85)
    fil.create_mask(border_masking=True, verbose=False, use_existing_mask=True)
    fil.medskel(verbose=False)
    fil.analyze_skeletons(branch_thresh=40 * u.pix, skel_thresh=10 * u.pix, prune_criteria='length')

    # Show the longest path
    # figure(figsize=(10, 10), dpi=80)
    # plt.title('Line skeletons of axon segments and their main paths')
    # plt.imshow(fil.skeleton, cmap='gray')
    # plt.contour(fil.skeleton_longpath, colors='r')
    # plt.show()
    return fil


def fil_info(fil):
    print(fil.lengths())
    print(fil.branch_lengths())


# Calculate the total length of an axon segment
def cal_dist(arr):
    prev = arr[0]
    dist = 0
    # print(branch_array)
    for pt in arr:
        if prev[0] != pt[0] and prev[1] != pt[1]:
            dist += math.sqrt((pt[0]-prev[0])**2+(pt[1]-prev[1])**2)
            prev = pt
    dist += math.sqrt((arr[-1][0]-prev[0])**2+(arr[-1][1]-prev[1])**2)
    return dist


# Get the pixel that is closest to the touching cell on the skeleton
def getTouchingPoint(filament, touch_coord):
    candidate_coords = filament.pixel_coords
    if len(candidate_coords) == 0:
        return [-1, -1]
    length = len(candidate_coords[0])
    x_closest = candidate_coords[0][0]
    y_closest = candidate_coords[1][0]
    min_dist = math.sqrt((touch_coord[0] - x_closest) ** 2 + (touch_coord[1] - y_closest) ** 2)
    for i in range(1, length):
        x = candidate_coords[0][i]
        y = candidate_coords[1][i]
        new_dist = math.sqrt((touch_coord[0] - x) ** 2 + (touch_coord[1] - y) ** 2)
        if new_dist < min_dist:
            x_closest = x
            y_closest = y
            min_dist = new_dist
    return [x_closest, y_closest]


def analyze_axons(fil, axon_adjacent, axons_to_cells):
    length_filament = len(fil.filaments)
    figure(figsize=(10, 10), dpi=80)
    plt.imshow(fil.skeleton, cmap='gray')
    info_list = []
    for i in range(length_filament):
        x = fil.filaments[i].pixel_coords[0][0]
        y = fil.filaments[i].pixel_coords[1][0]
        if not (axon_adjacent[x][y] in axons_to_cells.keys()):
            info_list.append({"touch_points": [], "end_points": [], "intersect_points": [], "dists": []})
            continue
        touching_points = axons_to_cells[axon_adjacent[x][y]]
        index = 0
        touch_points = []
        end_points = fil.filaments[i].end_pts
        dists = []
        intersect_points = []
        for touching_key in touching_points.keys():
            touching_coord = touching_points[touching_key]
            touch_point = getTouchingPoint(fil.filaments[i], touching_coord)
            touch_point.append(touching_key)
            touch_points.append(touch_point)
        for branch_array in fil.filaments[i].branch_pts():
            prev = [-1, -1]
            dist = cal_dist(branch_array)
            dists.append(dist)
        for ele in fil.filaments[i].intersec_pts:
            for e in ele:
                intersect_points.append(e)
        info_list.append({"touch_points": touch_points, "end_points": end_points, "intersect_points": intersect_points,
                          "dists": dists})
        # print(intersect_points)
        for coord in touch_points:
            plt.plot(coord[1], coord[0], 'ro')
        for coord in intersect_points:
            if not type(coord) is tuple:
                continue
            plt.plot(coord[1], coord[0], 'bo')
        for coord in end_points:
            if not type(coord) is tuple:
                continue
            plt.plot(coord[1], coord[0], 'go')
    for info in info_list:
        print(info)
    plt.show()
    return info_list
