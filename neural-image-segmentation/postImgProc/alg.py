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


# Return the orientation of axon in terms of degrees
def getOrientation(start, end):
    vertical_diff = start[0] - end[0] # The vertical coordinate of a smaller value is on the top
    horizontal_diff = end[1] - start[1]
    degree = math.degrees(math.atan2(vertical_diff, horizontal_diff))  # atan returns a value between -pi/2 and pi/2
    return degree


# Check if the pixel element is on the axon segment
def arr_in(ele, arr):
    for e in arr:
        if e[0] == ele[0] and e[1] == ele[1]:
            return True
    return False


def get_touch(ele, arr):
    for e in arr:
        if e[0] == ele[0] and e[1] == ele[1]:
            return e[2]
    return -1


# Since the intersection point might not be included in the branch pixel array,
# this branch is trying to find out if one end point is an intersect point
def close(ele1, ele2):
    dist = math.sqrt((ele1[0] - ele2[0]) ** 2 + (ele1[1] - ele2[1]) ** 2)
    return dist < 8


def getLineSegments(i, fil, info_list):
    line_segments = []
    touch_segments = set()
    for br_pts in fil.filaments[i].branch_pts():
        line = []
        for pt in br_pts:
            pt_wc = list(pt)
            pt_wc[0] += fil.filament_extents[i][0][0] - 1
            pt_wc[1] += fil.filament_extents[i][0][1] - 1
            line.append(pt_wc)
            if_touch = get_touch(pt_wc, info_list[i]["touch_points"])
            if if_touch != -1 and len(line) > 1:
                for inter in info_list[i]["intersect_points"]:
                    if close(line[-1], inter):
                        line.append(list(inter))
                    if close(line[0], inter):
                        line.insert(0, list(inter))
                touch_segments.add(len(line_segments))
                if_touch = get_touch(line[0], info_list[i]["touch_points"])
                if if_touch != -1:
                    touch_segments.add(len(line_segments))
                line_segments.append(line)
                line = [pt_wc]
        if len(line) > 1:
            for inter in info_list[i]["intersect_points"]:
                if close(line[-1], inter):
                    line.append(list(inter))
                if close(line[0], inter):
                    line.insert(0, list(inter))
            if_touch = get_touch(line[0], info_list[i]["touch_points"])
            if if_touch != -1:
                touch_segments.add(len(line_segments))
            if_touch = get_touch(line[-1], info_list[i]["touch_points"])
            if if_touch != -1:
                touch_segments.add(len(line_segments))
            line_segments.append(line)
    return line_segments, touch_segments


def getSegmentedAxons(fil, info_list, nr_cell):
    segmented_axons = []
    show_orientation = []
    for i in range(len(info_list)):
        line_segments, touch_segments = getLineSegments(i, fil, info_list)
        length = len(line_segments)
        print(length, touch_segments)
        line_used = [False for j in range(length)]
        count = 0
        itr = 0
        for j in touch_segments:
            line_used[j] = True
            count = count + 1
            segmented_axons.append(line_segments[j])
        while count < length and itr < 10:
            itr += 1
            for j in range(length):
                if line_used[j]:
                    continue
                pt_start_j = line_segments[j][0]
                pt_end_j = line_segments[j][-1]
                if (not arr_in(pt_start_j, info_list[i]["intersect_points"])) and (
                        not arr_in(pt_end_j, info_list[i]["intersect_points"])):
                    line_used[j] = True
                    count = count + 1
                    continue
                orientation_approx = 181
                assigned_seg = -1
                if arr_in(pt_start_j, info_list[i]["intersect_points"]):
                    for k in range(len(segmented_axons)):
                        sa = segmented_axons[k]
                        if pt_start_j == sa[0]:
                            ori1 = getOrientation(pt_start_j, pt_end_j)
                            ori2 = getOrientation(sa[0], sa[-1])
                            ori_approx = abs(ori1 - ori2) % 180
                            if ori_approx > 90:
                                ori_approx = 180 - ori_approx
                            print(ori_approx)
                            if ori_approx < orientation_approx:
                                orientation_approx = ori_approx
                                assigned_seg = k
                        if pt_start_j == sa[-1]:
                            ori1 = getOrientation(pt_start_j, pt_end_j)
                            ori2 = getOrientation(sa[0], sa[-1])
                            ori_approx = abs(ori1 - ori2) % 180
                            if ori_approx > 90:
                                ori_approx = 180 - ori_approx
                            print(ori_approx)
                            if ori_approx < orientation_approx:
                                orientation_approx = ori_approx
                                assigned_seg = k

                if arr_in(pt_end_j, info_list[i]["intersect_points"]):
                    for k in range(len(segmented_axons)):
                        sa = segmented_axons[k]
                        if pt_end_j == sa[0]:
                            ori1 = getOrientation(pt_start_j, pt_end_j)
                            ori2 = getOrientation(sa[0], sa[-1])
                            ori_approx = abs(ori1 - ori2) % 180
                            if ori_approx > 90:
                                ori_approx = 180 - ori_approx
                            print(ori_approx)
                            if ori_approx < orientation_approx:
                                orientation_approx = ori_approx
                                assigned_seg = k
                        if pt_end_j == sa[-1]:
                            ori1 = getOrientation(pt_start_j, pt_end_j)
                            ori2 = getOrientation(sa[0], sa[-1])
                            ori_approx = abs(ori1 - ori2) % 180
                            if ori_approx > 90:
                                ori_approx = 180 - ori_approx
                            print(ori_approx)
                            if ori_approx < orientation_approx:
                                orientation_approx = ori_approx
                                assigned_seg = k
                if assigned_seg != -1:
                    count = count + 1
                    line_used[j] = True
                    if segmented_axons[assigned_seg][0] == pt_start_j:
                        segmented_axons[assigned_seg].reverse()
                    elif segmented_axons[assigned_seg][0] == pt_end_j:
                        segmented_axons[assigned_seg].reverse()
                        line_segments[j].reverse()
                    elif segmented_axons[assigned_seg][-1] == pt_end_j:
                        line_segments[j].reverse()
                    segmented_axons[assigned_seg] = segmented_axons[assigned_seg] + line_segments[j]

    index_to_remove = []
    cell_axons_map = [[] for j in range(nr_cell)]
    all_touch_points = []
    for i in range(len(info_list)):
        for ele in info_list[i]["touch_points"]:
            all_touch_points.append(ele)
    for i in range(len(segmented_axons)):
        dist = cal_dist(segmented_axons[i])
        if dist <= 44:
            index_to_remove.append(i)
            continue
        touch_index1 = get_touch(segmented_axons[i][0], all_touch_points)
        if touch_index1 != -1:
            cell_axons_map[touch_index1 - 1].append(segmented_axons[i][:])
        touch_index2 = get_touch(segmented_axons[i][-1], all_touch_points)
        if touch_index2 != -1:
            cell_axons_map[touch_index2 - 1].append(segmented_axons[i][::-1])
        if touch_index1 != -1 and touch_index2 != -1:
            show_orientation.append(False)
        else:
            show_orientation.append(True)
    index_to_remove.reverse()
    for i in index_to_remove:
        del segmented_axons[i]
    return segmented_axons


def plotOriginal(segmented_axons):
    # File directory to the original image
    plt_img = Image.open(r'data/Axon2.tif')
    numpy_img = np.array(plt_img)
    opencv_img = cv2.cvtColor(numpy_img, cv2.COLOR_RGB2BGR)
    gray_img = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
    figure(figsize=(10, 10), dpi=80)
    plt.imshow(gray_img, cmap='gray')
    for i in range(len(segmented_axons)):
        for j in range(len(segmented_axons[i])):
            plt.scatter(segmented_axons[i][j][1], segmented_axons[i][j][0], c='orange', cmap='hot', marker=',', lw=1,
                        s=2, vmin=0, vmax=len(segmented_axons))
        plt.scatter(segmented_axons[i][0][1], segmented_axons[i][0][0], c='r', marker=',', lw=5, s=20)
        plt.scatter(segmented_axons[i][-1][1], segmented_axons[i][-1][0], c='r', marker=',', lw=5, s=20)
    plt.show()
