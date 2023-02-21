import cv2
import math

CLOSE = 8


def readImg(path):
    img_ori = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
    return img


# Translate distance from number of pixels to total length
def pixel_to_length(pixel_length):
    return pixel_length / 2.22


# Return the orientation of axon in terms of degrees
def getOrientation(start, end):
    vertical_diff = start[0] - end[0]  # The vertical coordinate of a smaller value is on the top
    horizontal_diff = end[1] - start[1]
    degree = math.degrees(math.atan2(vertical_diff, horizontal_diff))  # atan returns a value between -pi/2 and pi/2
    return degree


# Calculate the total length of an axon segment
def cal_dist(arr):
    prev = arr[0]
    dist = 0
    # print(branch_array)
    for pt in arr:
        if prev[0] != pt[0] and prev[1] != pt[1]:
            dist += math.sqrt((pt[0] - prev[0]) ** 2 + (pt[1] - prev[1]) ** 2)
            prev = pt
    dist += math.sqrt((arr[-1][0] - prev[0]) ** 2 + (arr[-1][1] - prev[1]) ** 2)
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


# Check if the pixel element is on the axon segment
def arr_in(ele, arr):
    for e in arr:
        if e[0] == ele[0] and e[1] == ele[1]:
            return True
    return False


# Get the touching points
def get_touch(ele, arr):
    for e in arr:
        if e[0] == ele[0] and e[1] == ele[1]:
            return e[2]
    return -1


# Since the intersection point might not be included in the branch pixel array,
# this branch is trying to find out if one end point is an intersect point
def close(ele1, ele2):
    dist = math.sqrt((ele1[0] - ele2[0]) ** 2 + (ele1[1] - ele2[1]) ** 2)
    return dist < CLOSE

