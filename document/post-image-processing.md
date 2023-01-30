# Post Image Processing
## Description
This module tries to analyze the segmentation mask of the
image to generate useful information including cell/axon 
counts, axon length and orientation.
## Data Structure Explanation
labeled_axon/labeled_cell: This is an image mask for the
entire picture for axons/cells respectively. Each 
different non-adjacent cells will have a different number
associated with it and the number will range from 1 to
[total number of axons/cells].<br />

axons_to_cells: This is a two-level dictionary that maps
each of the axon to its touching cell, the first level
dictionary uses the axon index as the key, the second level
dictionary uses the cell index as the key, the value will
be the pixel position of the touching point. <br />

info_list: This is a list of dictionary containing the
information associated with each axon segment. More
specifically, it contains "touch_points", which are the
pixels that connect to cells; "end_points", which are the
points at the end of all the axon segment branches;
"intersect_points", which are the intersections of the
axon branches; "dists"(debug purpose only), which are
the lengths of each axon branch. <br />

segmented_axons: This is the final results consisting of
a list of line skeletons for the axons. It will be a 2d
array containing several lists of pixels indicating the
line skeleton for each axon.
## Algorithm Design
