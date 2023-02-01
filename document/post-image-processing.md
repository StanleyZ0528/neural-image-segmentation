# Post Image Processing
## Description
This module tries to analyze the segmentation mask of the
image to generate useful information including cell/axon 
counts, axon length and orientation.
## Data Structure Explanation
`labeled_axon/labeled_cell`: This is an image mask for the
entire picture for axons/cells respectively. Each 
different non-adjacent cells will have a different number
associated with it and the number will range from 1 to
[total number of axons/cells].<br />

`axons_to_cells`: This is a two-level dictionary that maps
each of the axon to its touching cell, the first level
dictionary uses the axon index as the key, the second level
dictionary uses the cell index as the key, the value will
be the pixel position of the touching point. <br />

`fil`: This is the analysis result from the `fil-finder`
library. It contains useful analysis result including
branch points, filament length, etc.<br />

`info_list`: This is a list of dictionary containing the
information associated with each axon segment. More
specifically, it contains "touch_points", which are the
pixels that connect to cells; "end_points", which are the
points at the end of all the axon segment branches;
"intersect_points", which are the intersections of the
axon branches; "dists"(debug purpose only), which are
the lengths of each axon branch. <br />

`segmented_axons`: This is the final results consisting of
a list of line skeletons for the axons. It will be a 2d
array containing several lists of pixels indicating the
line skeleton for each axon.
## Algorithm Design
1. As the input, you will get the segmented result of the
full-size image with three colors representing background,
cells and axons. The respected mask for cell and axon along
can be filtered out by color and saved inside
`labeled_axon/labeled_cell`
2. Run the `fil-finder` library on top of 
the image, and then the analysis result will be inside
the `fil` data structure. The most important information
it contains will be line skeletons for the axons including
the pixels for all the branches of each individual axon.
3. Additionally, get the points that are connected to the
originating cells and saved it in `axons_to_cells`. Based
on the `fil` and `axons_to_cells` information, the `info_list`
data structure can be generated to further separate and
analyze each axon segment.
4. After getting the line skeleton of 1 pixel width as shown
in the above picture in `fil` and the 'special points'
information inside `info_list`<br />
For each separate line skeleton: <br />
   - Find the longest branch starting from at least one
   point that is connected to the cell, remove these branches
   from the branch list. Repeat this action until all the
   originated points of the cells have one branch found.
   - Continue exploring the leftover branches, if there
   exists branch that is connected to the previously found
   axon, add it to that branch, remove it from the skeleton
   branch list.
   - Note that when there are multiple branches found can
   be added to an existing found branch, the one with the
   most close orientation will be added.
   - The length that is less than 20μm will be filtered to
   not counting towards the final analysis (222 pixels
   length is 100μm, so length less than or equal to 44
   pixels will be filtered out)
5. After finishing the previous step, we can get the final
result inside `segmented_axons`, the length of the array will
be the total number of axons, and the distance can be
calculated with each sub-array of pixels following along the
axons.
6. Finally, useful information including cell cluster count,
axon count, axon length distribution and axon orientation
will be shown at the bottom of the user interface.