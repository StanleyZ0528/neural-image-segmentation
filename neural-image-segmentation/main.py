# This is the main script to start the application for our Neural Image Segmentation project
from postImgProc.utils import *
from postImgProc.alg import *
import warnings


def main():
    warnings.simplefilter("ignore")
    print("Starting application...")
    img = readImg('data/Axon2.png')
    # print(np.unique(img.reshape(-1, 3), axis=0))
    labeled_axon, labeled_cell, nr_cell = separate_axon_and_cell(img)
    axons_to_cells = get_touching_dict(labeled_axon, labeled_cell)
    fil = filter_axon(img)
    info_list = analyze_axons(fil, labeled_axon, axons_to_cells)
    segmented_axons = getSegmentedAxons(fil, info_list, nr_cell)
    # print(len(segmented_axons))
    # print(segmented_axons)
    plotOriginal(segmented_axons)


if __name__ == '__main__':
    main()
