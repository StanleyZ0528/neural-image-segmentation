# This is the main script to start the application for our Neural Image Segmentation project
from postImgProc.utils import *
import warnings


def main():
    warnings.simplefilter("ignore")
    print("Starting application...")
    img = readImg('data/Axon1.png')
    labeled_axon, labeled_cell = separate_axon_and_cell(img)
    axons_to_cells = get_touching_dict(labeled_axon, labeled_cell)
    fil = filter_axon(img)
    analyze_axons(fil, labeled_axon, axons_to_cells)


if __name__ == '__main__':
    main()
