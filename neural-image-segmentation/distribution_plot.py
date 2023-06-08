import os
import glob
import json
import matplotlib.pyplot as plt
import numpy as np

img_dir = r"/Users/christopher/Downloads/bulk_result" #Put here in the directory where the files are (located from the ipynb file)
save_dir = r"/Users/christopher/Downloads/bulk_result/bulk_result_cell_area.png"
data_path = os.path.join(img_dir, "*.json")
files = glob.glob(data_path) # finds all file names in the img_dir

all_cell_area = []
for file in files:
    with open(file, 'r') as f:
        cur_file = json.load(f)
        all_cell_area += cur_file['Cell areas'].values()
all_cell_area_np = np.array(all_cell_area)

print("Total files number: " + str(len(files)))
print("Total cells number: " + str(len(all_cell_area_np)))

# plot
q25, q75 = np.percentile(all_cell_area_np, [25, 75])
bin_width = 2 * (q75 - q25) * len(all_cell_area_np) ** (-1/3)
bins = round((all_cell_area_np.max() - all_cell_area_np.min()) / bin_width)
plt.hist(all_cell_area_np, bins=bins*2)
plt.xlabel('Cell area in [μm^2]')
plt.ylabel('N')
# plt.show()
plt.savefig(save_dir)