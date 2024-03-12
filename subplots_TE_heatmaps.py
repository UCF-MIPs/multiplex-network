import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from src import generate_edge_types

image_files = list()
influence_type = ['TM', 'UM', 'TF', 'UF']

for edge_type in influence_type:
    for dataset in generate_edge_types.dataset_list:
        image_files.append(f'plots/{dataset}/TUMF/{dataset}_{edge_type}_out_activity.png')

# Create a new figure for the subplot grid
fig, axs = plt.subplots(4, 3, figsize=(8, 8))

# Flatten the axes array for easier iteration if necessary
axs_flat = axs.flatten()

for ax, img_file in zip(axs_flat, image_files):
    img = mpimg.imread(img_file)
    ax.imshow(img)
    ax.axis('off')  # Hide axis since these are pre-generated plots

plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=5, hspace=1)
plt.tight_layout()
plt.savefig('plots/subplots_TE_heatmaps.png')
plt.show()
