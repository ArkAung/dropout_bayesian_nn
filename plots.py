import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import os


def plot_grid(rows, cols, figsize, image_root_path, labels, data_shape):
    f, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=figsize)

    for ax, label, name in zip(axes.ravel(), labels['Label'], labels['Common Name']):
        img = np.random.choice(os.listdir(os.path.join(image_root_path, label)))
        img = Image.open(os.path.join(image_root_path, label, img))
        img = img.resize(img, data_shape)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(name)


def visualize_probdist(distribution, images, labels, label_mapping):
    test_id = np.random.randint(0, high=len(labels), size=(6,))
    f, axes = plt.subplots(len(test_id), 2, figsize=(10, 24))
    f.tight_layout(h_pad=5, w_pad=0)
    axs = axes.ravel()

    ax_idx = 0
    for tid in test_id:
        current_ax = axs[ax_idx]
        for i in range(5):
            current_ax.hist(distribution[tid][:, i], alpha=0.3, label=label_mapping[i])
            current_ax.axvline(np.quantile(distribution[tid][:, i], 0.5), color='red', linestyle=':', alpha=0.4)
            current_ax.axvline(0.5, color='green', linestyle='--')
            current_ax.legend()
            current_ax.set_xlabel('probability')
            current_ax.set_ylabel('count')
            current_ax.title.set_text("Correct Label: {}".format(label_mapping[labels[tid]]))
        np.set_printoptions(False)
        ax_idx += 1
        current_ax = axs[ax_idx]
        current_ax.axis('off')
        current_ax.title.set_text("For Test Image Index: {}".format(tid))
        current_ax.imshow(images[tid])
        ax_idx += 1