import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def progress_bar(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(bar_length * iteration // total)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()  # Print a new line when done
        

def plot_image(data):
    print("\nPlotting images...")

    fig, axes = plt.subplots(1, len(data), figsize=(18, 6))
    
    for index, elem in enumerate(data):
        axes[index].imshow(elem["img"])
        axes[index].set_title(elem["title"])
    
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle("Preview", fontsize=16)
    plt.tight_layout()
    plt.show()
    
def save_reduced_image(reduced_img, title):
    reduced_img_uint8 = (reduced_img * 255).astype(np.uint8)
    reduced_img_bgr = cv.cvtColor(reduced_img_uint8, cv.COLOR_RGB2BGR)
    cv.imwrite(title, reduced_img_bgr, [cv.IMWRITE_JPEG_QUALITY, 90])
    print(f"Image saved successfully in the current directory.")