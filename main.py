import numpy as np
import cv2 as cv
import math
from algorithm.pca import pca_dimen_reduction
from algorithm.kmeans import k_means, initialize_color_centers, text_enhancer
from utility.main import plot_image, save_reduced_image

if __name__ == "__main__":
    print("\n************** Welcome to Image Reduction Program using K-Means Clustering **************")
    print("1. Image Color reduction (This program reduce the number of colors for a given image using K-Means Clustering algorithm and PCA.)\n")
    print("2. Text Enhancement (This program enhance the text features using K-Means Clustering algorithm.)\n")
    img_url = input("Paste your image in input folder and enter its name with format: ")
    
    print("\nLoading image...\n")
    img = cv.imread(f"./input/{img_url}")
    if img is None:
        print(f"Error: Image '{img_url}' not found in the input folder.")
        exit(1)
        
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = img / 255.0
    points = img.reshape((-1, img.shape[2]))
    
    m, n = points.shape
    
    work_choice = input("Choose the work you want to perform (1 for Color Reduction, 2 for Text Enhancement): ").strip()
    img_result_data = [
        {
            "img": img,
            "title": "Original Image",
        }
    ]
        
    if work_choice == '1':
        # memory check
        memory_gib = m / (1024 ** 3)
        memory_gib = memory_gib * 8
        possible_compr = math.ceil(1 / memory_gib)
        print(f"Maximum colors available for reduction: {possible_compr}")
        
        compr_factor = int(input("Enter the number of colors you want to reduce the image to: "))
        if compr_factor > possible_compr:
            print(f"Warning: Higher color reduction factor than available entered.")
            exit(1)
        
        pca_reduction = input("Do you want to perform PCA for dimensionality reduction? (yes/no): ").strip().lower()
        if pca_reduction not in ['yes', 'no']:
            print("Invalid input for PCA reduction. Please enter 'yes' or 'no'.")
            exit(1)

        if pca_reduction == 'yes':
            points, cnv_image = pca_dimen_reduction(2, points, img.shape)
            img_result_data.append({
                "img": cnv_image,
                "title": "PCA Reduced Image",
            })
        
        # Initialize color centers
        centroid_cords = initialize_color_centers(compr_factor, m, points)
        
        # k-means clustering
        new_centroid_cords, cluster_labels = k_means(points, centroid_cords, compr_factor, m, n, 20)
        compressed_img = new_centroid_cords[cluster_labels].reshape(img.shape)
        compressed_img = np.clip(compressed_img, 0.0, 1.0)
        
        img_result_data.append({
            "img": compressed_img,
            "title": f"Color Reduced Image ({compr_factor} Colors)",
        })
        plot_image(img_result_data)
        save_reduced_image(compressed_img, f"./reduced_{img_url}")
    elif work_choice == '2':
        clustered_image = text_enhancer(img)
        clustered_image = np.clip(clustered_image, 0.0, 1.0)
        img_result_data.append({
            "img": clustered_image,
            "title": "Text Enhanced Image",
        })
        plot_image(img_result_data)
        save_reduced_image(clustered_image, f"./enhanced_{img_url}")
    else:
        print("Invalid choice. Please enter 1 or 2.")
        exit(1)
    
    exit(0)