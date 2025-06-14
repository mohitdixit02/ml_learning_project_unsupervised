import numpy as np
from utility.main import progress_bar

# original code
def k_means(points, centroids, k, m, n, max_itr=20, progress_bar_val=True): # k is no of clusters
    if progress_bar_val:
        print("Performing K-Means Clustering...")

    for itr in range(max_itr):
        if progress_bar_val:
            progress_bar(itr + 1, max_itr, prefix='Progress', suffix='Complete', bar_length=50)
        
        distances = np.linalg.norm(points[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)  # shape: (m, k)
        centroid_number_for_pixel = np.argmin(distances, axis=1)
        
        new_centroids = np.zeros((k, n))            
        # Update centroids
        for i in range(k):
            pixels_belonging_to_cluster = points[centroid_number_for_pixel == i]
            if len(pixels_belonging_to_cluster) > 0:
                new_centroids[i] = np.mean(pixels_belonging_to_cluster, axis=0)
            else:
                new_centroids[i] = points[np.random.choice(m)]
            
        # Check for convergence
        if np.all(centroids == new_centroids):
            if progress_bar_val:
                progress_bar(max_itr, max_itr, prefix='Progress', suffix='Complete', bar_length=50)
            break
            
        centroids = new_centroids

    return centroids, centroid_number_for_pixel

def initialize_color_centers(no_of_centres, m, points):
    indexes = np.random.choice(m, no_of_centres, replace=False)
    return points[indexes]

def text_enhancer(img):
    h, w, c = img.shape
    clustered_image = np.zeros_like(img)
    windows_size = 10
    print("performing Text Enhancement...\n")
    
    for i in range(0, h, windows_size):
        for j in range(0, w, windows_size):
            progress_bar(i * w + j, h * w, prefix='Progress', suffix='Complete', bar_length=50)
            window = img[i:i + windows_size, j:j + windows_size]
            points = window.reshape((-1, c))
            m, n = points.shape
            # Initialize color centers
            centroid_cords = initialize_color_centers(2, m, points)
            # k-means clustering
            new_centroid_cords, cluster_labels = k_means(points, centroid_cords, 2, m, n, 20, False)
            centroid_diff = np.linalg.norm(new_centroid_cords[1] - new_centroid_cords[0])*100
            if centroid_diff > 2:
                compressed_img = new_centroid_cords[cluster_labels].reshape(window.shape)
                compressed_img = np.clip(compressed_img, 0.0, 1.0)
                clustered_image[i:i + windows_size, j:j + windows_size] = compressed_img
            else:
                clustered_image[i:i + windows_size, j:j + windows_size] = window
    progress_bar(1, 1, prefix='Progress', suffix='Complete', bar_length=50)
    
    
    return clustered_image