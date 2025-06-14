from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def pca_dimen_reduction(features, points, image_shape):
    print("Performing PCA for Dimensionality Reduction...")
    pca = PCA(n_components=features)
    reduced_points = pca.fit_transform(points)
    print("Reduced points shape:", reduced_points.shape)
    converted_reduced_points = pca.inverse_transform(reduced_points)
    converted_image = converted_reduced_points.reshape(image_shape[0], image_shape[1], 3)
    converted_image = np.clip(converted_image, 0.0, 1.0)

    
    return converted_reduced_points, converted_image
