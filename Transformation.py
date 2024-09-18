import cv2
import numpy as np
from matplotlib import pyplot as plt

def gaussian_blur(image):
    return cv2.GaussianBlur(image, (15, 15), 0)

def create_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    return mask

def analyze_object(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    return image

def display_image_transformations(image_path):
    image = cv2.imread(image_path)

    # Apply transformations
    blur = gaussian_blur(image)
    mask = create_mask(image)
    analyzed = analyze_object(image.copy())

    # Display results
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original")

    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB))
    plt.title("Gaussian Blur")

    plt.subplot(2, 2, 3)
    plt.imshow(mask, cmap='gray')
    plt.title("Mask")

    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(analyzed, cv2.COLOR_BGR2RGB))
    plt.title("Analyzed Object")

    plt.show()

# Example usage
# display_image_transformations("images/Apple/apple_healthy/image (1).JPG")
def plot_color_histogram(image_path):
    image = cv2.imread(image_path)
    chans = cv2.split(image)
    colors = ("b", "g", "r")
    plt.figure()
    plt.title("Color Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")

    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])

    plt.show()

# Example usage
# plot_color_histogram("images/Apple/apple_healthy/image (1).JPG")

