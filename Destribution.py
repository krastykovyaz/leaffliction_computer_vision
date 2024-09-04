import os
import matplotlib.pyplot as plt
from collections import Counter

def analyze_and_plot(directory):
    image_count = Counter()

    for root, dirs, files in os.walk(directory):
        if files:
            class_name = os.path.basename(root)
            image_count[class_name] = len(files)
    
    labels = list(image_count.keys())
    sizes = list(image_count.values())

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title(f'{directory.split("/")[-1]} class distribution')

    plt.subplot(1, 2, 2)
    plt.bar(labels, sizes, color=['blue', 'orange', 'green', 'purple'])
    plt.title(f'{directory.split("/")[-1]} class distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of images')

    plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python Distribution.py <directory>")
        sys.exit(1)
    # python3 Destribution.py /Users/aleksandr/Desktop/Leaffliction/images/Apple
    directory = sys.argv[1]
    analyze_and_plot(directory)
