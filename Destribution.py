import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import Counter
import random


def random_color_generator():
    color = random.choice(list(mcolors.CSS4_COLORS.keys()))
    return color


def analyze_and_plot(directory):
    image_count = Counter()
    colors = []
    for root, dirs, files in os.walk(directory):
        if files:
            class_name = os.path.basename(root)
            print(class_name)
            image_count[class_name] = len(files)
            print(image_count[class_name])
            colors.append(random_color_generator())
            # color += 1
    # print(colors)
    labels = list(image_count.keys())
    # print("labels", labels)
    sizes = list(image_count.values())
    # print("sizes", sizes)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140,
            colors=colors, wedgeprops={"edgecolor": "black",
                                       'linewidth': 1,
                                       'antialiased': True})
    # uncomment if you want to see a legend
    # plt.legend(labels, loc=4)
    plt.title(f'{directory.split("/")[-1].lower()} class distribution',
              x=0.0, y=1.05, loc="left", fontsize=20)
    fig = plt.subplot(1, 2, 2)
    plt.bar(labels, sizes, color=colors, edgecolor='black')
    # rotate x labels
    fig.set_xticklabels(fig.get_xticklabels(), rotation=90)
    # adjust labels
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python Distribution.py <directory>")
        sys.exit(1)
    # python3 Destribution.py /Users/aleksandr/Desktop/Leaffliction/images/Apple
    directory = sys.argv[1]
    analyze_and_plot(directory)
