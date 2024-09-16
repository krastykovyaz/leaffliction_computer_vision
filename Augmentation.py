from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import os


def augment_image(image_path):
    img = Image.open(image_path)
    img_name, img_extension = os.path.splitext(image_path)

    # Flip the image horizontally
    img_flip = ImageOps.mirror(img)
    img_flip.save(f"{img_name}_Flip{img_extension}")

    # Rotate the image by 90 degrees
    img_rotate = img.rotate(90, expand=True)
    img_rotate.save(f"{img_name}_Rotate{img_extension}")

    # Skew the image
    img_skew = img.transform(img.size, Image.AFFINE, (1, 0.3, 0, 0.3, 1, 0))
    img_skew.save(f"{img_name}_Skew{img_extension}")

    # Shear the image
    img_shear = img.transform(img.size, Image.AFFINE, (1, 0.3, 0, 0.3, 1, 0))
    img_shear.save(f"{img_name}_Shear{img_extension}")

    # Crop the image (10% from each side)
    width, height = img.size
    img_crop = img.crop((int(0.1 * width),
                         int(0.1 * height),
                         int(0.9 * width),
                         int(0.9 * height)))
    img_crop.save(f"{img_name}_Crop{img_extension}")

    # Distort the image
    img_distort = img.transform(img.size,
                                Image.QUAD,
                                (0, 0, width, 0,
                                    width - 30,
                                    height, 30,
                                    height))
    img_distort.save(f"{img_name}_Distortion{img_extension}")

    # Blur the image
    img_blur = img.filter(ImageFilter.GaussianBlur(5))
    img_blur.save(f"{img_name}_Blur{img_extension}")

    # Adjust contrast
    enhancer = ImageEnhance.Contrast(img)
    img_contrast = enhancer.enhance(2)
    img_contrast.save(f"{img_name}_Contrast{img_extension}")

    # Scaling (resize image)
    img_scaling = img.resize((int(width * 1.2), int(height * 1.2)))
    img_scaling.save(f"{img_name}_Scaling{img_extension}")

    # Adjust Illumination (Brightness)
    enhancer = ImageEnhance.Brightness(img)
    img_illumination = enhancer.enhance(1.5)
    img_illumination.save(f"{img_name}_Illumination{img_extension}")
    print('OK')


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python Augmentation.py <directory>")
        sys.exit(1)
    directory = sys.argv[1]
    augment_image(directory)
