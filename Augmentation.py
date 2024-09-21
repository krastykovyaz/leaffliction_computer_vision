#!/usr/bin/env python3

from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import os
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def augment_image(image_path, number):
    try:
        logger.info(f"Opening image: {image_path}")
        img = Image.open(image_path)
        img_name, img_extension = os.path.splitext(image_path)
        if number >= 1:
            # Flip the image horizontally
            img_flip = ImageOps.mirror(img)
            flip_path = f"{img_name}_Flip{img_extension}"
            img_flip.save(flip_path)
            logger.info(f"Saved flipped image: {flip_path}")
        if number >= 2:
            # Rotate the image by 90 degrees
            img_rotate = img.rotate(90, expand=True)
            rotate_path = f"{img_name}_Rotate{img_extension}"
            img_rotate.save(rotate_path)
            logger.info(f"Saved rotated image: {rotate_path}")
        if number >= 3:
            # Skew the image
            img_skew = img.transform(img.size,
                                     Image.AFFINE,
                                     (1, 0.3, 0, 0.3, 1, 0))
            skew_path = f"{img_name}_Skew{img_extension}"
            img_skew.save(skew_path)
            logger.info(f"Saved skewed image: {skew_path}")
        if number >= 4:
            # Shear the image
            img_shear = img.transform(img.size,
                                      Image.AFFINE,
                                      (1, 0.4, 0, 0.4, 1, 0))
            shear_path = f"{img_name}_Shear{img_extension}"
            img_shear.save(shear_path)
            logger.info(f"Saved sheared image: {shear_path}")
        if number >= 5:
            # Crop the image (10% from each side)
            width, height = img.size
            img_crop = img.crop((int(0.1 * width),
                                 int(0.1 * height),
                                 int(0.9 * width),
                                 int(0.9 * height)))
            crop_path = f"{img_name}_Crop{img_extension}"
            img_crop.save(crop_path)
            logger.info(f"Saved cropped image: {crop_path}")

        if number >= 6:
            # Distort the image
            img_distort = img.transform(img.size,
                                        Image.QUAD,
                                        (0, 0, width, 0,
                                         width - 30,
                                         height, 30,
                                         height))
            distort_path = f"{img_name}_Distortion{img_extension}"
            img_distort.save(distort_path)
            logger.info(f"Saved distorted image: {distort_path}")


    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
    finally:
        logger.info(f'Augmentation of image {image_path} completed.')


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        logger.error("Usage: python Augmentation.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    if not os.path.isfile(image_path):
        logger.error(f"Image file not found: {image_path}")
        sys.exit(1)

    augment_image(image_path, 10)
