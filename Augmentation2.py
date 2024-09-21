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
            # Blur the image
            img_blur = img.filter(ImageFilter.GaussianBlur(5))
            blur_path = f"{img_name}_Blur{img_extension}"
            img_blur.save(blur_path)
            logger.info(f"Saved blurred image: {blur_path}")

        if number >= 2:
            # Adjust contrast
            enhancer = ImageEnhance.Contrast(img)
            img_contrast = enhancer.enhance(2)
            contrast_path = f"{img_name}_Contrast{img_extension}"
            img_contrast.save(contrast_path)
            logger.info(f"Saved contrast-enhanced image: {contrast_path}")

        if number >= 3:
            width, height = img.size
            # Scaling (resize image)
            img_scaling = img.resize((int(width * 1.2), int(height * 1.2)))
            scaling_path = f"{img_name}_Scaling{img_extension}"
            img_scaling.save(scaling_path)
            logger.info(f"Saved scaled image: {scaling_path}")
        if number >= 4:
            # Adjust Illumination (Brightness)
            enhancer = ImageEnhance.Brightness(img)
            img_illumination = enhancer.enhance(1.5)
            illumination_path = f"{img_name}_Illumination{img_extension}"
            img_illumination.save(illumination_path)
            logger.info(f"Saved illumination-adjusted image: {illumination_path}")

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
