import os
import numpy as np
import cv2


def load_data(root_dir, contents=['jpg', 'png'], image_size=544):
    data = []

    for content in contents:
        # construct path to each image
        directory = os.path.join(root_dir, content)
        fps = [os.path.join(directory, filename) for filename in os.listdir(directory)]
        print(fps)
        # read images
        images = [cv2.imread(filepath) for filepath in fps]

        # if images have different sizes you have to resize them before:

        resized_images = [cv2.resize(image, dsize=(image_size, image_size)) for image in images]

        normalized_images = [image / 255 for image in resized_images]

        # stack to one np.array
        np_images = np.stack(normalized_images, axis=0)

        data.append(np_images)

    return data