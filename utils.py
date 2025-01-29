import numpy as np
import tensorflow as tf
from PIL import Image

def process_image(image_path, image_size=224):

    image = Image.open(image_path)
    image = np.asarray(image)
    
    image = tf.image.resize(image, (image_size, image_size))
    image = tf.cast(image, tf.float32) / 255.0

    image = np.expand_dims(image, axis=0)  
    return image

def load_label_map(label_map_path):
    import json
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    return label_map