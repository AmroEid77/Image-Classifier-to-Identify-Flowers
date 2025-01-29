import argparse
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from utils import process_image, load_label_map


class HubLayerWrapper(tf.keras.layers.Layer):
    def __init__(self, hub_url, **kwargs):
        super().__init__(**kwargs)
        self.hub_layer = hub.KerasLayer(hub_url, trainable=False) 
    
    def call(self, inputs):
        return self.hub_layer(inputs)
    



def predict(image_path, model, top_k=5):
    image = process_image(image_path)
    
    ps = model.predict(image)[0]  

    
    top_k_probabilities = np.sort(ps)[-top_k:][::-1] 
    top_k_classes = np.argsort(ps)[-top_k:][::-1] 


    return top_k_probabilities, top_k_classes



def main():

    parser = argparse.ArgumentParser(description="Predict the flower class for an image.")
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    parser.add_argument("model_path", type=str, help="Path to the saved model file.")
    parser.add_argument("--top_k", type=int, default=5, help="Return the top K most likely classes.")
    parser.add_argument("--category_names", type=str, default=None, help="Path to JSON file mapping labels to flower names.")
    
    args = parser.parse_args()
    
    model = tf.keras.models.load_model(args.model_path, custom_objects={'HubLayerWrapper': HubLayerWrapper, 'KerasLayer': hub.KerasLayer})
    
    probs, classes = predict(args.image_path, model, top_k=args.top_k)
    
    if args.category_names:
        label_map = load_label_map(args.category_names)
        classes = [label_map[str(cls)] for cls in classes]  
    
    print(f"Top {args.top_k} predictions:")
    for i in range(args.top_k):
        print(f"{classes[i]}: {probs[i]:.4f}")
    
if __name__ == "__main__":
    main()
