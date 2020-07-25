import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

import argparse
import numpy as np
import json

def process_image(numpy_image):
    image_size = 224
    image_shape = (image_size, image_size, 3)
    image = tf.image.convert_image_dtype(numpy_image, dtype=tf.int16, saturate=False)
    image = tf.image.resize(numpy_image, (image_size, image_size)).numpy()
    image /= 255
    return image

def load_model(model_path):
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer})
    print(model.summary())
    return model

#Label Mapping
def Map_classes():
    with open('label_map.json', 'r') as f:
        class_names = json.load(f)
    
    class_names1 = dict()
    for key in class_names:
        class_names1[str(int(key)-1)] = class_names[key]
    return class_names1

def predict(image_path, model_path, top_k):
    top_k = int(top_k)
  
    
    #Loading the model
    model = load_model(model_path)
    
    #Processing the image
    image = Image.open(image_path)
    test_image = np.asarray(image)
    processed_test_image = process_image(test_image)
    
    #Prediction probabilities
    prob_preds = model.predict(np.expand_dims(processed_test_image,axis=0))
    prob_preds = prob_preds[0].tolist()
    
    
    #top 1 prediction
    class_names1 = Map_classes()
    top_pred_class_id = model.predict_classes(np.expand_dims(processed_test_image,axis=0))
    top_pred_class_prob = prob_preds[top_pred_class_id[0]]
    pred_class = class_names1[str(top_pred_class_id[0])]
    print("\n\nMost likely class for this image is " ,top_pred_class_id," with the name: ", pred_class, "; class probability : ",top_pred_class_prob)
    
    #Highest k propabilities
    values, indices= tf.math.top_k(prob_preds, k=top_k)
    probs=values.numpy().tolist()
    classes=indices.numpy().tolist()
    
    class_names = [class_names1[str(i)] for i in classes]
    print('\nHighest ', top_k,' prediction propabilities:\n',probs,'\n')
    print('\nHighest ', top_k, 'classes:',classes,'\n')
    print('\nHighest ', top_k,' class_names:', class_names,'\n')
    
    
def main():
    
    image_path = input("Enter the image path:")
    saved_model = input("Enter the model path:")
    top_k = input("Enter the Number of propabilities:")
    
     # if __name__ == "__main__":
        #parser = argparse.ArgumentParser(description = "Description for my parser")
        #parser.add_argument("image_path",help="Image Path", default="")
        #parser.add_argument("saved_model",help="Model Path", default="")
        #parser.add_argument("--top_k", help="Fetch top k predictions", required = False, default = 3)
       # parser.add_argument("--category_names", help="Class map json file", required = False, default = "label_map.json")
        #args = parser.parse_args()
        
    predict(image_path,saved_model,top_k)
main()
    
    
    


  
    
    