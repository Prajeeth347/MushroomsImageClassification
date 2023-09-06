from distutils.log import debug
from fileinput import filename
from flask import *
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os 

model = tf.keras.models.load_model("../mushroom_image_classifier_model.h5")

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(128, 128))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0  # Normalize the image
    return img

class_names = ['Agaricus',
 'Amanita',
 'Boletus',
 'Cortinarius',
 'Entoloma',
 'Hygrocybe',
 'Lactarius',
 'Russula',
 'Suillus']
class_toxicity = [1,0,1,0,0,1,1,1,1]

def predict_mushroom_class_and_confidence_edible(image_path):
    img = preprocess_image(image_path)
    predictions = model.predict(img)
    class_index = np.argmax(predictions, axis=1)
    predicted_class_name = class_names[class_index[0]]
    confidence = predictions[0][class_index[0]]
    edible_binary = class_toxicity[class_index[0]]
    if edible_binary == 0:
      edible= 'Not Edible'
    else:
      edible = "Edible"
    return predicted_class_name, confidence,edible

app = Flask(__name__)

@app.route('/')
def main():
	return render_template("index.html")

@app.route('/success', methods = ['POST'])
def success():
	if request.method == 'POST':
		f = request.files['file']
		f.save(f.filename)
		predicted_class_name, confidence,edible = predict_mushroom_class_and_confidence_edible(f.filename)
		os.remove(f.filename)
		return render_template("Acknowledgement.html", name = predicted_class_name,edibility=edible,confidance=confidence)

if __name__ == '__main__':
	app.run(debug=True)
