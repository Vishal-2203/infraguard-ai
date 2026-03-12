from utils.image_processing import preprocess_image
import tensorflow as tf
from utils.image_processing import preprocess_image

model = tf.keras.models.load_model("model/crack_detector.h5")

def predict_crack(image_path):

    img = preprocess_image(image_path)

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        return "Crack Detected", float(prediction)
    else:
        return "No Crack", float(1 - prediction)
