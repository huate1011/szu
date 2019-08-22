import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import numpy as np

# model = tf.keras.models.load_model('model_512.h5')  # load weights
model = keras.models.load_model('model_auto_gen_20_epochs.h5')  # load weights



# ------------------------------
# function for drawing bar chart for emotion preditions
def emotion_analysis(emotions):
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(objects))

    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')

    plt.show()



# ------------------------------
# make prediction for custom image out of test set

img = image.load_img("../dataset/brando.png", grayscale=True, target_size=(48, 48))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

x /= 255

custom = model.predict(x)
emotion_analysis(custom[0])