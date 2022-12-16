import gradio as gr
from keras.models import load_model
from keras.layers import Layer, Softmax
import numpy as np

softmax = Softmax()


class Normalization(Layer):
    def __init__(self, name=None, **kwargs):
        super(Normalization, self).__init__()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def call(self, inputs):
        return (inputs - self.mean) / self.std


cnn_model = load_model("./files/spider.h5", custom_objects={'Normalization': Normalization})
class_names = ['Black Widow',
               'Blue Tarantula',
               'Bold Jumper',
               'Brown Grass Spider',
               'Brown Recluse Spider',
               'Deinopis Spider',
               'Golden Orb Weaver',
               'Hobo Spider',
               'Huntsman Spider',
               'Ladybird Mimic Spider',
               'Peacock Spider',
               'Red Knee Tarantula',
               'Spiny-backed Orb-weaver',
               'White Kneed Tarantula',
               'Yellow Garden Spider']
cnn_model.summary()


def predict_input_image(img):
    img_4d = img.reshape(1, 224, 224, 3)
    img_4d = img_4d / 255.0
    prediction = softmax(cnn_model.predict(img_4d)[0]).numpy()
    return {class_names[i]: float(prediction[i]) for i in range(len(class_names))}


image = gr.Image(shape=(224, 224))
label = gr.Label(num_top_classes=len(class_names))

gr.Interface(fn=predict_input_image, inputs=image, outputs=label, interpretation='default').launch(debug='True')
