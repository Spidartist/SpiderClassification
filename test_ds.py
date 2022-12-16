import tensorflow as tf
from keras.models import load_model
from keras.layers import Layer, Softmax
import numpy as np

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

test_dir = "./dataset/test"
batch_size = 1

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

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


class Normalization(Layer):
    def __init__(self, name=None, **kwargs):
        super(Normalization, self).__init__()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def call(self, inputs):
        return (inputs - self.mean) / self.std


cnn_model = load_model("./spider.h5", custom_objects={'Normalization': Normalization})
image, label = next(iter(test_data))
cnt = 0
max_num = 0
for (image, label) in test_data:
    max_num += 1
    y_hat = np.argmax(cnn_model.predict(image)[0])
    y_true = np.argmax(label)
    if y_true == y_hat:
        cnt += 1
    print("{:30s}{:30s}{}".format(class_names[y_hat], class_names[y_true], y_true == y_hat))
    if max_num == 200:
        break

print("Accuracy: ", cnt/max_num)
