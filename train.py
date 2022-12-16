import tensorflow as tf
from keras.layers import Conv2D, Dense, MaxPool2D, Input, Flatten, BatchNormalization, Dropout, Layer
from keras.models import Model, Sequential
from keras.utils import plot_model
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    horizontal_flip=True)

valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255)

train_dir = "./dataset/train"
val_dir = "./dataset/valid"
batch_size = 32

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

val_data = valid_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42
)
print(len(train_data))

lr = 0.001
epochs = 100

cnn_model = Sequential()


class Normalization(Layer):
    def __init__(self):
        super(Normalization, self).__init__()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def call(self, inputs):
        return (inputs - self.mean) / self.std


pretrained_model = tf.keras.applications.ResNet50(include_top=False,
                                                  input_shape=(224, 224, 3),
                                                  pooling='max', classes=15,
                                                  weights="imagenet")
# for layer in pretrained_model.layers:
#     layer.trainable = False
cnn_model.add(Input((224, 224, 3)))
cnn_model.add(Normalization())
cnn_model.add(pretrained_model)
cnn_model.add(Dense(15))
plot_model(cnn_model, "model1.png", show_shapes=True)
cnn_model.summary()
metrics = ["acc"]
cnn_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), optimizer=SGD(learning_rate=lr, momentum=0.9),
                  metrics=metrics)
callbacks = [
    ModelCheckpoint("files/model_new.h5"),
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=False),
]

cnn_model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    steps_per_epoch=len(train_data),
    validation_steps=len(val_data),
    callbacks=callbacks,
)
