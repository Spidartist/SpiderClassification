import glob
import random
import tensorflow as tf
import cv2
import os
from keras.layers import Conv2D, Dense, MaxPool2D, Input, Flatten, BatchNormalization, Dropout
from keras.models import Model
from keras.utils import plot_model
from keras.metrics import Recall, Precision
from keras.callbacks import EarlyStopping, ModelCheckpoint

labels = sorted(os.listdir("./dataset/train"))

char_to_int = dict((c, i) for i, c in enumerate(labels))
onehot_encoded = dict()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def get_path_X(type_data):
    data_path = []
    for label in labels:
        data_path.extend(glob.glob("./dataset/{}/{}/*".format(type_data, label)))
    random.shuffle(data_path)
    return data_path


def get_Y(X_data):
    Y_data = []
    for datum in X_data:
        Y_data.append(char_to_int[datum.split("/")[-2].strip()])
    return Y_data


def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (224, 224))
    x = x / 255.0
    # (224, 224, 3)
    return x


def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        temp = [0] * len(labels)
        temp[y] = 1
        y = temp
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.int64])
    x.set_shape([224, 224, 3])
    y.set_shape([15])
    return x, y


def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    return dataset


train_X_path = get_path_X("train")
test_X_path = get_path_X("test")
val_X_path = get_path_X("valid")

train_Y = get_Y(train_X_path)
test_Y = get_Y(test_X_path)
val_Y = get_Y(val_X_path)

train_ds = tf_dataset(train_X_path, train_Y)
valid_ds = tf_dataset(val_X_path, val_Y)

for x, y in train_ds:
    print(x.shape)
    print(y.shape)
    break


def build_model():
    size = 224
    inputs = Input((size, size, 3))
    x = inputs
    x = Conv2D(64, (3, 3), activation="relu")(x)
    x = Conv2D(128, (3, 3), activation="relu")(x)
    x = MaxPool2D((2, 2), strides=2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(len(labels), activation="softmax")(x)
    return Model(inputs, x)


if __name__ == "__main__":
    batch = 8
    lr = 1e-4
    epochs = 50

    cnn_model = build_model()
    plot_model(cnn_model, "model.png", show_shapes=True)
    cnn_model.summary()
    metrics = ["acc", Recall(), Precision()]
    cnn_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=metrics)
    callbacks = [
        ModelCheckpoint("files/model_new.h5"),
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=False)
    ]
    train_steps = len(train_X_path) // batch
    valid_steps = len(val_X_path) // batch
    if len(train_X_path) % batch != 0:
        train_steps += 1
    if len(val_X_path) % batch != 0:
        valid_steps += 1

    cnn_model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        callbacks=callbacks,
        shuffle=False
    )
