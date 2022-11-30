from keras.models import load_model
from read_dataset import test_X_path, test_Y, tf_dataset

batch_size = 8

test_steps = (len(test_X_path) // batch_size)
if len(test_X_path) % batch_size != 0:
    test_steps += 1

test_ds = tf_dataset(test_X_path, test_Y, batch_size)

cnn_model = load_model("files/model_new.h5")

cnn_model.evaluate(test_ds, steps=test_steps)
