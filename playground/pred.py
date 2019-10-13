import tensorflow as tf
from keras.models import load_model
import os
import numpy as np

# Mute excessively verbose Tensorflow output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

siamese_model_path = '/home/ubuntu/speaker-id/model/siamese__nseconds_3.0__filters_32__embed_64__drop_0.05__r_0.hdf5'
input_1_path = '/home/ubuntu/speaker-id/data/examples/input_1.npy'
input_2_path = '/home/ubuntu/speaker-id/data/examples/input_2.npy'

# Load model.
model = load_model(siamese_model_path)

# Load inputs
input_1 = np.load(input_1_path)
input_2 = np.load(input_2_path)

pred = model.predict([input_1, input_2])

print(pred)

n_correct = 0
if np.argmin(pred[:, 0]) == 0:
  # 0 is the correct result as by the function definition.
  n_correct += 1
print(n_correct)

