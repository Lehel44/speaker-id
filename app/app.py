# -*- coding: utf-8 -*-

import os
import pickle
import uuid
from pprint import pformat

import numpy as np
import soundfile as sf
import tensorflow as tf
from flask import Flask, request, Response

import paths
from model_loader import ModelLoader
from preprocess import BatchPreProcessor, preprocess_instances

# Flask app.
app = Flask(__name__)

# Load configuration.
app.config.from_object('config.Config')

# Mute excessively verbose Tensorflow output.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Load models.
model_loader = ModelLoader()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def check_audio_file(file_id):
    if file_id not in request.files:
        return False
    file = request.files[file_id]
    if file and allowed_file(file.filename):
        return True
    return False


def save_audio_file(file_id, path):
    username = request.form.get('username')
    user_id = str(uuid.uuid4())
    file = request.files[file_id]
    file.save(os.path.join(path, user_id + '.wav'))
    return user_id, username, file


@app.route('/enroll', methods=['POST'])
def enroll():
    if check_audio_file('audio_file'):
        #user_id, username, file = save_audio_file('audio_file', paths.UPLOAD_FOLDER)
        #print(file)
        audio_file = request.files['audio_file']

        # Save user data.
        user_dict = dict()
        user_dict_path = paths.USER_DICT_PATH
        try:
            user_dict = pickle.load(open(user_dict_path, 'rb'))
            user_dict[user_id] = username
            pickle.dump(user_dict, open(user_dict_path, 'wb'))
        except (OSError, IOError) as e:
            user_dict[user_id] = username
            pickle.dump(user_dict, open(user_dict_path, 'wb'))

        audio_path = os.path.join(paths.UPLOAD_FOLDER, user_id + '.wav')
        preprocess(user_id, audio_path, 3, 2)
    else:
        return Response(str(400), mimetype='text/plain')

    return Response(str(200), mimetype='text/plain')


@app.route('/identify', methods=['POST'])
def identify():
    if check_audio_file('audio_file'):
        user_id, _, _ = save_audio_file('audio_file', paths.IDENTIFY_PATH)
        audio_path = os.path.join(paths.IDENTIFY_PATH, user_id + '.wav')
        preprocess(user_id, audio_path, 3, 2)
        response = predict(user_id)
        app.logger.info(pformat(response))
        user_dict = pickle.load(open(paths.USER_DICT_PATH, 'rb'))
        app.logger.info(pformat(user_dict))
        return Response(str(200), mimetype='text/plain')
    return Response(str(400), mimetype='text/plain')


'''Preprocesses raw audio files in the enrolling phase. Preprocessing
   includes standardization and downsampling of the current audio sample.'''


def preprocess(user_id, audio_file, sample_length, downsampling):
    instance, sample_rate = sf.read(audio_file)
    # Cut 3 second
    middle = int(len(instance) / 2)
    dist = sample_length * sample_rate
    instance = instance[middle - dist / 2:middle + dist / 2]
    # Expand to 3 dimension.
    input = np.stack([instance])[:, :, np.newaxis]
    batch_preprocessor = BatchPreProcessor('classifier', preprocess_instances(downsampling))
    (input, _) = batch_preprocessor((input, []))
    # Save preprocessed file.
    np.save(os.path.join(paths.PREPROCESSED_PATH, user_id), input)


def predict(current_user_id):
    # Get preprocessed audio samples.
    all_audio_data = []
    all_audio_names = []
    current_audio_data = []
    user_dict = pickle.load(open(paths.USER_DICT_PATH))
    for user_id in user_dict.keys():
        path = os.path.join(paths.PREPROCESSED_PATH, user_id + '.npy')
        file = np.load(path)
        all_audio_data.append(file)
        all_audio_names.append(user_dict[user_id])

    current_audio_data = np.load(os.path.join(paths.PREPROCESSED_PATH, current_user_id + '.npy'))

    num_of_speakers = len(all_audio_data)

    input_1 = np.stack([current_audio_data[0]] * num_of_speakers)
    input_2 = np.concatenate(all_audio_data)

    result = model_loader.predict([input_1, input_2])

    print(pformat(result))
    print(pformat(all_audio_names))
    response = np.concatenate(result).tolist()
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
