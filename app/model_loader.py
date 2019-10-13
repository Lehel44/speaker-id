import logging
import tensorflow as tf
from keras.models import load_model
import paths



class ModelLoader:

    def __init__(self):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.compat.v1.Session(config=config)
        self.graph = self.session.graph
        self.siamese = self.load(paths.SIAMESE_MODEL_PATH)
        self.conv_encoder = self.load(paths.CONV_ENCODER_PATH)

    def load(self, file_name=None):
        with self.graph.as_default():
            with self.session.as_default():
                try:
                    return load_model(file_name, compile=False)
                except Exception as e:
                    logging.exception(e)
                    return False

    def predict(self, model_input, model='siamese'):
        with self.graph.as_default():
            with self.session.as_default():
                if model == 'siamese':
                    return self.siamese.predict(model_input)
                elif model == 'conv_encoder':
                    return self.conv_encoder.predict(model_input)
                else:
                    logging.info('Unrecognized model.')
                    raise ValueError('Unrecognized model.')
