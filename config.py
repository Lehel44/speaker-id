# Basic configuration file containing paths.

class Config(object):
  DEBUG = True
  DEVELOPMENT = True
  UPLOAD_FOLDER = '/home/ubuntu/speaker-id/data/wav'
  USER_DICT_PATH = '/home/ubuntu/speaker-id/data/wav/user_data.dict'
  PREPROCESSED_PATH = '/home/ubuntu/speaker-id/data/preprocessed_audio'
  IDENTIFY_PATH = '/home/ubuntu/speaker-id/identify'
  SIAMESE_MODEL_PATH = '/home/ubuntu/speaker-id/model/siamese__nseconds_3.0__filters_32__embed_64__drop_0.05__r_0.hdf5'
  CONV_ENCODER_PATH = '/home/ubuntu/speaker-id/model/conv_encoder.hdf5'
  ALLOWED_EXTENSIONS = set(['wav'])

class ProductionConfig(Config):
  DEBUG = False
  DEVELOPMENT = False
