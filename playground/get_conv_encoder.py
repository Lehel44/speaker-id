from keras.models import load_model, save_model

SIAMESE_MODEL_PATH = '/home/lehel/model/siamese__nseconds_3.0__filters_32__embed_64__drop_0.05__r_0.hdf5'

siamese = load_model(SIAMESE_MODEL_PATH)

conv_encoder = siamese.layers[2]

conv_encoder.save('/home/lehel/model/conv_encoder.hdf5')
