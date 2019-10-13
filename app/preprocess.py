import soundfile as sf
import numpy as np


def preprocess_instances(downsampling, whitening=True):
    """This is the canonical preprocessing function for this project.
    1. Downsampling audio segments to desired sampling rate
    2. Whiten audio segments to 0 mean and fixed RMS (aka volume)
    """

    def preprocess_instances_(instances):
        instances = instances[:, ::downsampling, :]
        if whitening:
            instances = whiten(instances)
        return instances

    return preprocess_instances_


class BatchPreProcessor(object):
    """Wrapper class for instance and label pre-processing.
    This class implements a __call__ method that pre-process classifier-style batches (inputs, outputs) and siamese
    network-style batches ([input_1, input_2], outputs) identically.
    # Arguments
        mode: str. One of {siamese, classifier)
        instance_preprocessor: function. Pre-processing function to apply to input features of the batch.
        target_preprocessor: function. Pre-processing function to apply to output labels of the batch.
    """

    def __init__(self, mode, instance_preprocessor, target_preprocessor=lambda x: x):
        assert mode in ('siamese', 'classifier')
        self.mode = mode
        self.instance_preprocessor = instance_preprocessor
        self.target_preprocessor = target_preprocessor

    def __call__(self, batch):
        """Pre-processes a batch of samples."""
        if self.mode == 'siamese':
            ([input_1, input_2], labels) = batch

            input_1 = self.instance_preprocessor(input_1)
            input_2 = self.instance_preprocessor(input_2)

            labels = self.target_preprocessor(labels)

            return [input_1, input_2], labels
        elif self.mode == 'classifier':
            instances, labels = batch

            instances = self.instance_preprocessor(instances)

            labels = self.target_preprocessor(labels)

            return instances, labels
        else:
            raise ValueError


def whiten(batch, rms=0.038021):
    """This function whitens a batch so each sample has 0 mean and the same root mean square amplitude i.e. volume."""
    if len(batch.shape) != 3:
        raise (ValueError, 'Input must be a 3D array of shape (n_segments, n_timesteps, 1).')

    # Subtract mean
    sample_wise_mean = batch.mean(axis=1)
    whitened_batch = batch - np.tile(sample_wise_mean, (1, 1, batch.shape[1])).transpose((1, 2, 0))

    # Divide through
    sample_wise_rescaling = rms / np.sqrt(np.power(batch, 2).mean())
    whitened_batch = whitened_batch * np.tile(sample_wise_rescaling, (1, 1, batch.shape[1])).transpose((1, 2, 0))

    return whitened_batch
