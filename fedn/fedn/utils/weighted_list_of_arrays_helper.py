import os
import tempfile
import numpy as np
import collections
import tempfile

from .helpers import HelperBase


class WeightedListOfArraysHelper(HelperBase):
    """ FEDn helper class for keras.Sequential. """

    def average_weights(self, weights):
        """ Average weights of Keras Sequential models. """
        raise NotImplemented

    def increment_average(self, weights: [np.ndarray], weights_next: [np.ndarray], _: int) -> [np.ndarray]:
        """ Update an incremental average. """
        n1, n2 = weights[0], weights_next[0]
        coeff1, coeff2 = n1.sum(), n2.sum()
        new_n = np.array([np.max([n1[0], n2[0]]),
                          np.sum([n1[1], n2[1]])])
        new_weights = [new_n]
        for w1, w2 in zip(weights[1:], weights_next[1:]):
            new_weights.append((coeff1 * w1 + coeff2 * w2) / (coeff1 + coeff2))
        return new_weights

    def set_weights(self, weights_, weights):
        """

        :param weights_:
        :param weights:
        """
        weights_ = weights

    def get_weights(self, weights):
        """

        :param weights:
        :return:
        """
        return weights

    def get_tmp_path(self):
        """ Return a temporary output path compatible with save_model, load_model. """
        fd, path = tempfile.mkstemp(suffix='.npz')
        os.close(fd)
        return path

    def save_model(self, weights, path=None):
        """

        :param weights:
        :param path:
        :return:
        """
        if not path:
            path = self.get_tmp_path()

        weights_dict = {}
        for i, w in enumerate(weights):
            weights_dict[str(i)] = w

        np.savez_compressed(path, **weights_dict)

        return path

    def load_model(self, path="weights.npz") -> [np.ndarray]:
        """

        :param path:
        :return:
        """
        a = np.load(path)
        weights = []
        for i in range(len(a.files)):
            weights.append(a[str(i)])
        return weights

    def load_model_from_BytesIO(self, model_bytesio) -> [np.ndarray]:
        """ Load a model from a BytesIO object. """
        path = self.get_tmp_path()
        with open(path, 'wb') as fh:
            fh.write(model_bytesio)
            fh.flush()
        model = self.load_model(path)
        os.unlink(path)
        return model

    def serialize_model_to_BytesIO(self, model):
        """

        :param model:
        :return:
        """
        outfile_name = self.save_model(model)

        from io import BytesIO
        a = BytesIO()
        a.seek(0, 0)
        with open(outfile_name, 'rb') as f:
            a.write(f.read())
        os.unlink(outfile_name)
        return a
