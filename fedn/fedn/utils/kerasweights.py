import os
import tempfile
import numpy as np
import tensorflow.keras.models as krm
import collections
import tempfile

from .helpers import HelperBase

class KerasWeightsHelper(HelperBase):
    """ FEDn helper class for keras.Sequential. """

    def average_weights(self, weights):
        """ Average weights """

        avg_w = []
        for l in range(len(weights[0])):
            lay_l = np.array([w[l] for w in weights])
            weight_l_avg = np.mean(lay_l, 0)
            avg_w.append(weight_l_avg)

        return avg_w

    def increment_average(self, model, model_next, n):
        """ Update an incremental average. """
        w_prev = self.get_weights(model)
        w_next = self.get_weights(model_next)
        w = np.add(w_prev, (np.array(w_next) - np.array(w_prev)) / n)
        self.set_weights(model, w)

    def set_weights(self, model, weights):
        model.set_weights(weights)

    def get_weights(self, model):
        return model.get_weights()

    def get_tmp_path(self):
        fod, path = tempfile.mkstemp(suffix='.h5')
        return path

    def save_model(self, model, path='weights.h5'):

        # saving model weights
        model.save_weights(path)

        #import tarfile
        #tar = tarfile.open("model.tar.gz", "w:gz")
        #tar.add(path)
        #tar.close()

        return path

    def load_model(self, path='weights.h5'):
        from .
        #import tarfile

        #tar = tarfile.open(path, "r:gz")
        #for tarinfo in tar:
        #    tar.extract(tarinfo)
        #tar.close()
        model =
        model.load_weights(path)
        return model

    def serialize_model_to_BytesIO(self,model):
        outfile_name = self.save_model(model)

        from io import BytesIO
        a = BytesIO()
        a.seek(0, 0)
        with open(outfile_name, 'rb') as f:
            a.write(f.read())
        os.unlink(outfile_name)
        return a

    def load_model_from_BytesIO(self,model_bytesio):
        """ Load a model from a BytesIO object. """
        path = self.get_tmp_path()
        with open(path, 'wb') as fh:
            fh.write(model_bytesio)
            fh.flush()

        return self.load_model(path)


