import numpy as np
import sys
import inspect

class compNorm(object):

    @staticmethod
    # ========================================================================================================================================
    def init(normalizer):
    # ========================================================================================================================================
        for name, obj in inspect.getmembers(sys.modules[__name__]):
            if inspect.isclass(obj):
                if hasattr(obj, 'name') and normalizer == obj.name:
                    return obj()
        else:
            raise NotImplementedError("Unknown normalization type {:s} (not implemented)".format(normalizer))


# ========================================================================================================================================
class Normalizator(object):
# ========================================================================================================================================

    def normalize(self, data):
        raise NotImplementedError()

    def normalize_by(self, raw_data, data):
        raise NotImplementedError()

    def denormalize_by(self, raw_data, data):
        raise NotImplementedError()


# ========================================================================================================================================
class VarianceNormalizator(Normalizator):
# ========================================================================================================================================

    name = 'var'

    def _mean_and_standard_dev(self, data):
        return np.mean(data, axis=0), np.std(data, axis=0)

    def normalize(self, data):
        me, st = self._mean_and_standard_dev(data)
        st[st == 0] = 1
        return (data-me)/st

    def normalize_by(self, raw_data, data):
        me, st = self._mean_and_standard_dev(raw_data)
        st[st == 0] = 1
        return (data-me)/st


# ========================================================================================================================================
class SimpleNormalizator(Normalizator):
# ========================================================================================================================================

    name = 'minmax'

    def _maxmin(self, data):
        return np.amax(data,axis=1).reshape(data.shape[0],1),np.amin(data,axis=1).reshape(data.shape[0],1)

    def normalize(self, data):
        mx, mn = self._maxmin(data)
        if (mx-mn).any() == 0:
            raise Exception('max = min')
        return np.divide((data-mn),(mx-mn))
