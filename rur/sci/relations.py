import numpy as np

@staticmethod
class mgal_mbh:
    params_dict = {
        'RV15a': [1.05, 1E11, 7.45], # Reines & Volonteri 2015 (AGNs)
        'RV15b': [1.40, 1E11, 8.95], # Reines & Volonteri 2015 (E/CBs)
        'BM19':  [1.64, 1E11, 7.88], # Baron & Ménard 2019 (All?)
    }
    def evaluate(self, tag):
        params = self.params_dict[tag]
        return lambda mgal: 10**(params[0]*np.log10(params[1])+params[2])

    __call__ = evaluate