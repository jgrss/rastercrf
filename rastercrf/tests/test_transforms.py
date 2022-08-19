import unittest

from rastercrf.model import CRFClassifier
from rastercrf.transforms import time_to_feas, transform_probas
from rastercrf import util

import numpy as np


array1 = np.array([[2, 4, 8, 16],
                   [32, 64, 128, 256],
                   [512, 1024, 2048, 4096]], dtype='float64')

array2 = np.array([[1, 3, 9, 18],
                   [36, 72, 144, 288],
                   [576, 1152, 2304, 4608]], dtype='float64')

# len() = 2 = ntime
# array.shape = (3, 4) = [samples x features]
X = [array1, array2]

x_names = ['1', '2', '3', '4']
y_names = ['a', 'b', 'c']
nsamps = 3

crf = CRFClassifier()
crfm = crf._setup(X, x_names, y_names, False)

x_names = util.check_x_names(X, x_names, False)

X_t = time_to_feas(np.ascontiguousarray(X, dtype='float64'),
                   x_names,
                   band_diffs=False)

# len(probas) -> samples (rows x columns)
# len(probas[0]) -> time
# len(probas[0][0]) -> features
X_t_out = [[{b'z.001': 2.0, b'z.002': 4.0, b'z.003': 8.0, b'z.004': 16.0},
            {b'z.001': 1.0, b'z.002': 3.0, b'z.003': 9.0, b'z.004': 18.0}],
           [{b'z.001': 32.0, b'z.002': 64.0, b'z.003': 128.0, b'z.004': 256.0},
            {b'z.001': 36.0, b'z.002': 72.0, b'z.003': 144.0, b'z.004': 288.0}],
           [{b'z.001': 512.0, b'z.002': 1024.0, b'z.003': 2048.0, b'z.004': 4096.0},
            {b'z.001': 576.0, b'z.002': 1152.0, b'z.003': 2304.0, b'z.004': 4608.0}]]

# len(probas) -> samples (rows x columns)
# len(probas[0]) -> time
# len(probas[0][0]) -> classes
probas = util.dict_keys_to_bytes([[{'a': 0.1, 'b': 0.2, 'c': 0.7}, {'a': 0.2, 'b': 0.3, 'c': 0.5}],
                                  [{'a': 0.4, 'b': 0.1, 'c': 0.5}, {'a': 0.6, 'b': 0.3, 'c': 0.1}],
                                  [{'a': 0.5, 'b': 0.4, 'c': 0.1}, {'a': 0.3, 'b': 0.1, 'c': 0.6}]])

probas_array = np.empty((crfm.ntime, crfm.nclasses, 3, 1), dtype='float64')

for t in range(0, crfm.ntime):
    for s in range(0, len(probas)):
        probas_array[t, :, s, :] = np.array(list(probas[s][t].values()))[:, np.newaxis]

# -> [time x classes x rows x columns]
out_probas = transform_probas(probas,
                              np.array([[[1]]], dtype='float64'),
                              crfm.class_labels,
                              crfm.nclasses,
                              crfm.ntime,
                              crfm.nbands,
                              nsamps, # three sample rows
                              1, # one column
                              insert_nodata=False,
                              n_jobs=1)


class TestTransforms(unittest.TestCase):

    def test_transform_probas(self):
        self.assertTrue(np.allclose(probas_array, out_probas))

    def test_transform_probas_ntime(self):
        self.assertEqual(out_probas.shape[0], crfm.ntime)

    def test_transform_probas_nclasses(self):
        self.assertEqual(out_probas.shape[1], crfm.nclasses)

    def test_transform_probas_nrows(self):
        self.assertEqual(out_probas.shape[2], nsamps)

    def test_transform_probas_ncols(self):
        self.assertEqual(out_probas.shape[3], 1)

    def test_crf_ntime(self):
        self.assertEqual(crfm.ntime, 2)

    def test_crf_nfeas(self):
        self.assertEqual(crfm.nbands, 4)

    def test_crf_xvars(self):
        self.assertEqual(crfm.X_t, X_t_out)

    def test_time_to_feas_nsamps(self):
        self.assertEqual(len(X_t), nsamps)

    def test_time_to_feas_ntime(self):
        self.assertEqual(len(X_t[0]), 2)

    def test_time_to_feas_nfeas(self):
        self.assertEqual((len(X_t[0][0])), 4)

    def test_time_to_feas_keys1(self):
        self.assertEqual(list(X_t[0][0].keys()), list(X_t_out[0][0].keys()))

    def test_time_to_feas_keys2(self):
        self.assertEqual(list(X_t[1][0].keys()), list(X_t_out[1][0].keys()))

    def test_time_to_feas_keys3(self):
        self.assertEqual(list(X_t[2][0].keys()), list(X_t_out[2][0].keys()))

    def test_bytes_class1(self):
        self.assertTrue(isinstance(list(probas[0][0].keys())[0], bytes))

    def test_bytes_class2(self):
        self.assertTrue(isinstance(list(probas[0][0].keys())[1], bytes))

    def test_bytes_class3(self):
        self.assertTrue(isinstance(list(probas[0][0].keys())[2], bytes))


if __name__ == '__main__':
    unittest.main()
