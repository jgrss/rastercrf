# distutils: language = c++
# cython: language_level=3
# cython: cdivision=True
# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False

import cython
cimport cython

# from cython.parallel import prange
# from cython.parallel import parallel

import numpy as np
cimport numpy as np

from libcpp.map cimport map as cpp_map
# from libcpp.vector cimport vector as cpp_vector
from libcpp.string cimport string as cpp_string

ctypedef char* char_ptr


cdef extern from "<vector>" namespace "std":
    cdef cppclass vector[T]:
        void push_back(T&) nogil
        size_t size() nogil
        T& operator[](size_t) nogil
        void clear() nogil


cdef extern from 'numpy/npy_math.h':
    bint npy_isnan(double value) nogil


cdef extern from 'numpy/npy_math.h':
    bint npy_isinf(double value) nogil


cdef extern from 'stdlib.h' nogil:
    double exp(double val)


cdef extern from 'stdlib.h' nogil:
    double fabs(double value)


cdef inline double _nan_check(double value) nogil:
    return 0.0 if (npy_isnan(value) or npy_isinf(value)) else value


cdef inline double _clip_low(double value) nogil:
    return 0.0 if value < 0 else value


cdef inline double _clip_high(double value) nogil:
    return 1.0 if value > 1 else value


cdef inline double _scale_min_max(double xv, double mno, double mxo, double mni, double mxi) nogil:
    return (((mxo - mno) * (xv - mni)) / (mxi - mni)) + mno


cdef inline double _clip(double value) nogil:
    return _clip_low(value) if value < 1 else _clip_high(value)


cdef inline double _band_diff(double value1, double value2) nogil:
    return _clip(fabs(value2 - value1))
    # return _clip(_scale_min_max(value2 - value1, 0.0, 1.0, -1.0, 1.0))


cdef inline double _avi(double red, double nir) nogil:
    """Advanced Vegetation Index"""
    return (nir * (1.0 - red) * (nir - red))**0.3334


cdef inline double _evi(double blue, double red, double nir) nogil:
    """Enhanced Vegetation Index"""
    return 2.5 * ((nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0))


cdef inline double _evi2(double red, double nir) nogil:
    """Two-band Enhanced Vegetation Index"""
    return 2.5 * ((nir - red) / (nir + 1.0 + (2.4 * red)))


# cdef inline double _bsi(double blue, double red, double nir, double swir2):
#     """Bare Soil Index"""
#     return ((swir2 + red) - (nir - blue)) / ((swir2 + red) + (nir - blue))


cdef inline double _brightness(double green, double red, double nir) nogil:
    """Brightness Index"""
    return (green**2 + red**2 + nir**2)**0.5


cdef inline double _brightness_swir(double green, double red, double nir, double swir1) nogil:
    """Brightness Index"""
    return (green**2 + red**2 + nir**2 + swir1**2)**0.5


# cdef inline double _dbsi(double green, double red, double nir, double swir1):
#     """Dry Bare Soil Index"""
#     return ((swir1 - green) / (swir1 + green)) - _ndvi(red, nir)

cdef inline double _bsi(double blue, double red, double nir, double swir1) nogil:
    """Bare Soil Index"""
    return ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue))


cdef inline double _gndvi(double green, double nir) nogil:
    """Green Normalized Difference Vegetation Index"""
    return (nir - green) / (nir + green)


cdef inline double _nbr(double nir, double swir2) nogil:
    """Normalized Burn Ratio"""
    return (nir - swir2) / (nir + swir2)


cdef inline double _ndmi(double nir, double swir1) nogil:
    """Normalized Difference Moisture Index"""
    return (nir - swir1) / (nir + swir1)


cdef inline double _ndsi(double green, double swir1) nogil:
    """Normalized Difference Snow Index"""
    return (green - swir1) / (green + swir1)


cdef inline double _ndwi(double green, double nir) nogil:
    """Normalized Difference Water Index"""
    return (green - nir) / (green + nir)


cdef inline double _ndvi(double red, double nir) nogil:
    """Normalized Difference Vegetation Index"""
    return (nir - red) / (nir + red)


cdef inline double _npcri(double blue, double red) nogil:
    """Normalized Pigment Chlorophyll Ratio Index"""
    return (red - blue) / (red + blue)


cdef inline double _si(double blue, double green, double red) nogil:
    """Shadow Index"""
    return ((1.0 - blue) * (1.0 - green) * (1.0 - red))**0.3334


cdef inline double _wi(double red, double swir1) nogil:
    """Woody Index"""
    return 0.0 if red + swir1 > 0.5 else 1.0 - ((red + swir1) / 0.5)


cdef cpp_map[cpp_string, double] _sample_to_dict_pan(double[::1] tsamp,
                                                     vector[cpp_string] string_ints,
                                                     double scale_factor) nogil:

    """
    Converts names and a 1d array to a dictionary for a panchromatic sensor
    """

    cdef:
        Py_ssize_t t
        unsigned int tsamp_len = tsamp.shape[0]
        cpp_map[cpp_string, double] features_map

    for t in range(0, tsamp_len):
        features_map[string_ints[t]] = tsamp[t] * scale_factor

    return features_map


cdef cpp_map[cpp_string, double] _sample_to_dict_bgrn(double[::1] tsamp,
                                                      vector[cpp_string] string_ints,
                                                      double brightness,
                                                      double evi,
                                                      double evi2,
                                                      double gndvi,
                                                      double ndvi,
                                                      cpp_string brightness_string,
                                                      cpp_string evi_string,
                                                      cpp_string evi2_string,
                                                      cpp_string gndvi_string,
                                                      cpp_string ndvi_string,
                                                      double scale_factor,
                                                      bint add_indices,
                                                      bint transform) nogil:

    """
    Converts names and a 1d array to a dictionary for a blue, green, red, NIR sensor
    """

    cdef:
        Py_ssize_t t
        unsigned int tsamp_len = tsamp.shape[0]
        double tsamp_scaled
        cpp_map[cpp_string, double] features_map

    for t in range(0, tsamp_len):

        tsamp_scaled = tsamp[t] * scale_factor

        if transform:
            features_map[string_ints[t]] = _scale_min_max(exp(_clip(tsamp_scaled)), 0, 5, exp(0.0), exp(1.0))
        else:
            features_map[string_ints[t]] = tsamp_scaled

    if add_indices:

        features_map[brightness_string] = _nan_check(brightness)
        features_map[evi_string] = _clip(_nan_check(evi))
        features_map[evi2_string] = _clip(_nan_check(evi2))
        features_map[gndvi_string] = _nan_check(gndvi)
        features_map[ndvi_string] = _nan_check(ndvi)

    return features_map


cdef cpp_map[cpp_string, double] _sample_to_dict_s220(double[::1] tsamp,
                                                      vector[cpp_string] string_ints,
                                                      double brightness,
                                                      double nbr,
                                                      double ndmi,
                                                      double ndvi,
                                                      double wi,
                                                      cpp_string brightness_string,
                                                      cpp_string nbr_string,
                                                      cpp_string ndmi_string,
                                                      cpp_string ndvi_string,
                                                      cpp_string wi_string,
                                                      double scale_factor,
                                                      bint add_indices,
                                                      bint transform) nogil:

    """
    Converts names and a 1d array to a dictionary for a Sentinel 2 20m sensor
    """

    cdef:
        Py_ssize_t t
        unsigned int tsamp_len = tsamp.shape[0]
        cpp_map[cpp_string, double] features_map
        double tsamp_scaled

    for t in range(0, tsamp_len):

        tsamp_scaled = tsamp[t] * scale_factor

        if transform:
            features_map[string_ints[t]] = _scale_min_max(exp(_clip(tsamp_scaled)), 0, 5, exp(0.0), exp(1.0))
        else:
            features_map[string_ints[t]] = tsamp_scaled

    if add_indices:

        features_map[brightness_string] = _scale_min_max(_clip(_nan_check(brightness)), 0.01, 1.0, 0.0, 1.0)
        features_map[nbr_string] = _clip(_scale_min_max(_nan_check(nbr), 0.01, 1.0, -1.0, 1.0))
        features_map[ndmi_string] = _clip(_scale_min_max(_nan_check(ndmi), 0.01, 1.0, -1.0, 1.0))
        features_map[ndvi_string] = _clip(_scale_min_max(_nan_check(ndvi), 0.01, 1.0, -1.0, 1.0))
        features_map[wi_string] = _scale_min_max(_clip(_nan_check(wi)), 0.01, 1.0, 0.0, 1.0)

    return features_map


cdef cpp_map[cpp_string, double] _sample_to_dict(double[::1] tsamp,
                                                 vector[cpp_string] labels_vect_,
                                                 double scale_factor,
                                                 bint transform,
                                                 bint skip_end=False) nogil:

    """
    Converts names and a 1d array to a dictionary for a generic sensor
    """

    cdef:
        Py_ssize_t t
        unsigned int tsamp_len
        double tsamp_scaled
        cpp_map[cpp_string, double] features_map

    if skip_end:
        tsamp_len = tsamp.shape[0] - 1
    else:
        tsamp_len = tsamp.shape[0]

    for t in range(0, tsamp_len):

        tsamp_scaled = tsamp[t] * scale_factor

        if transform:
            features_map[labels_vect_[t]] = _scale_min_max(exp(_clip(tsamp_scaled)), 0, 5, exp(0.0), exp(1.0))
        else:
            features_map[labels_vect_[t]] = tsamp_scaled

    return features_map


cdef cpp_map[cpp_string, double] _array_to_dict(vector[char*] labels_vct,
                                                double[:, :, ::1] tsamp_array_,
                                                Py_ssize_t tidx,
                                                Py_ssize_t sidx,
                                                unsigned int nvars) nogil:

    cdef:
        Py_ssize_t vidx
        cpp_map[cpp_string, double] features_map_load

    for vidx in range(0, nvars):
        features_map_load[labels_vct[vidx]] = tsamp_array_[tidx, sidx, vidx]

    return features_map_load


cdef vector[cpp_map[cpp_string, double]] _samples_to_dict_3d_diffs(vector[char*] labels,
                                                                   double[:, :, ::1] tsamp_array,
                                                                   unsigned int ntime,
                                                                   Py_ssize_t sidx,
                                                                   unsigned int nvars,
                                                                   double[:, :, ::1] data,
                                                                   double[::1] tsample) nogil:

    cdef:
        Py_ssize_t tidx, m, n
        cpp_map[cpp_string, double] features_map
        vector[cpp_map[cpp_string, double]] samples_load

    for tidx in range(0, ntime):

        if tidx == 0:

            for m in range(0, nvars):
                tsample[m] = 0.0

            for n in range(0, nvars):
                tsample[m + 1] = _band_diff(data[tidx + 1, sidx, n], data[tidx, sidx, n])
                m += 1

        elif tidx + 1 == ntime:

            for m in range(0, nvars):
                tsample[m] = _band_diff(data[tidx - 1, sidx, m], data[tidx, sidx, m])

            for n in range(0, nvars):
                tsample[m + 1] = 0.0
                m += 1

        else:

            for m in range(0, nvars):
                tsample[m] = _band_diff(data[tidx - 1, sidx, m], data[tidx, sidx, m])

            for n in range(0, nvars):
                tsample[m + 1] = _band_diff(data[tidx + 1, sidx, n], data[tidx, sidx, n])
                m += 1

        for n in range(0, nvars):
            tsample[m + 1] = data[tidx, sidx, n]
            m += 1

        features_map = _array_to_dict(labels,
                                      tsamp_array,
                                      tidx,
                                      sidx,
                                      nvars)

        samples_load.push_back(features_map)

    return samples_load


cdef vector[cpp_map[cpp_string, double]] _samples_to_dict_3d(vector[char*] labels,
                                                             double[:, :, ::1] tsamp_array,
                                                             unsigned int ntime,
                                                             Py_ssize_t sidx,
                                                             unsigned int nvars) nogil:

    """
    Converts names and a 1d array to a dictionary
    """

    cdef:
        Py_ssize_t tidx
        cpp_map[cpp_string, double] features_map
        vector[cpp_map[cpp_string, double]] samples_load

    for tidx in range(0, ntime):

        features_map = _array_to_dict(labels,
                                      tsamp_array,
                                      tidx,
                                      sidx,
                                      nvars)

        samples_load.push_back(features_map)

    return samples_load


cdef cpp_map[cpp_string, double] _samples_to_dict(vector[char*] labels,
                                                  double[::1] tsamp,
                                                  unsigned int nvars) nogil:

    """
    Converts names and a 1d array to a dictionary
    """

    cdef:
        Py_ssize_t v
        cpp_map[cpp_string, double] features_map

    for v in range(0, nvars):
        features_map[labels[v]] = tsamp[v]

    return features_map


cdef vector[double] _push_nodata(vector[double] vct,
                                 vector[cpp_string] labels_bytes,
                                 unsigned int n_classes) nogil:

    """
    Appends class predictions

    Args:
        vct (list): An empty list.
        labels_bytes (list): A list of class labels.
        n_classes (int): The number of potential classes.

    Returns:
        ``list`` of class probabilities
    """

    cdef:
        Py_ssize_t m
        double ps_label_value
        cpp_string m_label

    for m in range(0, n_classes):

        m_label = labels_bytes[m]

        if m_label == b'n':
            ps_label_value = 1.0
        else:
            ps_label_value = 0.0

        vct.push_back(ps_label_value)

    return vct


cdef void _push_nodata_array(vector[cpp_string] labels_bytes,
                             unsigned int n_classes,
                             Py_ssize_t jj,
                             Py_ssize_t ii,
                             double[:, :, ::1] out_array_) nogil:

    """
    Appends class predictions

    Args:
        labels_bytes (list): A list of class labels.
        n_classes (int): The number of potential classes.

    Returns:
        ``list`` of class probabilities
    """

    cdef:
        Py_ssize_t m
        double ps_label_value
        cpp_string m_label

    for m in range(0, n_classes):

        m_label = labels_bytes[m]

        if m_label == b'n':
            ps_label_value = 1.0
        else:
            ps_label_value = 0.0

        out_array_[ii, jj, m] = ps_label_value


cdef void _push_classes_array(cpp_map[cpp_string, double] ps,
                              vector[cpp_string] labels_bytes,
                              unsigned int n_classes,
                              Py_ssize_t jj,
                              Py_ssize_t ii,
                              double[:, :, ::1] out_array_) nogil:

    """
    Appends class predictions

    Args:
        ps (dict): A dictionary of class probabilities.
        labels_bytes (list): A list of class labels.
        n_classes (int): The number of potential classes.

    Returns:
        ``list`` of class probabilities
    """

    cdef:
        Py_ssize_t m
        double ps_label_value
        cpp_string m_label

    for m in range(0, n_classes):

        m_label = labels_bytes[m]

        if ps.count(m_label) > 0:
            ps_label_value = ps[m_label]
        else:
            ps_label_value = 0.0

        out_array_[ii, jj, m] = ps_label_value


# cdef vector[double] _push_classes(vector[double] vct,
#                                   cpp_map[cpp_string, double] ps,
#                                   vector[cpp_string] labels_bytes,
#                                   unsigned int n_classes) nogil:
#
#     """
#     Appends class predictions
#
#     Args:
#         vct (list): An empty list.
#         ps (dict): A dictionary of class probabilities.
#         labels_bytes (list): A list of class labels.
#         n_classes (int): The number of potential classes.
#
#     Returns:
#         ``list`` of class probabilities
#     """
#
#     cdef:
#         Py_ssize_t m
#         double ps_label_value
#         cpp_string m_label
#
#     for m in range(0, n_classes):
#
#         m_label = labels_bytes[m]
#
#         if ps.count(m_label) > 0:
#             ps_label_value = ps[m_label]
#         else:
#             ps_label_value = 0.0
#
#         vct.push_back(ps_label_value)
#
#     return vct


def transform_probas(vector[vector[cpp_map[cpp_string, double]]] pred,
                     double[:, :, ::1] data,
                     vector[cpp_string] labels,
                     unsigned int nclasses,
                     unsigned int ntime,
                     unsigned int nbands,
                     unsigned int nrows,
                     unsigned int ncols,
                     bint insert_nodata=False,
                     int n_jobs=1):

    """
    Transforms CRF probabilities in dictionary format to probabilities in array format

    Args:
        pred (list of lists of dicts): e.g., [[{}, {}], [{}, {}]]
        data (ndarray): Time x samples x bands.
        labels (list): The class labels.
        nclasses (int): The number of potential classes.
        ntime (int): The number of states.
        nrows (int): The number of rows.
        ncols (int): The number of columns.
        insert_nodata (Optional[bool])
        n_jobs (Optional[int])

    Returns:
        ``ndarray``
    """

    cdef:
        Py_ssize_t i, j, k
        vector[cpp_map[cpp_string, double]] pr
        cpp_map[cpp_string, double] ps
        unsigned int nstates
        unsigned int nsamples = pred.size()
        double[:, :, ::1] out_array

    if insert_nodata:
        nstates = data.shape[0]
    else:
        nstates = pred[0].size()

    out_array = np.empty((nsamples, nstates, nclasses), dtype='float64')

    # with nogil, parallel(num_threads=n_jobs):
    with nogil:

        # out_array = [samples x time x classes]
        # for i in prange(0, nsamples, schedule='static'):
        for i in range(0, nsamples):

            # The current sample
            pr = pred[i]

            k = 0
            for j in range(0, nstates):

                # If True then 'no data'
                if insert_nodata:

                    if data[j, i, nbands-1] > 0:

                        # Add the list forcing 'no data' to 1
                        _push_nodata_array(labels, nclasses, j, i, out_array)

                    else:

                        # The current sample state
                        ps = pr[k]

                        # Get the class probabilities
                        _push_classes_array(ps, labels, nclasses, j, i, out_array)

                        k += 1

                else:

                    # The current sample state
                    ps = pr[k]

                    # Get the class probabilities
                    _push_classes_array(ps, labels, nclasses, j, i, out_array)

                    k += 1

    return np.float64(out_array).transpose(1, 2, 0).reshape(ntime,
                                                            nclasses,
                                                            nrows,
                                                            ncols)


cdef vector[cpp_map[cpp_string, double]] _add_ffunc_diffs(double[:, :, ::1] data,
                                                          Py_ssize_t i,
                                                          unsigned int ntime,
                                                          unsigned int nbands,
                                                          vector[cpp_string] band_names_vect,
                                                          double scale_factor,
                                                          bint transform) nogil:

    cdef:
        Py_ssize_t j, m, n
        vector[cpp_map[cpp_string, double]] samples
        double[::1] tsample

    for j in range(0, ntime):

        # Time j, sample i, all bands
        tsample = data[j, i, :]

        if j == 0:

            for m in range(0, nbands):
                tsample[m] = 0.0

            for n in range(0, nbands):
                tsample[m + 1] = (data[j, i, n] - data[j + 1, i, n])
                m += 1

        elif j + 1 == ntime:

            for m in range(0, nbands):
                tsample[m] = (data[j, i, m] - data[j - 1, i, m])

            for n in range(0, nbands):
                tsample[m + 1] = 0.0
                m += 1

        else:

            for m in range(0, nbands):
                tsample[m] = (data[j, i, m] - data[j - 1, i, m])

            for n in range(0, nbands):
                tsample[m + 1] = (data[j, i, n] - data[j + 1, i, n])
                m += 1

        for n in range(0, nbands):
            tsample[m + 1] = data[j, i, n]
            m += 1

        samples.push_back(_sample_to_dict(tsample,
                                          band_names_vect,
                                          scale_factor,
                                          transform))

    return samples


cdef vector[cpp_map[cpp_string, double]] _add_ffunc(double[:, :, ::1] data,
                                                    Py_ssize_t i,
                                                    unsigned int ntime,
                                                    unsigned int nbands,
                                                    bint remove_nodata,
                                                    int nodata_layer_idx,
                                                    vector[cpp_string] band_names_vect,
                                                    double scale_factor,
                                                    bint transform) nogil:

    cdef:
        Py_ssize_t j, m, n
        vector[cpp_map[cpp_string, double]] samples
        double[::1] tsample
        bint skip_end = True if remove_nodata else False

    for j in range(0, ntime):

        # Time j, sample i, all bands
        tsample = data[j, i, :]

        if remove_nodata:

            # The 'no data' layer should be last.
            # 1 = no data
            # 0 = valid
            if tsample[nodata_layer_idx] != 0:
                continue

        samples.push_back(_sample_to_dict(tsample,
                                          band_names_vect,
                                          scale_factor,
                                          transform,
                                          skip_end=skip_end))  # Do not include the 'no data' band in the features

    return samples


def time_to_sensor_feas(double[:, :, ::1] data,
                        cpp_string sensor,
                        unsigned int ntime,
                        unsigned int nbands,
                        unsigned int nrows,
                        unsigned int ncols,
                        vector[char*] band_names,
                        double scale_factor=0.0001,
                        bint add_indices=False,
                        bint transform=False,
                        bint band_diffs=False,
                        bint remove_nodata=False,
                        int nodata_layer=-1):

    """
    Converts a time-shaped array to CRF features

    Args:
        data (ndarray): Time x Samples x Bands.
        sensor (str): The satellite sensor.
        ntime (int): The number of time dimensions.
        nbands (int): The number of band dimensions.
        nrows (int): The number of row dimensions.
        ncols (int): The number of column dimensions.
        band_names (array): The band band_names.
        scale_factor (Optional[float])
        add_indices (Optional[bool])
        transform (Optional[bool])
        band_diffs (Optional[bool])
        remove_nodata (Optional[bool])
        nodata_layer (Optional[int]): The 'no data' layer index. If -1, the 'no data' layer is taken as the
            last layer along axis 0. Valid samples should be 0.

    Returns:
        ``list`` of feature dictionaries
    """

    cdef:
        Py_ssize_t i, j, v, m, n
        double[::1] tsample
        vector[cpp_map[cpp_string, double]] samples
        vector[vector[cpp_map[cpp_string, double]]] samples_full

#        unsigned int blue_idx, green_idx, red_idx, nir_idx, swir1_idx, swir2_idx, nir1_idx, nir2_idx, nir3_idx, rededge_idx, ndiff_vars

#        double avi, brightness, evi, evi2, gndvi, nbr, ndmi, ndvi, si, wi

#        cpp_string avi_string = <cpp_string>'avi'.encode('utf-8')
#        cpp_string brightness_string = <cpp_string>'bri'.encode('utf-8')
#        cpp_string evi_string = <cpp_string>'evi'.encode('utf-8')
#        cpp_string evi2_string = <cpp_string>'evi2'.encode('utf-8')
#        cpp_string gndvi_string = <cpp_string>'gndvi'.encode('utf-8')
#        cpp_string nbr_string = <cpp_string>'nbr'.encode('utf-8')
#        cpp_string ndmi_string = <cpp_string>'ndmi'.encode('utf-8')
#        cpp_string ndvi_string = <cpp_string>'ndvi'.encode('utf-8')
#        cpp_string si_string = <cpp_string>'si'.encode('utf-8')
#        cpp_string wi_string = <cpp_string>'wi'.encode('utf-8')

        vector[cpp_string] band_names_vect
#        cpp_map[cpp_string, cpp_map[cpp_string, int]] sensor_bands
#        cpp_map[cpp_string, int] l7_like
#        cpp_map[cpp_string, int] l8
#        cpp_map[cpp_string, int] s210
#        cpp_map[cpp_string, int] s220
#        cpp_map[cpp_string, int] s2

        bint nodata
        int nodata_layer_idx

    if nodata_layer == -1:
        nodata_layer_idx = nbands - 1
    else:
        nodata_layer_idx = nodata_layer

    ndiff_vars = <int>(nbands*3)

    tsample = np.zeros(ndiff_vars, dtype='float64')

    if band_diffs:

        for v in range(1, ndiff_vars+1):
            band_names_vect.push_back(<cpp_string>str(v).encode('utf-8'))

    else:

        for v in range(0, nbands):
            band_names_vect.push_back(<cpp_string>band_names[v])

#    l7_like[b'blue'] = 0
#    l7_like[b'green'] = 1
#    l7_like[b'red'] = 2
#    l7_like[b'nir'] = 3
#    l7_like[b'swir1'] = 4
#    l7_like[b'swir2'] = 5
#
#    l8[b'coastal'] = 0
#    l8[b'blue'] = 1
#    l8[b'green'] = 2
#    l8[b'red'] = 3
#    l8[b'nir'] = 4
#    l8[b'swir1'] = 5
#    l8[b'swir2'] = 6
#
#    s210[b'blue'] = 0
#    s210[b'green'] = 1
#    s210[b'red'] = 2
#    s210[b'nir'] = 3
#
#    s220[b'nir1'] = 0
#    s220[b'nir2'] = 1
#    s220[b'nir3'] = 2
#    s220[b'rededge'] = 3
#    s220[b'swir1'] = 4
#    s220[b'swir2'] = 5
#
#    s2[b'blue'] = 0
#    s2[b'green'] = 1
#    s2[b'red'] = 2
#    s2[b'nir1'] = 3
#    s2[b'nir2'] = 4
#    s2[b'nir3'] = 5
#    s2[b'nir'] = 6
#    s2[b'rededge'] = 7
#    s2[b'swir1'] = 8
#    s2[b'swir2'] = 9
#
#    sensor_bands[b'l7'] = l7_like
#    sensor_bands[b'l8'] = l8
#    sensor_bands[b'l5bgrn'] = s210
#    sensor_bands[b'l7bgrn'] = s210
#    sensor_bands[b'l8bgrn'] = s210
#    sensor_bands[b'bgrn'] = s210
#    sensor_bands[b'qb'] = s210
#    sensor_bands[b'ps'] = s210
#    sensor_bands[b's210'] = s210
#    sensor_bands[b's220'] = s220
#    sensor_bands[b's2'] = s2
#    sensor_bands[b's2l7'] = l7_like

#    if sensor != b'pan':
#
#        if (sensor == b's210') or (sensor == b'l5bgrn') or (sensor == b'l7bgrn') or (sensor == b'l8bgrn') or (sensor == b'bgrn') or (sensor == b'qb') or (sensor == b'ps'):
#
#            blue_idx = sensor_bands[sensor][b'blue']
#            green_idx = sensor_bands[sensor][b'green']
#            red_idx = sensor_bands[sensor][b'red']
#            nir_idx = sensor_bands[sensor][b'nir']
#
#        elif sensor == b's220':
#
#            nir1_idx = sensor_bands[sensor][b'nir1']
#            nir2_idx = sensor_bands[sensor][b'nir2']
#            nir3_idx = sensor_bands[sensor][b'nir3']
#            rededge_idx = sensor_bands[sensor][b'rededge']
#            swir1_idx = sensor_bands[sensor][b'swir1']
#            swir2_idx = sensor_bands[sensor][b'swir2']
#
#        else:
#
#            blue_idx = sensor_bands[sensor][b'blue']
#            green_idx = sensor_bands[sensor][b'green']
#            red_idx = sensor_bands[sensor][b'red']
#            nir_idx = sensor_bands[sensor][b'nir']
#            swir1_idx = sensor_bands[sensor][b'swir1']
#            swir2_idx = sensor_bands[sensor][b'swir2']

    with nogil:

        if band_diffs:

            for i in range(0, nrows*ncols):

                samples_full.push_back(_add_ffunc_diffs(data,
                                                        i,
                                                        ntime,
                                                        nbands,
                                                        band_names_vect,
                                                        scale_factor,
                                                        transform))

        else:

            for i in range(0, nrows*ncols):

                samples_full.push_back(_add_ffunc(data,
                                                  i,
                                                  ntime,
                                                  nbands,
                                                  remove_nodata,
                                                  nodata_layer_idx,
                                                  band_names_vect,
                                                  scale_factor,
                                                  transform))

                # for j in range(0, ntime):
                #
                #     # Time j, sample i, all bands
                #     tsample = data[j, i, :]
                #
                #     if remove_nodata:
                #
                #         # The 'no data' layer should be last.
                #         # 1 = no data
                #         # 0 = valid
                #         if tsample[nodata_layer_idx] != 0:
                #             continue
                #
                #     if sensor == b'pan':
                #
                #         samples.push_back(_sample_to_dict_pan(tsample,
                #                                               band_names_vect,
                #                                               scale_factor))
                #
                #     else:
                #
                #         if (sensor == b's210') or (sensor == b'l5bgrn') or (sensor == b'l7bgrn') or (sensor == b'l8bgrn') or (sensor == b'bgrn') or (sensor == b'qb') or (sensor == b'ps'):
                #
                #             if (tsample[blue_idx] * scale_factor < 0.01) and (tsample[green_idx] * scale_factor < 0.01) and (tsample[red_idx] * scale_factor < 0.01):
                #
                #                 brightness = 0.0
                #                 evi = 0.0
                #                 evi2 = 0.0
                #                 gndvi = 0.0
                #                 ndvi = 0.0
                #
                #             else:
                #
                #                 brightness = _brightness(tsample[green_idx]*scale_factor,
                #                                          tsample[red_idx]*scale_factor,
                #                                          tsample[nir_idx]*scale_factor)
                #
                #                 evi = _evi(tsample[blue_idx]*scale_factor, tsample[red_idx]*scale_factor, tsample[nir_idx]*scale_factor)
                #                 evi2 = _evi2(tsample[red_idx]*scale_factor, tsample[nir_idx]*scale_factor)
                #                 gndvi = _gndvi(tsample[green_idx]*scale_factor, tsample[nir_idx]*scale_factor)
                #                 ndvi = _ndvi(tsample[red_idx]*scale_factor, tsample[nir_idx]*scale_factor)
                #
                #             samples.push_back(_sample_to_dict_bgrn(tsample,
                #                                                    band_names_vect,
                #                                                    brightness,
                #                                                    evi,
                #                                                    evi2,
                #                                                    gndvi,
                #                                                    ndvi,
                #                                                    brightness_string,
                #                                                    evi_string,
                #                                                    evi2_string,
                #                                                    gndvi_string,
                #                                                    ndvi_string,
                #                                                    scale_factor,
                #                                                    add_indices,
                #                                                    transform))
                #
                #         elif sensor == b's220':
                #
                #             brightness = _brightness(tsample[nir1_idx]*scale_factor,
                #                                      tsample[rededge_idx]*scale_factor,
                #                                      tsample[swir1_idx]*scale_factor)
                #
                #             nbr = _nbr(tsample[rededge_idx]*scale_factor, tsample[swir2_idx]*scale_factor)
                #             ndmi = _ndmi(tsample[rededge_idx]*scale_factor, tsample[swir1_idx]*scale_factor)
                #             ndvi = _ndvi(tsample[nir1_idx]*scale_factor, tsample[rededge_idx]*scale_factor)
                #             wi = _wi(tsample[nir1_idx]*scale_factor, tsample[swir1_idx]*scale_factor)
                #
                #             samples.push_back(_sample_to_dict_s220(tsample,
                #                                                    band_names_vect,
                #                                                    brightness,
                #                                                    nbr,
                #                                                    ndmi,
                #                                                    ndvi,
                #                                                    wi,
                #                                                    brightness_string,
                #                                                    nbr_string,
                #                                                    ndmi_string,
                #                                                    ndvi_string,
                #                                                    wi_string,
                #                                                    scale_factor,
                #                                                    add_indices,
                #                                                    transform))
                #
                #         else:
                #
                #             samples.push_back(_sample_to_dict(tsample,
                #                                               band_names_vect,
                #                                               scale_factor,
                #                                               transform))
                #
                # samples_full.push_back(samples)
                # samples.clear()

    return samples_full


def time_to_feas(double[:, :, ::1] data,
                 vector[char*] labels,
                 bint band_diffs=False):

    """
    Transforms time-formatted variables to CRF-formatted features

    Args:
        data (3d array): The data to transform. The shape should be [time x samples x predictors].
        labels (vector): The labels that correspond to each predictor.
        band_diffs (Optional[bool]): Whether to calculate band differences.

    Returns:
        ``list``
    """

    cdef:
        Py_ssize_t s, t, v, m, n
        unsigned int ntime = data.shape[0]
        unsigned int nsamples = data.shape[1]
        unsigned int nvars = data.shape[2]
        vector[cpp_map[cpp_string, double]] samples
        vector[vector[cpp_map[cpp_string, double]]] samples_full
        unsigned int ndiff_vars = <int>(nvars*3)
        double[::1] tsample = np.zeros(ndiff_vars, dtype='float64')

    with nogil:

        if band_diffs:

            for s in range(0, nsamples):

                samples = _samples_to_dict_3d_diffs(labels,
                                                    data,
                                                    ntime,
                                                    s,
                                                    nvars,
                                                    data,
                                                    tsample)

                samples_full.push_back(samples)

        else:

            for s in range(0, nsamples):

                samples = _samples_to_dict_3d(labels,
                                              data,
                                              ntime,
                                              s,
                                              nvars)

                samples_full.push_back(samples)

    return samples_full


cdef vector[int] _get_label(cpp_map[cpp_string, int] label_mappings,
                            vector[char*] sample,
                            unsigned int ntime) nogil:

    cdef:
        Py_ssize_t t
        vector[int] out_sample_

    for t in range(0, ntime):
        out_sample_.push_back(label_mappings[sample[t]])

    return out_sample_


def labels_to_values(vector[vector[char_ptr]] labels,
                     cpp_map[cpp_string, int] label_mappings,
                     unsigned int nsamples,
                     unsigned int ntime):

    """
    Transforms CRF str/byte labels to values

    Args:
        labels (list)
        label_mappings (dict)
        nsamples (int)
        ntime (int)

    Returns:
        ``list``
    """

    cdef:
        Py_ssize_t s, t
        vector[char*] sample
        vector[int] out_sample
        vector[vector[int]] out_full

    with nogil:

        for s in range(0, nsamples):

            sample = labels[s]
            out_sample = _get_label(label_mappings, sample, ntime)
            out_full.push_back(out_sample)

    return out_full
