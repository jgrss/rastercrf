import numpy as np
import numba as nb


def check_x_names(X, x_names, band_diffs):

    if band_diffs:

        ndiff_vars = int(X[0].shape[1] * 3)
        pred_labels = ['z.{:03d}'.format(v).encode('utf-8') for v in range(1, ndiff_vars + 1)]

    else:

        pred_labels = []

        for label in x_names:

            try:
                lab = 'z.{:03d}'.format(int(label)).encode('utf-8')
            except:
                lab = label.encode('utf-8')

            pred_labels.append(lab)

    x_names = pred_labels

    return x_names


def dict_keys_to_bytes(data):

    if isinstance(data, int) or isinstance(data, float): return data
    if isinstance(data, str): return str.encode(data)
    if isinstance(data, dict): return dict(map(dict_keys_to_bytes, data.items()))
    if isinstance(data, list): return list(map(dict_keys_to_bytes, data))
    if isinstance(data, tuple): return tuple(map(dict_keys_to_bytes, data))


def columns_to_nd(array, ndims, nrows, ncols, nlayers=None):

    """
    Reshapes an array from columns layout to [``ndims`` x ``nrows`` x ``ncols``] or
    [``ndims`` x ``nlayers`` x ``nrows`` x ``ncols``].

    Args:
        array (2d array)
        ndims (int)
        nrows (int)
        ncols (int)
        nlayers (Optional[int])

    Returns:
        3-d or 4-d ``numpy.ndarray``
    """

    if not isinstance(nlayers, int):
        return array.T.reshape(ndims, nrows, ncols)
    else:
        return array.reshape(nrows, ncols, ndims, nlayers).transpose(2, 3, 0, 1)


def nd_to_columns(array):

    """
    Reshapes an array from nd layout to [samples (``nrows`` * ``ncols``) x ``ndims``]

    Args:
        array (3d or 4d array)

    Returns:
        2-d ``numpy.ndarray``
    """

    if len(array.shape) == 3:

        ndims, nrows, ncols = array.shape
        return array.transpose(1, 2, 0).reshape(nrows*ncols, ndims)

    else:

        nsamps, ndims, nrows, ncols = array.shape
        return array.transpose(0, 2, 3, 1).reshape(nsamps*nrows*ncols, ndims)


def scale_min_max(xv, mno, mxo, mni, mxi):
    return np.clip((((mxo - mno) * (xv - mni)) / (mxi - mni)) + mno, mno, mxo)


def transform_data(data, scale_factor=1.0):

    return np.clip(np.float64(scale_min_max(np.exp(np.clip(data*scale_factor, 0, 1)),
                                             0, 5, np.exp(0), np.exp(1))), 0, 5)


@nb.jit
def sample_to_dict(tsp, band_names, *args):

    """
    Args:
        tsp (1d)
        band_names (list): Names to replace an integer range.
    """

    nt = len(tsp)

    if args:

        n_args = len(args)

        tlab = np.empty(nt + n_args, dtype=object)
        tval = np.zeros(nt + n_args, dtype='float64')

    else:

        tlab = np.empty(nt, dtype=object)
        tval = np.zeros(nt, dtype='float64')

    for r in range(0, nt):

        if band_names:
            tlab[r] = band_names[r].encode('utf-8')
        else:
            tlab[r] = str(r + 1).encode('utf-8')

        tval[r] = tsp[r]

    if args:

        for i, (arg_name, arg_value) in enumerate(args):

            tlab[r + i + 1] = arg_name.encode('utf-8')
            tval[r + i + 1] = arg_value

    return dict(zip(tlab.tolist(), tval))
    # return dict(zip(tlab[nt:].tolist(), tval[nt:]))


def _nan_check(value):
    return 0.0 if np.isnan(value) else value


def _min(a, b):
    return a if a < b else b


def _max(a, b):
    return a if a > b else b


def _clip_low(value):
    return 0.0 if value < 0 else value


def _clip_high(value):
    return 1.0 if value > 1 else value


def _clip(value):
    return _clip_low(value) if value < 1 else _clip_high(value)


def _avi(red, nir):
    """Advanced Vegetation Index"""
    return (nir * (1.0 - red) * (nir - red))**0.3334


def _evi(blue, red, nir):
    """Enhanced Vegetation Index"""
    return 2.5 * ((nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0))


def _evi2(red, nir):
    """Two-band Enhanced Vegetation Index"""
    return 2.5 * ((nir - red) / (nir + 1.0 + (2.4 * red)))


def _bsi(blue, red, nir, swir2):
    """Bare Soil Index"""
    return ((swir2 + red) - (nir - blue)) / ((swir2 + red) + (nir - blue))


def _brightness(green, red, nir):
    """Brightness Index"""
    return (green ** 2 + red ** 2 + nir ** 2) ** 0.5


def _brightness_swir(green, red, nir, swir1):
    """Brightness Index"""
    return (green ** 2 + red ** 2 + nir ** 2 + swir1 ** 2) ** 0.5


def _dbsi(green, red, nir, swir1):
    """Dry Bare Soil Index"""
    return ((swir1 - green) / (swir1 + green)) - _ndvi(red, nir)


def _nbr(nir, swir2):
    """Normalized Burn Ratio"""
    return (nir - swir2) / (nir + swir2)


def _ndmi(nir, swir1):
    """Normalized Difference Moisture Index"""
    return (nir - swir1) / (nir + swir1)


def _ndvi(red, nir):
    """Normalized Difference Vegetation Index"""
    return (nir - red) / (nir + red)


def _si(blue, green, red):
    """Shadow Index"""
    return ((1.0 - blue) * (1.0 - green) * (1.0 - red))**0.3334


def _wi(red, swir1):
    """Woody Index"""
    return 0.0 if red + swir1 > 0.5 else 1.0 - ((red + swir1) / 0.5)


SENSOR_BANDS = dict(l7=dict(blue=0,
                            green=1,
                            red=2,
                            nir=3,
                            swir1=4,
                            swir2=5),
                    l8=dict(blue=0,
                            green=1,
                            red=2,
                            nir=3,
                            swir1=4,
                            swir2=5),
                    l5bgrn=dict(blue=0,
                                green=1,
                                red=2,
                                nir=3),
                    l7bgrn=dict(blue=0,
                                green=1,
                                red=2,
                                nir=3),
                    l8bgrn=dict(blue=0,
                                green=1,
                                red=2,
                                nir=3),
                    bgrn=dict(blue=0,
                              green=1,
                              red=2,
                              nir=3),
                    qb=dict(blue=0,
                            green=1,
                            red=2,
                            nir=3),
                    ps=dict(blue=0,
                            green=1,
                            red=2,
                            nir=3),
                    s210=dict(blue=0,
                              green=1,
                              red=2,
                              nir=3),
                    s220=dict(nir1=0,
                              nir2=1,
                              nir3=2,
                              rededge=3,
                              swir1=4,
                              swir2=5),
                    s2l7=dict(blue=0,
                              green=1,
                              red=2,
                              nir=3,
                              swir1=4,
                              swir2=5))


def array_to_dict(sensor,
                  indices,
                  scale_factor,
                  transform,
                  band_names,
                  *args):

    """
    Converts an array sample to a CRF features

    Args:
        sensor (str): The satellite sensor.
        indices (bool): Whether to use indices with bands.
        scale_factor (float): A scale factor to apply.
        transform (bool): Whether to transform the bands.
        band_names (list): A list of names. If not given, names are set as an integer range (1-n features).

    Returns:
        ``list``
    """

    nargs = len(args)
    feas = list()

    if sensor != 'pan':

        if sensor in ['s210', 'l5bgrn', 'l7bgrn', 'l8bgrn', 'bgrn', 'qb', 'ps']:

            blue_idx = SENSOR_BANDS[sensor]['blue']
            green_idx = SENSOR_BANDS[sensor]['green']
            red_idx = SENSOR_BANDS[sensor]['red']
            nir_idx = SENSOR_BANDS[sensor]['nir']

        elif sensor == 's220':

            nir1_idx = SENSOR_BANDS[sensor]['nir1']
            nir2_idx = SENSOR_BANDS[sensor]['nir2']
            nir3_idx = SENSOR_BANDS[sensor]['nir3']
            rededge_idx = SENSOR_BANDS[sensor]['rededge']
            swir1_idx = SENSOR_BANDS[sensor]['swir1']
            swir2_idx = SENSOR_BANDS[sensor]['swir2']

        else:

            blue_idx = SENSOR_BANDS[sensor]['blue']
            green_idx = SENSOR_BANDS[sensor]['green']
            red_idx = SENSOR_BANDS[sensor]['red']
            nir_idx = SENSOR_BANDS[sensor]['nir']
            swir1_idx = SENSOR_BANDS[sensor]['swir1']
            swir2_idx = SENSOR_BANDS[sensor]['swir2']

    for si in range(0, nargs):

        tsamp = args[si] * scale_factor

        if sensor == 'pan':
            feas.append(sample_to_dict(tsamp, band_names))
        else:

            if indices:

                if sensor in ['s210', 'l5bgrn', 'l7bgrn', 'l8bgrn', 'bgrn', 'qb', 'ps']:

                    if (tsamp[blue_idx] < 0.01) and (tsamp[green_idx] < 0.01) and (tsamp[red_idx] < 0.01):

                        brightness = 0.0
                        evi = 0.0
                        evi2 = 0.0
                        gndvi = 0.0
                        ndvi = 0.0

                    else:

                        brightness = _brightness(tsamp[green_idx], tsamp[red_idx], tsamp[nir_idx])
                        evi = _evi(tsamp[blue_idx], tsamp[red_idx], tsamp[nir_idx])
                        evi2 = _evi2(tsamp[red_idx], tsamp[nir_idx])
                        gndvi = _ndvi(tsamp[green_idx], tsamp[nir_idx])
                        ndvi = _ndvi(tsamp[red_idx], tsamp[nir_idx])

                    indices = [('bri', _nan_check(brightness)),
                               ('evi', _clip(_nan_check(evi))),
                               ('evi2', _clip(_nan_check(evi2))),
                               ('gndvi', _nan_check(gndvi)),
                               ('ndvi', _nan_check(ndvi))]

                elif sensor == 's220':

                    brightness = _brightness(tsamp[nir1_idx], tsamp[rededge_idx], tsamp[swir1_idx])
                    nbr = _nbr(tsamp[rededge_idx], tsamp[swir2_idx])
                    ndmi = _ndvi(tsamp[rededge_idx], tsamp[swir1_idx])
                    ndvi = _ndvi(tsamp[nir1_idx], tsamp[rededge_idx])
                    wi = _wi(tsamp[nir1_idx], tsamp[swir1_idx])

                    brightness = scale_min_max(_clip(_nan_check(brightness)), 0.01, 1.0, 0.0, 1.0)
                    nbr = _clip(scale_min_max(_nan_check(nbr), 0.01, 1.0, -1.0, 1.0))
                    ndmi = _clip(scale_min_max(_nan_check(ndmi), 0.01, 1.0, -1.0, 1.0))
                    ndvi = _clip(scale_min_max(_nan_check(ndvi), 0.01, 1.0, -1.0, 1.0))
                    wi = scale_min_max(_clip(_nan_check(wi)), 0.01, 1.0, 0.0, 1.0)

                    indices = [('bri', brightness),
                               ('nbr', nbr),
                               ('ndmi', ndmi),
                               ('ndvi', ndvi),
                               ('wi', wi)]

                else:

                    if (tsamp[blue_idx] < 0.01) and (tsamp[green_idx] < 0.01) and (tsamp[red_idx] < 0.01):

                        avi = 0.0
                        brightness = 0.0
                        evi = 0.0
                        evi2 = 0.0
                        gndvi = 0.0
                        nbr = 0.0
                        ndmi = 0.0
                        ndvi = 0.0
                        wi = 0.0

                        tsamp[:] = 0.0

                    else:

                        avi = _avi(tsamp[red_idx], tsamp[nir_idx])
                        brightness = _brightness_swir(tsamp[green_idx], tsamp[red_idx], tsamp[nir_idx], tsamp[swir1_idx])
                        evi = _evi(tsamp[blue_idx], tsamp[red_idx], tsamp[nir_idx])
                        evi2 = _evi2(tsamp[red_idx], tsamp[nir_idx])
                        gndvi = _ndvi(tsamp[green_idx], tsamp[nir_idx])
                        nbr = _nbr(tsamp[nir_idx], tsamp[swir2_idx])
                        ndmi = _ndmi(tsamp[nir_idx], tsamp[swir1_idx])
                        ndvi = _ndvi(tsamp[red_idx], tsamp[nir_idx])
                        si = _si(tsamp[blue_idx], tsamp[green_idx], tsamp[red_idx])
                        wi = _wi(tsamp[red_idx], tsamp[swir1_idx])

                        avi = scale_min_max(_nan_check(avi), 0.01, 1.0, 0.0, 1.0)
                        brightness = scale_min_max(_nan_check(brightness), 0.01, 1.0, 0.0, 1.0)
                        evi = scale_min_max(_clip(_nan_check(evi)), 0.01, 1.0, 0.0, 1.0)
                        evi2 = scale_min_max(_clip(_nan_check(evi2)), 0.01, 1.0, 0.0, 1.0)
                        gndvi = scale_min_max(_nan_check(gndvi), 0.01, 1.0, -1.0, 1.0)
                        nbr = _clip(scale_min_max(_nan_check(nbr), 0.01, 1.0, -1.0, 1.0))
                        ndmi = _clip(scale_min_max(_nan_check(ndmi), 0.01, 1.0, -1.0, 1.0))
                        ndvi = _clip(scale_min_max(_nan_check(ndvi), 0.01, 1.0, -1.0, 1.0))
                        si = scale_min_max(_clip(_nan_check(si)), 0.01, 1.0, 0.0, 1.0)
                        wi = scale_min_max(_clip(_nan_check(wi)), 0.01, 1.0, 0.0, 1.0)

                    # indices = [('bri', brightness),
                    #            ('evi', evi),
                    #            ('evi2', evi2),
                    #            ('gndvi', gndvi),
                    #            ('nbr', nbr),
                    #            ('ndmi', ndmi),
                    #            ('ndvi', ndvi),
                    #            ('si', si),
                    #            ('wi', wi)]

                    indices = [('avi', avi),
                               ('evi2', evi2),
                               ('nbr', nbr),
                               ('si', si),
                               ('wi', wi)]

                if transform:

                    tsamp = np.clip(np.float64(scale_min_max(np.exp(np.clip(tsamp, 0, 1)),
                                                              0, 5, np.exp(0), np.exp(1))), 0, 5)

                feas.append(sample_to_dict(tsamp, band_names, *indices))

            else:

                if transform:

                    tsamp = np.clip(np.float64(scale_min_max(np.exp(np.clip(tsamp, 0, 1)),
                                                              0, 5, np.exp(0), np.exp(1))), 0, 5)

                feas.append(sample_to_dict(tsamp, band_names))

    return feas
