import concurrent.futures

from .util import array_to_dict
from .proc import Concurrency

import numpy as np
import pandas as pd
from tqdm import trange, tqdm


def str_to_bytes(string):

    ss = string.split('.')

    if len(ss) > 1:

        if ss[0].startswith('weekly'):
            string = 'a.{}.{:03d}'.format(ss[0], int(ss[1]))
        else:
            string = 'b.{}.{:03d}'.format(ss[0], int(ss[1]))

    else:

        try:
            string = 'z.{:03d}'.format(int(ss[0]))
        except:
            string = 'z.{}'.format(ss[0])

    return string.encode()


def sample_to_dict(df_sample, feature_cols):
    """Conversion of a DataFrame to a dictionary"""
    return dict(zip(sorted(list(map(str_to_bytes, feature_cols))), df_sample.values.tolist()))


# def sample_to_dict(nd_sample, feature_cols):
#     """Conversion of a ndarray to a dictionary"""
#     return dict(zip(sorted(list(map(str_to_bytes, feature_cols))), nd_sample.tolist()))


def _values_to_array(values):
    """Converts dictionary values to an array"""
    return np.array(list(values), dtype='float64')


def calc_diff(value1, value2):

    """
    Calculates the difference and scales to 0-1

    Args:
        value1 (1d array): Data range 0-1.
        value2 (1d array): Data range 0-1.
    """

    return np.clip(np.abs(value2 - value1), 0, 1)


def calc_band_diffs(xxx, keys_diff):

    """
    Calculates band differences
    """

    new_Xd_ = list()

    concat_list = []

    comb_keys = list(xxx[0].keys()) + keys_diff
    values = _values_to_array(xxx[0].values()).tolist()
    end_values = (_values_to_array(xxx[0].values()) * 0.0).tolist()

    a_diff = calc_diff(_values_to_array(end_values + list(xxx[0].values()) + end_values),
                       _values_to_array(end_values + list(xxx[1].values()) + values))

    a = dict(zip(comb_keys, a_diff))

    concat_list.append(a)

    for xxxx in range(1, len(xxx) - 1):

        b_diff = calc_diff(_values_to_array(list(xxx[xxxx - 1].values()) + list(xxx[xxxx + 1].values()) + end_values),
                           _values_to_array(list(xxx[xxxx].values()) + list(xxx[xxxx].values()) + list(xxx[xxxx].values())))

        # (current - previous) (current - following)
        b = dict(zip(comb_keys, b_diff))

        concat_list.append(b)

    xxxx = len(xxx) - 1

    c_diff = calc_diff(_values_to_array(list(xxx[xxxx - 1].values()) + end_values + end_values),
                       _values_to_array(list(xxx[xxxx].values()) + end_values + list(xxx[xxxx].values())))

    c = dict(zip(comb_keys, c_diff))

    concat_list.append(c)

    new_Xd_.append(concat_list)

    return new_Xd_[0]


def calc_band_diffs_nested(Xd_, keys_diff):

    """
    Calculates band differences
    """

    new_Xd_ = list()

    for xxx in Xd_:

        concat_list = []

        comb_keys = list(xxx[0].keys()) + keys_diff
        values = _values_to_array(xxx[0].values()).tolist()
        end_values = (_values_to_array(xxx[0].values()) * 0.0).tolist()

        a_diff = calc_diff(_values_to_array(end_values + list(xxx[1].values()) + end_values),
                           _values_to_array(end_values + list(xxx[0].values()) + values))

        a = dict(zip(comb_keys, a_diff))

        concat_list.append(a)

        for xxxx in range(1, len(xxx) - 1):

            b_diff = calc_diff(_values_to_array(list(xxx[xxxx - 1].values()) + list(xxx[xxxx + 1].values()) + end_values),
                               _values_to_array(list(xxx[xxxx].values()) + list(xxx[xxxx].values()) + list(xxx[xxxx].values())))

            # (current - previous) (current - following)
            b = dict(zip(comb_keys, b_diff))

            concat_list.append(b)

        c_diff = calc_diff(_values_to_array(list(xxx[xxxx - 1].values()) + end_values + end_values),
                           _values_to_array(list(xxx[xxxx].values()) + end_values + list(xxx[xxxx].values())))

        c = dict(zip(comb_keys, c_diff))
        concat_list.append(c)

        new_Xd_.append(concat_list)

    return new_Xd_


def sample_array_worker(X_data,
                        y_data,
                        class_iters,
                        class_sub_frac,
                        dataframe,
                        sensor,
                        min_time,
                        max_time,
                        idx_nodata,
                        nbands,
                        nrows,
                        ncols,
                        add_indices,
                        scale_factor,
                        transform,
                        band_names,
                        band_diffs,
                        single_frac,
                        multi_frac):

    if band_diffs:

        ncols_ = X_data.shape[1]
        keys_diff = [str(int(k)).encode('utf-8') for k in range(ncols_+1, ncols_+2+ncols_*2)]

    Xc, yc = [], []

    for a in class_iters:

        add_nulls = np.random.randint(0, high=2)

        # Create a pool of row indices to sample from
        if len(a) == 1:
            idx_nnull = np.where(y_data == a[0])[0]
        else:

            idx_nnull = dataframe.query(f"label == {a}")\
                                    .groupby('label', group_keys=False)\
                                    .apply(lambda x: x.sample(n=min(len(x), max(min_time, max_time))))\
                                    .sample(frac=1)

        if not isinstance(idx_nnull, np.ndarray):

            # Sub-sample to avoid overfitting common classes
            if class_sub_frac:

                if len(list(set(a).intersection(list(class_sub_frac.keys())))) == a:

                    idx_nnull_ = []

                    for class_key, frac_val in class_sub_frac.items():

                        idx_nnull_ += idx_nnull.query(f"label == '{class_key}'")\
                                            .sample(frac=frac_val).index.values.tolist()

                    idx_nnull_ += idx_nnull.query(f"label != {list(class_sub_frac.keys())}").index.values.tolist()
                    idx_nnull = np.array(idx_nnull_, dtype='int64')

                else:
                    idx_nnull = idx_nnull.index.values

            else:
                idx_nnull = idx_nnull.index.values

        if idx_nnull.shape[0] > 0:

            # Get a random season length
            if min_time == max_time:
                season_len = min_time
            else:
                season_len = np.random.randint(min_time, high=max_time)

            season_len = min(season_len, idx_nnull.shape[0])

            # Get a random subset of indices the length of ``season_len``
            idx = np.sort(np.random.choice(idx_nnull, size=season_len, replace=False))

            if (add_nulls == 1) and (idx_nodata.shape[0] > 0):

                idx_nodata_sub = np.random.choice(idx_nodata, size=np.random.randint(2, high=5), replace=False)
                idx_take = np.array(idx.tolist() + idx_nodata_sub.tolist(), dtype='int64')

            else:
                idx_take = idx

            # Transpose each temporal state --> samples x features
            Xd_ = [dlayer.transpose(1, 2, 0).reshape(nrows * ncols, nbands) for dlayer in X_data[idx_take]]

            # Get a random subset of the grid indices
            if len(a) == 1:
                grid_len = np.random.randint(int(single_frac[0]*(nrows*ncols)), high=int(single_frac[1]*(nrows*ncols)))
            else:
                grid_len = np.random.randint(int(multi_frac[0]*(nrows*ncols)), high=int(multi_frac[1]*(nrows*ncols)))

            grid_idx = np.random.choice(range(0, nrows*ncols), size=grid_len)

            # Flatten the data from [time x features x rows x columns] --> [s1, s2, ..., sn]
            # len(Xd_) = n samples
            # len(Xd_[0]) = n time
            Xd_ = [array_to_dict(sensor,
                                 add_indices,
                                 scale_factor,
                                 transform,
                                 band_names,
                                 *[Xd_[j][i] for j in range(0, idx_take.shape[0])]) for i in grid_idx]

            if band_diffs:
                Xd_ = calc_band_diffs_nested(Xd_, keys_diff)

            # len(y_) = n samples
            # len(y_[0]) = n time
            y_ = [np.array(y_data)[idx_take].tolist() for i in grid_idx]

            Xc += Xd_
            yc += y_

    return Xc, yc


def sample_array(X,
                 y,
                 X_data,
                 y_data,
                 sensor,
                 class_names,
                 common_classes,
                 iso_classes,
                 class_sub_frac=None,
                 remove_classes=None,
                 null_class=None,
                 n_iters=10,
                 min_time=5,
                 max_time=10,
                 max_workers=1,
                 chunk_size=10,
                 add_indices=False,
                 scale_factor=0.0001,
                 transform=False,
                 band_names=None,
                 band_diffs=False,
                 single_frac=None,
                 multi_frac=None):

    """
    Samples arrays for CRF-formatted features

    Args:
        X (list): A list of predictor features.
        y (list): A list of class labels.
        X_data (ndarray): The X data.
        y_data (ndarray): The y data.
        sensor (str): The satellite sensor.
        class_names (list): The list of class labels.
        common_classes (list): The class labels common in all random generations.
        iso_classes (list): The class labels to be kept in isolation.
        class_sub_frac (dict): A dictionary of fractions to subsample.
        remove_classes (Optional[list]): Classes to remove from ``y_data``.
        null_class (Optional[str]): The null data class.
        n_iters (Optional[int]): The number of iterations.
        min_time (Optional[int]): The minimum sequence length.
        max_time (Optional[int]): The maximum sequence length.
        max_workers (Optional[int]): The maximum number of parallel workers.
        chunk_size (Optional[int]): The parallel task chunk size.
        add_indices (Optional[bool]): Whether to use indices with bands.
        scale_factor (Optional[float]): A scale factor to apply.
        transform (Optional[bool]): Whether to transform the bands.
        band_names (Optional[list]): A list of names. If not given, names are set as an integer range (1-n features).
        band_diffs (Optional[bool]): Whether use band first differences.
        single_frac (Optional[tuple | list]): A pair of sub-sample fractions for single-class lists.
        multi_frac (Optional[tuple | list]): A pair of sub-sample fractions for multi-class lists.

    Returns:
        ``tuple`` of ``lists``

    Example:
        >>> import rastercrf as rcrf
        >>>
        >>> X, y = [], []
        >>>
        >>> X_data, y_data = rcrf.extract_samples('/path/to/images',
        >>>                                       'l7',
        >>>                                       '/path/to/vector_file.gpkg')
        >>>
        >>> X, y = rcrf.sample_array(X, y, X_data, y_data, 'l7', ['l', 'w', 'c', 's', 'n'], ['c', 's'], ['l', 'w'])
    """

    if not single_frac:
        single_frac = (0.1, 0.25)

    if not multi_frac:
        multi_frac = (0.25, 0.5)

    ntime, nbands, nrows, ncols = X_data.shape

    if isinstance(y_data, list):
        y_data = np.array(y_data)

    if remove_classes:

        for rclass in remove_classes:

            r_idx = np.where(y_data != rclass)
            y_data = y_data[r_idx]
            X_data = X_data[r_idx]

    if null_class:
        idx_nodata = np.where(y_data == null_class)[0]
    else:
        idx_nodata = np.array([], dtype='int64')

    class_iters = [common_classes + iso for iso in iso_classes] + [[c] for c in class_names]

    df = pd.DataFrame(data=y_data, columns=['label'])

    if max_workers == 1:

        for iter_ in trange(0, n_iters):

            X_, y_ = sample_array_worker(X_data,
                                         y_data,
                                         class_iters,
                                         class_sub_frac,
                                         df,
                                         sensor,
                                         min_time,
                                         max_time,
                                         idx_nodata,
                                         nbands,
                                         nrows,
                                         ncols,
                                         add_indices,
                                         scale_factor,
                                         transform,
                                         band_names,
                                         band_diffs,
                                         single_frac,
                                         multi_frac)

            X += X_
            y += y_

    else:

        tasks = (X_data,
                 y_data,
                 class_iters,
                 class_sub_frac,
                 df,
                 sensor,
                 min_time,
                 max_time,
                 idx_nodata,
                 nbands,
                 nrows,
                 ncols,
                 add_indices,
                 scale_factor,
                 transform,
                 band_names,
                 band_diffs,
                 single_frac,
                 multi_frac)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:

            # Submit all of the tasks as futures
            futures = [executor.submit(sample_array_worker, *tasks) for iter_ in range(0, n_iters)]

            for f in tqdm(concurrent.futures.as_completed(futures), total=n_iters):

                X_, y_ = f.result()

                X += X_
                y += y_

    return X, y


def apply_sample_to_dict(ndx, fcols):
    return sample_to_dict(ndx, fcols)


def sample_dataframe_worker(df_list,
                            min_time,
                            max_time,
                            feature_cols,
                            labels_dict,
                            id_column,
                            shuffle,
                            band_diffs,
                            fractions):

    if band_diffs:

        ncols_ = len(feature_cols)

        keys_diff = []
        for k in range(ncols_+1, ncols_+1+ncols_*2):

            try:
                kstring = 'z.{:03d}'.format(int(k))
            except:
                kstring = 'z.{}'.format(k)

            keys_diff.append(kstring)

        # keys_diff = ['{:03d}'.format(int(k)).encode('utf-8') for k in range(ncols_+1, ncols_+1+ncols_*2)]
        # keys_diff = [str(int(k)).encode('utf-8') for k in range(ncols_+1, ncols_+1+ncols_*2)]

    if not fractions:

        if len(df_list) == 1:
            fractions = [1.0]
        elif len(df_list) == 2:

            auni = np.random.uniform(low=0.1, high=0.9, size=1)[0]
            buni = 1.0 - auni
            fractions = [auni, buni]

        elif len(df_list) == 3:

            auni = np.random.uniform(low=0.1, high=0.3, size=1)[0]
            buni = np.random.uniform(low=0.1, high=0.3, size=1)[0]
            cuni = 1.0 - auni + buni
            fractions = [auni, buni, cuni]

        else:
            raise AttributeError('It is recommended to use DataFrame lists of length 3 or less.')

    if min_time == max_time:
        n_time_ = max_time
    else:
        n_time_ = np.random.choice(range(min_time, max_time), size=1, replace=False)[0]

    if len(df_list) > 1:

        dfc_samples = None

        for frac_, dfc_ in zip(fractions, df_list):

            ssize = int(n_time_*frac_) if int(n_time_*frac_) <= dfc_.shape[0] else dfc_.shape[0]

            if not isinstance(dfc_samples, pd.DataFrame):
                dfc_samples = dfc_.sample(n=ssize, replace=False)
            else:

                dfc_samples = pd.concat((dfc_samples,
                                         dfc_.sample(n=ssize, replace=False)), axis=0)

        if band_diffs and (dfc_samples.shape[0] < 2):
            return [], []

        if shuffle:
            dfc_samples = dfc_samples.sample(frac=1)

    else:

        dfc_ = df_list[0]
        frac_ = fractions[0]

        ssize = int(n_time_ * frac_) if int(n_time_ * frac_) <= dfc_.shape[0] else dfc_.shape[0]

        dfc_samples = dfc_.sample(n=ssize, replace=False)

        if band_diffs and (dfc_samples.shape[0] < 2):
            return [], []

    X = dfc_samples[feature_cols].apply(apply_sample_to_dict, 
                                        axis=1, 
                                        args=(feature_cols,)).values.tolist()

    # Old list comprehension
    # X = [sample_to_dict(dfc_samples.iloc[i], feature_cols) for i in range(dfc_samples.shape[0])]

    dfc_labels = np.int64(dfc_samples.loc[:, id_column].values)

    y = [labels_dict[dfc_labels[i]] if dfc_labels[i] in labels_dict else 'null' for i in range(dfc_samples.shape[0])]

    if band_diffs:
        X = calc_band_diffs(X, keys_diff)

    return X, y


def sample_dataframe(X,
                     y,
                     df_list,
                     columns,
                     labels_dict,
                     id_column='response',
                     n_iters=10,
                     min_time=5,
                     max_time=10,
                     shuffle=False,
                     max_workers=1,
                     chunk_size=10,
                     band_diffs=False,
                     fractions=None):

    """
    Samples DataFrames for CRF-formatted features

    Args:
        X (list): A list of predictor features.
        y (list): A list of class labels.
        df_list (list): A list of ``pandas.DataFrames`` or ``geopandas.DataFrames``.
        columns (list): The column variable names.
        labels_dict (dict): The labels dictionary.
        id_column (Optional[str]): The id column.
        n_iters (Optional[int]): The number of iterations.
        min_time (Optional[int]): The minimum sequence length.
        max_time (Optional[int]): The maximum sequence length.
        shuffle (Optional[bool]): Whether to shuffle samples.
        max_workers (Optional[int]): The maximum number of parallel workers.
        chunk_size (Optional[int]): The parallel task chunk size.
        band_diffs (Optional[bool]): Whether use band first differences.
        fractions (Optional[list]): Label fractions.

    Returns:
        ``tuple`` of X and y as ``lists``

    Example:
        >>> import rastercrf as rcrf
        >>>
        >>> X = list()
        >>> y = list()
        >>>
        >>> dfa = df.query('response == 1')
        >>> dfb = df.query('response == 2')
        >>>
        >>> df_list = [dfa, dfb]
        >>> columns = ['a', 'b', 'c']
        >>> labels_dict = {1: b'y', 2: b'n'}
        >>>
        >>> X, y = rcrf.sample_dataframe(X, y, df_list, columns, labels_dict)
    """

    if max_workers == 1:

        for m in range(0, n_iters):

            X_, y_ = sample_dataframe_worker(df_list,
                                             min_time,
                                             max_time,
                                             columns,
                                             labels_dict,
                                             id_column,
                                             shuffle,
                                             band_diffs,
                                             fractions)

            if X_:

                X.append(X_)
                y.append(y_)

    else:

        tasks = ((df_list, min_time, max_time, columns, labels_dict, id_column, shuffle, band_diffs, fractions) for m in range(0, n_iters))

        cexec = Concurrency(sample_dataframe_worker, tasks, chunk_size, max_workers=max_workers)

        X, y = cexec.exec(X, y)

    return X, y
