import dask


def xarray_to_features(data, band_names, years, window, month=1, num_workers=1):

    """
    Transforms a multi-temporal ``xarray.DataArray`` to CRF features

    Args:
        data (DataArray): The ``xarray.DataArray`` to load and transform.
            The coordinates should be ['time', 'band', 'y', 'x'].
        band_names (list): The band names to load.
        years (list): The years to load.
        window (namedtuple): The window ``rasterio.windows.Window`` object.
        month (Optional[int]): The month to load.
        num_workers (Optional[int]): The number of parallel ``dask.compute`` workers.

    Example:
        >>> import rastercrf as rcrf
        >>> import geowombat as gw
        >>> from geowombat.core.windows import get_window_offsets
        >>>
        >>> clf = rcrf.CRFClassifier()
        >>>
        >>> columns = ['000', '001', '002']
        >>> labels = ['y', 'n']
        >>>
        >>> band_names = ['b', 'g', 'r']
        >>> time_names = ['2010', '2011', '2012']
        >>>
        >>> with gw.open([...], band_names=band_names, time_names=time_names) as src:
        >>>
        >>>     windows = get_window_offsets(src.gw.nrows,
        >>>                                  src.gw.ncols,
        >>>                                  src.gw.row_chunks,
        >>>                                  src.gw.col_chunks,
        >>>                                  return_as='list')
        >>>
        >>>     for w in windows:
        >>>
        >>>         # Transform features
        >>>         X = rcrf.xarray_to_features(src, band_names, time_names, w, num_workers=8)
        >>>
        >>>         # Make predictions
        >>>         probas = clf.predict_probas(X, columns, labels, w.height, w.width)
    """

    X_list = list()

    # Get the block
    data_slice = data.sel(band=band_names)[:, :, window.row_off:window.row_off + window.height, window.col_off:window.col_off + window.width]

    # Iterate over each year
    for year in years:

        data_period = data_slice.sel(time=slice('{YEAR:d}-{MONTH:02d}-1'.format(YEAR=int(year), MONTH=int(month)),
                                                '{YEAR:d}-{MONTH:02d}-1'.format(YEAR=int(year)+1, MONTH=int(month))))

        # Transpose to [samples x [time x bands]
        src_x = data_period.stack(s=('y', 'x')).stack(X=('time', 'band')).astype('float64').data

        X_list.append(src_x)

    # Get a list of the predictor features
    return dask.compute(X_list, num_workers=num_workers)[0]
