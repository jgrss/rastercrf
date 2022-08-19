from pathlib import Path

import geowombat as gw

import numpy as np
import geopandas as gpd
from tqdm import tqdm


def extract_samples(raster_path,
                    sensor,
                    vector_path,
                    image_ext='',
                    drange=None):

    """
    Extracts samples from imagery

    Args:
        raster_path (str): The raster base directory path.
        sensor (str): The sensor.
        vector_path (str): The vector file path. The vector file should have columns 'label' and 'image'.
        image_ext (Optional[str]): The image extension.
        drange (Optional[tuple]): The raster dynamic range to clip to.

    Returns:
        ``tuple`` of 1-d ``numpy.ndarray`` of arrays shaped as [bands x rows x columns]

    Example:
        >>> import rastercrf as rcrf
        >>>
        >>> X_data, y_data = rcrf.extract_samples('/path/to/images',
        >>>                                       'l7',
        >>>                                       '/path/to/vector_file.gpkg')
    """

    rpath = Path(raster_path)

    minrow = 1e9
    mincol = 1e9

    X_data = list()
    y_data = list()

    df = gpd.read_file(vector_path)

    with gw.config.update(sensor=sensor):

        for row in tqdm(df.itertuples(index=True, name='Pandas'), total=df.shape[0]):

            rimage = rpath.joinpath(row.image + image_ext)

            if rimage.is_file():

                with gw.open(rimage.as_posix(), chunks=512) as src:

                    df_grid = df.query("index == {:d}".format(row.Index)).to_crs(src.crs)

                    clip = gw.clip(src,
                                   df_grid,
                                   mask_data=False)

                    if drange:
                        subset = clip.clip(drange[0], drange[1]).data.compute()
                    else:
                        subset = clip.data.compute()

                    minrow = min(subset.shape[1], minrow)
                    mincol = min(subset.shape[2], mincol)

                    X_data.append(subset)
                    y_data.append(row.label)

    return np.array([d[:, :minrow, :mincol] for d in X_data], dtype='float64'), np.array(y_data)
