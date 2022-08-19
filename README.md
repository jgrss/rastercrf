[![MIT license](https://img.shields.io/badge/License-MIT-black.svg)](https://lbesson.mit-license.org/)
[![Python 3.6](https://img.shields.io/badge/python-3.x-black.svg)](https://www.python.org/downloads/release/python-360/)
![Package version](https://img.shields.io/badge/version-1.4.2-blue.svg?cacheSeconds=2592000)

### Conditional Random Fields for rasters

```python
>>> import rastercrf as rcrf
```

Load data

```python
>>> X_data, y_data, X_data_test, y_data_test, name_data = rcrf.load_data('south_america')
```

Get the full path of a model

```python
>>> model_name = rcrf.model_path('clouds_l7')
```

Sample data for model fitting

```python
>>> X, y = [], []
>>>
>>> X_data, y_data = rcrf.extract_samples('/path/to/images',
>>>                                       'l7',
>>>                                       '/path/to/vector_file.gpkg')
>>>
>>> X, y = rcrf.sample_array(X, y, X_data, y_data, 'l7', ['l', 'w', 'c', 's', 'n'], ['c', 's'], ['l', 'w'])
```

Fit a model and test

```python
>>> from sklearn_crfsuite import metrics
>>> from sklearn.model_selection import train_test_split
>>>
>>> clf = rcrf.CRFClassifier()
>>>
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5)
>>> clf.fit(X_train, y_train)
>>>
>>> print(metrics.flat_classification_report(y_test, y_pred, labels=['c', 's', 'l', 'w', 'n']))
```

Predict on an image

```python
>>> import geowombat as gw
>>> from geowombat.core.windows import get_window_offsets
>>>
>>> columns = ['000', '001', '002']
>>> labels = ['c', 's', 'l', 'w', 'n']
>>>
>>> band_names = ['b', 'g', 'r']
>>> time_names = ['2010', '2011', '2012']
>>>
>>> with gw.open([...], band_names=band_names, time_names) as src:
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
```
