from abc import ABC, abstractmethod
from pathlib import Path
import logging
import types
from collections import namedtuple
import itertools
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from .handler import logger
from . import util
from .transforms import labels_to_values
from .transforms import time_to_feas
from .transforms import time_to_sensor_feas
from .transforms import transform_probas
from .errors import TimeError

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import joblib
import sklearn_crfsuite
from tqdm import tqdm


class BaseMixin(ABC):

    @abstractmethod
    def tune(self, X, y, test_size=0.5, labels=None, num_cpus=1):

        """
        Tunes model parameters
        """

        raise NotImplementedError('Hyperparameter tuning is not available for this classifier.')

    def to_file(self, filename, overwrite=False):

        """
        Saves a model to file

        Args:
            filename (str | Path): The file to save to.
            overwrite (Optional[bool]): Whether to overwrite an existing model.
        """

        if isinstance(filename, Path):
            filename = str(filename)

        fpath = Path(filename)

        if overwrite:

            if fpath.is_file():
                fpath.unlink()

        try:

            if hasattr(self, 'verbose'):

                if self.verbose > 0:
                    logger.info('  Saving model to file ...')

            joblib.dump((self.model, self._columns, self._labels, self._func),
                        filename,
                        compress=True)

        except:
            logger.warning('  Could not dump the model to file.')

    def from_file(self, filename):

        """
        Loads a model from file

        Args:
            filename (str | Path): The model file to load.
        """

        if isinstance(filename, Path):
            filename = str(filename)

        if not Path(filename).is_file():
            logger.warning('  The model file does not exist.')
        else:

            if hasattr(self, 'verbose'):

                if self.verbose > 0:
                    logger.info('  Loading the model from file ...')

        # Function added in 1.2.12
        try:
            self.model, self._columns, self._labels, self._func = joblib.load(filename)
        except:
            self.model, self._columns, self._labels = joblib.load(filename)


class ModelMixin(BaseMixin):

    @property
    def columns(self):
        return self._columns

    @property
    def labels(self):
        return self._labels


def lstm_crf(n_season, n_feas, n_classes, learn_mode='join'):

    """
    Long short-term memory recurrent neural network
    """

    from keras.models import Model, Input
    from keras.layers import LSTM, Dense, TimeDistributed, Bidirectional
    import keras as k
    from keras_contrib.layers import CRF
    from keras_contrib.metrics import crf_accuracy
    from keras_contrib.losses import crf_loss

    input_layer = Input(shape=(n_season, n_feas),
                        dtype='float32',
                        name='Input')

    # Embedding Layer
    # embed_layer = Embedding(input_dim=n_feas,
    #                         output_dim=embedding_size,
    #                         input_length=n_feas)(input_layer)

    # BI-LSTM Layer
    bi_lstm1 = Bidirectional(LSTM(units=200,  # hidden units for state vector `a`
                                  return_sequences=True,
                                  dropout=0.5,
                                  recurrent_dropout=0.5,
                                  kernel_initializer=k.initializers.he_normal()),
                             name='Bi-dir1')(input_layer)

    bi_lstm2 = Bidirectional(LSTM(units=150,  # hidden units for state vector `a`
                                  return_sequences=True,
                                  dropout=0.5,
                                  recurrent_dropout=0.5,
                                  kernel_initializer=k.initializers.he_normal()),
                             name='Bi-dir2')(bi_lstm1)

    bi_lstm3 = Bidirectional(LSTM(units=100,  # hidden units for state vector `a`
                                  return_sequences=True,
                                  dropout=0.5,
                                  recurrent_dropout=0.5,
                                  kernel_initializer=k.initializers.he_normal()),
                             name='Bi-dir3')(bi_lstm2)

    # TimeDistributed Layer
    time_dist = TimeDistributed(Dense(n_classes,
                                      activation='softmax'),
                                name='Time')(bi_lstm3)

    # CRF Layer
    # output_layer = CRF(n_classes,
    #                    learn_mode=learn_mode,  # return class posteriors
    #                    sparse_target=False,     # the labels are labels, not indices or one-hot encoded
    #                    unroll=True,
    #                    name='CRF')(time_dist)

    model = Model(input_layer, time_dist)

    # Optimizer
    adam = k.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

    # Compile model
    # model.compile(optimizer=adam,
    #               loss=crf_loss,
    #               metrics=[crf_accuracy, 'accuracy'])

    model.compile(optimizer=adam,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


class LSTMCRFClassifier(object):

    def __init__(self, labels, max_season_len, batch_size=256, epochs=1000):

        self.labels = labels
        self.max_season_len = max_season_len
        self.batch_size = batch_size
        self.epochs = epochs

        self.word_to_int = {}
        self.word_to_int_byte = {}

        for i in range(0, len(labels)):
            self.word_to_int[labels[i]] = i
            self.word_to_int_byte[labels[i].encode()] = i

        self.model = None

    def transform_x(self, X):

        """
        Transforms a list of dictionaries into a padded sequence of arrays

        Args:
            X (list): A list of lists of dictionaries, where [[{}_a1, {}_a2, {}_a3]_a, ...]

                where,

                    len([]_a) = season length
                    len({}_a1) = feature length
                    {}_a1 = {b'var1': val1, b'var2': val2, ...}

                The output array would have the equivalent sample 1, time 1 of

                    out[0, 0, :] = X[0][0]

                and sample 1, time 2 of

                    out[0, 1, :] = X[0][1]

        Returns:
            ``numpy.ndarray`` shaped [samples x time x features]
        """

        from keras.preprocessing.sequence import pad_sequences

        X = [[list(xseg.values()) for xseg in x] for x in X]
        X = pad_sequences(maxlen=self.max_season_len, sequences=X, dtype='float32', padding='post', value=0.0)

        return X

    def fit(self, X, y, split_val=False, test_size=0.6, pad_class=None, learn_mode='join'):

        """
        Args:
            learn_mode (Optional[str]): Choices are ['join', 'marginal'].
        """

        # import tensorflow.keras.backend as K
        # from keras.callbacks import ModelCheckpoint
        from keras.preprocessing.sequence import pad_sequences
        from keras.utils import to_categorical
        # pip install git+https://www.github.com/keras-team/keras-contrib.git

        n_feas = len(X[0][0])
        n_classes = len(self.labels)

        X = self.transform_x(X)

        try:
            y = [[self.word_to_int[yseg] for yseg in ysub] for ysub in y]
        except:
            y = [[self.word_to_int_byte[yseg] for yseg in ysub] for ysub in y]

        y = pad_sequences(maxlen=self.max_season_len,
                          sequences=y,
                          dtype='int64',
                          padding='post',
                          value=self.word_to_int[pad_class])

        y = [to_categorical(i, num_classes=n_classes) for i in y]

        self.model = lstm_crf(self.max_season_len, n_feas, n_classes, learn_mode=learn_mode)

        # Saving the best model only
        # filepath = "ner-bi-lstm-td-model-{val_accuracy:.2f}.hdf5"

        # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        # callbacks_list = [checkpoint]

        if split_val:

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

            # Fit the best model
            self.model.fit(X_train,
                           np.array(y_train),
                           batch_size=self.batch_size,
                           epochs=self.epochs,
                           validation_data=[X_test, np.array(y_test)],
                           verbose=1)

        else:

            # Fit the best model
            self.model.fit(X,
                           np.array(y),
                           batch_size=self.batch_size,
                           epochs=self.epochs,
                           validation_split=0.1,
                           verbose=1)

    def predict_sensor_probas(self,
                              X,
                              sensor,
                              labels=None,
                              scale_factor=0.0001,
                              add_indices=False,
                              transform=False,
                              band_names=None,
                              band_diffs=False,
                              remove_nodata=False,
                              nodata_layer=-1,
                              verbose=0,
                              n_jobs=1):

        """
        Predicts class probabilities for a specific satellite sensor

        Args:
            X (4d array): The variables to use for predictions, shaped [time x bands x rows x columns].
            sensor (str): The satellite sensor.
            labels (Optional[list]): The class labels.
            scale_factor (Optional[float]): The scale factor to apply to `X`.
            add_indices (Optional[bool]): Whether to use indices with bands.
            transform (Optional[bool]): Whether to transform the bands.
            band_names (Optional[list]): A list of names. If not given, names are set as an integer range (1-n features).
            band_diffs (Optional[bool]): Whether to use band first differences.
            remove_nodata (Optional[bool]): Whether to remove 'no data' samples from the predictions.
            nodata_layer (Optional[int]): The 'no data' layer index. If -1, the 'no data' layer is taken as the
                last layer along axis 0. Valid samples should be 0.
            verbose (Optional[int]): The level of verbosity.
            n_jobs (Optional[int]): The number of parallel jobs.

        Returns:
            ``4d array`` of predictions, shaped as [time x classes x rows x columns].
        """

        if not labels:
            labels = self.labels

        class_labels = [label.encode('utf-8') for label in labels]

        nclasses = len(class_labels)

        ntime, nbands, nrows, ncols = X.shape

        if verbose > 1:
            logger.info('  Reshaping arrays ...')

        # Reshape from [bands, rows, columns] --> [rows x columns, bands] --> [time, samples, bands].
        feature_array = np.ascontiguousarray([tlayer.transpose(1, 2, 0).reshape(nrows * ncols,
                                                                                nbands)
                                              for tlayer in X], dtype='float64')

        feature_array[np.isnan(feature_array)] = 0.0

        if band_names:
            band_names_byte = [lab.encode('utf-8') for lab in band_names]
        else:
            band_names_byte = [str(lab).encode('utf-8') for lab in range(1, nbands+1)]

        if verbose > 0:
            logger.info('  Transforming the array into dictionaries for predictions ...')

        # Convert the array to lists of dictionaries
        features = time_to_sensor_feas(feature_array,
                                       sensor.encode('utf-8'),
                                       ntime,
                                       nbands,
                                       nrows,
                                       ncols,
                                       band_names_byte,
                                       scale_factor=scale_factor,
                                       add_indices=add_indices,
                                       transform=transform,
                                       band_diffs=band_diffs,
                                       remove_nodata=remove_nodata,
                                       nodata_layer=nodata_layer)

        return self.predict_probas(features, ntime, nrows, ncols, nclasses)
        # return self.predict(features, ntime, nrows, ncols)

    def predict_probas(self, X, ntime, nrows, ncols, nclasses):

        """
        Args:
            X (ndarray):
            ntime (int):
            nrows (int):
            ncols (int):
            nclasses (int):
        """

        if isinstance(X, np.ndarray):

            X = time_to_feas(np.ascontiguousarray(X, dtype='float64'),
                             [f'{i:02d}'.encode() for i in range(1, X[0].shape[1]+1)],
                             band_diffs=False)

        X = self.transform_x(X)

        y = self.model.predict(X)

        # Reshape to [time x classes x rows x columns].
        return np.ascontiguousarray(util.columns_to_nd(y,
                                                       ntime,
                                                       nrows,
                                                       ncols,
                                                       nlayers=nclasses), dtype='float64')

    def predict(self, X, ntime, nrows, ncols):

        X = self.transform_x(X)

        y = self.model.predict(X)

        y = np.uint8(y.argmax(axis=-1))

        # Reshape to [time x rows x columns].
        return np.ascontiguousarray(util.columns_to_nd(y,
                                                       ntime,
                                                       nrows,
                                                       ncols), dtype='uint8')

    def to_file(self, filename, overwrite=False):

        if overwrite:

            if Path(filename).is_file():
                Path(filename).unlink()

        self.model.save(str(filename))

    def from_file(self, filename):

        """
        https://github.com/keras-team/keras-contrib
        """

        import keras as k
        import keras_contrib

        self.model = k.models.load_model(str(filename), {'CRF': keras_contrib.layers.CRF,
                                                         'crf_accuracy': keras_contrib.metrics.crf_accuracy,
                                                         'crf_loss': keras_contrib.losses.crf_loss})


class LGBMClassifier(BaseEstimator, ModelMixin):

    def __init__(self, **kwargs):

        self._columns = None
        self._labels = None
        self._func = None

        self.model = lgb.LGBMClassifier(**kwargs)

    def tune(self, X, y, test_size=0.5, labels=None, num_cpus=1):

        import ray
        from ray import tune
        from sklearn import metrics

        ray.init(num_cpus=num_cpus)

        def trainable(config):

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

            params = dict(boosting_type='dart',
                          num_leaves=config['num_leaves'],
                          max_depth=config['max_depth'],
                          learning_rate=config['learning_rate'],
                          n_estimators=config['n_estimators'],
                          objective='multiclass',
                          n_jobs=1,
                          class_weight='balanced',
                          bagging_fraction=config['bagging_fraction'],
                          feature_fraction=config['feature_fraction'],
                          num_iterations=config['num_iterations'],
                          reg_alpha=config['reg_alpha'],
                          reg_lambda=config['reg_lambda'])

            clf = lgb.LGBMClassifier(**params)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            f1_score = metrics.f1_score(y_test, y_pred, average='weighted')

            tune.report(mean_accuracy=f1_score, ray='tune')

        search_space = {'num_leaves': tune.choice([10, 50, 100]),
                        'max_depth': tune.choice([10, 25, 50]),
                        'learning_rate': tune.uniform(0.001, 0.1),
                        'n_estimators': tune.choice([50, 100, 200]),
                        'bagging_fraction': tune.uniform(0.25, 0.75),
                        'feature_fraction': tune.uniform(0.25, 0.75),
                        'num_iterations': tune.choice([50, 100, 200]),
                        'reg_alpha': tune.uniform(0.001, 0.1),
                        'reg_lambda': tune.uniform(0.001, 0.1)}

        analysis = tune.run(trainable, config=search_space)

        ray.shutdown()

        return analysis.trial_dataframes

    @property
    def classes_(self):
        return self.model.classes_

    def fit(self, X, y, calibrate=False, **kwargs):

        if calibrate:

            if 'cv' not in kwargs:

                # Stratification object for calibrated cross-validation
                skf_cv = StratifiedShuffleSplit(n_splits=3,
                                                test_size=0.5,
                                                train_size=0.5)

                kwargs['cv'] = skf_cv

            self.model = CalibratedClassifierCV(self.model, **kwargs)

        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class CRFMixin(ModelMixin):

    def tune(self, X, y, test_size=0.5, labels=None, num_cpus=1, num_samples=5):

        """
        Tunes CRF hyperparameters
        
        Args:
            X (list): The predictors.
            y (list): The class labels.
            test_size (Optional[float]): The test fraction [0-1].
            labels (Optional[list]): The label names.
            num_cpus (Optional[int]): The number of concurrent CPUs.
            num_samples (Optional[int]): The number of trial samples.
            
        Returns:
            ``DataFrame``
        """
        
        import ray
        from ray import tune
        from sklearn_crfsuite import metrics

        if not isinstance(labels[0], bytes):
            labels = [lab.encode('utf-8') for lab in labels]

        ray.init(num_cpus=num_cpus)

        def trainable(config):

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

            # y_test = [[ylab.decode() for ylab in ylist] for ylist in y_test]

            params = dict(algorithm='lbfgs',
                          c1=config['c1'],
                          c2=config['c2'],
                          max_iterations=config['max_iterations'],
                          num_memories=config['num_memories'],
                          epsilon=config['epsilon'],
                          delta=config['delta'],
                          period=config['period'],
                          linesearch=config['linesearch'],
                          max_linesearch=config['max_linesearch'],
                          all_possible_states=config['all_possible_states'],
                          all_possible_transitions=config['all_possible_transitions'],
                          verbose=False)

            clf = sklearn_crfsuite.CRF(**params)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            if not isinstance(y_test[0][0], bytes):
                y_test = [[lab.encode() for lab in seq] for seq in y_test]

            if not isinstance(y_pred[0][0], bytes):
                y_pred = [[lab.encode() for lab in seq] for seq in y_pred]

            f1_score = metrics.flat_f1_score(y_test, y_pred, labels=labels, average='weighted')

            tune.report(mean_accuracy=f1_score, ray='tune')

        search_space = {'c1': tune.uniform(0.001, 0.1),
                        'c2': tune.uniform(0.001, 0.1),
                        'max_iterations': tune.choice([5000]),
                        'num_memories': tune.choice([3, 5, 10, 20, 30, 40, 50]),
                        'epsilon': tune.uniform(0.001, 0.1),
                        'delta': tune.uniform(0.001, 0.1),
                        'period': tune.choice([10, 20, 30, 40, 50]),
                        'max_linesearch': tune.choice([10, 20, 30, 40, 50]),
                        'all_possible_states': tune.choice([True, False]),
                        'all_possible_transitions': tune.choice([True, False]),
                        'linesearch': tune.choice(['MoreThuente'])}

        analysis = tune.run(trainable, 
                            num_samples=num_samples,
                            name='CRF-tune',
                            config=search_space)

        ray.shutdown()

        # Return the DataFrame of parameters with the highest accuracy
        return analysis.dataframe(metric='mean_accuracy', mode='max')

    def fit(self, X, y, columns=None, labels=None, func=None):

        """
        Fits a Conditional Random Fields classifier

        Args:
            X (list): The variables (list of dictionaries).
            y (list): The class labels (list of strings).
            columns (Optional[list]): A list of column names.
            labels (Optional[list]): A list of class labels.
            func (Optional[object]): A function applier.
        """

        self._columns = columns
        self._labels = labels
        self._func = func

        self.model.fit(X, y)

        return self

    def _setup(self, X, x_names, y_names, band_diffs, **func_kwargs):

        """
        Args:
            X (list): The data to predict on.
            x_names (list): A list of predictor attributes.
            y_names (list): A list of class names.
            band_diffs (bool): Whether to calculate band differences.
            **func_kwargs (Optional[dict]): Extra keyword arguments.
        """

        Vars = namedtuple('Vars', 'X_t class_labels nclasses ntime nbands')

        if x_names:
            check_ints = True
        else:
            x_names = [xlab.encode('utf-8') for xlab in self.model.attributes_]
            check_ints = False

        if not y_names:
            y_names = self.model.classes_

        # Apply a user function for additional features
        if isinstance(self._func, types.FunctionType):
            X = [self._func(x_sub_array, **func_kwargs) for x_sub_array in X]

        ntime = len(X)
        nattrs = len(x_names)

        for xi, x_state in enumerate(X):

            if x_state.shape[1] != nattrs:

                try:
                    logger.warning(f"  Feature columns: {','.join(x_names)}")
                except:
                    logger.warning(f"  Feature columns: {','.join([xlab.decode() for xlab in x_names])}")

                raise TimeError(xi, x_state.shape[1], nattrs)

        class_labels = [ylab.encode('utf-8') for ylab in y_names]

        if check_ints:
            x_names = util.check_x_names(X, x_names, band_diffs)

        X_t = time_to_feas(np.ascontiguousarray(X, dtype='float64'),
                           x_names,
                           band_diffs=band_diffs)

        return Vars(X_t=X_t,
                    class_labels=class_labels,
                    nclasses=len(class_labels),
                    ntime=ntime,
                    nbands=nattrs)

    def predict(self, X, nrows, ncols, labels_dict_r, x_names=None, y_names=None, band_diffs=False, **func_kwargs):

        """
        Predicts labels from features

        Args:
            X (4d array): The variables to use for predictions, shaped [time x bands x rows x columns].
            nrows (int): The number of rows in the image.
            ncols (int): The number of columns in the image.
            labels_dict_r (dict)
            x_names (Optional[list]): The X labels.
            y_names (Optional[list]): The y labels.
            band_diffs (Optional[bool])
        """

        x_names = x_names if x_names else self._columns

        var_names = self._setup(X, x_names, y_names, band_diffs, **func_kwargs)

        pred = np.array(self.model.predict(var_names.X_t), dtype=bytes).tolist()

        pred = np.array(labels_to_values(pred,
                                         labels_dict_r,
                                         len(pred),
                                         len(pred[0])), dtype='uint8')

        return np.uint8(util.columns_to_nd(pred, len(X), nrows, ncols))

    def _predict_and_transform(self,
                               X,
                               class_labels,
                               ntime,
                               nbands,
                               nrows,
                               ncols,
                               insert_nodata=False,
                               X_array=None,
                               verbose=0,
                               n_jobs=1):

        """
        Predicts classes and transforms from lists to an array

        Args:
            X (list): The data to predict on.
            class_labels (list): The class labels.
            ntime (int): The number of states.
            nbands (int): The number of bands, or features.
            nrows (int): The number of rows.
            ncols (int): The number of columns.
            insert_nodata (Optional[bool])
            X_array (Optional[ndarray]): The feature array [time x samples x bands]. Used to fill in 'no data'
                if 'no data' values were removed for predictions.
            verbose (Optional[int]): The level of verbosity.
            n_jobs (Optional[int]): The number of parallel jobs.

        Returns:
            ``ndarray``
        """

        if not insert_nodata:
            X_array = np.array([[[1]]], dtype='float64')

        if verbose > 1:
            logger.info('  Predicting marginal probabilities ...')

        # Predict class probabilities and translate
        probas = util.dict_keys_to_bytes(self.model.predict_marginals(X))

        if verbose > 1:
            logger.info('  Transforming probabilities into an array ...')

        # Convert from list of dictionaries to array
        return transform_probas(probas,
                                X_array,
                                class_labels,
                                len(class_labels),
                                ntime,
                                nbands,
                                nrows,
                                ncols,
                                insert_nodata=insert_nodata)

    def predict_probas(self,
                       X,
                       nrows,
                       ncols,
                       x_names=None,
                       y_names=None,
                       band_diffs=False,
                       group_probas=None,
                       **func_kwargs):

        """
        Predicts probabilities for a customized dataset

        Args:
            X (list): A list of 2d arrays shaped [(rows x columns) x bands].
            nrows (int): The number of rows in the image.
            ncols (int): The number of columns in the image.
            x_names (Optional[list]): The X labels.
            y_names (Optional[list]): The y labels.
            band_diffs (Optional[bool]): Whether to apply band differencing.
            group_probas (Optional[list]): A list of class probabilities to group.

        Returns:
            ``4d array`` of predictions, shaped as [time x classes x rows x columns].
        """

        x_names = x_names if x_names else self._columns

        if group_probas:

            # Locate the label indices
            group_indices = np.array([y_names.index(g) for g in group_probas], dtype='int64')

        var_names = self._setup(X, x_names, y_names, band_diffs, **func_kwargs)

        if group_probas:

            probas = self._predict_and_transform(var_names.X_t, var_names.class_labels, var_names.ntime, var_names.nbands, nrows, ncols)

            # Sum over the probability groups
            probas[:, group_indices[0], :, :] = probas[:, group_indices, :, :].sum(axis=1)

            # Set all group probabilities except 1 to 0
            probas[:, group_indices[1:], :, :] = 0

        else:
            probas = self._predict_and_transform(var_names.X_t, var_names.class_labels, var_names.ntime, var_names.nbands, nrows, ncols)

        return probas

    def predict_sensor_probas(self,
                              X,
                              sensor,
                              labels=None,
                              scale_factor=0.0001,
                              add_indices=False,
                              transform=False,
                              band_names=None,
                              band_diffs=False,
                              remove_nodata=False,
                              nodata_layer=-1,
                              verbose=0,
                              n_jobs=1):

        """
        Predicts class probabilities for a specific satellite sensor

        Args:
            X (4d array): The variables to use for predictions, shaped [time x bands x rows x columns].
            sensor (str): The satellite sensor.
            labels (Optional[list]): The class labels.
            scale_factor (Optional[float]): The scale factor to apply to `X`.
            add_indices (Optional[bool]): Whether to use indices with bands.
            transform (Optional[bool]): Whether to transform the bands.
            band_names (Optional[list]): A list of names. If not given, names are set as an integer range (1-n features).
            band_diffs (Optional[bool]): Whether to use band first differences.
            remove_nodata (Optional[bool]): Whether to remove 'no data' samples from the predictions.
            nodata_layer (Optional[int]): The 'no data' layer index. If -1, the 'no data' layer is taken as the
                last layer along axis 0. Valid samples should be 0.
            verbose (Optional[int]): The level of verbosity.
            n_jobs (Optional[int]): The number of parallel jobs.

        Returns:
            ``4d array`` of predictions, shaped as [time x classes x rows x columns].
        """

        if not labels:
            labels = self.labels

        class_labels = [label.encode('utf-8') for label in labels]

        ntime, nbands, nrows, ncols = X.shape

        if verbose > 1:
            logger.info('  Reshaping arrays ...')

        # Reshape from [bands, rows, columns] --> [rows x columns, bands] --> [time, samples, bands].
        feature_array = np.ascontiguousarray([tlayer.transpose(1, 2, 0).reshape(nrows * ncols,
                                                                                nbands)
                                              for tlayer in X], dtype='float64')

        feature_array[np.isnan(feature_array)] = 0.0

        if band_names:
            band_names_byte = [lab.encode('utf-8') for lab in band_names]
        else:
            band_names_byte = [str(lab).encode('utf-8') for lab in range(1, nbands+1)]

        if verbose > 0:
            logger.info('  Transforming the array into dictionaries for predictions ...')

        # Convert the array to lists of dictionaries
        features = time_to_sensor_feas(feature_array,
                                       sensor.encode('utf-8'),
                                       ntime,
                                       nbands,
                                       nrows,
                                       ncols,
                                       band_names_byte,
                                       scale_factor=scale_factor,
                                       add_indices=add_indices,
                                       transform=transform,
                                       band_diffs=band_diffs,
                                       remove_nodata=remove_nodata,
                                       nodata_layer=nodata_layer)

        if verbose > 0:
            logger.info('  Predicting class labels ...')

        return self._predict_and_transform(features,
                                           class_labels,
                                           ntime,
                                           nbands,
                                           nrows,
                                           ncols,
                                           insert_nodata=remove_nodata,
                                           X_array=feature_array,
                                           verbose=verbose,
                                           n_jobs=n_jobs)


class CRFClassifier(CRFMixin):

    """
    A class for Conditional Random Fields classification

    Example keyword arguments:

        algorithm='lbfgs',
        c1=0,
        c2=1,
        max_iterations=2000,
        num_memories=20,
        epsilon=1e-5,
        delta=1e-5,
        period=20,
        linesearch='MoreThuente',
        max_linesearch=20,
        all_possible_states=True,
        all_possible_transitions=True,
        verbose=False)
    """

    def __init__(self, **kwargs):

        self._columns = None
        self._labels = None
        self._func = None

        self.model = sklearn_crfsuite.CRF(**kwargs)
