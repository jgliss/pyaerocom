#!/usr/bin/env python3
# -*- coding: utf-8 -*-
########################################################################
#
# This python module is part of the pyaerocom software
#
# License: GNU General Public License v3.0
# More information: https://github.com/metno/pyaerocom
# Documentation: https://pyaerocom.readthedocs.io/en/latest/
# Copyright (C) 2017 met.no
# Contact information: Norwegian Meteorological Institute (MET Norway)
#
########################################################################

import os, logging
from pathlib import Path
from pyaerocom.combine_vardata_ungridded import combine_vardata_ungridded
from pyaerocom.exceptions import (DataRetrievalError,
                                  NetworkNotImplemented, NetworkNotSupported)

from pyaerocom.io.read_aeronet_sdav2 import ReadAeronetSdaV2
from pyaerocom.io.read_aeronet_sdav3 import ReadAeronetSdaV3
from pyaerocom.io.read_aeronet_invv2 import ReadAeronetInvV2
from pyaerocom.io.read_aeronet_invv3 import ReadAeronetInvV3
from pyaerocom.io.read_aeronet_sunv2 import ReadAeronetSunV2
from pyaerocom.io.read_aeronet_sunv3 import ReadAeronetSunV3
from pyaerocom.io.read_earlinet import ReadEarlinet
from pyaerocom.io.read_ebas import ReadEbas
from pyaerocom.io.read_aasetal import ReadAasEtal
from pyaerocom.io.read_gaw import ReadGAW
from pyaerocom.io.read_ghost import ReadGhost
from pyaerocom.io.read_eea_aqerep import ReadEEAAQEREP
from pyaerocom.io.read_eea_aqerep_v2 import ReadEEAAQEREP_V2
from pyaerocom.io.read_airnow import ReadAirNow

from pyaerocom.io.cachehandler_ungridded import CacheHandlerUngridded
from pyaerocom.ungriddeddata import UngriddedData
from pyaerocom.helpers import varlist_aerocom

from pyaerocom import const, print_log

class ReadUngridded(object):
    """Factory class for reading of ungridded data based on obsnetwork ID

    This class also features reading functionality that goes beyond reading
    of inidividual observation datasets; including, reading of multiple
    datasets and post computation of new variables based on datasets that can
    be read.
    """
    SUPPORTED_READERS = [ReadAeronetInvV3,
                         ReadAeronetInvV2,
                         ReadAeronetSdaV2,
                         ReadAeronetSdaV3,
                         ReadAeronetSunV2,
                         ReadAeronetSunV3,
                         ReadEarlinet,
                         ReadEbas,
                         ReadGAW,
                         ReadAasEtal,
                         ReadGhost,
                         ReadAirNow,
                         ReadEEAAQEREP,
                         ReadEEAAQEREP_V2]

                         

    DONOTCACHE_NAME = 'DONOTCACHE'
    def __init__(self, datasets_to_read=None, vars_to_retrieve=None,
                 ignore_cache=False, data_dir=None):

        #will be assigned in setter method of dataset_to_read
        self._datasets_to_read = []
        self._vars_to_retrieve = None
        self._data_dir = {}

        #: dictionary containing reading classes for each dataset to read (will
        #: be filled in setter of datasets_to_read)
        self._readers = {}

        if datasets_to_read is not None:
            self.datasets_to_read = datasets_to_read

        if vars_to_retrieve is not None:
            self.vars_to_retrieve = vars_to_retrieve

        # initiate a logger for this class
        self.logger = logging.getLogger(__name__)

        if data_dir is not None:
            self.data_dir=data_dir

        if vars_to_retrieve is not None:
            self.vars_to_retrieve = vars_to_retrieve

        if ignore_cache:
            self.logger.info('Deactivating caching')
            const.CACHING = False

    @property
    def data_dir(self):
        """Data directory(ies) for dataset(s) to read"""
        return self._data_dir

    @data_dir.setter
    def data_dir(self, val):
        if isinstance(val, Path):
            val = str(val)
        dsr = self.datasets_to_read
        if len(dsr) < 2 and isinstance(val, str):
            val = {dsr[0] : val}
        elif not isinstance(val, dict):
            raise ValueError('Invalid input data_dir ({}); needs to be a '
                             'dictionary for each dataset that is supposed to '
                             'be read ({})'.format(val, dsr))

        for ds, data_dir in val.items():
            assert os.path.exists(data_dir), f'{data_dir} does not exist'
        self._data_dir = val

    @property
    def vars_to_retrieve(self):
        """Variables to retrieve (list or dict)

        Dictionary can be used in case different variables from multiple
        datasets are supposed to be read.
        """
        return self._vars_to_retrieve

    @vars_to_retrieve.setter
    def vars_to_retrieve(self, val):
        if isinstance(val, str):
            val = [val]
        elif not isinstance(val, (list, dict)):
            raise ValueError('Invalid input for vars_to_retrieve ({}). '
                             'Need either str, list or dict')
        self._vars_to_retrieve = val

    @property
    def DATASET_PATH(self):
        """Data directory of dataset to read

        Raises exception if more than one dataset to read is specified
        """
        if not len(self._datasets_to_read) == 1:
            raise ValueError('Conflict: multiple datasets are assigned to be '
                             'read, but dataset path can only be retrieved for '
                             'single dataset retrievals')
        return self.get_reader(self._datasets_to_read[0]).DATASET_PATH

    def _check_donotcachefile(self):
        """Check if donotcache file exists

        Returns
        -------
        bool
            True if file exists, else False
        """
        try:
            if os.path.exists(os.path.join(const.cache_basedir,
                                           self.DONOTCACHE_NAME)):
                return True
        except:
            pass
        return False

    @property
    def ignore_cache(self):
        """Boolean specifying whether caching is active or not"""
        if self._check_donotcachefile() or not const.CACHING:
            return True
        return False

    @property
    def dataset_to_read(self):
        """Helper that returns the dataset to be read

        Note
        ----
        Only works if a single dataset is assigned in :attr:`datasets_to_read`,
        else throws an ValueError.

        Raises
        ------
        ValueError
            if :attr:`datasets_to_read` contains no or more than one entry.
        """
        dsr = self.datasets_to_read
        if len(dsr) == 0:
            raise ValueError('Could not fetch reader class. No dataset '
                             'assigned in attr. datasets_to_read')
        elif len(dsr) > 1:
            raise ValueError('Could not fetch reader class. More than one '
                             'dataset is assigned in attr. '
                             'datasets_to_read')
        return dsr[0]

    @property
    def datasets_to_read(self):
        """List of datasets supposed to be read"""
        return self._datasets_to_read

    @datasets_to_read.setter
    def datasets_to_read(self, datasets):
        if isinstance(datasets, str):
            datasets = [datasets]
        elif not isinstance(datasets, (tuple, list)):
            raise IOError('Invalid input for parameter datasets_to_read')
        avail = []
        for ds in datasets:
            if ds in const.OBS_UNGRIDDED_POST: # no reader available
                avail.append(ds)
            else:
                try:
                    self.find_read_class(ds)
                    avail.append(ds)
                except NetworkNotSupported:
                    print_log.warning('Removing {} from list of datasets to read '
                                      'in ReadUngridded class. Reason: network '
                                      'not supported or data is not available'
                                      .format(ds))
        self._datasets_to_read = avail

    def dataset_provides_variables(self, dataset_to_read=None):
        """List of variables provided by a certain dataset"""
        if dataset_to_read is None:
            dataset_to_read = self.dataset_to_read
        if dataset_to_read in self._readers:
            return self._readers[dataset_to_read].PROVIDES_VARIABLES
        return self.find_read_class(dataset_to_read).PROVIDES_VARIABLES

    def get_reader(self, dataset_to_read=None):
        """Helper method that returns loaded reader class

        Parameters
        -----------
        dataset_to_read : str
            Name of dataset

        Returns
        -------
        ReadUngriddedBase
            instance of reading class (needs to be implementation of base
            class :class:`ReadUngriddedBase`)

        Raises
        ------
        NetworkNotSupported
            if network is not supported by pyaerocom
        NetworkNotImplemented
            if network is supported but no reading routine is implemented yet
        """
        if dataset_to_read is None:
            dataset_to_read = self.dataset_to_read
        elif not dataset_to_read in self.supported_datasets:
            raise NetworkNotSupported('Could not fetch reader class: Input '
                                      'network {} is not supported by '
                                      'ReadUngridded'.format(dataset_to_read))
        if not dataset_to_read in self._readers:
            self.find_read_class(dataset_to_read)
        return self._readers[dataset_to_read]

    def find_read_class(self, dataset_to_read):
        """Find reading class for dataset name

        Loops over all reading classes available in :attr:`SUPPORTED_READERS`
        and finds the first one that matches the input dataset name, by
        checking the attribute :attr:`SUPPORTED_DATASETS` in each respective
        reading class.

        Parameters
        -----------
        dataset_to_read : str
            Name of dataset

        Returns
        -------
        ReadUngriddedBase
            instance of reading class (needs to be implementation of base
            class :class:`ReadUngriddedBase`)

        Raises
        ------
        NetworkNotSupported
            if network is not supported by pyaerocom
        NetworkNotImplemented
            if network is supported but no reading routine is implemented yet

        """
        if dataset_to_read in self._readers:
            return self._readers[dataset_to_read]

        for _cls in self.SUPPORTED_READERS:
            if dataset_to_read in _cls.SUPPORTED_DATASETS:
                try:
                    self._readers[dataset_to_read] = _cls(dataset_to_read)
                except Exception as e:
                    raise DataRetrievalError('Failed to instantiate reader for '
                                             'dataset {}. Reason: {}'
                                             .format(dataset_to_read, repr(e)))
                return self._readers[dataset_to_read]
        raise NetworkNotImplemented('Could not find reading class for dataset '
                                    '{}'.format(dataset_to_read))

    def _get_data_dir(self, dataset_to_read):
        """Helper to retrieve data directory for a given dataset (if available)

        Parameters
        ----------
        dataset_to_read : str
            ID of dataset to be imported

        Returns
        -------
        str or None
            directory path if a data directory is specified, else None.
        """
        data_dir = self.data_dir
        if isinstance(data_dir, str):
            return data_dir
        elif isinstance(data_dir, dict) and dataset_to_read in data_dir:
            return data_dir[dataset_to_read]
        return None

    def _get_vars_to_retrieve(self, dataset_to_read):
        """Helper to retrieve variables to be read for a given dataset

        Parameters
        ----------
        dataset_to_read : str
            ID of dataset to be imported

        Raises
        ------
        DataRetrievalError
            if :attr:`vars_to_retrieve` is a dictionary (i.e., dataset
            specific) and the input data ID is not specified therein.

        Returns
        -------
        list
            list of variable to be retrieved
        """
        vtr = self.vars_to_retrieve
        if isinstance(vtr, list):
            return vtr
        #vars_to_retrieve is a dict (dataset specific)
        if not dataset_to_read in vtr:
            raise DataRetrievalError(
                'Missing specification of vars_to_retrieve for {}'
                .format(dataset_to_read))
        return vtr[dataset_to_read]

    def read_dataset(self, dataset_to_read, vars_to_retrieve=None,
                     only_cached=False, filter_post=None, **kwargs):
        """Read dataset into an instance of :class:`ReadUngridded`

        Parameters
        ----------
        dataset_to_read : str
            name of dataset
        vars_to_retrieve : str or list
            variable or list of variables to be imported
        only_cached : bool
            if True, then nothing is reloaded but only data is loaded that is
            available as cached objects (not recommended to use but may be
            used if working offline without connection to database)
        filter_post : dict, optional
            filters applied to `UngriddedData` object AFTER it is read into
            memory, via :func:`UngriddedData.apply_filters`. This option was
            introduced in pyaerocom version 0.10.0 and should be used
            preferably over **kwargs. There is a certain flexibility with
            respect to how these filters can be defined, for instance, sub
            dicts for each `dataset_to_read`. The most common way would be
            to provide directly the input needed for
            `UngriddedData.apply_filters`. If you want to read multiple variables
            from one or more datasets, and if you want to apply variable
            specific filters, it is recommended to read the data individually
            for each variable and corresponding set of filters and then
            merge the individual filtered `UngriddedData` objects afterwards,
            e.g. using `data_var1 & data_var2`.
        **kwargs
            Additional input options for reading of data, which are applied
            WHILE the data is read. If any such additional options are
            provided that are applied during the reading, then automatic
            caching of the output `UngriddedData` object will be deactivated.
            Thus, it is recommended to handle data filtering via `filter_post`
            argument whenever possible, which will result in better performance
            as the unconstrained original data is read in and cached, and then
            the filtering is applied.

        Returns
        --------
        UngriddedData
            data object
        """
        _caching = None
        if len(kwargs) > 0:
            _caching = const.CACHING
            const.CACHING = False

            print_log.info('Received additional reading constraints, '
                           'ignoring caching')

        reader = self.get_reader(dataset_to_read)

        if vars_to_retrieve is not None:
            # Note: self.vars_to_retrieve may be None as well, then
            # default variables of each network are read
            self.vars_to_retrieve = vars_to_retrieve

        if self.vars_to_retrieve is None:
            self.vars_to_retrieve = reader.PROVIDES_VARIABLES

        vars_to_retrieve = varlist_aerocom(self.vars_to_retrieve)

        # data_dir will be None in most cases, but can be specified when
        # creating the instance, by default, data_dir is inferred automatically
        # in the reading class, using database location
        data_dir = self._get_data_dir(dataset_to_read)
        if data_dir is not None:
            if not os.path.exists(data_dir):
                raise FileNotFoundError(
                    'Trying to read {} from specified data_dir {} failed. '
                    'Directory does not exist'.format(dataset_to_read, data_dir)
                    )
            reader._dataset_path = data_dir
            const.print_log.info('Reading {} from specified data loaction: {}'
                                 .format(dataset_to_read, data_dir))

        # Since this interface enables to load multiple datasets, each of
        # which support a number of variables, here, only the variables are
        # considered that are supported by the dataset
        vars_available = [var for var in vars_to_retrieve if
                          reader.var_supported(var)]
        if len(vars_available) == 0:
            raise DataRetrievalError('None of the input variables ({}) is '
                                     'supported by {} interface'
                                     .format(vars_to_retrieve, dataset_to_read))
        cache = CacheHandlerUngridded(reader)
        if not self.ignore_cache:
            # initate cache handler
            for var in vars_available:
                try:
                    cache.check_and_load(var,
                                         force_use_outdated=only_cached)
                except Exception:
                    self.logger.exception('Fatal: compatibility error between '
                                          'old cache file {} and current version '
                                          'of code ')

        if not only_cached:
            vars_to_read = [v for v in vars_available if not v in cache.loaded_data]
        else:
            vars_to_read = []

        data_read = None
        if len(vars_to_read) > 0:

            _loglevel = print_log.level
            print_log.setLevel(logging.INFO)
            data_read = reader.read(vars_to_read, **kwargs)
            print_log.setLevel(_loglevel)

            for var in vars_to_read:
                # write the cache file
                if not self.ignore_cache:
                    try:
                        cache.write(data_read, var)
                    except Exception as e:
                        _caching = False
                        print_log.warning('Failed to write to cache directory. '
                                          'Error: {}. Deactivating caching in '
                                          'pyaerocom'.format(repr(e)))

        if len(vars_to_read) == len(vars_available):
            data_out = data_read
        else:
            data_out = UngriddedData()
            for var in vars_available:
                if var in cache.loaded_data:
                    data_out.append(cache.loaded_data[var])
            if data_read is not None:
                data_out.append(data_read)

        if _caching is not None:
            const.CACHING = _caching

        if filter_post is not None:
            filters = self._eval_filter_post(filter_post,
                                             dataset_to_read,
                                             vars_available)
            if not isinstance(filter_post, dict):
                raise ValueError(
                    f'Invalid input for filter_post in ReadUngridded: '
                    f'{filter_post}. Need dictionary.')
            data_out = data_out.apply_filters(**filters)
        return data_out

    def _eval_filter_post(self, filter_post, dataset_to_read, vars_available):
        if not isinstance(filter_post, dict):
            raise ValueError(f'input filter_post must be dict, got '
                             f'{type(filter_post)}')

        if dataset_to_read in filter_post:
            # filters are specified specifically for that dataset
            subset = filter_post[dataset_to_read]
            return self._eval_filter_post(subset,
                                          dataset_to_read,
                                          vars_available)
        filters = {}
        for key, val in filter_post.items():
            if key == 'ignore_station_names':
                if isinstance(val, (str, list)):
                    filters['station_name'] = val
                    if not 'negate' in filters:
                        filters['negate'] = []
                    filters['negate'].append('station_name')

                elif isinstance(val, dict): #variable specific station filtering
                    if len(vars_available) > 1:
                        raise NotImplementedError(
                            f'Cannot filter different sites for multivariable '
                            f'UngriddedData objects (i.e. apply filter '
                            f'ignore_station_names={val} for UngriddedData '
                            f'object containing {vars_available}')
                    else:
                        # the variable that is available in the UngriddedData
                        # object
                        var = vars_available[0]
                        try:
                            filters['station_name'] = val[var]
                            if not 'negate' in filters:
                                filters['negate'] = []
                            filters['negate'].append('station_name')
                        except KeyError:
                            continue
                else:
                    raise ValueError(f'Invalid input for ignore_station_names: {val}')
            else:
                filters[key] = val
        return filters

    def read_dataset_post(self, dataset_to_read, vars_to_retrieve,
                          only_cached=False, filter_post=None, **kwargs):
        """Read dataset into an instance of :class:`ReadUngridded`

        Parameters
        ----------
        dataset_to_read : str
            name of dataset
        vars_to_retrieve : list
            variable or list of variables to be imported
        only_cached : bool
            if True, then nothing is reloaded but only data is loaded that is
            available as cached objects (not recommended to use but may be
            used if working offline without connection to database)
        filter_post : dict, optional
            filters applied to `UngriddedData` object AFTER it is read into
            memory, via :func:`UngriddedData.apply_filters`. This option was
            introduced in pyaerocom version 0.10.0 and should be used
            preferably over **kwargs. There is a certain flexibility with
            respect to how these filters can be defined, for instance, sub
            dicts for each `dataset_to_read`. The most common way would be
            to provide directly the input needed for
            `UngriddedData.apply_filters`. If you want to read multiple variables
            from one or more datasets, and if you want to apply variable
            specific filters, it is recommended to read the data individually
            for each variable and corresponding set of filters and then
            merge the individual filtered `UngriddedData` objects afterwards,
            e.g. using `data_var1 & data_var2`.
        **kwargs
            Additional input options for reading of data, which are applied
            WHILE the data is read. If any such additional options are
            provided that are applied during the reading, then automatic
            caching of the output `UngriddedData` object will be deactivated.
            Thus, it is recommended to handle data filtering via `filter_post`
            argument whenever possible, which will result in better performance
            as the unconstrained original data is read in and cached, and then
            the filtering is applied.

        Returns
        --------
        UngriddedData
            data object
        """
        aux_info = self.post_compute[dataset_to_read]
        loaded = []
        for var in vars_to_retrieve:
            input_data_ids_vars = []
            aux_info_var = aux_info['aux_requires'][var]
            for aux_id, aux_vars in aux_info_var.items():
                if aux_id in self.post_compute:
                    aux_data = self.read_dataset_post(
                                    dataset_to_read=aux_id,
                                    vars_to_retrieve=aux_vars,
                                    only_cached=only_cached,
                                    **kwargs)
                    for aux_var in aux_vars:
                        input_data_ids_vars.append((aux_data, aux_id, aux_var))
                else:

                    # read variables individually, so filter_post is more
                    # flexible if some post filters are specified for
                    # individual variables...
                    for aux_var in aux_vars:
                        _data = self.read_dataset(
                            aux_id,
                            aux_var,
                            only_cached=only_cached,
                            filter_post=filter_post,
                            **kwargs)
                        input_data_ids_vars.append((_data, aux_id, aux_var))


            aux_merge_how = aux_info['aux_merge_how'][var]

            if var in aux_info['aux_units']:
                var_unit_out = aux_info['aux_units'][var]
            else:
                var_unit_out = None

            if aux_merge_how == 'eval':
                # function MUST be defined
                aux_fun = aux_info['aux_funs'][var]
            else:
                aux_fun = None

            merged_stats = combine_vardata_ungridded(
                data_ids_and_vars=input_data_ids_vars,
                merge_eval_fun=aux_fun,
                merge_how=aux_merge_how,
                var_name_out=var,
                var_unit_out=var_unit_out,
                data_id_out=aux_info['data_id'])
            loaded.append(UngriddedData.from_station_data(merged_stats))
        first = loaded[0]
        if len(loaded) == 1:
            return first
        for data in loaded[1:]:
            first.append(data)
        return first

    def read(self, datasets_to_read=None, vars_to_retrieve=None,
             only_cached=False, filter_post=None, **kwargs):
        """Read observations

        Iter over all datasets in :attr:`datasets_to_read`, call
        :func:`read_dataset` and append to data object

        Parameters
        ----------
        datasets_to_read : str or list
            data ID or list of all datasets to be imported
        vars_to_retrieve : str or list
            variable or list of variables to be imported
        only_cached : bool
            if True, then nothing is reloaded but only data is loaded that is
            available as cached objects (not recommended to use but may be
            used if working offline without connection to database)
        filter_post : dict, optional
            filters applied to `UngriddedData` object AFTER it is read into
            memory, via :func:`UngriddedData.apply_filters`. This option was
            introduced in pyaerocom version 0.10.0 and should be used
            preferably over **kwargs. There is a certain flexibility with
            respect to how these filters can be defined, for instance, sub
            dicts for each `dataset_to_read`. The most common way would be
            to provide directly the input needed for
            `UngriddedData.apply_filters`. If you want to read multiple variables
            from one or more datasets, and if you want to apply variable
            specific filters, it is recommended to read the data individually
            for each variable and corresponding set of filters and then
            merge the individual filtered `UngriddedData` objects afterwards,
            e.g. using `data_var1 & data_var2`.
        **kwargs
            Additional input options for reading of data, which are applied
            WHILE the data is read. If any such additional options are
            provided that are applied during the reading, then automatic
            caching of the output `UngriddedData` object will be deactivated.
            Thus, it is recommended to handle data filtering via `filter_post`
            argument whenever possible, which will result in better performance
            as the unconstrained original data is read in and cached, and then
            the filtering is applied.

        Example
        -------
        >>> import pyaerocom.io.readungridded as pio
        >>> from pyaerocom import const
        >>> obj = pio.ReadUngridded(dataset_to_read=const.AERONET_SUN_V3L15_AOD_ALL_POINTS_NAME)
        >>> obj.read()
        >>> print(obj)
        >>> print(obj.metadata[0.]['latitude'])

        """
        if datasets_to_read is not None:
            self.datasets_to_read = datasets_to_read
        if vars_to_retrieve is not None:
            self.vars_to_retrieve = vars_to_retrieve

        data = UngriddedData()
        for ds in self.datasets_to_read:
            read_vars = self._get_vars_to_retrieve(ds)
            self.logger.info('Reading {} data, variables: {}'
                             .format(ds, read_vars))
            if ds in self.post_compute:
                data.append(self.read_dataset_post(ds, read_vars,
                                                   only_cached=only_cached,
                                                   filter_post=filter_post,
                                                   **kwargs))
            else:
                data.append(self.read_dataset(ds, read_vars,
                                              only_cached=only_cached,
                                              filter_post=filter_post,
                                              **kwargs))

            self.logger.info('Successfully imported {} data'.format(ds))
        return data

    @property
    def post_compute(self):
        """Information about datasets that can be computed in post"""
        return const.OBS_UNGRIDDED_POST

    @property
    def SUPPORTED_DATASETS(self):
        """
        Returns list of strings containing all supported dataset names
        """
        lst = []
        for reader in self.SUPPORTED_READERS:
            lst.extend(reader.SUPPORTED_DATASETS)
        lst.extend(self.post_compute)
        return lst

    @property
    def supported_datasets(self):
        """
        Wrapper for :attr:`SUPPORTED_DATASETS`
        """
        return self.SUPPORTED_DATASETS

    def get_vars_supported(self, obs_id, vars_desired):
        """
        Filter input list of variables by supported ones for a certain data ID

        Parameters
        ----------
        obs_id : str
            ID of observation network
        vars_desired : list
            List of variables that are desired

        Returns
        -------
        list
            list of variables that can be read through the input network

        """
        obs_vars = []
        if isinstance(vars_desired, str):
            vars_desired = [vars_desired]
        if obs_id in self.post_compute:
            # check if all required are accessible
            postinfo = self.post_compute[obs_id]
            for var in varlist_aerocom(vars_desired):
                if not var in postinfo['vars_supported']:
                    continue
                requires = postinfo['aux_requires'][var]
                all_good = True
                for ds, vars_required in requires.items():
                    if isinstance(vars_required,str):
                        vars_required = [vars_required]
                    vars_avail = self.get_vars_supported(ds, vars_required)
                    if not len(vars_required) == len(vars_avail):
                        all_good = False
                        break
                if all_good:
                    obs_vars.append(var)

        else:
            # check if variable can be read from a dataset on disk
            _oreader = self.get_reader(obs_id)
            for var in varlist_aerocom(vars_desired):
                if _oreader.var_supported(var):
                    obs_vars.append(var)
        return obs_vars

    def __str__(self):
        s=''
        for ds in self.datasets_to_read:
            s += '\n{}'.format(self.get_reader(ds))
        return s

if __name__=="__main__":

    reader = ReadUngridded()

    data = reader.read('AirNow', 'concpm10')