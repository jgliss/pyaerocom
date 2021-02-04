"""base class for reading EEA AqERep files (formerly known as Airbase data).

This file is part of the pyaerocom package.

#################################################################
# Created 20120128 by Jan Griesfeller for Met Norway
#
# Last changed: See git log
#################################################################

# Copyright (C) 2021 met.no
# Contact information:
# Norwegian Meteorological Institute
# Box 43 Blindern
# 0313 OSLO
# NORWAY
# E-mail: jan.griesfeller@met.no
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
# MA 02110-1301, USA

Example
-------
look at the end of the file
"""

import numpy as np
from collections import OrderedDict as od
from pyaerocom.io.readungriddedbase import ReadUngriddedBase
from pyaerocom.ungriddeddata import UngriddedData
from pyaerocom import const
from pyaerocom.stationdata import StationData
import pandas as pd
import os
import sys
from pyaerocom.io.helpers import get_country_name_from_iso
import matplotlib.pyplot as plt

class ReadEEAAQEREPBase(ReadUngriddedBase):
    """Class for reading EEA AQErep data

    Extended class derived from  low-level base class :class: ReadUngriddedBase
    that contains some more functionallity.
    """
    #: Mask for identifying datafiles
    _FILEMASK = '*.csv'

    #: Version log of this class (for caching)
    __version__ = '0.02'

    #: Name of the dataset (OBS_ID)
    DATA_ID = ''

    #: List of all datasets supported by this interface
    SUPPORTED_DATASETS = [DATA_ID]

    #: Temporal resolution flag for the supported dataset that is provided in a
    #: defined temporal resolution
    TS_TYPE = 'hourly'

    #: Dictionary specifying values corresponding to invalid measurements
    #: there's no value for NaNs in this data set. It uses an empty string
    NAN_VAL ={}

    #: Dictionary specifying the file column names (values) for each Aerocom
    #: variable (keys)
    # There's only one variable in each file named concentration
    VAR_NAMES_FILE = {}
    VAR_NAMES_FILE['concso2'] = 'concentration'
    VAR_NAMES_FILE['concpm10'] = 'concentration'
    VAR_NAMES_FILE['conco3'] = 'concentration'
    VAR_NAMES_FILE['concno2'] = 'concentration'
    VAR_NAMES_FILE['concco'] = 'concentration'
    VAR_NAMES_FILE['concno'] = 'concentration'
    VAR_NAMES_FILE['concpm25'] = 'concentration'

    #: file masks for the data files
    FILE_MASKS = {}
    FILE_MASKS['concso2'] = '**/*_1_*_timeseries.csv'
    FILE_MASKS['concpm10'] = '**/*_5_*_timeseries.csv'
    FILE_MASKS['conco3'] = '**/*_7_*_timeseries.csv'
    FILE_MASKS['concno2'] = '**/*_8_*_timeseries.csv'
    FILE_MASKS['concco'] = '**/*_10_*_timeseries.csv'
    FILE_MASKS['concno'] = '**/*_38_*_timeseries.csv'
    FILE_MASKS['concpm25'] = '**/*_6001_*_timeseries.csv'

    #: field name of the start time of the measurement (in lower case)
    START_TIME_NAME = 'datetimebegin'

    #: filed name of the end time of the measurement (in lower case)
    END_TIME_NAME = 'datetimeend'

    #: dictionary that connects the EEA variable codes with aerocom variable names
    VAR_CODES = {}
    VAR_CODES['1'] = 'concso2'
    VAR_CODES['5'] = 'concpm10'
    VAR_CODES['7'] = 'conco3'
    VAR_CODES['8'] = 'concno2'
    VAR_CODES['10'] = 'concco'
    VAR_CODES['38'] = 'concno'
    VAR_CODES['6001'] = 'concpm25'

    #: column name that holds the EEA variable code
    VAR_CODE_NAME = 'airpollutantcode'

    #: List of variables that are provided by this dataset (will be extended
    #: by auxiliary variables on class init, for details see __init__ method of
    #: base class ReadUngriddedBase)
    PROVIDES_VARIABLES = list(VAR_NAMES_FILE.keys())

    #: there's no general instrument name in the data
    INSTRUMENT_NAME = 'unknown'

    #: max time steps to read per file (hourly data)
    # to make numpy array allocation size static
    MAX_LINES_TO_READ = 24 * 366

    #: file name of the metadata file
    #: this will be prepended with a data path later on
    # this file is in principe updated once a day.
    # so we night consider updating it from within the code later on.
    # URL: https://discomap.eea.europa.eu/map/fme/metadata/PanEuropean_metadata.csv
    DEFAULT_METADATA_FILE = 'metadata.csv'

    #: Name of latitude variable in metadata file
    LATITUDENAME = 'latitude'

    #: name of longitude variable in metadata file
    LONGITUDENAME= 'longitude'

    #: name of altitude variable in metadata file
    ALTITUDENAME = 'altitude'

    #: this class reads the European Environment Agency's Eionet data
    #: for details please read
    #: https://www.eea.europa.eu/about-us/countries-and-eionet
    WEBSITE = 'https://discomap.eea.europa.eu/map/fme/AirQualityExport.htm'

    #: Eionet offers 2 data revisions
    #: E2a (near real time) and E1a (quality controlled)
    #: this class reads the E2a data for now.
    # But by changing the base path
    # and this constant, it can also read the E1a data set
    DATA_PRODUCT = 'E2a'

    def __init__(self, data_dir=None):
        super(ReadEEAAQEREPBase, self).__init__(None, dataset_path=data_dir)

    #: default variables for read method
    @property
    def DEFAULT_VARS(self):
        return [self.VAR_CODES['7']]

    @property
    def DATASET_NAME(self):
        """Name of the dataset"""
        return self.data_id

    def read_file(self, filename, vars_to_retrieve=None,
                  vars_as_series=False):
        """Read a single EEA file

        Note that there's only a single variable in the file

        Parameters
        ----------
        filename : str
            Absolute path to filename to read.
        vars_to_retrieve : :obj:`list`, optional
            List of strings with variable names to read. If None, use :attr:
                `DEFAULT_VARS`.
        vars_as_series : bool
            If True, the data columns of all variables in the result dictionary
            are converted into pandas Series objects.

        Returns
        -------
        StationData
            Dict-like object containing the results.

        """
        if vars_to_retrieve is None:
            vars_to_retrieve = self.DEFAULT_VARS
        elif isinstance(vars_to_retrieve, str):
            vars_to_retrieve = [vars_to_retrieve]

        for var in vars_to_retrieve:
            if not var in self.PROVIDES_VARIABLES:
                raise ValueError('Invalid input variable {}'.format(var))

        # there's only one variable in the file
        aerocom_var_name = vars_to_retrieve[0]

        # Iterate over the lines of the file
        self.logger.info("Reading file {}".format(filename))
        file_delimiter = ','
        # this lists the data to keep from the original read string
        # this becomes a time series
        file_indexes_to_keep = [11,13,14,15,16]
        # this is some header information
        header_indexes_to_keep = [0,3,8,9,10,12,]
        # These are the indexes with a time and are stored as np.datetime64
        time_indexes = [13,14]

        # read the file
        # file_data = []
        with open(filename, 'r') as f:
            # read header...
            # Countrycode,Namespace,AirQualityNetwork,AirQualityStation,AirQualityStationEoICode,SamplingPoint,SamplingProcess,Sample,AirPollutant,AirPollutantCode,AveragingTime,Concentration,UnitOfMeasurement,DatetimeBegin,DatetimeEnd,Validity,Verification
            header = f.readline().lower().rstrip().split(file_delimiter)
            # create output dict
            data_dict = {}
            for idx in header_indexes_to_keep:
                data_dict[header[idx]] = ''

            for idx in file_indexes_to_keep:
                if idx in time_indexes:
                    data_dict[header[idx]] = np.zeros(self.MAX_LINES_TO_READ, dtype='datetime64[s]')
                else:
                    data_dict[header[idx]] = np.empty(self.MAX_LINES_TO_READ, dtype=np.float_)

            # read the data...
            # DE,http://gdi.uba.de/arcgis/rest/services/inspire/DE.UBA.AQD,NET.DE_BB,STA.DE_DEBB054,DEBB054,SPO.DE_DEBB054_PM2_dataGroup1,SPP.DE_DEBB054_PM2_automatic_light-scat_Duration-30minute,SAM.DE_DEBB054_2,PM2.5,http://dd.eionet.europa.eu/vocabulary/aq/pollutant/6001,hour,3.2000000000,µg/m3,2020-01-04 00:00:00 +01:00,2020-01-04 01:00:00 +01:00,1,2
            lineidx = 0
            for line in f:
                rows = line.rstrip().split(file_delimiter)
                if lineidx == 0:
                    header_info = {}
                    for idx in header_indexes_to_keep:
                        if header[idx] != self.VAR_CODE_NAME:
                            data_dict[header[idx]] = rows[idx]
                        else:
                            #extract the EEA var code from the URL noted in the data file
                            data_dict[header[idx]] = rows[idx].split('/')[-1]

                for idx in file_indexes_to_keep:
                    # if the data is a time
                    if idx in time_indexes:
                        # make the time string ISO compliant so that numpy can directly read it
                        # this is not very time string forgiving but fast
                        data_dict[header[idx]][lineidx] = np.datetime64(rows[idx][0:10]+'T'+rows[idx][11:19]+rows[idx][20:])
                    else:
                        # data is not a time
                        # sometimes there's no value in the file. Set that to nan
                        try:
                            data_dict[header[idx]][lineidx] = np.float_(rows[idx])
                        except ValueError:
                            data_dict[header[idx]][lineidx] = np.nan

                lineidx += 1

        # Empty data object (a dictionary with extended functionality)
        data_out = StationData()
        data_out.data_id = self.data_id
        data_out.dataset_name = self.DATASET_NAME
        data_out.station_id = data_dict['airqualitystation']
        data_out.station_name = data_dict['airqualitystation']
        data_out.filename = filename
        data_out.instrument_name = self.INSTRUMENT_NAME
        data_out.country_code = data_dict['countrycode']
        data_out.ts_type = data_dict['averagingtime'] + 'ly'
        data_out['variables'] = [aerocom_var_name]
        data_out['var_info'][aerocom_var_name] = od()
        data_out['var_info'][aerocom_var_name]['units'] = 'ug m-3'
        data_out['var_info'][aerocom_var_name]['units_original'] = data_dict['unitofmeasurement']
        data_out['var_info'][aerocom_var_name]['ts_type'] = 'hourly'

        for key, value in data_dict.items():
            # adjust the variable name to aerocom standard
            if key != self.VAR_NAMES_FILE[aerocom_var_name]:
                data_out[key] = value[:lineidx]
            else:
                data_out[aerocom_var_name] = value[:lineidx]

        # just assume hourly data for now
        time_diff = np.timedelta64(30,'m')
        data_out['dtime'] = data_dict[self.START_TIME_NAME][:lineidx] + time_diff
        # convert data vectors to pandas.Series (if attribute
        # vars_as_series=True)
        if vars_as_series:
            data_out[aerocom_var_name] = pd.Series(data_out[aerocom_var_name], index=data_out['dtime'])

        return data_out


    def _read_metadata_file(self, filename=None):
        """Read EEA metadata file

        Parameters
        ----------
        filename : str
            Absolute path to filename to read.

        Returns
        -------
        metadata
            Dict-like object containing the results with the keys being a combination of the
            station name and the variable and the values being all fields of the metadata file

        """

        if filename is None:
            filename = os.path.join(self.DATASET_PATH, self.DEFAULT_METADATA_FILE)
        self.logger.warning("Reading file {}".format(filename))

        struct_data = {}
        with open(filename, 'r') as f:
            # read header...
            # Countrycode Timezone Namespace   AirQualityNetwork AirQualityStation AirQualityStationEoICode   AirQualityStationNatCode   SamplingPoint  SamplingProces Sample   AirPollutantCode  ObservationDateBegin ObservationDateEnd   Projection  Longitude   Latitude Altitude MeasurementType   AirQualityStationType   AirQualityStationArea   EquivalenceDemonstrated MeasurementEquipment InletHeight BuildingDistance  KerbDistance
            header = f.readline().lower().rstrip().split()
            # create output dict
            data_dict = {}
            for key in header:
                data_dict[key] = []
            lineidx = 0
            bad_line_no = 0
            bad_line_arr = []
            for line in f:
                rows = line.rstrip().split('\t')

                if len(rows) < 24:
                    print(line)
                    bad_line_no += 1
                    bad_line_arr.append(line)
                    continue
                temp_dict = {}
                for idx, key in enumerate(header):
                    if header[idx] != self.VAR_CODE_NAME:
                        # data_dict[header[idx]] = rows[idx]
                        temp_dict[header[idx]] = rows[idx]
                    else:
                        #extract the EEA var code from the URL noted in the data file
                        # data_dict[header[idx]] = rows[idx].split('/')[-1]
                        temp_dict[header[idx]] = rows[idx].split('/')[-1]

                meta_key = '{}__{}'.format(temp_dict['airqualitystation'], temp_dict['airpollutantcode'])
                if meta_key not in struct_data:
                    struct_data[meta_key] = temp_dict.copy()
                else:
                    pass
                lineidx += 1

        self.logger.info("Reading file {} done".format(filename))
        return struct_data


    def get_file_list(self, pattern=None):
        """Search all files to be read

        Uses :attr:`_FILEMASK` (+ optional input search pattern, e.g.
        station_name) to find valid files for query.

        Parameters
        ----------
        pattern : str, optional
            file name pattern applied to search

        Returns
        -------
        list
            list containing retrieved file locations

        Raises
        ------
        IOError
            if no files can be found
        """
        import glob, os
        from pyaerocom._lowlevel_helpers import list_to_shortstr
        from pyaerocom.exceptions import DataSourceError

        if pattern is None:
            const.print_log.warning('_FILEMASK attr. must not be None...'
                                    'using default pattern *.* for file search')
            pattern = '*.*'
        self.logger.info('Fetching data files. This might take a while...')
        files = sorted(glob.glob(os.path.join(self.DATASET_PATH,
                                              pattern), recursive=True))
        if not len(files) > 0:
            all_str = list_to_shortstr(os.listdir(self.DATASET_PATH))
            raise DataSourceError('No files could be detected matching file '
                                  'mask {} in dataset {}, files in folder {}:\n'
                                  'Files in folder:{}'.format(pattern,
                                  self.dataset_to_read,
                                  self.DATASET_PATH,
                                  all_str))
        self.files = files
        return files

    def get_station_coords(self, meta_key):
        """
        get a station's coordinates

        Parameters
        ----------
        meta_key : `str`
            string with the internal station key
        """

        ret_data = {}
        ret_data['latitude'] = self._metadata[meta_key][self.LATITUDENAME]
        ret_data['longitude'] = self._metadata[meta_key][self.LONGITUDENAME]
        ret_data['altitude'] = self._metadata[meta_key][self.ALTITUDENAME]
        return ret_data

    def read(self, vars_to_retrieve=None,
             files=None,
             first_file=None,
             last_file=None,
             metadatafile=None):
        """Method that reads list of files as instance of :class:`UngriddedData`

        Parameters
        ----------
        vars_to_retrieve : :obj:`list` or similar, optional
            List containing variable IDs that are supposed to be read. If None,
            all variables in :attr:`PROVIDES_VARIABLES` are loaded.
        files : :obj:`list`, optional
            List of files to be read. If None, then the file list used is the
            returned from :func:`get_file_list`.
        first_file : :obj:`int`, optional
            Index of the first file in :obj:'file' to be read. If None, the
            very first file in the list is used.
        last_file : :obj:`int`, optional
            Index of the last file in :obj:'file' to be read. If None, the very
            last file in the list is used.
        metadatafile : :obj:'str', optional
            full qualified path to metadata file. If None, the default metadata
            file will be used

        Returns
        -------
        UngriddedData
            data object
        """

        if vars_to_retrieve is None:
            vars_to_retrieve = self.DEFAULT_VARS
        elif isinstance(vars_to_retrieve, str):
            vars_to_retrieve = [vars_to_retrieve]

        if files is None:
            if len(self.files) == 0:
                self.get_file_list(self.FILE_MASKS[vars_to_retrieve[0]])
            files = self.files

        if first_file is None:
            first_file = 0
        if last_file is None:
            last_file = len(files)

        if metadatafile is None:
            metadatafile = os.path.join(self.DATASET_PATH,self.DEFAULT_METADATA_FILE)

        files = files[first_file:last_file]

        data_obj = UngriddedData()
        meta_key = 0.0
        idx = 0

        # Assign metadata object and index
        metadata = data_obj.metadata
        meta_idx = data_obj.meta_idx

        # non compliant, but efficiently indexed metadata
        self._metadata = self._read_metadata_file(metadatafile)

        # returns a dict with country codes as keys and the country names as value
        _country_dict = get_country_name_from_iso()

        for i, _file in enumerate(files):
            station_data = self.read_file(_file,
                                          vars_to_retrieve=vars_to_retrieve)

            # only the variables in the file
            num_vars = len(station_data.var_info.keys())
            # to find the metadata quickly, we use a string internally
            _meta_key = '{}__{}'.format(station_data['station_id'], station_data['airpollutantcode'])

            # Fill the metadata dict.
            # The location in the data set is time step dependant
            if _meta_key not in self._metadata:
                self.logger.warning('metadata for station {} not found! skipping that station!'.format(_meta_key))
                continue
            metadata[meta_key] = od()
            meta_idx[meta_key] = od()
            metadata[meta_key].update(station_data.get_meta())
            metadata[meta_key].update(self.get_station_coords(_meta_key))
            metadata[meta_key]['variables'] = list(station_data.var_info.keys()) #vars_to_retrieve
            metadata[meta_key]['station_classification'] = self._metadata[_meta_key]['airqualitystationtype']
            metadata[meta_key]['area_classification'] = self._metadata[_meta_key]['airqualitystationarea']
            try:
                metadata[meta_key]['country'] = _country_dict[metadata[meta_key]['country_code']]
            except KeyError:
                pass
            metadata[meta_key]['var_info'] = station_data['var_info']
            metadata[meta_key]['website'] = self.WEBSITE
            metadata[meta_key]['data_product'] = self.DATA_PRODUCT
            metadata[meta_key]['station_name'] = self._metadata[_meta_key]['airqualitystationeoicode']

            # List with indices of this station for each variable
            num_times = len(station_data['dtime'])
            totnum = num_times * num_vars

            # Check whether the size of the data object needs to be extended
            if (idx + totnum) >= data_obj._ROWNO:
                # if totnum < data_obj._CHUNKSIZE, then the latter is used
                data_obj.add_chunk(totnum)

            for var_idx, var in enumerate(list(station_data.var_info)):
                # set invalid data to np.nan
                # https://dd.eionet.europa.eu/vocabulary/aq/observationvalidity/view
                station_data[var][station_data['validity'] < 1] = np.nan
                # there's also a verification flag that we don't use for now
                # which probably only makes sense to be used with the non NRT data
                # http://dd.eionet.europa.eu/vocabulary/aq/observationverification/view
                values = station_data[var]
                start = idx + var_idx * num_times
                stop = start + num_times

                # Write common meta info for this station (data lon, lat and
                # altitude are set to station locations)
                data_obj._data[start:stop, data_obj._LATINDEX
                ] = station_data['latitude']
                data_obj._data[start:stop, data_obj._LONINDEX
                ] = station_data['longitude']
                data_obj._data[start:stop, data_obj._ALTITUDEINDEX
                ] = station_data['altitude']
                data_obj._data[start:stop, data_obj._METADATAKEYINDEX
                ] = meta_key
                data_obj._data[start:stop, data_obj._DATAFLAGINDEX
                ] = station_data['validity']
                data_obj._data[start:stop, data_obj._TIMEINDEX
                ] = station_data['dtime']
                data_obj._data[start:stop, data_obj._DATAINDEX
                ] = values
                data_obj._data[start:stop, data_obj._VARINDEX
                ] = var_idx
                meta_idx[meta_key][var] = np.arange(start, stop)

                if not var in data_obj.var_idx:
                    data_obj.var_idx[var] = var_idx

            idx += totnum
            meta_key = meta_key + 1.

        # Shorten data_obj._data to the right number of points
        data_obj._data = data_obj._data[:idx]
        #data_obj.data_revision[self.DATASET_NAME] = self.data_revision
        self.data = data_obj
        self._metadata = None

        return data_obj



if __name__ == "__main__":

    # Test that the reading routine works
    from pyaerocom.io.read_eea_aqerep import ReadEEAAQEREP
    import logging
    r = ReadEEAAQEREP()
    r.logger.setLevel(logging.INFO)
    var_names_to_test = ['concpm25', 'vmro3','concpm10', ]
    station_id = 'AT2KA71'
    for var_name in var_names_to_test:
        data = r.read(vars_to_retrieve = [var_name])
        print('data read')
        print(data[station_id])
        print(data[station_id][var_name])


