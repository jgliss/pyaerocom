#!/usr/bin/env python3
# -*- coding: utf-8 -*-
################################################################
# readoddata.py
#
# observational data reading class
#
# this file is part of the aerocom_pt package
#
#################################################################
# Created 20171024 by Jan Griesfeller for Met Norway
#
# Last changed: See git log
#################################################################

# Copyright (C) 2017 met.no
# Contact information:
# Norwegian Meteorological Institute
# Box 43 Blindern
# 0313 OSLO
# NORWAY
# E-mail: jan.griesfeller@met.no, jonas.gliss@met.no
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


import os
import glob
# import pdb
import numpy as np
import sys
import pandas as pd

#from pyaerocom.io.read_aeronet_sdav2 import ReadAeronetSDAV2
from pyaerocom.io.read_aeronet_sdav3 import ReadAeronetSdaV3
from pyaerocom.io.read_aeronet_invv2 import ReadAeronetInvV2
from pyaerocom.io.read_aeronet_sunv2 import ReadAeronetSunV2
from pyaerocom.io.read_aeronet_sunv3 import ReadAeronetSunV3
from pyaerocom.io.read_earlinet import ReadEarlinet

from pyaerocom import const

class ReadUngridded():
    """High-level reading class for ungridded files based on obsnetwork ID
    """

    #SUPPORTED_DATASETS = [const.AERONET_SUN_V2L2_AOD_DAILY_NAME, const.AERONET_SUN_V2L2_SDA_DAILY_NAME]
    SUPPORTED_DATASETS = [const.AERONET_SUN_V2L2_AOD_DAILY_NAME,
                          const.AERONET_INV_V2L2_DAILY_NAME,
                          const.AERONET_SUN_V3L15_AOD_DAILY_NAME,
                          const.AERONET_SUN_V3L15_SDA_DAILY_NAME,
                          const.AERONET_SUN_V3L2_SDA_DAILY_NAME,
                          const.AERONET_SUN_V3L15_AOD_ALL_POINTS_NAME,
                          const.EARLINET_NAME]
    #
    _METADATAKEYINDEX = 0
    _TIMEINDEX = 1
    _LATINDEX = 2
    _LONINDEX = 3
    _ALTITUDEINDEX = 4
    _VARINDEX = 5
    _DATAINDEX = 6

    _COLNO = 11
    _ROWNO = 10000

    # The following number denotes the kept precision after the decimal dot of
    # the location (e.g denotes lat = 300.12345)
    # used to code lat and long in a single number for a uniqueness test
    _LOCATION_PRECISION = 5
    _LAT_OFFSET = np.float(90.)

    # when this file exists, an existing cache file is not read
    _DONOTCACHEFILE = os.path.join(const.OBSDATACACHEDIR, 'DONOTCACHE')

    def __init__(self, data_set_to_read=[const.AERONET_SUN_V2L2_AOD_DAILY_NAME],
                 vars_to_read=ReadAeronetSunV2.PROVIDES_VARIABLES,
                 verbose=False):
        if isinstance(data_set_to_read, list):
            self.data_sets_to_read = data_set_to_read
        else:
            self.data_sets_to_read = [data_set_to_read]

        self.verbose = verbose
        self.vars_to_read = vars_to_read
        self.metadata = {}
        self.data = []
        self.filemasks = []
        self.__version__ = 0.02
        self.datasetnames = []
        self.infiles = []
        # file caching
        self.writecachefile = True
        # pointer to first unused index of self.data
        self.index_pointer = 0
        self.index = len(self.data)
        self.revision = {}
        self.data_version = {}
        self.data_dirs = {}
        self.cache_file = {}

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == 0:
            raise StopIteration
        self.index = self.index - 1
        return self.data[self.index]

    def __str__(self):
        stat_names = []
        for key in self.metadata:
            stat_names.append(self.metadata[key]['station name'])
        return ','.join(stat_names)

    def __getitem__(self, item):
        return self.data[item]

    ###################################################################################
    def code_lat_lon_in_float(self):
        """method to code lat and lon in a single number so that we can use np.unique to
        determine single locations"""

        # multiply lons with 10 ** (three times the needed) precision and add the lats muliplied with 1E(precision) to it
        self.coded_loc = self.data[:, self._LONINDEX] * 10 ** (3 * self._LOCATION_PRECISION) + (
                self.data[:, self._LATINDEX] + self._LAT_OFFSET) * (10 ** self._LOCATION_PRECISION)
        return self.coded_loc

    ###################################################################################
    def decode_lat_lon_from_float(self):
        """method to decode lat and lon from a single number calculated by code_lat_lon_in_float
        """

        lons = np.trunc(self.coded_loc / 10 ** (2 * self._LOCATION_PRECISION)) / 10 ** self._LOCATION_PRECISION
        lats = (self.coded_loc - np.trunc(self.coded_loc / 10 ** (2 * self._LOCATION_PRECISION)) * 10 ** (
                2 * self._LOCATION_PRECISION)) / (10 ** self._LOCATION_PRECISION) - self._LAT_OFFSET

        return lats, lons

    ###################################################################################

    @property
    def longitude(self):
        """Longitudes of data"""

        lons = [self.metadata[np.float(x)]['longitude'] for x in range(len(self.metadata))]
        return lons

    @longitude.setter
    def longitude(self, value):
        raise AttributeError("Longitudes cannot be changed, please check "
                             "underlying data type stored in attribute grid")

    @property
    def latitude(self):
        """Latitudes of data"""
        lats = [self.metadata[np.float(x)]['latitude'] for x in range(len(self.metadata))]
        return lats

    @latitude.setter
    def latitude(self, value):
        raise AttributeError("Latitudes cannot be changed, please check "
                             "underlying data type stored in attribute grid")

    @property
    def station_name(self):
        """Latitudes of data"""
        stat_names = [self.metadata[np.float(x)]['station name'] for x in range(len(self.metadata))]
        return stat_names

    @station_name.setter
    def station_name(self, value):
        raise AttributeError("Station names cannot be changed, please check "
                             "underlying data type stored in attribute grid")

    @property
    def time(self):
        """Time dimension of data"""
        pass

    @time.setter
    def time(self, value):
        raise AttributeError("Time array cannot be changed, please check "
                             "underlying data type stored in attribute grid")

    ###################################################################################

    def read(self):
        """Read observations

        Example
        -------
        >>> import pyaerocom.io.readungridded as pio
        >>> from pyaerocom import const
        >>> obj = pio.ReadUngridded(data_set_to_read=[const.AERONET_SUN_V3L15_AOD_ALL_POINTS_NAME])
        >>> obj.read()
        >>> print(obj)
        >>> print(obj.metadata[0.]['latitude'])
        """

        import pickle

        for data_set_to_read in self.data_sets_to_read:
            # read the data sets
            cache_hit_flag = False

            data_dir = const.OBSCONFIG[data_set_to_read]['PATH']
            revision = self.get_data_revision(data_set_to_read)
            newest_file_in_read_dir = max(glob.iglob(os.path.join(data_dir, '*')), key=os.path.getctime)
            newest_file_date_in_read_dir = os.path.getctime(newest_file_in_read_dir)

            cache_file = os.path.join(
                const.OBSDATACACHEDIR,
                '_'.join([data_set_to_read, 'AllYears', 'AllVars.plk']))
            # TODO check for yearly data sets as well and single variables as well
            if os.path.isfile(cache_file):
                # CACHE HIT
                if self.verbose:
                    print("Importing from cache file: {}".format(cache_file))
                    
                cache_hit_flag = True
                # read cache file
                in_handle = open(cache_file, 'rb')
                newest_file_in_read_dir_saved = pickle.load(in_handle)
                newest_file_date_in_read_dir_saved = pickle.load(in_handle)
                revision_saved = pickle.load(in_handle)
                if (newest_file_in_read_dir_saved != newest_file_in_read_dir
                        or newest_file_date_in_read_dir_saved != newest_file_date_in_read_dir
                        or revision_saved != revision):
                    cache_hit_flag = False
                    in_handle.close()

                # The rest can only be checked when the reading object is initialised
                object_version_saved = pickle.load(in_handle)

            # self.ReadCacheData(data_set_to_read)
            # CACHE MISS
            # call the different obs data reading classes
            if data_set_to_read == const.AERONET_SUN_V2L2_AOD_DAILY_NAME:
                # read AERONETSUN V2 L2 daily data set
                read_dummy = ReadAeronetSunV2(index_pointer=self.index_pointer, 
                                              verbose=self.verbose)
                if cache_hit_flag and object_version_saved == read_dummy.__version__:
                    read_dummy = pickle.load(in_handle)
                    if self.verbose:
                        sys.stdout.write('cache file ' + cache_file + ' read\n')
                    # TODO we might need to adjust self.index_pointer in case we really work with more than one data set!
                    in_handle.close()
                else:
                    # re-read data
                    read_dummy.read(self.vars_to_read)

            elif data_set_to_read == const.AERONET_SUN_V3L15_SDA_DAILY_NAME or \
                    data_set_to_read == const.AERONET_SUN_V3L2_SDA_DAILY_NAME:
                # read AERONET SDA V3 L1.5 daily data set
                read_dummy = ReadAeronetSdaV3(index_pointer=self.index_pointer,
                                              data_set_to_read=data_set_to_read,
                                              verbose=self.verbose)
                if cache_hit_flag and object_version_saved == read_dummy.__version__:
                    read_dummy = pickle.load(in_handle)
                    if self.verbose:
                        sys.stdout.write('cache file ' + cache_file + ' read\n')
                    # TODO we might need to adjust self.index_pointer in case we really work with more than one data set!
                    in_handle.close()
                else:
                    # re-read data
                    read_dummy.read(self.vars_to_read)

            elif data_set_to_read == const.AERONET_INV_V2L2_DAILY_NAME:
                # read AERONET inversions V2 L2.0 daily data set
                read_dummy = ReadAeronetInvV2(index_pointer=self.index_pointer,
                                              data_set_to_read=data_set_to_read,
                                              verbose=self.verbose)
                if cache_hit_flag and object_version_saved == read_dummy.__version__:
                    read_dummy = pickle.load(in_handle)
                    if self.verbose:
                        sys.stdout.write('cache file ' + cache_file + ' read\n')
                    # TODO we might need to adjust self.index_pointer in case we really work with more than one data set!
                    in_handle.close()
                else:
                    # re-read data
                    read_dummy.read(self.vars_to_read)

            elif data_set_to_read == const.AERONET_SUN_V3L15_AOD_DAILY_NAME:
                # read AERONETSUN V3 L1.5 daily data set
                read_dummy = ReadAeronetSunV3(index_pointer=self.index_pointer,
                                              verbose=self.verbose)
                if cache_hit_flag and object_version_saved == read_dummy.__version__:
                    read_dummy = pickle.load(in_handle)
                    if self.verbose:
                        sys.stdout.write('cache file ' + cache_file + ' read\n')
                    # TODO we might need to adjust self.index_pointer in case we really work with more than one data set!
                    in_handle.close()
                else:
                    # re-read data
                    read_dummy.read(self.vars_to_read)

            elif data_set_to_read == const.AERONET_SUN_V3L15_AOD_ALL_POINTS_NAME:
                # read AERONETSUN V3 L1.5 all points data set
                read_dummy = ReadAeronetSunV3(index_pointer=self.index_pointer,
                                              data_set_to_read=data_set_to_read,
                                              verbose=self.verbose)
                if cache_hit_flag and object_version_saved == read_dummy.__version__:
                    read_dummy = pickle.load(in_handle)
                    if self.verbose:
                        sys.stdout.write('cache file ' + cache_file + ' read\n')
                    # TODO we might need to adjust self.index_pointer in case we really work with more than one data set!
                    in_handle.close()
                else:
                    # re-read data
                    read_dummy.read(self.vars_to_read)


            elif data_set_to_read == const.AERONET_SUN_V3L2_AOD_DAILY_NAME:
                print("Not implemented at this point.")

            elif data_set_to_read == const.AERONET_SUN_V3L2_AOD_ALL_POINTS_NAME:
                print("Not implemented at this point.")

            elif data_set_to_read == const.EARLINET_NAME:
                read_dummy = ReadEarlinet(index_pointer=self.index_pointer, verbose=self.verbose)
                if cache_hit_flag and object_version_saved == read_dummy.__version__:
                    read_dummy = pickle.load(in_handle)
                    if self.verbose:
                        sys.stdout.write('cache file ' + cache_file + ' read\n')
                    # TODO we might need to adjust self.index_pointer in case we really work with more than one data set!
                    in_handle.close()
                else:
                    # re-read data
                    read_dummy.read(self.vars_to_read)

            else:
                continue
            self.infiles.append(read_dummy.files)
            self.index_pointer = read_dummy.index_pointer
            self.data_dirs[data_set_to_read] = const.OBSCONFIG[data_set_to_read]['PATH']
            self.revision[data_set_to_read] = revision
            self.data_version[data_set_to_read] = read_dummy.__version__
            # TODO do not forget to adjust pointers between metadata and data in case there's more than one data set!!!
            if isinstance(self.data, list):
                # this is the 1st data set
                self.data = read_dummy.data
                # we might want to check for double station names here
                # for the second data set we DO need to update the dict keys of metadata and data!
                # since we do want to keep the point cloud
                # we might want to move the variables to read to a dictionary
            else:
                # extend self.data numpy array
                self.data = np.append(self.data, read_dummy.data, axis=0)
                # also adjust self.index_pointer!
                # sys.stderr.write("ERROR: can only read one obs data network at the time at this point!")
            self.metadata.update(read_dummy.metadata)
            # write the cache file
            if not cache_hit_flag:
                # write cache file in case the data was newly read
                if self.verbose:
                    sys.stdout.write('Writing cache file ' + cache_file + '\n')
                # OutHandle = gzip.open(c__cache_file, 'wb') # takes too much time
                out_handle = open(cache_file, 'wb')
                newest_file_in_read_dir_saved = newest_file_in_read_dir
                newest_file_date_in_read_dir_saved = newest_file_date_in_read_dir
                revision_saved = revision
                object_version_saved = read_dummy.__version__
                pickle.dump(newest_file_in_read_dir_saved, out_handle, pickle.HIGHEST_PROTOCOL)
                pickle.dump(newest_file_date_in_read_dir_saved, out_handle, pickle.HIGHEST_PROTOCOL)
                pickle.dump(revision_saved, out_handle, pickle.HIGHEST_PROTOCOL)
                pickle.dump(object_version_saved, out_handle, pickle.HIGHEST_PROTOCOL)
                pickle.dump(read_dummy, out_handle, pickle.HIGHEST_PROTOCOL)
                out_handle.close()
                if self.verbose:
                    sys.stdout.write('done\n')

    ###################################################################################

    def to_timeseries(self, station_name = None, start_date = None, end_date = None, freq = None):
        """method to get the ObsData object data as dict using pd.Series for the variables

        Parameters
        ----------
        station_names : :obj:`tuple` or :obj:`str:`, optional
            station name or list of station names to return
        start_date, end_date : :obj:`str:`, optional
            date strings with start and end date to return
        freq : obj:`str:`, optional
            frequency to resample to using the pandas resample method
            us the offset aliases as noted in
            http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

        Returns
        -------
        list or dictionary
            station_names is a string: dictionary with station data
            station_names is list or None: list of dictionaries with station data

        Example
        -------
        >>> import pyaerocom.io.readobsdata
        >>> obj = pyaerocom.io.readobsdata.ReadUngridded(verbose=True)
        >>> obj.read()
        >>> pdseries = obj.to_timeseries()
        >>> pdseriesmonthly = obj.to_timeseries(station_name='Avignon',start_date='2011-01-01', end_date='2012-12-31', freq='M')
        """

        out_data = []
        if station_name is None:
            # return all stations
            for index, val in self.metadata.items():
                data = self._to_timeseries_helper(val, start_date, end_date, freq)
                if data is not None:
                    out_data.append(data)
        elif isinstance(station_name, str):
            # user asked for a single station name
            # return a single dictionary in this case
            for index, val in self.metadata.items():
                if station_name == val['station name']:
                    # we might change this to return a list at some point
                    return self._to_timeseries_helper(val, start_date, end_date, freq)
        elif isinstance(station_name, list):
            # user asked for a list of station names
            # return list with matching station names
            for index, val in self.metadata.items():
                # print(val['station name'])
                if val['station name'] in station_name:
                    data = self._to_timeseries_helper(val)
                    if data is not None:
                        out_data.append(self._to_timeseries_helper(val, start_date, end_date, freq))

        return out_data

    ###################################################################################

    def get_data_revision(self, data_set_name):
        """method to read the revision string from the file Revision.txt in the main data directory"""

        revision = 'unset'
        try:
            revision_file = os.path.join(const.OBSCONFIG[data_set_name]['PATH'], const.REVISION_FILE)
            if os.path.isfile(revision_file):
                with open(revision_file, 'rt') as in_file:
                    revision = in_file.readline().strip()
                    in_file.close()
        except:
            pass

        return revision

    ###################################################################################

    def get_data_dir(self, data_set_name):
        """method to return the path of an obs data set"""

        try:
            return const.OBSCONFIG[data_set_name]['PATH']
        except:
            raise AttributeError("data set name " + data_set_name + "not found ")

    ###################################################################################

    def _to_timeseries_helper(self, val, start_date = None, end_date = None, freq = None):
        """small helper routine for self.to_timeseries to not to repeat the same code fragment three times"""

        data_found_flag = False
        temp_dict = {}
        # do not return anything for stations without data
        temp_dict['station name'] = val['station name']
        temp_dict['latitude'] = val['latitude']
        temp_dict['longitude'] = val['longitude']
        temp_dict['altitude'] = val['altitude']
        temp_dict['PI'] = val['PI']
        temp_dict['data_set_name'] = val['data_set_name']
        if 'files' in val:
            temp_dict['files'] = val['files']
        for var in val['indexes']:
            if var in self.vars_to_read:
                data_found_flag = True
                temp_dict[var] = pd.Series(self.data[val['indexes'][var], self._DATAINDEX],
                                           index=pd.to_datetime(
                                               self.data[
                                                   val['indexes'][var], self._TIMEINDEX],
                                               unit='s'))

                temp_dict[var] = temp_dict[var][start_date:end_date].drop_duplicates()
                if freq is not None:
                    temp_dict[var] = temp_dict[var][start_date:end_date].resample(freq).mean()

        if data_found_flag:
            return temp_dict
        else:
            return None

    ###################################################################################

    def CheckObsnetworkName(self, obs_network_name):
        """method to quick check if the user supplied obs network string is right"""

        if obs_network_name in self.SUPPORTED_DATASETS:
            return True
        else:
            return False