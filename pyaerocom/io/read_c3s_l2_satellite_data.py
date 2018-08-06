################################################################
# read_aeronet_sunv3.py
#
# read Aeronet direct sun V3 data
#
# this file is part of the pyaerocom package
#
#################################################################
# Created 20180731 by Jan Griesfeller for Met Norway
#
# Last changed: See git log
#################################################################

# Copyright (C) 2018 met.no
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

"""
read Aeronet direct sun V3 data
"""
import os
import glob
import sys

import numpy as np

import pandas as pd

from pyaerocom import const


class ReadC3sL2SatelliteData:
    """Interface for reading C3S L2 data (e.g. SLSTR)

    Attributes
    ----------
    data : numpy array of dtype np.float64 initially of shape (10000,8)
        data point array
    metadata : dict
        meta data dictionary

    Parameters
    ----------
    verbose : Bool
        if True some running information is printed

    """
    _FILEMASK = '*L2P*.nc'
    __version__ = "0.01"
    # DATASET_NAME = const.AERONET_SUN_V3L15_AOD_DAILY_NAME
    # DATASET_PATH = const.OBSCONFIG[const.AERONET_SUN_V3L15_AOD_DAILY_NAME]['PATH']
    DATASET_NAME = 'C3S-SLSTR-L2'
    DATASET_PATH = '/lustre/storeA/project/aerocom/aerocom-users-database/C3S-Aerosol/SLSTR_SU_v1.00/L2_download/2017/2017_07_01/'
    # Flag if the dataset contains all years or not
    DATASET_IS_YEARLY = False

    _METADATAKEYINDEX = 0
    _TIMEINDEX = 1
    _LATINDEX = 2
    _LONINDEX = 3
    _ALTITUDEINDEX = 4
    _VARINDEX = 5
    _DATAINDEX = 6
    _OBSLATINDEX = 7
    _OBSLONINDEX = 8
    _OBSALTITUDEINDEX = 9
    _DISTANCEINDEX = 10

    _COLNO = 11
    _ROWNO = 100000
    _CHUNKSIZE = 10000

    # variable names of dimension data
    _LATITUDENAME = 'latitude'
    _LONGITUDENAME = 'longitude'
    _ALTITUDENAME = 'altitude'
    _TIMENAME = 'time'

    # data vars
    # will be stored as pandas time series
    DATA_COLNAMES = {}
    DATA_COLNAMES['od550aer'] = 'AOD550'
    DATA_COLNAMES['ang4487aer'] = 'ANG550_870'
    # DATA_COLNAMES['od865aer'] = 'AOD_865nm'

    # meta data vars
    # will be stored as array of strings
    METADATA_COLNAMES = {}
    METADATA_COLNAMES['latitude'] = 'latitude'
    METADATA_COLNAMES['longitude'] = 'longitude'
    # METADATA_COLNAMES['altitude'] = 'altitude'
    METADATA_COLNAMES['time'] = 'time'

    # additional vars
    # calculated
    AUX_COLNAMES = []

    PROVIDES_VARIABLES = list(DATA_COLNAMES.keys())
    for col in AUX_COLNAMES:
        PROVIDES_VARIABLES.append(col)

    # max distance between point on the erth's surface for a match
    # in meters
    MAX_DISTANCE = 50000.

    def __init__(self, index_pointer=0, dataset_to_read=None, verbose=False):
        self.verbose = verbose
        self.metadata = {}
        self.data = []
        self.index = len(self.metadata)
        self.files = []
        # the reading actually works for all V3 direct sun data sets
        # so just adjust the name and the path here
        # const.AERONET_SUN_V3L15_AOD_DAILY_NAME is the default
        if dataset_to_read is None:
            pass
            # self.dataset_name = const.AERONET_SUN_V3L15_AOD_DAILY_NAME
            # self.dataset_path = const.OBSCONFIG[const.AERONET_SUN_V3L15_AOD_DAILY_NAME]['PATH']


        # set the revision to the one from Revision.txt if that file exist
        # self.revision = self.get_data_revision()

        # pointer to 1st free row in self.data
        # can be externally set so that in case the super class wants to read more than one data set
        # no data modification is needed to bring several data sets together
        self.index_pointer = index_pointer

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == 0:
            raise StopIteration
        self.index = self.index - 1
        return self.metadata[float(self.index)]

    def __str__(self):
        stat_names = []
        for key in self.metadata:
            stat_names.append(self.metadata[key]['station name'])

        return ','.join(stat_names)

    ###################################################################################

    def read_file(self, filename, vars_to_read=None, backend_to_use='netcdf4',verbose=False):
        """method to read an data file entirely

        Parameters
        ----------
        filename : str
            absolute path to filename to read
        vars_to_read : list
            list of str with variable names to read; defaults to ['od550aer']
        verbose : Bool
            set to True to increase verbosity

        Returns
        --------
        xarray dataset of the entire file


        Example
        -------
        >>> import pyaerocom.io.read_c3s_l2_satellite_data
        >>> obj = pyaerocom.io.read_c3s_l2_satellite_data.ReadC3sL2SatelliteData(verbose=True)
        >>> filename = '/lustre/storeA/project/aerocom/aerocom-users-database/C3S-Aerosol/SLSTR_SU_v1.00/L2_download/2017/2017_12_06/20171206190704-C3S-L2P_AEROSOL-AER_PRODUCTS-SLSTR_Sentinel_S3A-SU_9397-v1.00.nc'
        >>> filedata = obj.read_file(filename, vars_to_read=['od550aer'], verbose = True)
        >>> print(filedata)
        """

        import time
        start_time = time.perf_counter()
        if backend_to_use == 'xarray':
            import xarray as xr
            file_data = xr.open_dataset(filename)
            # delete unwanted variables to save space
            # if vars_to_read is not None:
            #     vars_to_delete = []
            #     for var in file_data.data_vars:
            #         if var not in vars_to_read:
            #             vars_to_delete.append(var)
            #     file_data = file_data.drop(vars_to_delete)
        else:
            from netCDF4 import Dataset
            import datetime
            start_time = time.perf_counter()
            # read
            rootgrp = Dataset(filename, "r")
            file_data = {}
            if verbose:
                # print(rootgrp.variables.keys())
                print(filename)

            # read global attributes
            file_data['global_attributes'] = {}
            for name in rootgrp.ncattrs():
                file_data['global_attributes'][name] = getattr(rootgrp, name)

            # read the variables for the dimensions
            # take an easy way for now and use static list
            # we might want to look into the dimensions of the variables to read here
            for var in self.METADATA_COLNAMES:
                # read variable
                # set NaNs to np.nan in between
                if self.METADATA_COLNAMES[var] not in rootgrp.variables:
                    continue
                file_data[var]={}
                file_data[var]['data'] = np.float_(rootgrp.variables[self.METADATA_COLNAMES[var]][:]).filled(np.nan)
                # read variable's Attributes
                file_data[var]['attributes'] = {}
                for name in rootgrp[self.METADATA_COLNAMES[var]].ncattrs():
                    file_data[var]['attributes'][name] = getattr(rootgrp.variables[self.METADATA_COLNAMES[var]], name)

            for var in vars_to_read:
                # read variable
                # set NaNs to np.nan and get rid of the masked array in between
                file_data[var]={}
                file_data[var]['data'] = np.float_(rootgrp.variables[self.DATA_COLNAMES[var]][:]).filled(np.nan)
                # read variable's Attributes
                file_data[var]['attributes'] = {}
                for name in rootgrp[self.DATA_COLNAMES[var]].ncattrs():
                    file_data[var]['attributes'][name] = getattr(rootgrp.variables[self.DATA_COLNAMES[var]], name)
                # #read dimensiions
                # file_data[var]['dimensions'] = {}
                # for name in rootgrp.dimensions[self.DATA_COLNAMES[var]]:
                #     file_data[var]['attributes'][name] = getattr(rootgrp.variables[self.DATA_COLNAMES[var]], name)


            rootgrp.close()
            end_time = time.perf_counter()
            if verbose:
                elapsed_sec = end_time - start_time
                temp = 'elapsed seconds: {:.3f}'.format(elapsed_sec)
                print(temp)
        return file_data

    ###################################################################################

    def read(self, vars_to_read=['od550aer'], locs=None, backend='geopy', verbose=False):
        """method to read all files in self.files into self.data and self.metadata

        Example
        -------
        >>> import pyaerocom.io.read_c3s_l2_satellite_data
        >>> obj = pyaerocom.io.read_c3s_l2_satellite_data.ReadC3sL2SatelliteData(verbose=True)
        >>> locations = [(49.093,8.428,0.),(58.388, 8.252, 0.)]
        >>> obj.read(locs=locations,verbose=True)
        >>> obj.read(verbose=True)
        """

        # Metadata key is float because the numpy array holding it is float

        import time
        import geopy.distance

        self.files = self.get_file_list()
        self.data = np.empty([self._ROWNO, self._COLNO], dtype=np.float_)
        MODLINENO=10000

        for _file in sorted(self.files):
            if self.verbose:
                sys.stdout.write(_file + "\n")
            file_data = self.read_file(_file, vars_to_read=vars_to_read)
            # the metatdata dict is left empty for L2 data
            # the location in the data set is time step dependant!

            # this is a list with indexes of this station for each variable
            # not sure yet, if we really need that or if it speeds up things
            # self.metadata[met_data_key]['indexes'] = {}
            # start_index = self.index_pointer

            # variable index
            obs_var_index = 0

            # separate the code between returning
            # - all data
            # - just a subset at locations (but all time steps
            # - subset of locations and certain time steps

            if locs is None:
                # return all data points
                start_time = time.perf_counter()
                for var in sorted(vars_to_read):
                    for idx in range(file_data['time']['data'].size):
                        if self.index_pointer % MODLINENO == 0:
                            print('{} copied'.format(self.index_pointer))
                        self.data[self.index_pointer, self._DATAINDEX] = file_data[var]['data'][idx]
                        self.data[self.index_pointer, self._TIMEINDEX] = file_data['time']['data'][idx]
                        self.data[self.index_pointer, self._LATINDEX] = file_data['latitude']['data'][idx]
                        self.data[self.index_pointer, self._LONINDEX] = file_data['longitude']['data'][idx]
                        # self.data[self.index_pointer, self._ALTITUDEINDEX] = np.float_(var_data.altitude)
                        self.data[self.index_pointer, self._VARINDEX] = obs_var_index

                        self.index_pointer += 1
                        if self.index_pointer >= self._ROWNO:
                            # add another array chunk to self.data
                            self.data = np.append(self.data, np.zeros([self._CHUNKSIZE, self._COLNO], dtype=np.float_),
                                                  axis=0)
                            self._ROWNO += self._CHUNKSIZE

                    obs_var_index += 1

                    end_time = time.perf_counter()
                    if verbose:
                        elapsed_sec = end_time - start_time
                        temp = 'elapsed seconds: {:.3f}'.format(elapsed_sec)
                        print(temp)
            elif isinstance(locs, list):
                if backend == 'geopy':
                    # return just the data points at given locations
                    # try using geopy for distance calculation
                    # might not be extremely fast though

                    start_time = time.perf_counter()
                    for loc_index in range(len(locs)):
                        for var in sorted(vars_to_read):
                            for idx in range(file_data['time']['data'].size):
                                if self.index_pointer > 0 and self.index_pointer % MODLINENO == 0:
                                    print('{} copied'.format(self.index_pointer))

                                # one magnitude slower than geopy.distance.great_circle
                                # distance = geopy.distance.distance(locs[0],(file_data['latitude']['data'][idx], file_data['longitude']['data'][idx])).m
                                #exclude wrong coordinates
                                if np.isnan(file_data['latitude']['data'][idx] + file_data['longitude']['data'][idx]):
                                    continue
                                distance = geopy.distance.great_circle(locs[loc_index],
                                                                       (file_data['latitude']['data'][idx],
                                                                        file_data['longitude']['data'][idx])).m
                                if distance <= self.MAX_DISTANCE:
                                    print('idx: {} dist [m]: {}'.format(idx, distance))
                                    self.data[self.index_pointer, self._DATAINDEX] = file_data[var]['data'][idx]
                                    self.data[self.index_pointer, self._TIMEINDEX] = file_data['time']['data'][idx]
                                    self.data[self.index_pointer, self._LATINDEX] = file_data['latitude']['data'][idx]
                                    self.data[self.index_pointer, self._LONINDEX] = file_data['longitude']['data'][idx]
                                    self.data[self.index_pointer, self._OBSLATINDEX] = locs[loc_index][0]
                                    self.data[self.index_pointer, self._OBSLONINDEX] = locs[loc_index][1]
                                    self.data[self.index_pointer, self._OBSALTITUDEINDEX] = locs[loc_index][2]
                                    self.data[self.index_pointer, self._DISTANCEINDEX] = distance
                                    # self.data[self.index_pointer, self._ALTITUDEINDEX] = np.float_(var_data.altitude)
                                    self.data[self.index_pointer, self._VARINDEX] = obs_var_index

                                    self.index_pointer += 1
                                    if self.index_pointer >= self._ROWNO:
                                        # add another array chunk to self.data
                                        self.data = np.append(self.data, np.zeros([self._CHUNKSIZE, self._COLNO], dtype=np.float_),
                                                              axis=0)
                                        self._ROWNO += self._CHUNKSIZE
                                # else:
                                #     print('dist [m]: {}'.format(distance))
                            obs_var_index += 1

                    # print some time statistics
                    end_time = time.perf_counter()
                    if verbose:
                        elapsed_sec = end_time - start_time
                        temp = 'elapsed seconds: {:.3f}'.format(elapsed_sec)
                        print(temp)

                elif backend == 'pyaerocom':
                    # this will be using a self written method to calculate the distance

                    #first copy entire data to temporary numpy array
                    # Then do the calculation with a numpy vector op

                    # return all data points
                    start_time = time.perf_counter()
                    for var in sorted(vars_to_read):
                        for idx in range(file_data['time']['data'].size):
                            if self.index_pointer % MODLINENO == 0:
                                print('{} copied'.format(self.index_pointer))
                        #     self.data[self.index_pointer, self._DATAINDEX] = file_data[var]['data'][idx]
                        #     self.data[self.index_pointer, self._TIMEINDEX] = file_data['time']['data'][idx]
                        #     self.data[self.index_pointer, self._LATINDEX] = file_data['latitude']['data'][idx]
                        #     self.data[self.index_pointer, self._LONINDEX] = file_data['longitude']['data'][idx]
                        #     # self.data[self.index_pointer, self._ALTITUDEINDEX] = np.float_(var_data.altitude)
                        #     self.data[self.index_pointer, self._VARINDEX] = obs_var_index
                        #
                        #     self.index_pointer += 1
                        #     if self.index_pointer >= self._ROWNO:
                        #         # add another array chunk to self.data
                        #         self.data = np.append(self.data,
                        #                               np.zeros([self._CHUNKSIZE, self._COLNO], dtype=np.float_),
                        #                               axis=0)
                        #         self._ROWNO += self._CHUNKSIZE
                        #
                        # obs_var_index += 1

                        end_time = time.perf_counter()
                        if verbose:
                            elapsed_sec = end_time - start_time
                            temp = 'elapsed seconds: {:.3f}'.format(elapsed_sec)
                            print(temp)

                    pass
                else:
                    print('Error: unknown backend name provided')

            else:
                print('locations not recognised')


        # shorten self.data to the right number of points
        if verbose:
            print('size of data object: {}'.format(self.index_pointer))

        self.data = self.data[0:self.index_pointer]

    ###################################################################################

    def get_file_list(self):
        """search for files to read

        Example
        -------
        >>> import pyaerocom.io.read_c3s_l2_satellite_data
        >>> obj = pyaerocom.io.read_c3s_l2_satellite_data.ReadC3sL2SatelliteData(verbose=True)
        >>> obj.get_file_list()
        """

        if self.verbose:
            print('searching for data files. This might take a while...')
        files = glob.glob(os.path.join(self.DATASET_PATH,'**',
                                       self._FILEMASK),
                          recursive=True)
        return files

    ###################################################################################

    def get_data_revision(self):
        """method to read the revision string from the file Revision.txt in the main data directory"""

        revision_file = os.path.join(self.DATASET_PATH, const.REVISION_FILE)
        revision = 'unset'
        if os.path.isfile(revision_file):
            with open(revision_file, 'rt') as in_file:
                revision = in_file.readline().strip()
                in_file.close()

            self.revision = revision
###################################################################################
