################################################################
# read_aeolus_l2b_data.py
#
# read binary ESA L2B files of the ADM Aeolus mission
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
import logging


class ReadAeolusL2bData:
    """Interface for reading ADM AEOLUS L2B data Aerosol product for the AEOLUS PRODEX project

    IMPORTANT:
    This module requires the coda package to be installed in the local python distribution.
    The coda package can be obtained from http://stcorp.nl/coda/
    In addition it needs a definition file (named AEOLUS-20170913.codadef at the time of
    this writing) that came with the test data from ESA and seems to be available also via the coda
    github page at https://github.com/stcorp/codadef-aeolus/releases/download/20170913/AEOLUS-20170913.codadef.

    A description of the data format can be found here: http://stcorp.nl/coda/codadef/AEOLUS/index.html

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
    _FILEMASK = '*.DBL'
    __version__ = "0.01"
    DATASET_NAME = 'AEOLUS-L2B'
    DATASET_PATH = '/lustre/storeA/project/aerocom/aerocom1/ADM_CALIPSO_TEST/download/'
    # Flag if the dataset contains all years or not
    DATASET_IS_YEARLY = False

    _TIMEINDEX = 0
    _LATINDEX = 1
    _LONINDEX = 2
    _ALTITUDEINDEX = 3
    _EC550INDEX = 4
    _BS550INDEX = 5
    _SRINDEX = 6
    _LODINDEX = 7

    _COLNO = 11
    _ROWNO = 100000
    _CHUNKSIZE = 10000

    # variable names
    # dimension data
    _LATITUDENAME = 'latitude'
    _LONGITUDENAME = 'longitude'
    _ALTITUDENAME = 'altitude'
    _TIMENAME = 'time'
    # variable_data
    _EC550NAME = 'ec550aer'
    _BS550NAME = 'bs550aer'
    _LODNAME = 'lod'
    _SRNAME = 'sr'


    GROUP_DELIMITER = '/'
    # data vars
    # will be stored as pandas time series
    DATA_COLNAMES = {}
    DATA_COLNAMES[_EC550NAME] = 'sca_optical_properties/sca_optical_properties/extinction'
    DATA_COLNAMES[_BS550NAME] = 'sca_optical_properties/sca_optical_properties/backscatter'
    DATA_COLNAMES[_LODNAME] = 'sca_optical_properties/sca_optical_properties/lod'
    DATA_COLNAMES[_SRNAME] = 'sca_optical_properties/sca_optical_properties/sr'

    # meta data vars
    # will be stored as array of strings
    METADATA_COLNAMES = {}
    #METADATA_COLNAMES['time'] = 'sca_optical_properties/starttime'
    METADATA_COLNAMES[_LATITUDENAME] = 'sca_optical_properties/geolocation_middle_bins/latitude'
    METADATA_COLNAMES[_LONGITUDENAME] = 'sca_optical_properties/geolocation_middle_bins/longitude'
    METADATA_COLNAMES[_ALTITUDENAME] = 'sca_optical_properties/geolocation_middle_bins/altitude'

    #Alle vars to loop over them
    _COLNAMES = DATA_COLNAMES
    _COLNAMES.update(METADATA_COLNAMES)

    # because the time is only stored once for an entire profile, we have tp treat that separately
    TIME_NAME = 'time'
    TIME_PATH = 'sca_optical_properties/starttime'

    # additional vars
    # calculated
    AUX_COLNAMES = []

    _INDEX_DICT = {}
    _INDEX_DICT.update({_LATITUDENAME: _LATINDEX})
    _INDEX_DICT.update({_LONGITUDENAME: _LONINDEX})
    _INDEX_DICT.update({_ALTITUDENAME: _ALTITUDEINDEX})
    _INDEX_DICT.update({TIME_NAME:_TIMEINDEX})
    _INDEX_DICT.update({_EC550NAME:_EC550INDEX})
    _INDEX_DICT.update({_BS550NAME:_BS550INDEX})
    _INDEX_DICT.update({_LODNAME:_LODINDEX})
    _INDEX_DICT.update({_SRNAME:_SRINDEX})

    PROVIDES_VARIABLES = list(DATA_COLNAMES.keys())
    PROVIDES_VARIABLES.append(list(METADATA_COLNAMES.keys()))

    # max distance between point on the earth's surface for a match
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

    # def read_file(self, filename, vars_to_read=None, backend_to_use='netcdf4',verbose=False):
    def read_file(self, filename, vars_to_read=None, return_as='dict', verbose=False):
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

        This is whats in one DBL file
        codadump list /lustre/storeA/project/aerocom/aerocom1/ADM_CALIPSO_TEST/download/AE_OPER_ALD_U_N_2A_20070101T002249149_002772000_003606_0001.DBL

        /mph/product
        /mph/proc_stage
        /mph/ref_doc
        /mph/acquisition_station
        /mph/proc_center
        /mph/proc_time
        /mph/software_ver
        /mph/baseline
        /mph/sensing_start
        /mph/sensing_stop
        /mph/phase
        /mph/cycle
        /mph/rel_orbit
        /mph/abs_orbit
        /mph/state_vector_time
        /mph/delta_ut1
        /mph/x_position
        /mph/y_position
        /mph/z_position
        /mph/x_velocity
        /mph/y_velocity
        /mph/z_velocity
        /mph/vector_source
        /mph/utc_sbt_time
        /mph/sat_binary_time
        /mph/clock_step
        /mph/leap_utc
        /mph/leap_sign
        /mph/leap_err
        /mph/product_err
        /mph/tot_size
        /mph/sph_size
        /mph/num_dsd
        /mph/dsd_size
        /mph/num_data_sets
        /sph/sph_descriptor
        /sph/intersect_start_lat
        /sph/intersect_start_long
        /sph/intersect_stop_lat
        /sph/intersect_stop_long
        /sph/sat_track
        /sph/num_brc
        /sph/num_meas_max_brc
        /sph/num_bins_per_meas
        /sph/num_prof_sca
        /sph/num_prof_ica
        /sph/num_prof_mca
        /sph/num_group_tot
        /dsd[?]/ds_name
        /dsd[?]/ds_type
        /dsd[?]/filename
        /dsd[?]/ds_offset
        /dsd[?]/ds_size
        /dsd[?]/num_dsr
        /dsd[?]/dsr_size
        /dsd[?]/byte_order
        /geolocation[?]/start_of_obs_time
        /geolocation[?]/num_meas_eff
        /geolocation[?]/measurement_geolocation[?]/centroid_time
        /geolocation[?]/measurement_geolocation[?]/mie_geolocation_height_bin[25]/longitude_of_height_bin
        /geolocation[?]/measurement_geolocation[?]/mie_geolocation_height_bin[25]/latitude_of_height_bin
        /geolocation[?]/measurement_geolocation[?]/mie_geolocation_height_bin[25]/altitude_of_height_bin
        /geolocation[?]/measurement_geolocation[?]/rayleigh_geolocation_height_bin[25]/longitude_of_height_bin
        /geolocation[?]/measurement_geolocation[?]/rayleigh_geolocation_height_bin[25]/latitude_of_height_bin
        /geolocation[?]/measurement_geolocation[?]/rayleigh_geolocation_height_bin[25]/altitude_of_height_bin
        /geolocation[?]/measurement_geolocation[?]/longitude_of_dem_intersection
        /geolocation[?]/measurement_geolocation[?]/latitude_of_dem_intersection
        /geolocation[?]/measurement_geolocation[?]/altitude_of_dem_intersection
        /geolocation[?]/geoid_separation
        /meas_pcd[?]/start_of_obs_time
        /meas_pcd[?]/l1b_input_screening/l1b_obs_screening
        /meas_pcd[?]/l1b_input_screening/l1b_obs_screening_flags[40]
        /meas_pcd[?]/l1b_input_screening/l1b_mie_meas_screening[?]/l1b_mie_meas_qc
        /meas_pcd[?]/l1b_input_screening/l1b_mie_meas_screening[?]/l1b_mie_meas_qc_flags[8]
        /meas_pcd[?]/l1b_input_screening/l1b_rayleigh_meas_screening[?]/l1b_rayleigh_meas_qc
        /meas_pcd[?]/l1b_input_screening/l1b_rayleigh_meas_screening[?]/l1b_rayleigh_meas_qc_flags[8]
        /meas_pcd[?]/l1b_cal_screening/cal_valid
        /meas_pcd[?]/l2a_processing_qc/sca_applied
        /meas_pcd[?]/l2a_processing_qc/ica_applied
        /meas_pcd[?]/l2a_processing_qc/mca_applied
        /meas_pcd[?]/l2a_processing_qc/feature_finder_indicators/layer_information[24]/bin_loaded
        /meas_pcd[?]/l2a_processing_qc/feature_finder_indicators/layer_information[24]/seed[30]
        /meas_pcd[?]/l2a_processing_qc/feature_finder_indicators/lowest_computable_bin[30]
        /sca_pcd[?]/starttime
        /sca_pcd[?]/firstmatchingbin
        /sca_pcd[?]/qc_flag
        /sca_pcd[?]/profile_pcd_bins[24]/extinction_variance
        /sca_pcd[?]/profile_pcd_bins[24]/backscatter_variance
        /sca_pcd[?]/profile_pcd_bins[24]/lod_variance
        /sca_pcd[?]/profile_pcd_bins[24]/processing_qc_flag
        /sca_pcd[?]/profile_pcd_mid_bins[23]/extinction_variance
        /sca_pcd[?]/profile_pcd_mid_bins[23]/backscatter_variance
        /sca_pcd[?]/profile_pcd_mid_bins[23]/lod_variance
        /sca_pcd[?]/profile_pcd_mid_bins[23]/ber_variance
        /sca_pcd[?]/profile_pcd_mid_bins[23]/processing_qc_flag
        /ica_pcd[?]/starttime
        /ica_pcd[?]/first_matching_bin
        /ica_pcd[?]/qc_flag
        /ica_pcd[?]/ica_processing_qc_flag_bin[24]
        /mca_pcd[?]/starttime
        /mca_pcd[?]/processing_qc_flag_bin[24]
        /amd_pcd[?]/starttime
        /amd_pcd[?]/l2b_amd_screening_qc
        /amd_pcd[?]/l2b_amd_screening_qc_flags
        /amd_pcd[?]/l2b_amd_collocations[?]/l2b_amd_collocation_qc
        /amd_pcd[?]/l2b_amd_collocations[?]/l2b_amd_collocation_qc_flags
        /group_pcd[?]/starttime
        /group_pcd[?]/brc_start
        /group_pcd[?]/measurement_start
        /group_pcd[?]/brc_end
        /group_pcd[?]/measurement_end
        /group_pcd[?]/height_bin_index
        /group_pcd[?]/upper_problem_flag
        /group_pcd[?]/particle_extinction_variance
        /group_pcd[?]/particle_backscatter_variance
        /group_pcd[?]/particle_lod_variance
        /group_pcd[?]/qc_flag
        /group_pcd[?]/mid_particle_extinction_variance_top
        /group_pcd[?]/mid_particle_backscatter_variance_top
        /group_pcd[?]/mid_particle_lod_variance_top
        /group_pcd[?]/mid_particle_ber_variance_top
        /group_pcd[?]/mid_particle_extinction_variance_bot
        /group_pcd[?]/mid_particle_backscatter_variance_bot
        /group_pcd[?]/mid_particle_lod_variance_bot
        /group_pcd[?]/mid_particle_ber_variance_bot
        /sca_optical_properties[?]/starttime
        /sca_optical_properties[?]/sca_optical_properties[24]/extinction
        /sca_optical_properties[?]/sca_optical_properties[24]/backscatter
        /sca_optical_properties[?]/sca_optical_properties[24]/lod
        /sca_optical_properties[?]/sca_optical_properties[24]/sr
        /sca_optical_properties[?]/geolocation_middle_bins[24]/longitude
        /sca_optical_properties[?]/geolocation_middle_bins[24]/latitude
        /sca_optical_properties[?]/geolocation_middle_bins[24]/altitude
        /sca_optical_properties[?]/sca_optical_properties_mid_bins[23]/extinction
        /sca_optical_properties[?]/sca_optical_properties_mid_bins[23]/backscatter
        /sca_optical_properties[?]/sca_optical_properties_mid_bins[23]/lod
        /sca_optical_properties[?]/sca_optical_properties_mid_bins[23]/ber
        /ica_optical_properties[?]/starttime
        /ica_optical_properties[?]/ica_optical_properties[24]/case
        /ica_optical_properties[?]/ica_optical_properties[24]/extinction
        /ica_optical_properties[?]/ica_optical_properties[24]/backscatter
        /ica_optical_properties[?]/ica_optical_properties[24]/lod
        /mca_optical_properties[?]/starttime
        /mca_optical_properties[?]/mca_optical_properties[24]/climber
        /mca_optical_properties[?]/mca_optical_properties[24]/extinction
        /mca_optical_properties[?]/mca_optical_properties[24]/lod
        /amd[?]/starttime
        /amd[?]/amd_properties[24]/pressure_fp
        /amd[?]/amd_properties[24]/temperature_fp
        /amd[?]/amd_properties[24]/frequencyshift_fp
        /amd[?]/amd_properties[24]/relativehumidity_fp
        /amd[?]/amd_properties[24]/molecularlod_fp
        /amd[?]/amd_properties[24]/molecularbackscatter_fp
        /amd[?]/amd_properties[24]/pressure_fiz
        /amd[?]/amd_properties[24]/temperature_fiz
        /amd[?]/amd_properties[24]/frequencyshift_fiz
        /amd[?]/amd_properties[24]/relativehumidity_fiz
        /amd[?]/amd_properties[24]/molecularlod_fiz
        /amd[?]/amd_properties[24]/molecularbackscatter_fiz
        /group_optical_properties[?]/starttime
        /group_optical_properties[?]/height_bin_index
        /group_optical_properties[?]/group_optical_property/group_extinction
        /group_optical_properties[?]/group_optical_property/group_backscatter
        /group_optical_properties[?]/group_optical_property/group_lod
        /group_optical_properties[?]/group_optical_property/group_sr
        /group_optical_properties[?]/group_geolocation_middle_bins/start_longitude
        /group_optical_properties[?]/group_geolocation_middle_bins/start_latitude
        /group_optical_properties[?]/group_geolocation_middle_bins/start_altitude
        /group_optical_properties[?]/group_geolocation_middle_bins/mid_longitude
        /group_optical_properties[?]/group_geolocation_middle_bins/mid_latitude
        /group_optical_properties[?]/group_geolocation_middle_bins/mid_altitude
        /group_optical_properties[?]/group_geolocation_middle_bins/stop_longitude
        /group_optical_properties[?]/group_geolocation_middle_bins/stop_latitude
        /group_optical_properties[?]/group_geolocation_middle_bins/stop_altitude
        /group_optical_properties[?]/group_optical_property_middle_bins/mid_extinction_top
        /group_optical_properties[?]/group_optical_property_middle_bins/mid_backscatter_top
        /group_optical_properties[?]/group_optical_property_middle_bins/mid_lod_top
        /group_optical_properties[?]/group_optical_property_middle_bins/mid_ber_top
        /group_optical_properties[?]/group_optical_property_middle_bins/mid_extinction_bot
        /group_optical_properties[?]/group_optical_property_middle_bins/mid_backscatter_bot
        /group_optical_properties[?]/group_optical_property_middle_bins/mid_lod_bot
        /group_optical_properties[?]/group_optical_property_middle_bins/mid_ber_bot
        /scene_classification[?]/starttime
        /scene_classification[?]/height_bin_index
        /scene_classification[?]/aladin_cloud_flag/clrh
        /scene_classification[?]/aladin_cloud_flag/clsr
        /scene_classification[?]/aladin_cloud_flag/downclber
        /scene_classification[?]/aladin_cloud_flag/topclber
        /scene_classification[?]/nwp_cloud_flag
        /scene_classification[?]/l2a_group_class_reliability

        The question mark indicates a variable size array

        It is not entirely clear what we actually have to look at.
        For simplicity the dat of the group 'sca_optical_properties' is returned at this point

        Example
        -------
        >>> import pyaerocom.io.read_aeolus_l2b_data
        >>> obj = pyaerocom.io.read_aeolus_l2b_data.ReadAeolusL2bData(verbose=True)
        >>> filename = '/lustre/storeA/project/aerocom/aerocom1/ADM_CALIPSO_TEST/download/AE_OPER_ALD_U_N_2A_20070101T002249149_002772000_003606_0001.DBL'
        >>> filedata = obj.read_file(filename, vars_to_read=['ec550aer'], return_as='numpy')
        >>> filedata = obj.read_file(filename, vars_to_read=['ec550aer'])
        >>> print(filedata['time'])
        """

        import time
        import coda

        # BaseTimeNympy = np.datetime64('2000-01-01')
        base_time = pd.DatetimeIndex(['2000-01-01'])
        # that's the seconds to add to the the time coming from coda
        seconds_to_add = (base_time.view('int64') // pd.Timedelta(1, unit='s'))[0]

        point_index = 0
        start_time = time.perf_counter()
        file_data = {}

        logging.warning('reading file {}'.format(filename))

        # read file
        product = coda.open(filename)
        if vars_to_read is None:
            # read all variables
            vars_to_read = list(self.DATA_COLNAMES.keys())
        vars_to_read.extend(list(self.METADATA_COLNAMES.keys()))

        # read data
        # start with the time because it is only stored once
        groups = self.TIME_PATH.split(self.GROUP_DELIMITER)
        file_data[self.TIME_NAME] = coda.fetch(product,
                                    groups[0],
                                    -1,
                                    groups[1])
        # epoch is 1 January 2000 at ESA
        # so add offset to move that to 1 January 1970
        # and save it into a np.datetime64 object

        file_data[self.TIME_NAME] = \
            ((file_data[self.TIME_NAME]+seconds_to_add)*1.E3).astype(np.int).astype('datetime64[ms]')
        # scalar
        # single_ts = np.datetime64(np.int((file_data[self.TIME_NAME][0]+seconds_to_add)*1.E3), 'ms')
        # 1167610969347

        # read data in a simple dictionary
        for var in vars_to_read:
            groups = self._COLNAMES[var].split(self.GROUP_DELIMITER)
            if len(groups) == 3:
                file_data[var] = {}
                idx=0
                for idx, key in enumerate(file_data[self.TIME_NAME]):
                    # np_key = np.datetime64(np.int(key*1.E3), 'ms')
                    # file_data[var][key]={}
                    file_data[var][key]=coda.fetch(product,
                                           groups[0],
                                           idx,
                                           groups[1],
                                           -1,
                                           groups[2])

            elif len(groups) == 2:
                file_data[var][key] = coda.fetch(product,
                                       groups[0],
                                       -1,
                                       groups[1])
            else:
                file_data[var][key] = coda.fetch(product,
                                       groups[0])
        if return_as == 'numpy':
            # return as one multidimensional numpy array that can be put into self.data (semi-)directly
            # MODLINENO = 10000
            index_pointer = 0
            data = np.empty([self._ROWNO, self._COLNO], dtype=np.float_)
            # for idx, _time in enumerate(file_data['time']):
            for idx, _time in enumerate(file_data['time'].astype(np.float_)):
                # print(idx)
                for _index in range(len(file_data[var][file_data['time'][idx]])):
                    data[index_pointer, self._TIMEINDEX] = _time
                    for var in vars_to_read:
                        data[index_pointer, self._INDEX_DICT[var]] = file_data[var][file_data['time'][idx]][_index]

                    index_pointer += 1
                    if index_pointer >= self._ROWNO:
                        # add another array chunk to self.data
                        data = np.append(data, np.zeros([self._CHUNKSIZE, self._COLNO], dtype=np.float_),
                                              axis=0)
                        self._ROWNO += self._CHUNKSIZE
                # print(_time)
            file_data = data[0:index_pointer]

        end_time = time.perf_counter()
        elapsed_sec = end_time - start_time
        temp = 'elapsed seconds: {:.3f}'.format(elapsed_sec)
        logging.warning(temp)
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
