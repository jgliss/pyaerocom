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

from pyaerocom import const
import logging
import time
import geopy.distance


class ReadAeolusL2bData:
    """Interface for reading ADM AEOLUS L2B data Aerosol product for the AEOLUS PRODEX project

    IMPORTANT:
    This module requires the coda package to be installed in the local python distribution.
    The coda package can be obtained from http://stcorp.nl/coda/
    In addition, it needs a definition file (named AEOLUS-20170913.codadef at the time of
    this writing) that came with the test data from ESA and seems to be available also via the coda
    github page at https://github.com/stcorp/codadef-aeolus/releases/download/20170913/AEOLUS-20170913.codadef.

    A description of the data format can be found here: http://stcorp.nl/coda/codadef/AEOLUS/index.html

    Attributes
    ----------
    data : numpy array of dtype np.float64 initially of shape (10000,8)
        data point array

    Parameters
    ----------

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
    # for distance calculations we need the location in radians
    # so store these for speed in self.data
    # the following indexes indicate the column in that is stored
    _RADLATINDEX = 8
    _RADLONINDEX = 9
    _DISTINDEX = 10

    _COLNO = 11
    _ROWNO = 100000
    _CHUNKSIZE = 10000

    # variable names
    # dimension data
    _LATITUDENAME = 'latitude'
    _LONGITUDENAME = 'longitude'
    _ALTITUDENAME = 'altitude'
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
    METADATA_COLNAMES[_LATITUDENAME] = 'sca_optical_properties/geolocation_middle_bins/latitude'
    METADATA_COLNAMES[_LONGITUDENAME] = 'sca_optical_properties/geolocation_middle_bins/longitude'
    METADATA_COLNAMES[_ALTITUDENAME] = 'sca_optical_properties/geolocation_middle_bins/altitude'

    # Alle vars to loop over them
    _COLNAMES = DATA_COLNAMES
    _COLNAMES.update(METADATA_COLNAMES)

    # because the time is only stored once for an entire profile, we have tp treat that separately
    _TIME_NAME = 'time'
    TIME_PATH = 'sca_optical_properties/starttime'

    # additional vars
    # calculated
    AUX_COLNAMES = []

    # create a dict with the aerocom variable name as key and the index number in the
    # resulting numpy array as value.
    _INDEX_DICT = {}
    _INDEX_DICT.update({_LATITUDENAME: _LATINDEX})
    _INDEX_DICT.update({_LONGITUDENAME: _LONINDEX})
    _INDEX_DICT.update({_ALTITUDENAME: _ALTITUDEINDEX})
    _INDEX_DICT.update({_TIME_NAME: _TIMEINDEX})
    _INDEX_DICT.update({_EC550NAME: _EC550INDEX})
    _INDEX_DICT.update({_BS550NAME: _BS550INDEX})
    _INDEX_DICT.update({_LODNAME: _LODINDEX})
    _INDEX_DICT.update({_SRNAME: _SRINDEX})

    # NaN values are variable specific
    _NAN_DICT = {}
    _NAN_DICT.update({_LATITUDENAME: -1.E-6})
    _NAN_DICT.update({_LONGITUDENAME: -1.E-6})
    _NAN_DICT.update({_ALTITUDENAME: -1.})
    _NAN_DICT.update({_EC550NAME: -1.E6})
    _NAN_DICT.update({_BS550NAME: -1.E6})
    _NAN_DICT.update({_LODNAME: -1.})
    _NAN_DICT.update({_SRNAME: -1.})

    PROVIDES_VARIABLES = list(DATA_COLNAMES.keys())
    PROVIDES_VARIABLES.append(list(METADATA_COLNAMES.keys()))

    # max distance between point on the earth's surface for a match
    # in meters
    MAX_DISTANCE = 50000.
    EARTH_RADIUS = geopy.distance.EARTH_RADIUS
    NANVAL_META = -1.E-6
    NANVAL_DATA = -1.E6

    def __init__(self, index_pointer=0, loglevel=logging.INFO, verbose=False):
        self.verbose = verbose
        self.metadata = {}
        self.data = []
        self.index = len(self.metadata)
        self.files = []
        self.index_pointer = index_pointer
        # that's the flag to indicate if the location of a data point in self.data has been
        # stored in rads in self.data already
        # trades RAM for speed
        self.rads_in_array_flag = False

        if loglevel is not None:
            self.logger = logging.getLogger(__name__)
            # self.logger = logging.getLogger('pyaerocom')
            default_formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(default_formatter)
            self.logger.addHandler(console_handler)
            self.logger.setLevel(loglevel)
            self.logger.debug('init')

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

    def read_file(self, filename, vars_to_read=None, return_as='dict', loglevel=None):
        """method to read an ESA binary data file entirely

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
        Either:
            dictionary (default):
                keys are 'time', 'latitude', 'longitude', 'altitude' and the variable names
                'ec550aer', 'bs550aer', 'sr', 'lod' if the whole file is read
                'time' is a 1d array, while the other dict values are a another dict with the
                time as keys (the same ret['time']) and a numpy array as values. These values represent the profile.
                Note 1: latitude and longitude are height dependent due to the tilt of the measurement.
                Note 2: negative values indicate a NaN

            2d ndarray of type float:
                representing a 'point cloud' with all points
                    column 1: time in seconds since the Unix epoch with ms accuracy (same time for every height
                    in a profile)
                    column 2: latitude
                    column 3: longitude
                    column 4: altitude
                    column 5: extinction
                    column 6: backscatter
                    column 7: sr
                    column 8: lod

                    Note: negative values are put to np.nan already

                    The indexes are noted in pyaerocom.io.read_aeolus_l2b_data.ReadAeolusL2bData.<index_name>
                    e.g. the time index is named pyaerocom.io.read_aeolus_l2b_data.ReadAeolusL2bData._TIMEINDEX
                    have a look at the example to access the values

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
        For simplicity the data of the group 'sca_optical_properties' is returned at this point

        Example
        -------
        >>> import pyaerocom.io.read_aeolus_l2b_data
        >>> obj = pyaerocom.io.read_aeolus_l2b_data.ReadAeolusL2bData(verbose=True)
        >>> filename = '/lustre/storeA/project/aerocom/aerocom1/ADM_CALIPSO_TEST/download/AE_OPER_ALD_U_N_2A_20070101T002249149_002772000_003606_0001.DBL'
        >>> # read returning a ndarray
        >>> filedata_numpy = obj.read_file(filename, vars_to_read=['ec550aer'], return_as='numpy')
        >>> time_as_numpy_datetime64 = filedata_numpy[0,obj._TIMEINDEX].astype('datetime64[s]')
        >>> print('time: {}'.format(time_as_numpy_datetime64))
        >>> print('latitude: {}'.format(filedata_numpy[1,obj._LATINDEX]))
        >>> # read returning a dictionary
        >>> filedata = obj.read_file(filename, vars_to_read=['ec550aer'])
        >>> print('time: {}'.format(filedata['time'][0].astype('datetime64[s]')))
        >>> print('all latitudes of 1st time step: {}'.format(filedata['latitude'][filedata['time'][0]]))
        """

        import time
        import coda

        # coda uses 2000-01-01T00:00:00 as epoch unfortunately.
        # so calculate the difference in seconds to the Unix epoch
        seconds_to_add = np.datetime64('2000-01-01T00:00:00') - np.datetime64('1970-01-01T00:00:00')
        seconds_to_add = seconds_to_add.astype(np.float_)

        # the same can be achieved using pandas, but we stick to numpy here
        # base_time = pd.DatetimeIndex(['2000-01-01'])
        # seconds_to_add = (base_time.view('int64') // pd.Timedelta(1, unit='s'))[0]

        start_time = time.perf_counter()
        file_data = {}

        self.logger.info('reading file {}'.format(filename))
        # read file
        product = coda.open(filename)
        if vars_to_read is None:
            # read all variables
            vars_to_read = list(self.DATA_COLNAMES.keys())
        vars_to_read.extend(list(self.METADATA_COLNAMES.keys()))

        # read data
        # start with the time because it is only stored once
        groups = self.TIME_PATH.split(self.GROUP_DELIMITER)
        file_data[self._TIME_NAME] = coda.fetch(product,
                                                groups[0],
                                                -1,
                                                groups[1])
        # epoch is 1 January 2000 at ESA
        # so add offset to move that to 1 January 1970
        # and save it into a np.datetime64[ms] object

        file_data[self._TIME_NAME] = \
            ((file_data[self._TIME_NAME] + seconds_to_add) * 1.E3).astype(np.int).astype('datetime64[ms]')

        # read data in a simple dictionary
        for var in vars_to_read:
            groups = self._COLNAMES[var].split(self.GROUP_DELIMITER)
            if len(groups) == 3:
                file_data[var] = {}
                for idx, key in enumerate(file_data[self._TIME_NAME]):
                    file_data[var][key] = coda.fetch(product,
                                                     groups[0],
                                                     idx,
                                                     groups[1],
                                                     -1,
                                                     groups[2])

            elif len(groups) == 2:
                file_data[var] = {}
                for idx, key in enumerate(file_data[self._TIME_NAME]):
                    file_data[var][key] = coda.fetch(product,
                                                     groups[0],
                                                     -1,
                                                     groups[1])
            else:
                file_data[var] = {}
                for idx, key in enumerate(file_data[self._TIME_NAME]):
                    file_data[var][key] = coda.fetch(product,
                                                     groups[0])
        if return_as == 'numpy':
            # return as one multidimensional numpy array that can be put into self.data directly
            # (column wise because the column numbers do not match)
            index_pointer = 0
            data = np.empty([self._ROWNO, self._COLNO], dtype=np.float_)

            for idx, _time in enumerate(file_data['time'].astype(np.float_) / 1000.):
                # file_data['time'].astype(np.float_) is milliseconds after the (Unix) epoch
                # but we want to save the time as seconds since the epoch
                for _index in range(len(file_data['latitude'][file_data['time'][idx]])):
                    # this works because all variables have to have the same size
                    # (aka same number of height levels)
                    # This loop could be avoided using numpy index slicing
                    # do that in case we need more optimisations
                    data[index_pointer, self._TIMEINDEX] = _time
                    for var in vars_to_read:
                        data[index_pointer, self._INDEX_DICT[var]] = file_data[var][file_data['time'][idx]][_index]
                        # put negative values to np.nan if the variable is not a metadata variable
                        if data[index_pointer, self._INDEX_DICT[var]] == self._NAN_DICT[var]:
                            data[index_pointer, self._INDEX_DICT[var]] = np.nan

                    index_pointer += 1
                    if index_pointer >= self._ROWNO:
                        # add another array chunk to self.data
                        data = np.append(data, np.empty([self._CHUNKSIZE, self._COLNO], dtype=np.float_),
                                         axis=0)
                        self._ROWNO += self._CHUNKSIZE

            # return only the needed elements...
            file_data = data[0:index_pointer]

        end_time = time.perf_counter()
        elapsed_sec = end_time - start_time
        temp = 'time for single file read [s]: {:.3f}'.format(elapsed_sec)
        self.logger.info(temp)
        # self.logger.info('{} points read'.format(index_pointer))
        return file_data

    ###################################################################################

    def read(self, base_dir=None, vars_to_read=['ec550aer'], locs=None, backend='geopy', verbose=False):
        """method to read all files in self.files into self.data and self.metadata
        At this point the data format is NOT the same as for the ungridded base class


        Example
        -------
        >>> import logging
        >>> import pyaerocom.io.read_aeolus_l2b_data
        >>> obj = pyaerocom.io.read_aeolus_l2b_data.ReadAeolusL2bData(loglevel=logging.DEBUG)
        >>> obj.read(vars_to_read=['ec550aer'])
        >>> locations = [(49.093,8.428,0.),(58.388, 8.252, 0.)]
        >>> obj.read(locs=locations,vars_to_read=['ec550aer'],verbose=True)
        >>> obj.read(verbose=True)
        """

        import time
        import geopy.distance

        start_time = time.perf_counter()
        self.files = self.get_file_list()
        after_file_search_time = time.perf_counter()
        elapsed_sec = after_file_search_time - start_time
        temp = 'time for file find: {:.3f}'.format(elapsed_sec)
        self.logger.info(temp)

        # self.data = np.empty([self._ROWNO, self._COLNO], dtype=np.float_)

        for idx, _file in enumerate(sorted(self.files)):
            file_data = self.read_file(_file, vars_to_read=vars_to_read, return_as='numpy')
            # the metadata dict is left empty for L2 data
            # the location in the data set is time step dependant!

            # start_time_read = time.perf_counter()
            # return all data points
            num_points = len(file_data)
            if idx == 0:
                self.data = file_data
                self._ROWNO = num_points
                self.index_pointer = num_points

            else:
                # append to self.data
                # add another array chunk to self.data
                self.data = np.append(self.data, np.zeros([num_points, self._COLNO], dtype=np.float_),
                                      axis=0)
                self._ROWNO = num_points
                # copy the data
                self.data[self.index_pointer:, :] = file_data
                self.index_pointer = self.index_pointer + num_points

                # end_time = time.perf_counter()
                # elapsed_sec = end_time - start_time_read
                # temp = 'time for single file read seconds: {:.3f}'.format(elapsed_sec)
                # self.logger.warning(temp)

        end_time = time.perf_counter()
        elapsed_sec = end_time - start_time
        temp = 'overall time for file read [s]: {:.3f}'.format(elapsed_sec)
        self.logger.info(temp)
        self.logger.info('size of data object: {}'.format(self.index_pointer))

    ###################################################################################

    def get_file_list(self, basedir=None):
        """search for files to read

        Example
        -------
        >>> import pyaerocom.io.read_aeolus_l2b_data
        >>> obj = pyaerocom.io.read_aeolus_l2b_data.ReadAeolusL2bData(verbose=True)
        >>> files = obj.get_file_list()
        """

        self.logger.info('searching for data files. This might take a while...')
        if basedir is None:
            files = glob.glob(os.path.join(self.DATASET_PATH, '**',
                                           self._FILEMASK),
                              recursive=True)
        else:
            files = glob.glob(os.path.join(basedir, '**',
                                           self._FILEMASK),
                              recursive=True)

        return files

    ###################################################################################

    def calc_dist_in_data(self, location=(49.093, 8.428, 0.), backend='pyaerocom'):
        """calculate the distance between a given coordinate and all points in self.data using numpy and
        put that in self.data[*,self._DISTINDEX]

        This method will likely never be used by a user, but serves as helper method for the colocate method

        Because the average earth radius in geopy.distance.EARTH_RADIUS is given in km, the result is also given in km

        The algorithm used to calculate the distance is needs the coordinates in rads. In order to not calculate that
        for for all the data points for every station, this is done only at the first call and the stored in
        self.data[:, self._RADLATINDEX] and self.data[:, self._RADLONINDEX]

        Using the pyaerocom backend of this method returns the same values as geopy.distance.great_circle
        but is roughly 2 magnitudes faster due to much less overhead and usage of the numpy vector functions for the
        calculation

        Example
        -------
        >>> import logging
        >>> import pyaerocom.io.read_aeolus_l2b_data
        >>> obj = pyaerocom.io.read_aeolus_l2b_data.ReadAeolusL2bData(loglevel=logging.DEBUG)
        >>> obj.read(vars_to_read=['ec550aer'])
        >>> location = (49.093,8.428,0.)
        >>> obj.calc_dist_in_data(location)
        >>> import numpy as np
        >>> print('min distance: {:.3f} km'.format(np.nanmin(obj.data[:, obj._DISTINDEX])))
        >>> print('max distance: {:.3f} km'.format(np.nanmax(obj.data[:, obj._DISTINDEX])))
        """

        start_time = time.perf_counter()
        if backend == 'pyaerocom':
            if self.rads_in_array_flag:
                # lat1 = self.data[:, self._LATINDEX]
                # lon1 = self.data[:, self._LONINDEX]
                pass
            else:
                # put the lats and lons in rad in self.data
                np.deg2rad(self.data[:, self._LATINDEX], out=self.data[:, self._RADLATINDEX])
                np.deg2rad(self.data[:, self._LONINDEX], out=self.data[:, self._RADLONINDEX])
                # lat1 = np.deg2rad(self.data[:, self._LATINDEX], out=self.data[:, self._RADLATINDEX])
                # lon1 = np.deg2rad(self.data[:, self._LONINDEX], out=self.data[:, self._RADLONINDEX])
                self.rads_in_array_flag = True

            lat2 = np.deg2rad(location[0])
            lon2 = np.deg2rad(location[1])

            # code from IDL to match:
            # d =!const.R_Earth * 2 * asin(
            # sqrt((sin((lat1 - lat2) / 2)) ^ 2 + cos(lat1) * cos(lat2) * (sin((lon1 - lon2) / 2)) ^ 2))

            self.data[:, self._DISTINDEX] = (self.EARTH_RADIUS * 2. * np.arcsin(
                np.sqrt(
                    np.power(np.sin((self.data[:, self._RADLATINDEX] - lat2) / 2.), 2) + np.cos(
                        self.data[:, self._RADLATINDEX]) * np.cos(lat2) * np.power(
                        np.sin((self.data[:, self._RADLONINDEX] - lon2) / 2),
                        2))))

            end_time = time.perf_counter()
            elapsed_sec = end_time - start_time
            temp = 'time for single station distance calc [s]: {:.3f}'.format(elapsed_sec)
            self.logger.info(temp)

        elif backend == 'geopy':
            # use geopy to calculate distance
            # two magnitudes slower than the pyaerocom backend
            # but told us that we did the calculation right and opens a possibility to use
            # a geopy supported geoid for the calculation. Although that will then be another
            # magnitude slower than geopy.distance.great_circle
            start_time = time.perf_counter()
            for idx in range(len(self.data[:,self._TIMEINDEX])):
                # blend out NaNs in lat and long
                if np.isnan(self.data[idx,self._LATINDEX] + self.data[idx,self._LONINDEX]):
                    self.data[idx,self._DISTINDEX] = np.nan
                    continue

                # one magnitude slower than geopy.distance.great_circle
                # distance = geopy.distance.distance(locs[0],(file_data['latitude']['data'][idx], file_data['longitude']['data'][idx])).m
                # exclude wrong coordinates
                self.data[idx, self._DISTINDEX] = geopy.distance.great_circle(location,
                                                       (self.data[idx,self._LATINDEX],
                                                        self.data[idx, self._LONINDEX])).km

            end_time = time.perf_counter()
            elapsed_sec = end_time - start_time
            temp = 'time for single station distance calc using geopy [s]: {:.3f}'.format(elapsed_sec)
            self.logger.info(temp)
        else:
            pass

        # return self.data

    ###################################################################################
    def colocate(self, location=None, max_dist=50., bbox=None, resample_to_grid=None):
        """return a point cloud of points that fall within a given distance in km around a given location

        Because the average earth radius in geopy.distance.EARTH_RADIUS is given in km, the result is also given in km

        This method returns the same values as geopy.distance.great_circle

        Example
        -------
        >>> import logging
        >>> import pyaerocom.io.read_aeolus_l2b_data
        >>> obj = pyaerocom.io.read_aeolus_l2b_data.ReadAeolusL2bData(loglevel=logging.DEBUG)
        >>> obj.read(vars_to_read=['ec550aer'])
        >>> # return the data points within a radius of a certain station
        >>> location = [(49.093,8.428,0.)]
        >>> max_dist = 300
        >>> result = obj.colocate(location=location, max_dist=max_dist)
        >>> import numpy as np
        >>> print('min distance: {:.3f} km'.format(np.nanmin(obj.data[:, obj._DISTINDEX])))
        >>> print('max distance: {:.3f} km'.format(np.nanmax(obj.data[:, obj._DISTINDEX])))
        >>> # return the data belonging to a certain area (here: EUROPE)
        >>> bbox = [(30., 80., -20., 70.)]
        >>> result = obj.colocate(bbox = bbox)
        >>> # return the data for regridding (selects the data for every grid point using the bboxes)
        >>> result = obj.colocate(resample_to_grid=True)
        >>> # find non empty bboxes
        >>> length=[len(result[x]['data']) for x in range(len(result))]
        >>> indexes=np.where(np.asarray(length) > 0.)[0].tolist()
        >>> result[indexes[0]]['bbox']
        >>> result[indexes[0]]['data'].shape
        """

        start_time = time.perf_counter()

        ret_data = np.empty([self._ROWNO, self._COLNO], dtype=np.float_)
        index_counter = 0
        cut_flag = True
        if location is not None:
            if isinstance(location,list):
                # parameter is a list
                # iterate
                for idx_lat, loc in enumerate(location):
                    # calculate distance
                    self.calc_dist_in_data(loc)
                    matching_indexes = np.where(self.data[:,self._DISTINDEX] < max_dist)
                    matching_length = len(matching_indexes[0])
                    if matching_length > 0:
                        # check if the data fits into ret_data
                        end_index = index_counter + matching_length
                        if end_index > len(ret_data):
                            # append the needed elements
                            logging.info('adding {} elements to ret_data'.format(end_index - len(ret_data)))
                            ret_data = np.append(ret_data,
                                np.zeros([end_index - len(ret_data),
                                          self._COLNO], dtype=np.float_,axis=0))
                            cut_flag = False

                        ret_data[index_counter:index_counter+matching_length,:] = self.data[matching_indexes,:]
                        index_counter += matching_length

            elif isinstance(location, tuple):
                logging.error('passing one location as tuple not supported at this point')
                pass
            else:
                logging.error('locations have to be passed as a list of tuples with (lat, lon)')
                pass
            # return only the part of the array containing some values
            if cut_flag:
                ret_data = ret_data[:index_counter, :]
            end_time = time.perf_counter()
            elapsed_sec = end_time - start_time
            temp = 'time for single station distance calc [s]: {:.3f}'.format(elapsed_sec)
            self.logger.info(temp)
            # log the found times
            unique_times = np.unique(self.data[matching_indexes, self._TIMEINDEX]).astype('datetime64[s]')
            self.logger.info('matching times:')
            self.logger.info(unique_times)

        elif bbox is not None:
            # return points in the bounding box given in bbox
            if isinstance(bbox,list):
                # parameter is a list
                # iterate
                ret_data = {}
                for idx, _bbox in enumerate(bbox):
                    ret_data[idx] = {}
                    ret_data[idx]['data'] = self.select_bbox(_bbox)
                    ret_data[idx]['bbox'] = _bbox
            else:
                pass
            end_time = time.perf_counter()
            elapsed_sec = end_time - start_time
            temp = 'time for bbox calc [s]: {:.3f}'.format(elapsed_sec)
            self.logger.info(temp)
        elif resample_to_grid:
            # resample to 1x1 degree grid at this point
            # create a list of tuple with the bounding boxes for a 1x1 degree grid
            # there's more efficient ways of doing that
            start_lat = -90.
            end_lat = 90.
            lat_spacing = 1.
            start_lon = -180.
            end_lon = 180.
            lon_spacing = 1.
            grid_lats_start = list(np.arange(start_lat,end_lat,lat_spacing))
            grid_lats_end = list(np.arange(start_lat+lat_spacing,end_lat+lat_spacing,lat_spacing))
            grid_lons_start = list(np.arange(start_lon,end_lon,lon_spacing))
            grid_lons_end = list(np.arange(start_lon+lon_spacing,end_lon+lon_spacing,lon_spacing))

            bbox_temp = []
            for idx_lat in range(len(grid_lats_start)):
                for idx_lon in range(len(grid_lons_start)):
                    bbox_temp.append((grid_lats_start[idx_lat], grid_lats_end[idx_lat],
                                 grid_lons_start[idx_lon], grid_lons_end[idx_lon]))
            ret_data = {}
            for idx, _bbox in enumerate(bbox_temp):
                # select the values
                ret_data[idx] = {}
                ret_data[idx]['data'] = self.select_bbox(_bbox)
                ret_data[idx]['bbox'] = _bbox
                if idx > 0 and idx % 1000 == 0:
                    self.logger.warning('{} boxes co-located'.format(idx))

            end_time = time.perf_counter()
            elapsed_sec = end_time - start_time
            temp = 'time for gridding calc [s]: {:.3f}'.format(elapsed_sec)
            self.logger.info(temp)
        else:
            pass

        return ret_data

    ###################################################################################

    def select_bbox(self,bbox=None):
        """method to return all points of self.data laying within a certain latitude and longitude range

        This method will likely never be used by a user, but serves as helper method for the colocate method

        EXAMPLE
        =======
        >>> import logging
        >>> import pyaerocom.io.read_aeolus_l2b_data
        >>> obj = pyaerocom.io.read_aeolus_l2b_data.ReadAeolusL2bData(loglevel=logging.DEBUG)
        >>> obj.read(vars_to_read=['ec550aer'])
        >>> bbox = (-62.,-61.,7.,8.)
        >>> result = obj.select_bbox(bbox)
        >>> import numpy as np
        >>> print('min distance: {:.3f} km'.format(np.nanmin(obj.data[:, obj._DISTINDEX])))
        >>> print('max distance: {:.3f} km'.format(np.nanmax(obj.data[:, obj._DISTINDEX])))


        """
        start_time = time.perf_counter()

        # ret_data = np.empty([self._ROWNO, self._COLNO], dtype=np.float_)
        # index_counter = 0
        # cut_flag = True

        if bbox is not None:
            logging.info(bbox)
            lat_min = bbox[0]
            lat_max = bbox[1]
            lon_min = bbox[2]
            lon_max = bbox[3]

            # remove NaNs at this point
            matching_indexes = np.where(np.isfinite(self.data[:, self._LATINDEX]))
            ret_data = self.data[matching_indexes[0],:]

            # np.where can unfortunately only work with a single criterion
            matching_indexes = np.where(ret_data[:, self._LATINDEX] <= lat_max)
            ret_data = ret_data[matching_indexes[0],:]
            # logging.warning('len after lat_max: {}'.format(len(ret_data)))
            matching_indexes = np.where(ret_data[:, self._LATINDEX] > lat_min)
            ret_data = ret_data[matching_indexes[0], :]
            # logging.warning('len after lat_min: {}'.format(len(ret_data)))
            matching_indexes = np.where(ret_data[:, self._LONINDEX] <= lon_max)
            ret_data = ret_data[matching_indexes[0], :]
            # logging.warning('len after lon_max: {}'.format(len(ret_data)))
            matching_indexes = np.where(ret_data[:, self._LONINDEX] > lon_min)
            ret_data = ret_data[matching_indexes[0], :]
            # logging.warning('len after lon_min: {}'.format(len(ret_data)))
            # matching_length = len(matching_indexes[0])

            # end_time = time.perf_counter()
            # elapsed_sec = end_time - start_time
            # temp = 'time for single station bbox calc [s]: {:.3f}'.format(elapsed_sec)
            # self.logger.info(temp)
            # log the found times
            # unique_times = np.unique(self.data[matching_indexes,self._TIMEINDEX]).astype('datetime64[s]')
            # self.logger.info('matching times:')
            # self.logger.info(unique_times)
            return ret_data
    ###################################################################################

