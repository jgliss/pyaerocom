"""Interface for reading EEA AqERep files (formerly known as Airbase data).

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

from pyaerocom import const
from pyaerocom.io.read_eea_aqerep_base import ReadEEAAQEREPBase


class ReadEEAAQEREP_V2(ReadEEAAQEREPBase):
    """Class for reading EEA AQErep data

    Extended class derived from  low-level base class :class: ReadUngriddedBase
    that contains the main functionality.
    """
    #: Name of the dataset (OBS_ID)
    DATA_ID = const.EEA_V2_NAME

    #: List of all datasets supported by this interface
    SUPPORTED_DATASETS = [DATA_ID]

    #: Eionet offers 2 data revisions
    #: E2a (near real time) and E1a (quality controlled)
    #: this class reads the E2a data for now.
    # But by changing the base path
    # and this constant, it can also read the E1a data set
    DATA_PRODUCT = 'E1a'

    #: file masks for the data files
    FILE_MASKS = {}
    FILE_MASKS['concso2'] = '**/AT*_1_*_timeseries.csv'
    FILE_MASKS['concpm10'] = '**/XK*_5_*_timeseries.csv'
    FILE_MASKS['conco3'] = '**/XK*_7_*_timeseries.csv'
    FILE_MASKS['concno2'] = '**/XK*_8_*_timeseries.csv'
    FILE_MASKS['concno2'] = '**/AT*_8_*_timeseries.csv'
    FILE_MASKS['concco'] = '**/AT*_10_*_timeseries.csv'
    FILE_MASKS['concno'] = '**/AT*_38_*_timeseries.csv'
    FILE_MASKS['concpm25'] = '**/XK*_6001_*_timeseries.csv'



if __name__ == "__main__":

    # Test that the reading routine works
    from pyaerocom.io.read_eea_aqerep_v2 import ReadEEAAQEREP_V2
    import logging
    station_id = {}
    station_id['concso2'] = 'AT31703'
    station_id['concpm10'] = 'XK0001A'
    station_id['conco3'] = 'XK0002A'
    station_id['concno2'] = 'XK0002A'
    station_id['concno2'] = 'AT31703'
    station_id['concco'] = 'XK0002A'
    station_id['concco'] = 'AT4S416'
    station_id['concno'] = 'XK0002A'
    station_id['concno'] = 'AT4S416'
    station_id['concpm25'] = 'XK0002A'

    var_names_to_test = station_id.keys()
    # var_names_to_test = ['concpm25']
    for var_name in var_names_to_test:
        r = ReadEEAAQEREP_V2()
        r.logger.setLevel(logging.INFO)
        data = None
        data = r.read(vars_to_retrieve = [var_name])
        print('{} data read'.format(var_name))
        try:
            print(data[station_id[var_name]])
            print(data[station_id[var_name]][var_name])
        except:
            print('failed test var {}'.format(var_name))
            pass


