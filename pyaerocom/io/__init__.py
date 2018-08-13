################################################################
# read/__init__.py
#
# init for data reading
#
# this file is part of the aerocom_pt package
#
#################################################################
# Created 20171030 by Jan Griesfeller for Met Norway
#
# Last changed: See git log
#################################################################

#Copyright (C) 2017 met.no
#Contact information:
#Norwegian Meteorological Institute
#Box 43 Blindern
#0313 OSLO
#NORWAY
#E-mail: jan.griesfeller@met.no
#This program is free software; you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation; either version 3 of the License, or
#(at your option) any later version.
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU General Public License for more details.
#You should have received a copy of the GNU General Public License
#along with this program; if not, write to the Free Software
#Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#MA 02110-1301, USA

# =============================================================================
# from .read_aeronet_sdav2 import ReadAeronetSDAV2
from .read_aeronet_invv2 import ReadAeronetInvV2
from .read_aeronet_sdav3 import ReadAeronetSdaV3
from .read_aeronet_sunv2 import ReadAeronetSunV2
from .read_aeronet_sunv3 import ReadAeronetSunV3
from .ebas_nasa_ames import EbasNasaAmesFile
from .ebas_sqlite_query import EbasSQLRequest, EbasFileIndex


from .readgridded import ReadGridded, ReadGriddedMulti
from .readungridded import ReadUngridded
from .fileconventions import FileConventionRead

from .read_c3s_l2_satellite_data import ReadC3sL2SatelliteData
from .read_aeolus_l2b_data import ReadAeolusL2bData
from . import testfiles