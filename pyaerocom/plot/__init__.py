################################################################
# plot/__init__.py
#
# init for map plotting
#
# this file is part of the aerocom_pt package
#
#################################################################
# Created 20171130 by Jan Griesfeller for Met Norway
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


# Note that these are temporary imports, namespaces will be redefined in the
# near future
from .plotmaps import plotmaps, annotate_aerocom, plot_ts_map
from .mapping import plot_map
from .plotscatter import plotscatter
from .plotsitelocation import plotsitelocation
from .plotprofile import plotcurtain
