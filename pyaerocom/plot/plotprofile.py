#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains scatter plot routines for Aerocom data.
"""

from pyaerocom import const
import pyaerocom.io as pio
import pyaerocom as pa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.basemap import Basemap
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# text positions for the annotations
XYPOS=[]
XYPOS.append((.01, 0.95))
XYPOS.append((0.01, 0.90))
XYPOS.append((0.3, 0.90))
XYPOS.append((0.01, 0.86))
XYPOS.append((0.3, 0.86))
XYPOS.append((0.01, 0.82))
XYPOS.append((0.3, 0.82))
XYPOS.append((0.01, 0.78))
XYPOS.append((0.3, 0.78))
XYPOS.append((0.8, 0.1))
XYPOS.append((0.8, 0.06))


def plotcurtain(obj, Options = None, filename=None, var_to_plot=None, what='levels',height_lev_no=24,
                vmin=2.,
                vmax=500.,
                logging=True):
    """low level method to plot 'curtain' plots with data in obk.data
    Essentially a scatter density plot along a track

    At this point we do not care what kind of object we get unless obj.data is existing
    and has the right format"""
    import numpy as np
    import matplotlib.pyplot as plt

    NAN_VAL = -999.
    plt_name = 'CURTAIN'
    if var_to_plot is None:
        try:
            var_to_plot = Options['var_to_plot']
        except TypeError:
            pass

    if what == 'levels':
        ec = obj.data[:, obj._INDEX_DICT[var_to_plot]]
        nan_indexes = np.where(np.isnan(ec))
        ec[nan_indexes] = NAN_VAL
        times = np.int(len(ec) / height_lev_no)
        ec = ec.reshape(times, height_lev_no).transpose()
        plot = plt.pcolormesh(ec, cmap='jet', vmin=vmin, vmax=vmax)
        plot.axes.set_xlabel('time step number')
        plot.axes.set_ylabel('height step number')
    elif what == 'mpl_scatter_density':
        import mpl_scatter_density
        nonnan_indexes = np.where(np.isfinite(obj.data[:,obj._ALTITUDEINDEX]))
        ec = obj.data[nonnan_indexes, obj._INDEX_DICT[var_to_plot]]
        altitudes = obj.data[nonnan_indexes,obj._ALTITUDEINDEX]
        times = obj.data[nonnan_indexes,obj._TIMEINDEX]
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
        ax.scatter_density(times, altitudes, c=ec, vmin=0., vmax=500, cmap='jet')
        ax.set_xlabel('time')
        ax.axes.set_ylabel('height')
    elif what == 'resample':
        pass
    else:
        pass
    ax.axes.set_title(what)
    obj.logger.info('plotting file: {}'.format(filename))
    plt.savefig(filename, dpi=300)
    plt.close()



    # var_to_run = Options['VariablesToRun'][0]
    # obs_network_name = Options['ObsNetworkName'][0]






    #
    # filter_name = 'WORLD-wMOUNTAINS'
    # filter_name = 'WORLD'
    # time_step_name = 'mALLYEARdaily'
    # # OD550_AER_an2008_YEARLY_WORLD_SCATTERLOG_AeronetSunV3Lev2.0.daily.ps.png
    # # if df_time[model_name].index[0].year != df_time[model_name].index[-1].year:
    # years_covered = df_time[model_name].index[:].year.unique().sort_values()
    # if len(years_covered) > 1:
    #     filename = '{}_{}_an{}-{}_{}_{}_{}_{}.png'.format(model_name,
    #                                                      var_to_run,years_covered[0],
    #                                                   years_covered[-1],
    #                                                   time_step_name, filter_name, plt_name,
    #                                                   obs_network_name)
    #     plotname = "{}-{} {}".format(years_covered[0], years_covered[-1], 'daily')
    #     title = "{} {} station list {}-{}".format(var_to_run, filter_name, years_covered[0], years_covered[-1])
    # else:
    #     filename = '{}_{}_an{}_{}_{}_{}_{}.png'.format(model_name,
    #                                                   var_to_run,years_covered[0],
    #                                                   time_step_name, filter_name, plt_name,
    #                                                   obs_network_name)
    #     plotname = "{} {}".format(years_covered[0], 'daily')
    #     title = "{} {} station list {}".format(var_to_run, filter_name, years_covered[0])
    #
    # if verbose:
    #     sys.stdout.write(filename+"\n")
    #
    # LatMin = -90
    # LatEnd = 90.
    # LonMin = -180.
    # LonEnd = 180.
    #
    # basemap_flag=False
    # if basemap_flag:
    #     m = Basemap(projection='cyl', llcrnrlat=LatMin, urcrnrlat=LatEnd,
    #                 llcrnrlon=LonMin, urcrnrlon=LonEnd, resolution='c', fix_aspect=False)
    #
    #     x, y = m(obs_lons, obs_lats)
    #     # m.drawmapboundary(fill_color='#99ffff')
    #     # m.fillcontinents(color='#cc9966', lake_color='#99ffff')
    #     plot = m.scatter(x, y, 4, marker='o', color='r', )
    #     m.drawmeridians(np.arange(-180,220,40),labels=[0,0,0,1], fontsize=10)
    #     m.drawparallels(np.arange(-90,120,30),labels=[1,1,0,0], fontsize=10)
    #     # axis = plt.axis([LatsToPlot.min(), LatsToPlot.max(), LonsToPlot.min(), LonsToPlot.max()])
    #     ax = plot.axes
    #     m.drawcoastlines()
    # else:
    #     ax = plt.axes([0.15,0.1,0.8,0.8],projection=ccrs.PlateCarree())
    #     ax.set_ylim([LatMin, LatEnd])
    #     ax.set_xlim([LonMin, LonEnd])
    #     #ax.set_aspect(2)
    #
    #     ax.coastlines()
    #     plot = plt.scatter(obs_lons, obs_lats, 8, marker='o', color='r')
    #     #plot.axes.set_aspect(1.8)
    #
    #
    #     # lon_formatter = LongitudeFormatter(number_format='.1f', degree_symbol='')
    #     # lat_formatter = LatitudeFormatter(number_format='.1f', degree_symbol='')
    #     # ax.xaxis.set_major_formatter(lon_formatter)
    #     # ax.yaxis.set_major_formatter(lat_formatter)
    #     xticks = ax.set_xticks([-180., -120., -60., 0., 60, 120, 180])
    #     yticks = ax.set_yticks([-90., -60, -30, 0., 30, 60, 90])
    #     # ax.annotate('source: AEROCOM', xy=(0.93, 0.04), xycoords='figure fraction',
    #     #             horizontalalignment='right', fontsize=9, bbox=dict(boxstyle='square', facecolor='none',
    #     #                                                                 edgecolor='black'))
    #     ax.annotate('longitude', xy=(0.55, 0.12), xycoords='figure fraction',
    #                 horizontalalignment='center', fontsize=12, )
    #     ax.annotate('latitude', xy=(0.07, 0.55), xycoords='figure fraction', rotation=90,
    #                 horizontalalignment='center', fontsize=12, )
    #     # ax.set_xlabel = 'longitude'
    #     # ax.set_ylabel = 'latitude'
    #     ax.annotate(model_name, xy=(174., -83.), xycoords='data', horizontalalignment='right', fontsize=13,
    #                 fontweight='bold',
    #                 color='black', bbox=dict(boxstyle='square', facecolor='white', edgecolor='none', alpha=0.7))
    #     ax.annotate("No of stations: {}".format(station_no), xy=(-174., -83.), xycoords='data',
    #                 fontweight='bold',
    #                 horizontalalignment='left', fontsize=13,
    #                 color='black', bbox=dict(boxstyle='square', facecolor='white', edgecolor='none', alpha=0.7))
    #
    #
    #
    # plt.title(title, fontsize=13)
    # plt.xticks(fontsize=11)
    # plt.yticks(fontsize=11)
    # plt.xlabel = 'longitude'
    # plt.ylabel = 'latitude'
    #
    #
    #
