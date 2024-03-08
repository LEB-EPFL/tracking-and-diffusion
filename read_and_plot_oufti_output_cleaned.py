# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
# import matplotlib
from matplotlib.gridspec import GridSpec
import os
import re
import pylab
import pandas
import numpy
import warnings
import json
import datetime
import random
import pdb

with open('segmentation_general_info.json') as jf:
    data = json.load(jf)

px_to_micron = data['px_to_micron']  # μm per pixel
basic_directory = data['basic_directory']
basic_directory_paper = data['basic_directory_paper']
all_starvation_times = data['all_starvation_times']
days = data['all_days']


all_segs = {}  # all segmentation results
area_results = {}
seg_coords = {}


def load_segmentation_results(days=data['all_days'], loc_type='file'):
    print(days)
    for d in days:
        for i in data[loc_type][d].keys():
            if isinstance(data[loc_type][d][i], str):
                movies = [data[loc_type][d][i]]
            else:
                movies = [x[0] for x in data[loc_type][d][i]]

            for idx, movie in enumerate(movies):
                print(i)
                print(movie)
                if len(movies) == 1:
                    label = d + '_' + i
                else:
                    label = movie.split('h/')[1]
                print(label)
                # pdb.set_trace()
                if loc_type == 'localized_metrics':
                    location = os.path.join(data["basic_directory"], d, i,
                                            data[loc_type][d][i][idx][0],
                                            'segmentation.csv')
                elif loc_type == 'single_origin':
                    location = os.path.join(data["basic_directory"], d, 'single_origin',
                                            data[loc_type][d][i][idx][0],
                                            'segmentation.csv')
                else:
                    location = data[loc_type][d][i]
                skip = data['skip'][d][i] if loc_type == 'file' else data[loc_type][d][i][idx][1]
                all_segs[label] = pandas.read_csv(location, skiprows=skip, index_col=13)
                # to use particle ids as the index. this also somehow fixes the problem of
                # misalignment,
                # where column titles did not match between the loaded dataframe and the csv file.
                # let's see how this goes.

                area_results[label] = all_segs[label].copy(deep=True)
        #            seg_coords[label]

                try:
                    area_results[label].drop(columns=['ancestors', 'frameNumber', 'birthframe', 'descendants', 'divisions', 'spots', 'polarity', 'cellId;', 'box'], inplace = True) # get rid of columns we won't use (most are empty)
                except KeyError:
                    area_results[label].drop(columns=['ancestors', 'frameNumber', 'birthframe', 'descendants', 'divisions', 'spots', 'polarity', 'box'], inplace = True)

                area_results[label]['nucleoid_xcoords'] = numpy.nan  # x coordinates of the nucleoid boundary
                area_results[label]['nucleoid_ycoords'] = numpy.nan
                area_results[label]['nucleoid_area'] = numpy.nan
                area_results[label]['sum_intensity'] = numpy.nan
                area_results[label]['mean_intensity'] = numpy.nan
                try:
                    area_results[label]['image'] = data['image'][d][i]
                except:
                    print("No image loaded")
                for p in area_results[label].index:
                    intensity = numpy.nan
                    n_area = numpy.nan
                    c_xs = []
                    c_ys = []
                    n_xs = []
                    n_ys = []
                    nucleoid_info = area_results[label].loc[p, 'objects'].split(';')
                    cell_info = area_results[label].loc[p, 'mesh'].split(';')
                    c_xs = (list(map(numpy.float64, cell_info[0].split())) +
                            list(map(numpy.float64, cell_info[2].split())))  # [1:][::-1][1:][::-1]
                    c_ys = (list(map(numpy.float64, cell_info[1].split())) +
                            list(map(numpy.float64, cell_info[3].split())))  # [1:][::-1][1:][::-1]

                    if len(nucleoid_info) > 1:
                        n_area = numpy.float(nucleoid_info[4])

                        n_xs = list(map(numpy.float64, nucleoid_info[0].split()))
                        n_ys = list(map(numpy.float64, nucleoid_info[1].split()))
                        # Added by Willi
                        intensity = nucleoid_info[3].split()
                        intensity = [int(i) for i in intensity]
                        area_results[label].loc[p, 'sum_intensity'] = numpy.sum(intensity)
                        area_results[label].loc[p, 'mean_intensity'] = numpy.sum(intensity)/n_area
                    if len(nucleoid_info) > 5:
                        n_area = n_area + numpy.float(nucleoid_info[9])
                        n_xs = n_xs + list(map(numpy.float64, nucleoid_info[5].split()))
                        n_ys = n_ys + list(map(numpy.float64, nucleoid_info[6].split()))
                        # Added by Willi
                        intensity = nucleoid_info[8].split()
                        intensity = [int(i) for i in intensity]
                        area_results[label].loc[p, 'sum_intensity'] = numpy.sum(intensity)
                        area_results[label].loc[p, 'mean_intensity'] = numpy.sum(intensity)/n_area
        #                n_xs = numpy.array(n_xs)
        #                n_ys = numpy.array(n_ys)
                    area_results[label].loc[p, 'cell_xcoords'] = str(c_xs)
                    area_results[label].loc[p, 'cell_ycoords'] = str(c_ys)
                    area_results[label].loc[p, 'nucleoid_area'] = n_area
                    area_results[label].loc[p, 'nucleoid_xcoords'] = str(n_xs)
                    area_results[label].loc[p, 'nucleoid_ycoords'] = str(n_ys)

                area_results[label]['nc_ratio'] = (area_results[label]['nucleoid_area'] /
                                                   area_results[label]['area'])
                print(label)
                try:
                    strain_re = re.compile(r".*_(\w{2,3}\d{1,3})_.*")
                    strain = strain_re.search(label).group(1)
                except AttributeError:
                    strain = 'WT' if 'wt' in label.lower() else 'DpolyP'
                area_results[label]['strain'] = strain
                try:
                    starvation_time_re = re.compile(r".*_([06]h)\w{0,4}_.*")
                    starvation_time = starvation_time_re.search(label).group(1)
                except AttributeError:
                    starvation_time = '0h' if '0h' in label.lower() else '6h'
                area_results[label]['starvation_time'] = starvation_time
                print(starvation_time)
                if starvation_time != '0h':
                    try:
                        area_results[label]['condition'] = i.split('_')[2]
                    except IndexError:
                        area_results[label]['condition'] = i.split('_')[0]
                else:
                    area_results[label]['condition'] = ''
                area_results[label]['day'] = d
                area_results[label].drop(columns=['mesh', 'signal0', 'objects'], inplace=True)
                # no longer need these columns
                for p in area_results[label].index:
                    area_results[label].loc[p, 'new_id'] = str(d) + str(p).zfill(4)
                area_results[label].set_index('new_id', inplace=True)

    return area_results, all_segs


def pool_segmentation_results(days=data['all_days']):
    if days is None:
        days = data['all_days']
    r = load_segmentation_results(days)[0]
    rp = {}

    for i in data['labels']:
        print(i)
        dfs_now = []
        for d in days:
            dfs_now.append(r[d + '_' + i])
        rp[i] = pandas.concat(dfs_now)

    return rp

##########################################################

pylab.rcParams.update({'font.size': 16})
pylab.rcParams.update({'ytick.major.width': 2})
pylab.rcParams.update({'ytick.major.size': 8})
pylab.rcParams.update({'ytick.direction': 'inout'})
pylab.rcParams.update({'xtick.major.width': 2})
pylab.rcParams.update({'xtick.major.size': 8})
pylab.rcParams.update({'xtick.direction': 'inout'})
pylab.rcParams.update({'ytick.minor.width': 1.2})
pylab.rcParams.update({'ytick.minor.size': 4})
pylab.rcParams.update({'xtick.minor.width': 1.2})
pylab.rcParams.update({'xtick.minor.size': 4})
pylab.rcParams.update({'axes.linewidth' : 2})
pylab.rcParams.update({'hatch.linewidth':  2})
pylab.rcParams.update({'hatch.color':  'w'})
pylab.rcParams.update({'axes.linewidth':2})
pylab.rcParams.update({'axes.linewidth':2})

colors = {}

colors['WT'] = {}
colors['WT']['0h'] = '#00BDF9' #64F003'
colors['WT']['6h_lowN'] =  '#0088F9' #419E01'
colors['WT']['6h_lowC'] =  '#00559C'
#colors['bLR31']['24h'] = '#0067BC' #2E6E01'

colors['DpolyP'] = {}
colors['DpolyP']['0h'] = '#FFB900'
colors['DpolyP']['6h_lowN'] = '#C08B00'
colors['DpolyP']['6h_lowC'] = '#7B5900'
#colors['DpolyP']['24h'] = '#886200'

ms = 12  # markersize

htc = {}  # hatches for the histograms

htc['0h'] = '//'
htc['6h'] = ''
# define symbol shape for each condition (previously timepoint) #
markershapes = {}

markershapes['0h'] = 'o'
markershapes['6h_lowN'] = '^'
markershapes['6h_lowC'] = 's'

xpositions = {}
xpositions['WT'] = {}
xpositions['WT_0h'] = 1
xpositions['WT_6h_lowN'] = 1.2
#xpositions['WT_24h'] = 1.4

xpositions['DpolyP'] = {}
xpositions['DpolyP_0h'] = 1.4
xpositions['DpolyP_6h_lowN'] = 1.6
#xpositions['DpolyP_24h'] = 2.4

position_subplot = {}  # the positions of the subplots for each condition and strain
position_subplot['WT_0h'] = 0
position_subplot['WT_6h_lowN'] = 1#5#2#1
position_subplot['WT_6h_lowC'] = 11#2  #1
position_subplot['DpolyP_0h'] = 2
position_subplot['DpolyP_6h_lowN'] = 3#4#3#4 #3
position_subplot['DpolyP_6h_lowC'] = 5 #3

position_subplot['201110'] = 0
position_subplot['201215'] = 1
position_subplot['201225'] = 2#5#2#1


def plot_nc_ratios_per_day_paper(days = ['201110', '201215', '201225']):

    xmin = 0.3
    xmax = 1.0
    ymin = 0
    ymax = 43
    bins_now = numpy.linspace(xmin, xmax, 25)

    results = load_segmentation_results()[0]

    for label in ['WT_0h', 'WT_6h_lowN',
                  'DpolyP_0h',
                  'DpolyP_6h_lowN'
                  ]:
        s = label.split('_')[0]
        t = label.split('_')[1]
        if t != '0h':
            t = t + '_' + label.split('_')[2]
        fig_label = 'NC ratio ' + label
        fig = pylab.figure(fig_label, figsize = (7.5, 7))
        fig.suptitle(fig_label)
        gs = GridSpec(len(days), 10, figure=fig)
        ax = {}
        fig.subplots_adjust(top=0.9)
        pylab.subplots_adjust(hspace=0.5)
        for j,d in enumerate(days):
            data_label = d + '_' + label
            data_now = results[data_label].nc_ratio
            ax[j] = fig.add_subplot(gs[j:j+1, :8])
            weights = 100 * numpy.ones_like(data_now) / len(data_now)
            j = position_subplot[d]
            ax[j].hist(data_now, bins = bins_now, weights = weights, color = colors[s][t], alpha = 1, label = label, hatch = htc[t.split('_')[0]])
            if t == '0h':
                ax[j].hist(data_now, bins = bins_now, weights = weights, histtype = 'step', color = colors[s][t], alpha = 1, label = label, lw = 2)
            ax[j].set_ylim(ymin, ymax)
            ax[j].set_xlim(xmin, xmax)
            ax[j].text(xmax - 0.23, ymax - 15, 'N = ' + str(len(data_now)) + '\nday: ' + d)

        ax[j].set_xlabel('nucleoid area / cell area')
        fig.text(0.04, 0.5, '% of particles', va='center', rotation='vertical')
        pylab.savefig(basic_directory + '_segmented/day_to_day_' + label + '.svg')
        pylab.savefig(basic_directory + '_segmented/day_to_day_' + label + '.png')

    return results

def plot_nc_ratios_hist_paper(pooled_results = None):

    now = datetime.datetime.now()
    string_now = now.strftime("%Y%m%d")

    xmin = 0.3  # 0.4
    xmax = 1  # 0.9
    ymin = 0
    ymax = 32  # 23
    bins_now = numpy.linspace(xmin, xmax, 25)

    if not isinstance(pooled_results, dict):
        results = load_segmentation_results()[0]
        pooled_results = pool_segmentation_results()

    fig_label = 'NC ratio '
    fig = pylab.figure(fig_label, figsize = (7.5, 7))
    fig.suptitle(fig_label)
    gs = GridSpec(4, 9, figure=fig)
    ax = {}
    fig.subplots_adjust(top=0.9)
    pylab.subplots_adjust(hspace=0.1)

    for label in data['labels']:
        s = label.split('_')[0]
        t = label.split('_')[1]
        if t != '0h':
            t = t + '_' + label.split('_')[2]
        data_now = pooled_results[label].nc_ratio
        weights = 100 * numpy.ones_like(data_now) / len(data_now)
        j = position_subplot[label]
        ax[j] = fig.add_subplot(gs[j:j+1, :8])
        ax[j].hist(data_now, bins = bins_now, weights = weights, color = colors[s][t], alpha = 1, label = label, hatch = htc[t.split('_')[0]])
        if t == '0h':
            ax[j].hist(data_now, bins = bins_now, weights = weights, histtype = 'step', color = colors[s][t], alpha = 1, label = label, lw = 2)
        ax[j].set_ylim(ymin, ymax)
        ax[j].set_xlim(0.4, 0.9)
        ax[j].text(0.75, ymax - 7, 'N = ' + str(len(data_now)))

        if j < 3:
            ax[j].set_xticks([])
        if j == 3:
            ax[3].set_xlabel('nucleoid area / cell area')

        fig.text(0.04, 0.5, '% of particles', va='center', rotation='vertical')
    pylab.savefig(basic_directory + '_segmented/' + string_now + '_nc_ratio.svg')
    pylab.savefig(basic_directory + '_segmented/' + string_now + '_nc_ratio.png')

    pylab.savefig(basic_directory_paper + string_now + '_nc_ratio.svg')
    pylab.savefig(basic_directory_paper + string_now + '_nc_ratio.png')

    return pooled_results

def plot_nc_ratios_dots_paper(pooled_results = None, quantity = 'nc_ratio'):

    now = datetime.datetime.now()
    string_now = now.strftime("%Y%m%d")

    ymin = 0.55
    ymax = 0.67

    if not isinstance(pooled_results, dict):
        results = load_segmentation_results()[0]
        pooled_results = pool_segmentation_results()

    fig_label = 'NC ratio '
    fig = pylab.figure(fig_label)
    pylab.title(fig_label)

    for label in data['labels']:
        s = label.split('_')[0]
        t = label.split('_')[1]
        if t != '0h':
            t = t + '_' + label.split('_')[2]
        data_now = pooled_results[label].loc[:,quantity]
        pylab.errorbar(xpositions[label], numpy.median(data_now), xerr = None, yerr = data_now.std() / numpy.sqrt(len(data_now)), color = colors[s][t], alpha = 1, label = label, fmt = markershapes[t], markersize = ms, ls = '--', markeredgecolor = 'k', markeredgewidth = 2, capsize = 10, ecolor = colors[s][t])
        pylab.ylim(ymin, ymax)
        pylab.ylabel('nucleoid area / cell area')
        pylab.xticks([1, 1.2, 1.4, 1.6], ['WT\n0h', 'WT\n6h lowN', 'ΔpolyP\n0h', 'ΔpolyP\n6h lowN'])

    pylab.savefig(basic_directory + '_segmented/' + string_now + '_nc_ratio_summary.svg')
    pylab.savefig(basic_directory + '_segmented/' + string_now + '_nc_ratio_summary.png')

    pylab.savefig(basic_directory_paper + string_now + '_nc_ratio_summary.svg')
    pylab.savefig(basic_directory_paper + string_now + '_nc_ratio_summary.png')

    return pooled_results

def plot_segmented_cell(label, seg_results = None, particle_id = 'random', nparticles = 1, channel = 'phase'):
    '''
    Can loop over neighboring particles if we want more than one.
    ΔpolyP 6h lowN 201215: 2012150399, 2012150114, 2012150553, 2012150187, 2012150074, 2012150544
    ΔpolyP 0h: 2012150008, 2012150168, 2012150065
    WT 0h: 2012150036, 2012150299
    WT 6h lowN: 2012150039, 2012150192, 2012150049
    '''

    if not isinstance(seg_results, dict):
        seg_results = load_segmentation_results()[0]

    if particle_id == 'random':
        particles = set(seg_results[label].index.to_numpy())
        particle_id = random.sample(particles, 1)[0]

    day = label.split('_')[0]
    case = label.split(day)[1][1:]
    if channel == 'phase':
        image = pylab.imread(data['phase_image'][day][case])
        colormap_now = pylab.gray()
    elif channel == 'fluorescence':
        image = pylab.imread(data['fl_image'][day][case])
        colormap_now = pylab.viridis()

    particle_id = str(particle_id)
    print(particle_id)

    c_coords = seg_results[label].loc[:,['cell_xcoords', 'cell_ycoords']]
    n_coords = seg_results[label].loc[:,['nucleoid_xcoords', 'nucleoid_ycoords']]
    cx_coords = c_coords.cell_xcoords
    cy_coords = c_coords.cell_ycoords
    nx_coords = n_coords.nucleoid_xcoords
    ny_coords = n_coords.nucleoid_ycoords

    cx_coords = cx_coords.loc[particle_id]
    nx_coords = nx_coords.loc[particle_id]
    cx_coords = [float(x.strip('[').strip(']')) for x in cx_coords.split(', ')]
    nx_coords = [float(x.strip('[').strip(']')) for x in nx_coords.split(', ')]
    cx_coords = numpy.array(cx_coords)  # set because two points are repeated, and then you end up plotting a line in the middle of the cell
    nx_coords = numpy.array(nx_coords)

    cy_coords = cy_coords.loc[particle_id]
    ny_coords = ny_coords.loc[particle_id]
    cy_coords = [float(x.strip('[').strip(']')) for x in cy_coords.split(', ')]
    ny_coords = [float(x.strip('[').strip(']')) for x in ny_coords.split(', ')]
    cy_coords = numpy.array(cy_coords)
    ny_coords = numpy.array(ny_coords)

    center = [cx_coords.mean(), cy_coords.mean()]
    xo = int(round(center[0], 0))
    yo = int(round(center[1], 0))
    xmin, xmax = xo - 20, xo + 20
    ymin, ymax = yo - 20, yo + 20
 #    subimage = image[(xo - 25):(xo + 25), (yo - 25):(yo + 25)]
    pylab.figure(particle_id + ', ' + channel)
    pylab.title(day + ', ' + case + '\nparticle ' + str(particle_id))
    #    fig.suptitle(particle_id)
    #        gs = GridSpec(12, 10, figure=fig)
    #`    ax     = {}
    pylab.imshow(image, cmap = colormap_now)
    pylab.xlim(xmin, xmax)
    pylab.ylim(ymin, ymax)
    pylab.plot(cx_coords[:int(len(cx_coords)/2)]-1, cy_coords[:int(len(cx_coords)/2)]-1, '-', color = '#FF9900', linewidth = 1)
    pylab.plot(cx_coords[int(len(cx_coords)/2):]-1, cy_coords[int(len(cx_coords)/2):]-1, '-', color = '#FFAA00', linewidth = 1)

    pylab.plot(nx_coords-1, ny_coords-1, 'w-', linewidth = 1)
    pylab.hlines(ymin + 3, xmax - 8 - (1/ (2 * px_to_micron)), xmax - 8 + (1/ (2 * px_to_micron)), color = 'w', linewidth = 3)

    now = datetime.datetime.now()
    string_now = now.strftime("%Y%m%d")

    pylab.savefig(data['basic_directory_paper'] + string_now + '_' + channel + '_' + day + '_' + case + '_' + particle_id + '.svg')
    pylab.savefig(data['basic_directory_paper'] + string_now + '_' + channel + '_' + day + '_' + case + '_' + particle_id + '.png')

    return cx_coords, cy_coords
 #pylab.plot(center[0], center[1], 'ro')
