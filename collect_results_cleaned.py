# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
import process_msds
import pdb
import matplotlib
import pylab
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import pandas
import numpy
import scipy
import trackpy
import plot_tracking_results
import os
import re
import sys
#from IPython.display import clear_output
import time
import json
import datetime
import load_all_results
import random
import intersect
import collect_results

##### SETUP #####

px_to_micron = 0.10748 # mum per pixel


with open('general_info.json') as jf:
    data = json.load(jf)

px_to_micron = data['px_to_micron']# mum per pixel
basic_directory = data['basic_directory']
all_starvation_times = data['all_starvation_times']

basic_directory_paper_figs = '//lebsrv2.epfl.ch/LEB_SHARED/SHARED/_Scientific projects/polyP_SoM/Figures/'
basic_directory_paper_data = '//lebsrv2.epfl.ch/LEB_SHARED/SHARED/_Scientific projects/polyP_SoM/paper_material/data/data_bank/'

##### DATA HANDLING FUNCTIONS ###

def combine_particle_results(trajectories, fit_results_alpha_D_app_individual, spot_type, label = None, avoid = None, transfer_columns = ['starting_snr', 'average_starting_magnitude', 'average_starting_offset', 'below_size_limit']):  # no longer transfering diagonal_size
    '''
    INPUT
    -----

    trajectories : a dictionary of pandas DataFrames
        This contains the particle trajectories.

    fit_results_alpha_D_app_individual : a dictionary of pandas DataFrames
        This contains the fit results to the imsds for each particle, i.e. the estimated D_app and α. Typically, it is the outcome of one of the fit functions from process_msds.py.

    spot_type : str
        This can be 'origins', 'muNS', 'fixed_origins', 'fixed_muNS', depending on the object you have tracked.

    label : str, defaults to None
        If you want to attach a label to the filename of the final result, you specify here.

    avoid : list of strings, defaults to None
        If you want to avoid certain datasets, here you can specify characteristic strings in their filename - for example 'lowC'.

    transfer_columns : list of strings, defaults to ['starting_snr', 'diagonal_size', 'starting_magnitude', 'starting_offset']
        A list of the quantities from the trajectories DataFrame that you want to append to the results DataFrame. These strings need to be names of columns in the trajectories DataFrame. Note that it makes most sense to transfer them to the results DataFrame if they are the same for all frames.

    OUTPUT
    ------
    '''
    analysis_directories = load_all_results.define_analysis_directories(spot_type)

    results = fit_results_alpha_D_app_individual.copy()

    now = datetime.datetime.now()
    string_now = now.strftime("%Y%m%d")

    if isinstance(label, str):
        label = label + '_' + string_now
    else:
        label = string_now

    filename = 'combined_results_per_particle_' + label + '.pkl'

    keys_of_interest = list(trajectories.keys())

    if isinstance(avoid, list):
        for l in avoid:
            keys_of_interest = [x for x in keys_of_interest if l not in x]

    for i in keys_of_interest[:]:
        if i not in list(fit_results_alpha_D_app_individual.keys()):
            print(i + ' is included in the trajectories but not in the fit results for α, D_app.')

    for i in list(fit_results_alpha_D_app_individual.keys()):
        if i not in keys_of_interest:
            print(i + ' is included in the fit results for α, D_app but not in the trajectories.')


    keys_of_interest = [x for x in keys_of_interest if x in list(fit_results_alpha_D_app_individual.keys())]
#    print(keys_of_interest)

    number_of_movies = len(keys_of_interest)

    for j, k in list(enumerate(keys_of_interest))[:]:  # note that the keys of the two dictionaries may not always be the same
        print(k)
        d = load_all_results.read('day', k, spot_type)
        s = load_all_results.read('starvation_time', k, spot_type) + 'h'

        for p in fit_results_alpha_D_app_individual[k].index:
            for q in transfer_columns:
                results[k].loc[p, q] = trajectories[k].loc[trajectories[k].particle == p, q].to_numpy().mean()#[0]  # only transfering the first entry for this particle's quantity, since we are only interested in quantities that have the same value across all frames

        sys.stdout.write(str(j) + ' out of ' + str(number_of_movies) + '\n')

        location = basic_directory + d + '/' + spot_type + '/' + s + '/' + k + analysis_directories[d][k]

        results[k].to_pickle(location + filename)

    return results

#### POOLING ####

### First, define categories of data. ###
### Level 1: all movies of the same condition and time lag within a day. Keep track of days.
### Level 2: all movies of the same condition and time lag across all days.

def define_pool_categories(spot_type, time_between_frames = ['5000'], strains = ['bLR1', 'bLR2'], starvation_times = all_starvation_times, conditions = ['lowN', 'lowC']):

    days = load_all_results.select_specifications(spot_type)['all_days']
    categories_with_day = []
    categories_with_day_without_timelags = []
    categories = []
    categories_without_timelags = []

    for t in time_between_frames:
        for s in strains:
            for st in starvation_times:
                if st == '0h':
                    k = s + '_' + st + '_' + str(t).zfill(4) + 'ms'
                    categories.append(k)
                    categories_without_timelags.append(s + '_' + st)
                    for d in days:
                        k = s + '_' + st + '_' + str(t).zfill(4) + 'ms' + '_' + str(d)
                        print(k)
                        categories_with_day.append(k)
                        categories_with_day_without_timelags.append(s + '_' + st + '_' + str(d))
                elif st == '6h':
                    for c in conditions:
                        k = s + '_' + st + '_' + str(t).zfill(4) + 'ms' + '_' + c
                        categories.append(k)
                        print(k)
                        categories_without_timelags.append(s + '_' + st + '_' + c)
                        for d in days:
                            k = s + '_' + st + '_' + str(t).zfill(4) + 'ms' + '_' + c + '_' + str(d)
                            categories_with_day.append(k)
                            categories_with_day_without_timelags.append(s + '_' + st + '_' + c + '_' + str(d))

    categories_without_timelags = list(set(categories_without_timelags))
    categories_with_day_without_timelags = list(set(categories_with_day_without_timelags))

    location = basic_directory + '_' + spot_type + '/'
    #print(location)

    with open(location + 'pool_categories.txt', 'w') as f:
        #indent=2 is not needed but makes the file human-readable
        json.dump('categories_with_day:', f, indent='       ')
        json.dump(categories_with_day, f, indent=2)
        json.dump('categories_with_day_without_timelags:', f, indent='       ')
        json.dump(categories_with_day_without_timelags, f, indent=2)
        json.dump('categories:', f, indent='       ')
        json.dump(categories, f, indent=2)
        json.dump('categories_without_timelags:', f, indent='       ')
        json.dump(categories_without_timelags, f, indent=2)

    #print(test)
    f.close()

    return categories_with_day, categories, categories_without_timelags, categories_with_day_without_timelags

def allocate_movies_to_categories(spot_type, movies = 'all_movies', time_between_frames = ['10000','5000'], strains = ['bLR1', 'bLR2'], starvation_times = all_starvation_times, conditions = ['lowN', 'lowC']):

    # avoid = load_all_results.select_specifications(spot_type)['typically_avoid']
    categories = define_pool_categories(spot_type,
                                        time_between_frames=time_between_frames,
                                        strains=strains,
                                        starvation_times=starvation_times,
                                        conditions=conditions)
    print(categories)

    directories = load_all_results.collect_all_directories(spot_type)

    if movies == 'all_movies':
        movies = []
        for d in directories.keys():
            for m in directories[d]:
                movies.append(m)

    pooled_directories_by_day = {}
    pooled_directories_by_day_without_timelag = {}
    pooled_directories = {}
    pooled_directories_without_timelag = {}

    for i in categories[0]:
        print('current category per day: ' + str(i))
        pooled_directories_by_day[i] = []
    for i in categories[1]:
        pooled_directories[i] = []
    for i in categories[2]:
        pooled_directories_without_timelag[i] = []
    for i in categories[3]:
        pooled_directories_by_day_without_timelag[i] = []

    for m in movies:
        print(m)
        s = load_all_results.read('strain', m, spot_type)
        t = load_all_results.read('time_between_frames', m, spot_type)
        st = load_all_results.read('starvation_time', m, spot_type)
        d = load_all_results.read('day', m, spot_type)
        con = load_all_results.read('condition', m, spot_type)
        if st == '0':
            k_day = s + '_' + st + 'h_' + str(t).zfill(4) + 'ms_' + d
            k = s + '_' + st + 'h_' + str(t).zfill(4) + 'ms'
            k_without_timelag = s + '_' + st + 'h'
            k_day_without_timelag = s + '_' + st + 'h_' + d
        elif st == '6':
            k_day = s + '_' + st + 'h_' + str(t).zfill(4) + 'ms' + '_' + con + '_' + d
            k = s + '_' + st + 'h_' + str(t).zfill(4) + 'ms' + '_' + con
            k_without_timelag = s + '_' + st + 'h_' + con
            k_day_without_timelag = s + '_' + st + 'h_' + con + '_' + d

        pooled_directories_by_day[k_day].append(m)
        pooled_directories_by_day_without_timelag[k_day_without_timelag].append(m)
        pooled_directories[k].append(m)
        pooled_directories_without_timelag[k_without_timelag].append(m)

    now = datetime.datetime.now()
    string_now = now.strftime("%Y%m%d_%H%M%S")
    for archive, string_now in zip(['', '/archive/'], ['', "_" + string_now]):
        f = open(basic_directory + '/_' + spot_type + archive + '/' + 'pooled_movies' + string_now + '.txt', 'w+')

        f.write('POOLED MOVIES BY DAY:\n\n')
        for i in pooled_directories_by_day.keys():
            f.write(i + ':\n')
            for j in pooled_directories_by_day[i]:
                f.write(j + '\n')
            f.write('\n')

        f.write('POOLED MOVIES BY DAY WITHOUT TIMELAGS:\n\n')
        for i in pooled_directories_by_day_without_timelag.keys():
            f.write(i + ':\n')
            for j in pooled_directories_by_day_without_timelag[i]:
                f.write(j + '\n')
            f.write('\n')

        f.write('\nPOOLED MOVIES:\n\n')
        for i in pooled_directories.keys():
            f.write(i + ':\n')
            for j in pooled_directories[i]:
                f.write(j + '\n')
            f.write('\n')

        f.write('\nPOOLED MOVIES WITHOUT TIMELAGS:\n\n')
        for i in pooled_directories_without_timelag.keys():
            f.write(i + ':\n')
            for j in pooled_directories_without_timelag[i]:
                f.write(j + '\n')
                f.write('\n')

    print("====== ", basic_directory)
    for archive, string_now in zip(['', '/archive/'], ['', "_" + string_now]):
        numpy.save(basic_directory + '/_' + spot_type + archive + '/' + 'pooled_directories_by_day' + string_now + '.npy', pooled_directories_by_day)
        numpy.save(basic_directory + '/_' + spot_type + archive + '/' + 'pooled_directories_by_day_without_timelag' + string_now + '.npy', pooled_directories_by_day_without_timelag)
        numpy.save(basic_directory + '/_' + spot_type + archive + '/' + 'pooled_directories' + string_now + '.npy', pooled_directories)
        numpy.save(basic_directory + '/_' + spot_type + archive + '/' + 'pooled_directories_without_timelag' + string_now + '.npy', pooled_directories_without_timelag)

    return pooled_directories_by_day, pooled_directories, pooled_directories_without_timelag, pooled_directories_by_day_without_timelag

def concatenate(spot_type, type, per_day = False, days = 'all_days', starvation_times = 'all', load = False, avoid = None, ax = 1, ignore_timelag = False):
    '''
        type : string input to load_all_results


    '''
    print('basic_directory:')
    print(basic_directory)

    if load:
        if isinstance(type, str):
            data = load_all_results.load_all_results(spot_type, type, days = days, starvation_times = starvation_times)
        else:
            print('You need to specify the filename of the files I should load.')
    else:
        data = type.copy()

    if isinstance(avoid, list):
        keys_of_interest = [x for x in data.keys() if not any([y in x for y in avoid])]
    else:
        keys_of_interest = [x for x in data.keys()]
        print(len(keys_of_interest))

    for i in type.keys():
        if i not in keys_of_interest:
            del data[i]

    if per_day:
        if not ignore_timelag:
            print(1)
            categories = numpy.load(basic_directory + '_' + spot_type + '/' + 'pooled_directories_by_day.npy', allow_pickle = True).item()
        else:
            print(2)
            categories = numpy.load(basic_directory + '_' + spot_type + '/' + 'pooled_directories_by_day_without_timelag.npy', allow_pickle = True).item()
    else:
        if not ignore_timelag:
            print(3)
            categories = numpy.load(basic_directory + '_' + spot_type + '/' + 'pooled_directories.npy', allow_pickle = True).item()
        else:
            print(4)
            categories = numpy.load(basic_directory + '_' + spot_type + '/' + 'pooled_directories_without_timelag.npy', allow_pickle = True).item()

    if isinstance(type, str):
        if 'imsd' in type:
            ax = 1
        else:
            ax = 0

    print(ax)
    result = {}
    for i in categories.keys():
        print(str(i) + ':\n')
        print('There are ' + str(len(categories[i])) + ' movies.\n')
        if len(categories[i]) > 0:
            l = []
            for j in categories[i]:
                if j in keys_of_interest:
                    print('this belongs')
                    print(j)
                    l.append(data[j])
                    print('including: ' + str(j))
                else:
                    print('You do not have the results for movie ' + str(j) + '.')
            if len(l) > 0:
                result[i] = pandas.concat(l, axis = ax)

    return result

def filter_results(spot_type, results, below_size_limit_only = True, movies_to_avoid = ['201004_0h_bLR31_210msDelay_004/', '201004_0hat440pm_bLR31_210msDelay_027/', '201004_6h_bLR32_210msDelay_037/']):
    '''
    Filter out particles with unphysical results.

    INPUT
    -----
    spot_type : str
        This can be 'origins', 'muNS', 'fixed_origins', 'fixed_muNS', depending on    the object you have tracked.

    results : dictionary of pandas DataFrames
        Each entry in this dictionary contains the combined, per particle,  results, for a given movie.

    movies_to_avoid : list of str
        A list of movies that should be omitted in the final results, at least at the moment. These are movies on which the Gaussian fits did not work alltogether, and whose Gaussian fit "results" are all 0. This seems like a bug, and it only concerns a fraction of particles, so for now I will ignore these movies.
        These are: 201004_0h_bLR31_210msDelay_004/
        201004_0hat440pm_bLR31_210msDelay_027/
        201004_6h_bLR32_210msDelay_037/.

    OUTPUT
    ------
    A dictionary of pandas DataFrames, similar to the input results, where all particles with unphysical properties have been removed.
    '''

    keys_of_interest = [x for x in results.keys() if not any([y in x for y in movies_to_avoid])]

    filtered_results = results.copy()

    for i in results.keys():
        if i in movies_to_avoid:
            del filtered_results[i]

    for i in filtered_results.keys():
        filtered_results[i] = filtered_results[i].dropna(axis = 0, how = 'any')  # remove all rows that have an NaN entry
        indices_neg_entries = filtered_results[i].index[numpy.where(filtered_results[i] < 0)[0]]
        filtered_results[i] = filtered_results[i].drop(labels = indices_neg_entries, axis = 0) # remove all rows that have a negative entry, which is unphysical for all quantities in the results DataFrames (and anyway I was not plotting them)
        if below_size_limit_only:  # relevant for muNS particles with Gaussian fits for estimating their size
            filtered_results[i] = filtered_results[i][filtered_results[i].below_diffraction_limit == 1]

    return filtered_results

def filter_msds(spot_type, msds, reference_results_file):
    '''
    Filter out particles with unphysical results, from the msds. The particles that will remain are the ones that have passed the filtering done by filter_results() above.

    INPUT
    -----
    spot_type : str
        This can be 'origins', 'muNS', 'fixed_origins', 'fixed_muNS', depending on    the object you have tracked.

    msds : dictionary of pandas DataFrames
        Each entry in this dictionary contains the msd, per particle, for a given condition; in other words, the input is the pooled msds. Each particle in each entry is represented by a column. Note that you need to use the renamed msds, where the particle id contains its date and movie number.

    reference_results_file : str
        The full path + filename to the file that contains the filtered results, generated with the function filter_results() above. Particles not present in this DataFrame will be removed from the msds.

    OUTPUT
    ------
    A dictionary of pandas DataFrames, similar to the input results, where all particles with unphysical properties have been removed.
    '''
    if isinstance(reference_results_file, str):
        filtered_results = numpy.load(reference_results_file, allow_pickle = True).item()
    else:
        filtered_results = reference_results_file

    if set(filtered_results.keys()) != set(msds.keys()):
        raise ValueError('The msds dictionary does not match the dictionary of the filtered results. Check the keys of these two dictionaries.')

    for i in msds.keys():
        print(i)
        keep_particles = list(filtered_results[i].index)
        print('initial N:' + str(len(msds[i].columns)))
        for j in msds[i].columns:
            if j not in keep_particles:
                msds[i] = msds[i].drop(labels = [j], axis = 1)
        print('final N:' + str(len(msds[i].columns)))

    return msds

def filter_traj(spot_type, trajectories, reference_results):
    '''
    Filter out particles with unphysical results, from the trajectories. The particles that will remain are the ones that have passed the filtering done by filter_results() on the DataFrames that contain the results per particle (D_app, α, and particle size).

    INPUT
    -----
    spot_type : str
        This can be 'origins', 'muNS', 'fixed_origins', 'fixed_muNS', depending on    the object you have tracked.

    trajectories : dictionary of pandas DataFrames
        Each entry in this dictionary contains the trajectories dataframe for a given condition; in other words, the input is the pooled trajectories. Note that you need to use the renamed trajectories, where the particle id contains its date and movie number.

    reference_results : str or dictionary of pandas DataFrames
        If string, the full path + filename to the file that contains the filtered results, generated with the function filter_results() above. Particles not present in this DataFrame will be removed from the msds.

    OUTPUT
    ------
    A dictionary of pandas DataFrames, similar to the input trajectories, where all particles with unphysical properties have been removed.

    '''

    if isinstance(reference_results, str):
        filtered_results = numpy.load(reference_results, allow_pickle = True).item()
    else:
        filtered_results = reference_results

    if set(filtered_results.keys()) != set(trajectories.keys()):
        raise ValueError('The trajectories dictionary does not match the dictionary of the filtered results. Check the keys of these two dictionaries.')

    for i in trajectories.keys():
        print(i)
        keep_particles = list(filtered_results[i].index)
        all_particles = set(trajectories[i].particle)
        print('initial N:' + str(len(all_particles)))
        trajectories[i] = trajectories[i].loc[trajectories[i]['particle'].isin(keep_particles)]
        print('final N:' + str(len(keep_particles)))

    return trajectories

##### PLOT PARAMETERS #####

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

color1 = '#FF8D04'
color2 = '#041FFF'

ms = 12  # markersize

colors = {}

colors['bLR31'] = {}
colors['bLR31']['0h'] = '#00BDF9' #64F003'
colors['bLR31']['6hlowN'] =  '#0088F9' #419E01'
colors['bLR31']['6hlowC'] =  '#00559C'
colors['bLR31']['24h'] = '#0067BC' #2E6E01'

colors['bLR32'] = {}
colors['bLR32']['0h'] = '#FFB900'
colors['bLR32']['6hlowN'] = '#C08B00'
colors['bLR32']['6hlowC'] = '#7B5900'
colors['bLR32']['24h'] = '#886200'

colors['bLR33'] = {}
colors['bLR33']['0h'] = '#F906AF'
colors['bLR33']['6h'] = '#A30372'
colors['bLR33']['24h'] = '#800159'

colors['bLR1'] = {}
colors['bLR1']['0h'] = '#00BDF9' #'#64F003'
colors['bLR1']['6hlowN'] = '#0088F9' #'#419E01'
colors['bLR1']['6hlowC'] = '#00559C' #'#419E01'

colors['bLR2'] = {}
colors['bLR2']['0h'] = '#FFB900'
colors['bLR2']['6hlowN'] = '#AB7C00'#'#C08B00'
colors['bLR2']['6hlowC'] = '#7B5900'

colors['bLR3'] = {}
colors['bLR3']['0h'] = '#F906AF'
colors['bLR3']['6h'] = '#A30372'

colors['LR262'] = {}
colors['LR262']['0h'] = '#00BDF9'
colors['LR262']['6hlowN'] = '#0088F9'

colors['LR264'] = {}
colors['LR264']['0h'] = '#FFB900'
colors['LR264']['6hlowN'] = '#AB7C00'

htc = {}  # hatches for the histograms

htc['0h'] = '//'
htc['6h'] = ''

markershapes = {}

markershapes['0h'] = 'o'
markershapes['6hlowN'] = '^'
markershapes['6hlowC'] = 's'

#markerfacecolors = {}
#markerfacecolors['0h'] = 'o'
#markerfacecolors['6hlowN'] = '^'
#markerfacecolors['6hlowC'] = 's'
#

symbols = {}
symbols['origins'] = {}
symbols['origins']['200518'] = 'o'
symbols['origins']['200520'] = 'v'
symbols['origins']['200602'] = '^'

symbols['muNS'] = {}
symbols['muNS']['201004'] = 'o'
symbols['muNS']['201006'] = '*'
symbols['muNS']['201009'] = 's'
symbols['muNS']['201226'] = 'd'
symbols['muNS']['210323'] = '+'

positions = {}

positions['bLR31'] = {}
positions['bLR31']['0h'] = 1
positions['bLR31']['6hlowN'] = 1.2#2#1.2
positions['bLR31']['6hlowC'] = 1.2#1.2#1.4
#positions['bLR31']['24h'] = 1.4

positions['bLR32'] = {}
positions['bLR32']['0h'] = 1.4
positions['bLR32']['6hlowN'] = 1.6
positions['bLR32']['6hlowC'] = 1.6#1.6#2.0
#positions['bLR32']['24h'] = 2.4

positions['bLR1'] = {}
positions['bLR1']['0h'] = 1
positions['bLR1']['6hlowN'] = 1.2
#positions['bLR1']['24h'] = 1.4

positions['bLR2'] = {}
positions['bLR2']['0h'] = 1.4
positions['bLR2']['6hlowN'] = 1.6
positions['bLR2']['24h'] = 2.4

positions['LR262'] = {}
positions['LR262']['0h'] = 1
positions['LR262']['6hlowN'] = 1.2

positions['LR264'] = {}
positions['LR264']['0h'] = 1.4
positions['LR264']['6hlowN'] = 1.6

position_subplot = {}               # the positions of the subplots for each condition and strain
position_subplot['origins'] = {}
position_subplot['origins']['bLR1_0h'] = 0
position_subplot['origins']['bLR1_6h_lowN'] = 1#5#2#1
position_subplot['origins']['bLR1_6h_lowC'] = 11#2  #1
position_subplot['origins']['bLR2_0h'] = 2
position_subplot['origins']['bLR2_6h_lowN'] = 3#4#3#4 #3
position_subplot['origins']['bLR2_6h_lowC'] = 5 #3

            # the positions of the subplots for each condition and strain
position_subplot['origins_comp'] = {}
position_subplot['origins_comp']['bLR1_0h'] = 0
position_subplot['origins_comp']['bLR1_6h_lowN'] = 1#5#2#1
# position_subplot['origins_comp']['bLR1_6h_lowC'] = 11#2  #1
position_subplot['origins_comp']['bLR2_0h'] = 2
position_subplot['origins_comp']['bLR2_6h_lowN'] = 3#4#3#4 #3
# position_subplot['origins_comp']['bLR2_6h_lowC'] = 5 #3
position_subplot['origins_comp']['LR262_0h'] = 4
position_subplot['origins_comp']['LR262_6h_lowN'] = 5
position_subplot['origins_comp']['LR264_0h'] = 6
position_subplot['origins_comp']['LR264_6h_lowN'] = 7


position_subplot['muNS'] = {}
position_subplot['muNS']['bLR31_0h'] = 0
position_subplot['muNS']['bLR31_6h_lowN'] = 1#10
position_subplot['muNS']['bLR31_6h_lowC'] = 2#6#1 #2
position_subplot['muNS']['bLR32_0h'] = 3 #2
position_subplot['muNS']['bLR32_6h_lowN'] = 4
position_subplot['muNS']['bLR32_6h_lowC'] = 5

position_subplot['muNS_lowC'] = {}
position_subplot['muNS_lowC']['bLR31_0h'] = 0
position_subplot['muNS_lowC']['bLR31_6h_lowN'] = 10
position_subplot['muNS_lowC']['bLR31_6h_lowC'] = 1#10
position_subplot['muNS_lowC']['bLR32_0h'] = 2 #3
position_subplot['muNS_lowC']['bLR32_6h_lowN'] = 4
position_subplot['muNS_lowC']['bLR32_6h_lowC'] = 3



# function to generate color gradients, copied on 2021.02.09 from https://stackoverflow.com/questions/25668828/how-to-create-colour-gradient-in-python #

def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=numpy.array(matplotlib.colors.to_rgb(c1))
    c2=numpy.array(matplotlib.colors.to_rgb(c2))

    return matplotlib.colors.to_hex((1-mix)*c1 + mix*c2)

### compare a subset of imsds to all other imsds ###

def plot_new_imsds(spot_type, ims, new_feature, avoid = ['1000ms']):
    '''
    Plot the new imsds on top of all previous imsds of the same conditions. Each new movie will be represented by a new color.

    INPUT
    -----
    spot_type : str
        This can be 'origins', 'muNS', 'fixed_origins', 'fixed_muNS', depending on the object you have tracked.

    ims : dictionary of pandas DataFrames
        All imsds (old and new), not pooled.

    new_feature : str
        A description of what is new about these new imsds; for instance, if there is a new day's worth of data, this could be the day's date. In general, it should be a string common to the keys of all imsds in the subset you want to compare to the rest of the imsds.

    avoid : list of strings, defaults to '1000ms'
        If you want to avoid certain datasets, here you can specify characteristic strings in their filename - for example '1000ms'.

    OUTPUT
    ------
    The dictionaries of the new and old imsds.
    '''

    central_directory = load_all_results.select_specifications(spot_type)['central_directory']

    c = numpy.load(basic_directory + '_' + spot_type + '/' + 'pooled_directories.npy', allow_pickle=True).item()  # the pool categories (here without "per_day", so bLR31_0h_0030ms etc)

    categories_of_interest = []

    color_loop = ['#3DB9FC', '#26E7F0']

    for i in c.keys():
        if len(c[i]) > 0:    # i.e. if there are movies that belong to the condition described by i
            categories_of_interest.append(i)

    categories_of_interest = [x for x in categories_of_interest if '0h' in x]  # further specify a feature you want to focus on; here, 0h

    if isinstance(avoid, list):
        for l in avoid:
            categories_of_interest = [x for x in categories_of_interest if l not in x]

    print('Movies with the new feature, ' + str(new_feature) + ':')

    now = datetime.datetime.now()
    string_now = now.strftime("%Y%m%d")

    to = time.time()

    for i in categories_of_interest[:]:
        print(i)
        pylab.figure(i)   # go to the figure for this category
        for j in c[i]:     # for every movie in this category (e.x. bLR31_0h_0030ms)
            if new_feature not in j:  # if it does not have the new feature you want to highlight
                plot_tracking_results.plot_msds(imsds = ims[j][:], emsd = None, loglog = True, color = 'b', symbol = '.', alpha_imsd = 0.025, alpha_emsd = 1.0, linewidth = 2, legend_content ='', legend_location = 2, emsd_label = 'movie ' + str(j), interactive_plot = False, new_imsd_figure = False, new_emsd_figure = False, plot_imsds = True, imsd_title = i, emsd_title = i, edgecolor = 'None')
        for j in c[i]:  # loop again to make sure that the new curves end up on top of the plot, so that they are visible
            k = 0
            if new_feature in j:
                plot_tracking_results.plot_msds(imsds = ims[j][:], emsd = None, loglog = True, color = color_loop[k], symbol = '.', alpha_imsd = 0.05, alpha_emsd = 1.0, linewidth = 2, legend_content = '', legend_location = 2, emsd_label = 'movie ' + str(j), interactive_plot = False, new_imsd_figure = False, new_emsd_figure = False, plot_imsds = True, imsd_title = i, emsd_title = i, edgecolor = 'None')
                k = k + 1
                print(color_loop[k])
                print(j)          # print the movie name

        pylab.xlim(1e-2, 1e2)
        pylab.ylim(1e-5, 10)
        pylab.xlabel('time lag (s)')
        pylab.ylabel('msd (mum^2)')
        pylab.savefig(basic_directory + central_directory + '/plots/new_imsds/' + new_feature + '/' + string_now + '_' + i + '.png', bbox_inches = 'tight')

    t1 = time.time()
    print('This took ' + str(t1-to) + 's.')

    return categories_of_interest

### within day variability, alphas ###

def plot_alpha_per_cat_per_day(spot_type, results, avoid = ['1s'], days = 'all_days', starvation_times = all_starvation_times):
    '''
    Make one figure per condition per day, each containing one histogram of alpha per movie. The point is to see whether alpha changes during the course of movie-taking.

    INPUT
    -----
    results : dictionary of DataFrames
       A dictionary of DataFrames of the results per particle. These are files saved as 'results_per_particle.pkl' that contain, for each particle, its D, alpha, diagonal size and starting snr.

    avoid : list of strings or None; defaults to None
        A string describing a feature that you are not interested in, found in the movie filenames.

    OUTPUT
    ------
    Graphs as described above, saved in the general folder for the spot type (origins or muNS).
    '''

    categories = 'pooled_directories_by_day.npy'

    c = numpy.load(basic_directory + '_' + spot_type + '/' + categories, allow_pickle=True).item()

    if days == 'all_days':
        days = load_all_results.select_specifications(spot_type)['all_days']

    central_directory = load_all_results.select_specifications(spot_type)['central_directory']

    now = datetime.datetime.now()
    string_now = now.strftime("%Y%m%d")
    destination_directory = basic_directory + central_directory + 'plots/alphas_per_cat/' + string_now + '/'

    #bins_now = numpy.arange(-1.1, 1.25, 0.05)
    bins_now = numpy.arange(-0.05, 1.25, 0.05)

    keys_of_interest = [x for x in c.keys() if not any([y in x for y in avoid])]

    if days != 'all_days':
        keys_of_interest = [x for x in keys_of_interest if any([d in x for d in days])]

    for i in keys_of_interest:
        n = len(c[i])
        if n > 1:
            fig = pylab.figure()
                               #, figsize = (10, 2.5*n))
            fig.suptitle(i)
            gs = GridSpec(12, 10, figure=fig)
            ax = {}
            print(i)
            fig.subplots_adjust(top=0.9)
            pylab.subplots_adjust(hspace=0.5)
            for j in numpy.arange(0, 6):
                ax[j] = fig.add_subplot(gs[(2*j):(2*(j+1)), :8])
                #for j,k in enumerate(c[i]):
                if j < len(c[i]):
                    k = c[i][j]
                    m = load_all_results.read('movie', k, spot_type).zfill(4)
                    #col = colorFader(color1,color2, mix = j / float(n))
                    nparticles = len(results[k])
                    alphas_now = results[k].alpha.to_numpy(dtype = numpy.float64)
                    alphas_now = alphas_now[~numpy.isnan(alphas_now)]
                    nparticles_notnan = len(alphas_now)
                    alphas_now = alphas_now[numpy.greater(alphas_now, numpy.zeros_like(alphas_now))]
                    nparticles_neither_negative = len(alphas_now)
                    alphas_now_median = round(numpy.median(alphas_now),2)
                    alphas_now_std = numpy.std(alphas_now)
                    alphas_now_sem = round(alphas_now_std / numpy.sqrt(nparticles_neither_negative),2)
                    percent_included = 100 * round(float(nparticles_neither_negative) / nparticles, 2)
                    weights = 100 * numpy.ones_like(alphas_now) / float(len(alphas_now))
                    h = ax[j].hist(alphas_now, bins = bins_now, #density = True,
                                   color = 'b', weights = weights)
                    pylab.axvline(numpy.median(alphas_now), color = 'k')
                    ax[j].text(bins_now[17], 10, 'N = ' + str(nparticles_neither_negative))
                    ax[j].text(bins_now[::-1][0] + 0.075, 12, 'movie ' + m)
                    ax[j].text(bins_now[::-1][0] + 0.075, 6, 'α = ' + format(alphas_now_median, '.2f')                                                     + '+/-' + str(alphas_now_sem))
                    print(str(percent_included) + '% of all tracked particles')
                    ax[j].set_ylim(0, 16)
                    pylab.yticks([0, 10])
                    if j < len(c[i]) - 1:
                        pylab.xticks([])

                else:
                    #ax[j].hist(results[c[i][0]].alpha.to_numpy(dtype = numpy.float64), bins = bins_now, normed = True, color = 'b')
                    ax[j].axis('off')

            pylab.savefig(destination_directory + 'alpha_histograms' + i + '.png')

    return h

### dependence on α on time (aka movienumber) ###

def collect_alpha_vs_movienumber(results, spot_type, avoid = ['1s'], label = '', days = 'all_days'):
    '''
    Collect - Make one figure per condition per day, each containing one histogram of alpha per movie. The point is to see whether alpha changes during the course of movie-taking.

    INPUT
    -----
    results : dictionary of DataFrames
        A dictionary of DataFrames of the results per particle. These are files saved as 'results_per_particle.pkl' that contain, for each particle, its D, alpha, diagonal size and starting snr.

    avoid : list of strings or None (default)
        A list of strings describing a feature that you are not interested in, found in the movie filenames.

    OUTPUT
    ------
    Graphs as described above, saved in the general folder for the spot type (origins or muNS).
    '''

    c = numpy.load(basic_directory + '_' + spot_type + '/' + 'pooled_directories_by_day.npy', allow_pickle=True).item()
    c2 = numpy.load(basic_directory + '_' + spot_type + '/' + 'pooled_directories.npy', allow_pickle=True).item()
    c2 = [x for x in c2.keys() if not any([y in x for y in avoid])]
    if days == 'all_days':
        days = load_all_results.select_specifications(spot_type)['all_days']

    central_directory = load_all_results.select_specifications(spot_type)['central_directory']

    keys_of_interest = [x for x in c.keys() if not any([y in x for y in avoid])]

    if isinstance(days, list):
        keys_of_interest = [x for x in keys_of_interest if any([d in x for d in days])]

    answer = {}
    for i in c2:
        answer[i] = {}
        for d in days:
            answer[i][d] = pandas.DataFrame(columns = ['movie', 'alpha', 'alpha_std', 'alpha_sem'])

    f = open(basic_directory + central_directory + 'alphas_percent_of_unphysical_particles' + label + '.txt', 'w')

    for i in keys_of_interest[:]:
        print('current key: ' + str(i))
        n = len(c[i])
        if n > 0:
            print(i)
            for k in c[i]:
                if k in results.keys():
                    print(k)
                    f.write(k + '\n')
                    movie = round(int(load_all_results.read('movie', k, spot_type)),0)
                    strain = load_all_results.read('strain', k, spot_type)
                    time_between_frames = load_all_results.read('time_between_frames', k, spot_type).zfill(4) +     'ms'
                    starvation_time = load_all_results.read('starvation_time', k, spot_type) + 'h'
                    con = load_all_results.read('condition', k, spot_type)
                    if len(con) > 0:
                        condition = strain + '_' + starvation_time + '_' + time_between_frames + '_' + con
                    else:
                        condition = strain + '_' + starvation_time + '_' + time_between_frames
                    day = load_all_results.read('day', k, spot_type)

                    nparticles = len(results[k])
                    alphas_now = results[k].alpha.to_numpy(dtype = numpy.float64)
                    alphas_now = alphas_now[~numpy.isnan(alphas_now)]
                    alphas_now = alphas_now[numpy.greater(alphas_now, numpy.zeros_like(alphas_now))]
                    nparticles_neither_negative = len(alphas_now)
                    f.write(str(round(100 - 100 * float(nparticles_neither_negative) / nparticles, 1)) + '% of particles have α = nan or negative.\n')
                    alphas_greater_1p2 = alphas_now[numpy.greater(alphas_now, 1.2 * numpy.ones_like(alphas_now))]
                    f.write(str(round(100 * float(len(alphas_greater_1p2)) / nparticles, 1)) + '% of particles have α > 1.2.\n\n')
                #alphas_now = alphas_now[numpy.less(alphas_now, 1.2 * numpy.ones_like(alphas_now))]

                    alphas_now_median = round(numpy.median(alphas_now),2)
                    alphas_now_std = numpy.std(alphas_now)
                    alphas_now_sem = round(alphas_now_std / numpy.sqrt(nparticles_neither_negative), 3)
                    percent_included = 100 * round(float(nparticles_neither_negative) / nparticles, 2)

                    little_answer =  pandas.DataFrame(data = numpy.array([[movie, alphas_now_median, alphas_now_std, alphas_now_sem]]), columns = ['movie', 'alpha', 'alpha_std', 'alpha_sem'])

                    answer[condition][day] = answer[condition][day].append(little_answer)

    now = datetime.datetime.now()
    string_now = now.strftime("%Y%m%d")
    filename = string_now + '_alpha_vs_movienumber' + label + '.npy'
    numpy.save(basic_directory + central_directory + filename, answer)
               #plots/alphas_per_cat_per_day/alpha_histograms' + i + '.png')
    f.close()

    colors_now = {}
    colors_now['muNS'] = {}
    colors_now['muNS']['210323'] = '#33B8FF'
    colors_now['muNS']['201226'] = '#2002FF'
    colors_now['muNS']['201009'] = '#FFA702'
    colors_now['muNS']['201006'] = '#01B10B'
    colors_now['muNS']['201004'] = '#B10101'

    colors_now['origins'] = {}
    colors_now['origins']['210227'] = 'r' #'#33B8FF' #'#2002FF'
    colors_now['origins']['210330'] = 'g' #'#2002FF' #'#FFA702'
    colors_now['origins']['210518'] =  'b' #'#FFA702' #'#01B10B'

    ms = 12
    legend_elements = []
    g1 = Line2D([0], [0], linewidth = 0, marker=markershapes['0h'], color=colors['bLR1']['0h'], label='WT 0h', markersize=ms, markeredgecolor = 'k', markeredgewidth = 1, alpha = 0.9)
    g2 = Line2D([0], [0], linewidth = 0, marker=markershapes['6hlowN'], color=colors['bLR1']['6hlowN'], label='WT 6h', markersize=ms, markeredgecolor = 'k', markeredgewidth = 1,alpha = 0.9)
    g3 = Line2D([0], [0], linewidth = 0, marker=markershapes['0h'], color=colors['bLR2']['0h'], label='ΔpolyP 0h', markersize=ms, markeredgecolor = 'k', markeredgewidth = 1,alpha = 0.9)
    g4 = Line2D([0], [0], linewidth = 0, marker=markershapes['6hlowN'], color=colors['bLR2']['6hlowN'], label='ΔpolyP 6h', markersize=ms,markeredgecolor = 'k', markeredgewidth = 1, alpha = 0.9)
    legend_elements = [g1, g2, g3, g4]

    for c in list(answer.keys())[:]:
        print(c)
        time_between_frames = load_all_results.read('time_between_frames', c, spot_type)
        strain = load_all_results.read('strain', c, spot_type)
        starvation_time = load_all_results.read('starvation_time', c, spot_type) + 'h'
        condition = load_all_results.read('condition', c, spot_type)
        time_between_frames = load_all_results.read('time_between_frames', c, spot_type)
        if strain == 'bLR1':
            strain_name = 'WT'
        elif strain == 'bLR2':
            strain_name = 'ΔpolyP'
        fig_label = 'alpha per movie number'
        fig_label = fig_label + label
#        fig_label = strain_name + ' ' + starvation_time + ' ' + condition + ' ' + label
        pylab.figure(fig_label, figsize = (6, 6)) #(7, 5.5)
        pylab.title(fig_label)
        if '6h' in c:
            if spot_type == 'muNS':
                pylab.xlabel('movie number \n(shifted by 6000 for 201226)')
            elif spot_type == 'origins':
                pylab.xlabel('movie number \n(shifted by the first number for each day)')
        else:
            pylab.xlabel('movie number \n(shifted by the first number for each day)')
        pylab.ylabel('α', rotation = 0, fontsize = 18, labelpad = 10)
        for d in answer[c].keys():
            movies = answer[c][d].movie.to_numpy(dtype = numpy.float64)
            if len(movies) > 0:
                min_movie = movies.min()
                new_movienumbers = movies - min_movie
                new_movienumbers = [x-20000 if x>=20000 else x for x in new_movienumbers]
                print('condition: ')
                print(c)
                print('day: ')
                print(d)
#                pylab.errorbar(new_movienumbers, answer[c][d].alpha.to_numpy(dtype = numpy.float64), fmt = 'o-', yerr = answer[c][d].alpha_sem.to_numpy(dtype = numpy.float64), label = d, color = colors_now[spot_type][d], capsize = 5, linewidth = 2, markersize = 10)
#                pylab.errorbar(new_movienumbers, answer[c][d].alpha.to_numpy(dtype = numpy.float64), fmt = 'o-', yerr = answer[c][d].alpha_std.to_numpy(dtype = numpy.float64), color = colors_now[d], capsize = 7, elinewidth = 1, linewidth = 2)
                #pylab.plot(new_movienumbers, answer[c][j].alpha.to_numpy(dtype = numpy.float64), '--', label = j)

                pylab.plot(new_movienumbers, answer[c][d].alpha.to_numpy(dtype = numpy.float64), markershapes[starvation_time + condition], label = d, color = colors[strain][starvation_time + condition],  markersize = ms, markeredgecolor = 'k', markeredgewidth = 1, alpha = 0.8)

        pylab.legend(handles=legend_elements, frameon = True, loc = 5, handletextpad = 0, borderaxespad = 0.2, bbox_to_anchor=(1.4, 0.85))

        if spot_type == 'origins':
            pylab.xlim(-0.5, 7)
            pylab.ylim(0.3, 0.55)

        elif spot_type == 'muNS':
            pylab.xlim(-1, 25)
            pylab.ylim(0.10, 0.81)

        pylab.show()

#        filename = string_now + '_' + strain_name + '_' + starvation_time + '_' + condition + '_' + time_between_frames + 'ms'
#
#        pylab.savefig(basic_directory + central_directory + 'plots/alphas_per_movienumber' + label + '/' + filename + '_alpha_vs_time_' + string_now + '.png')
#
#        pylab.savefig(basic_directory_paper_figs + filename + '_alpha_vs_time_' + '.png', bbox_inches = 'tight')
#        pylab.savefig(basic_directory_paper_figs + filename + '_alpha_vs_time_' + '.svg', bbox_inches = 'tight')

    filename = string_now + '_origins_alpha_vs_movienumber'

    pylab.savefig(basic_directory + central_directory + 'plots/alphas_per_movienumber' + label + '/' + filename + '.png')

    pylab.savefig(basic_directory_paper_figs + filename + '.png', bbox_inches = 'tight')
    pylab.savefig(basic_directory_paper_figs + filename + '.svg', bbox_inches = 'tight')

    return answer

### the function below does not pool properly per day; best to use pooled DataFrames as input for now ###

import ori_comp_analysis

def plot_imsds_pooled(spot_type, loglog = True, per_day = False, pool_now = True, load = False,
                      imsds = None, results = None, trajectories = None, label  = '', emsds=None,
                      avoid = ['1s'], avoid_movies = None, days = 'all_days', n_subset = 1000,
                      all_same_n_subset = False):

    '''
        The label needs to be in the following format:  'tstart10sec_tend40sec' for example, for the origins. Then the line that shows the interval where α comes from is drawn at the right place with respect to the time axis.
    '''

    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)

    central_directory = load_all_results.select_specifications(spot_type)['central_directory']
    empirical_noise_floor =  pandas.read_pickle(data[spot_type]['empirical_noise_floor_file'])

    if results is None:
        results = load_all_results.load_all_results('origins',
                                              ['fit_results'],
                                              days='all_days')
        results = collect_results.pool('origins', results, 0, per_day=False)
    if imsds is None :
        imsds = load_all_results.load_all_results('origins',
                                              ['imsds_all_renamed'],
                                              days='all_days')
        imsds = process_msds.rename_particles_in_imsds(imsds, 'origins')
        imsds = collect_results.pool('origins', imsds, 0, per_day=False)
    if trajectories is None:
        trajectories = load_all_results.load_all_results('origins',
                                              ['filtered_trajectories_all_renamed'],
                                              days='all_days')
        trajectories = collect_results.pool('origins', trajectories, 0,per_day=False)

    if emsds is None and spot_type == 'origins':
        emsds = ori_comp_analysis.real_emsds(plot=False)
        print(emsds.keys())
    # empirical_noise_floor = pandas.read_pickle(load_all_results.select_specifications(spot_type)['empirical_noise_floor_file'])

    if days == 'all_days':
        days = load_all_results.select_specifications(spot_type)['all_days']

    if 'tstart' in label:
        start = label.split('start')[1]
        start = start.split('sec')[0]
        start = start.replace('p', '.')
        t_start = float(start)
        print('t_start: ' + str(t_start))
    else:
        t_start = 'no start'

    if 'tend' in label:
        end = label.split('end')[1]
        end = end.split('sec')[0]
        end = end.replace('p', '.')
        t_end = float(end)
        print('t_end: ' + str(t_end))
    else:
        t_end = 'no end'

    now = datetime.datetime.now()
    string_now = now.strftime("%Y%m%d")

    keys_of_interest = [x for x in results.keys()]
    keys_of_interest = [x for x in keys_of_interest if not any([y in x for y in avoid])]

    to = time.time()

    selected_particles = {}

    for k in keys_of_interest[:]:
        to = time.time()
        print(k)
        fig = plt.figure(k, figsize = (7.5, 5.5))
        plt.figure(fig.number)
        strain = load_all_results.read('strain', k, spot_type)
        starvation_time = load_all_results.read('starvation_time', k, spot_type) + 'h'
        condition = load_all_results.read('condition', k, spot_type)
        ax = fig.gca()
        if loglog:
            ax.set_xscale('log')
            ax.set_yscale('log')
        if 'tstart' in label:
            start = numpy.where(imsds[k].index >= t_start)[0][0]
        if 'tend' in label:
            end = numpy.where(imsds[k].index <= t_end)[0].max()
        else:
            start = 0
            end = 4
        print('start: ' + str(start))
        print('end: ' + str(end))
        try:
            nparticles = len(imsds[k].columns)
        except KeyError:
            print(k, " did not work")
            k = k + "_lowN"
            print(k)
            pdb.set_trace()
            nparticles = len(imsds[k].columns)
        alphas = results[k].alpha.to_numpy(dtype = numpy.float64)
        alphas_now = alphas[~numpy.isnan(alphas)]
        alphas_now = alphas_now[numpy.greater(alphas_now, numpy.zeros_like(alphas_now))]
        nparticles_neither_negative = len(alphas_now)
        alpha_median = numpy.median(alphas_now)
        nparticles_now = len(alphas_now)
        print('Excluded from the α calculation because they were nan or negative:' )
        print(str(100 * (1 - float(nparticles_now) / nparticles)) + '%')
        localization_uncertainties = trajectories[k].ep.to_numpy(dtype = numpy.float64)
        localization_uncertainties_mean = localization_uncertainties.mean() * px_to_micron
        localization_uncertainties_std = localization_uncertainties.std() * px_to_micron
        print('\n' + str(k))
        print('average loc. uncertainty: ')
        print(str(localization_uncertainties_mean) + ' mum')
        print('average loc. uncertainty + 2 * std: ')
        print(str(localization_uncertainties_mean + 2 * localization_uncertainties_std) + ' mum')
        #imsds[k].plot(y = imsds[k].columns)

#        if starvation_time == '0h':
#            alpha_value = 20./nparticles_now
#            print('20')
#        else:
#            alpha_value = 20./nparticles_now # if you put this at 20, the 6h plots do not show up! there might be a lower limit for alpha
        particles = list(imsds[k].columns)[:]
        n_subset_now = {}
        if isinstance(n_subset, int):
            n_subset_now['bLR31_0h_0210ms'] = n_subset * 0.2 #0.6
            n_subset_now['bLR31_6h_0210ms_lowN'] = n_subset * 0.55
            n_subset_now['bLR31_6h_0210ms_lowC'] = n_subset * 0.3

            n_subset_now['bLR32_0h_0210ms'] = n_subset * 0.2
            n_subset_now['bLR32_6h_0210ms_lowN'] = n_subset
            n_subset_now['bLR32_6h_0210ms_lowC'] = n_subset * 0.25

            n_subset_now['bLR1_0h_5000ms'] = n_subset * 0.9
            n_subset_now['bLR1_6h_5000ms_lowN'] = n_subset * 1.1

            n_subset_now['bLR2_0h_5000ms'] = n_subset * 0.7
            n_subset_now['bLR2_6h_5000ms_lowN'] = n_subset * 1.4

            n_subset_now['LR262_0h_5000ms'] = n_subset*0.5
            n_subset_now['LR262_6h_5000ms_lowN'] = n_subset*0.5

            n_subset_now['LR264_0h_5000ms'] = n_subset*0.5
            n_subset_now['LR264_6h_5000ms_lowN'] = n_subset*0.5

            if all_same_n_subset:
                particles = random.sample(particles, n_subset)
                for key in n_subset_now.keys():
                    n_subset_now[key] = n_subset
            else:
                particles = random.sample(particles, int(n_subset_now[k]))
            selected_particles[k] = numpy.array(particles)

        print("PARTICLES", len(particles))
        for i,j in enumerate(particles):
            # breakpoint()
            plt.plot(imsds[k].index.to_numpy()[:60], imsds[k][imsds[k].columns[i]].to_numpy()[:60], color = 'b', alpha = 0.2)# 400 color = colors[strain][starvation_time + condition], alpha = 50./nparticles_now)
            # print("ALPHA SET")
        if "origins" in spot_type:
            constant = 3e-2 / imsds[k].index[start] ** alpha_median
        elif spot_type == 'muNS':
            constant = 1e-1 / imsds[k].index[start] ** alpha_median
        elif spot_type == 'muNS_lowC':
            constant = 1e-1 / imsds[k].index[start] ** alpha_median

        # figure3:  3e-1, 1.5e-1 / imsds[k].index[start] ** alpha_median
        plt.plot(imsds[k].index[start:end], constant * imsds[k].index[start:end] ** alpha_median, '-', color = 'k', label = str(round(alpha_median,2)))

#        plt.axvline(2.1, color = '#777777', linestyle = '--')

        plt.plot(empirical_noise_floor.index, empirical_noise_floor.msd, '--', color = 'r')#'#AA0000')

        if spot_type == 'muNS':
            plt.text(1.5e0, 1.2e-3, '-- empirical noise floor', color = 'r') #'#AA0000')  # !!! 1e1 for figure 3 !!! (1e1, 6e-5) figure3: (1e1, 1.5e-4)
            if isinstance(n_subset, int):
                plt.text(0.25, 6e-1, 'N = ' + str(int(n_subset_now[k]))) #(1e1, 1e0)
            else:
                plt.text(0.25, 6e-1, 'N = ' + str(len(particles))) #(1e1, 1e0)
            plt.text(0.25, 2.5e-1, 'α = ' + str(round(alpha_median,2)))# figure 3: (0.25, 7e-1) + '\nstd: ' + str(round(alphas_now.std(),2))) ## 1, 2e-5
            plt.xlim(0.21, 10) #figure 3: (0.21, 50); figure 2: (0.21, 10)
            plt.ylim(1e-3, 1) # me:(1e-4, 1e1), Suliana: (1e-3, 1), (4e-5, 5e0) figure 3: (1e-4, 5e0), (5e-5, 2e1), plt.ylim(1e-5, 5e0)

        if spot_type == 'muNS_lowC':
            plt.text(1.5e0, 1.2e-3, '-- empirical noise floor', color = 'r') #'#AA0000')  # !!! 1e1 for figure 3 !!! (1e1, 6e-5) figure3: (1e1, 1.5e-4)
            if isinstance(n_subset, int):
                plt.text(0.25, 6e-1, 'N = ' + str(int(n_subset_now[k]))) #(1e1, 1e0)
            else:
                plt.text(0.25, 6e-1, 'N = ' + str(len(particles))) #(1e1, 1e0)
            plt.text(0.25, 2.5e-1, 'α = ' + str(round(alpha_median,2)))# figure 3: (0.25, 7e-1) + '\nstd: ' + str(round(alphas_now.std(),2))) ## 1, 2e-5
            plt.xlim(0.21, 10) #figure 3: (0.21, 50); figure 2: (0.21, 10)
            plt.ylim(4e-4, 1) # me:(1e-4, 1e1), Suliana: (1e-3, 1), (4e-5, 5e0) figure 3: (1e-4, 5e0), (5e-5, 2e1), plt.ylim(1e-5, 5e0)

#            plt.text(1.5e0, 4.5e-4, '-- empirical noise floor', color = 'r') #'#AA0000')  # !!! 1e1 for figure 3 !!! (1e1, 6e-5) figure3: (1e1, 1.5e-4)
#            if isinstance(n_subset, int):
#                plt.text(0.25, 2.5e-1, 'N = ' + str(int(n_subset_now[k]))) #(1e1, 1e0)
#            else:
#                plt.text(0.25, 2.5e-1, 'N = ' + str(len(particles))) #(1e1, 1e0)
#            plt.text(0.25, 1.5e-1, 'α = ' + str(round(alpha_median,2)))# figure 3: (0.25, 7e-1) + '\nstd: ' + str(round(alphas_now.std(),2))) ## 1, 2e-5
#            plt.xlim(0.21, 10) #figure 3: (0.21, 50); figure 2: (0.21, 10)
#            plt.ylim(3.5e-4, 4e-1) # me:(1e-4, 1e1), Suliana: (1e-3, 1), (4e-5, 5e0) figure 3: (1e-4, 5e0), (5e-5, 2e1), plt.ylim(1e-5, 5e0)

        elif spot_type == 'origins':
            plt.text(3e1, 6e-4, '-- empirical noise floor', color = 'r')  # (4e1, 1.5e-4)
            plt.text(5.25e0, 6.5e-2, 'N = ' + str(int(n_subset_now[k]))) #(1e1, 1e1), (200, 1.5)
            plt.text(10, 4e-2, 'α = ' + str(round(alpha_median,2))) #+ '\nstd: ' + str(round(alphas_now.std(),2))) # (10, 3e-1) (1, 2e-5)
            plt.xlim(5, 200)
            plt.ylim(5e-4, 2E-1)#(1e-4, 1e-1), (5.5e-4, 7e-2)(5e-4, 1e-1), (1e-4, 1e-1)plt.ylim(1e-5, 5e0)
            focus = emsds[k[:7]]
            test = numpy.zeros([len(focus), 2])
            test[:,0] = focus.lagt
            test[:,1] = focus.msd
            alpha_emsd, D_emsd = numpy.polyfit(numpy.log10(test[1:4,0]), numpy.log10(test[1:4,1]), 1)
            plt.plot(emsds[k[:7]].lagt, emsds[k[:7]].msd, linewidth=5, color='b')
            plt.plot(list(imsds[k].index)[:50], list(imsds[k].median(1))[:50], color='g')
            plt.plot(list(imsds[k].index)[:50], list(imsds[k].mean(1))[:50], color='r')
            plt.text(10, 6e-2, '$α_{emsd}$ = ' + str(round(alpha_emsd,2)))

        ### the following two are for the presentation ###
#        plt.plot(imsds[k].index[start:end], constant * imsds[k].index[start:end] ** alpha_median, 'k-', label = str(round(alpha_median,2)))
#        plt.plot(imsds[k].index[start:end], constant * imsds[k].index[start:end] ** alpha_median, 'k-', label = str(round(alpha_median,2)))

#        plt.text(imsds[k].index[start] + 24, 3e-1, 'alpha = ' + str(round(alpha_median,2)) + '\nstd: ' + str(round(alphas_now.std(),2))) ## 1, 2e-5 #
#        plt.xlim(1e0, 1e2) #
#        plt.ylim(1e-5, 10) #

        plt.xlabel('time lag ' + r'$(s)$')
        plt.ylabel('msd ' + r'$(\mu m^{2})$')
        plt.title(k)

        figname = 'imsds_per_cat'
        if per_day:
            figname = figname + '_per_day'
        figname = figname + '_with_noise_floor' + label

        os.makedirs(basic_directory + central_directory + 'plots/' + figname + '/', exist_ok=True)
        plt.savefig(basic_directory + central_directory + 'plots/' + figname + '/' + string_now + '_' + k.strip('/') + '.png', bbox_inches = 'tight')

        os.makedirs(basic_directory_paper_figs + '_' + spot_type + '/imsds/' + str(n_subset).zfill(3) + "/", exist_ok=True)
        plt.savefig(basic_directory_paper_figs + '_' + spot_type + '/imsds/' + str(n_subset).zfill(3) + '/' + string_now + '_' + figname + '_' + k.strip('/') + '.png', bbox_inches = 'tight')
        plt.savefig(basic_directory_paper_figs + '_' + spot_type + '/imsds/' + str(n_subset).zfill(3) + '/' + string_now + '_' + figname + '_' + k.strip('/') + '.svg',  bbox_inches = 'tight')

        print('\033[1m FIGURE LOCATION \033[0m')
        print(basic_directory_paper_figs + '_' + spot_type + '/imsds/' + string_now + '_' + figname + '_' + k.strip('/') + '.svg')
        t1 = time.time()
        print('This took ' + str(t1-to) + ' s.')
        plt.clf()
        plt.close()
#    numpy.save(basic_directory_paper_data + '_' + spot_type + '/imsds/' + string_now + '_traj_' + figname + '.npy', trajectories)
#    numpy.save(basic_directory_paper_data + '_' + spot_type + '/imsds/' + string_now + '_results_' + figname + '.npy', results)
#    numpy.save(basic_directory_paper_data + '_' + spot_type + '/imsds/' + string_now + '_imsds_' + figname + '.npy', imsds)
    if spot_type != "origins":
        os.makedirs(basic_directory_paper_data + '_' + spot_type + '/imsds/', exist_ok=True)
        numpy.save(basic_directory_paper_data + '_' + spot_type + '/imsds/' + string_now + '_selected_particles_' + figname + '.npy', selected_particles)

    return imsds, results

def plot_percent_of_particles_below_theoretical_noise_floor(per_day = False, pool_now = False, imsds = None, results = None, trajectories = None, n_time_lags = 5, avoid = ['1000ms']):

    if pool_now:
        imsds = concatenate('imsds_all_renamed', per_day = per_day)
        results = concatenate('results_per_particle', per_day = per_day)
        trajectories = concatenate('trajectories_all_renamed', per_day = per_day)

    c = {}
    c['0030ms'] = '#003AFF'
    c['0090ms'] = '#FF8300'
    c['0210ms'] = '#00D66B'
    c['1000ms'] = '#B20000'

    nparticles_below_noise = {}
    keys_of_interest = [x for x in imsds.keys() if not any([y in x for y in avoid])]

#    keys_of_interest = [x for x in imsds.keys() if not avoid[0] in x]
#    keys_of_interest = [x for x in keys_of_interest if not avoid[1] in x]

    for k in keys_of_interest[:]:
        l = k.split('h')
        time_between_frames = load_all_results.read('time_between_frames', k, spot_type)
        time_between_frames_c = str(time_between_frames).zfill(4) + 'ms'
        time_between_frames = float(time_between_frames) * 0.001
        nparticles_below_noise[k] = pandas.DataFrame(columns = ['percent_below_noise'], index = numpy.arange(time_between_frames, (n_time_lags + 1) * time_between_frames, time_between_frames))
        nparticles = len(imsds[k].columns)
        localization_uncertainties = trajectories[k].ep.to_numpy(dtype = numpy.float64)
        localization_uncertainties_mean = localization_uncertainties.mean() * px_to_micron
        localization_uncertainties_std = localization_uncertainties.std() * px_to_micron
        noise_floor = (localization_uncertainties_mean + 2 * localization_uncertainties_std)**2
        print(k)
        print(str(noise_floor) + ' mum^2')

        for j in numpy.arange(0, n_time_lags):
            temp = imsds[k].iloc[j,:] < noise_floor
            nparticles_above_noise_now = temp.value_counts().loc[False]
            nparticles_below_noise_now = nparticles - nparticles_above_noise_now
            percent = 100 * (nparticles_below_noise_now / float(nparticles))
            percent = round(percent, 2)
            nparticles_below_noise[k].iloc[j, 0] = percent

        l = l[0] + 'h'
        pylab.figure(l)
        pylab.plot(nparticles_below_noise[k].index, nparticles_below_noise[k].percent_below_noise, '-o', color = c[time_between_frames_c], label = time_between_frames_c)
        pylab.legend()
#pylab.xticks(numpy.arange(0, n_time_lags * 0.21 - 0.030, 0.030))
        pylab.xlabel('time lag (s)')
        pylab.ylabel('% of particles with msd < noise floor')
        pylab.title(l)
        pylab.ylim(-0.025, 0.5)

        pylab.savefig(basic_directory + central_directory + 'plots/imsds_per_cat_with_noise_floor/percent_below_noise_floor_' + l + '.png')

    return nparticles_below_noise

def plot_D_hist_per_cat_per_day(results, spot_type, avoid = ['1s'], label = None, days = 'all_days'):
    '''
    Make one figure per condition per day, each containing one histogram of D per movie. The point is to see whether D changes during the course of movie-taking.

    INPUT
    -----
    results : dictionary of DataFrames
        A dictionary of DataFrames of the results per particle. These are files saved as 'results_per_particle.pkl' that contain, for each particle, its D, alpha, diagonal size and starting snr.

    avoid : list of strings or None (default)
        A list of strings describing a feature that you are not interested in, found in the movie filenames.

    OUTPUT
    ------
    Graphs as described above, saved in the general folder for the spot type (origins or muNS).
    '''

    categories = 'pooled_directories_by_day.npy'

    c = numpy.load(basic_directory + '_' + spot_type + '/' + categories, allow_pickle=True).item()
    #bins_now = numpy.arange(-1.1, 1.25, 0.05)
    bins_now = numpy.logspace(-5, -2, num = 30)
    central_directory = load_all_results.select_specifications(spot_type)['central_directory']
    if days == 'all_days':
        days = load_all_results.select_specifications(spot_type)['all_days']
    keys_of_interest = [x for x in c.keys()]

    if isinstance(avoid, list):
        for l in avoid:
            keys_of_interest = [x for x in keys_of_interest if l not in x]

    if isinstance(days, list):
        keys_of_interest = [x for x in keys_of_interest if any([d in x for d in days])]

    print(keys_of_interest)

    for i in keys_of_interest[:]:
        n = len(c[i])
        if n > 1:
            fig = pylab.figure()
            #, figsize = (10, 2.5*n))
            fig.suptitle(i)
            gs = GridSpec(12, 10, figure=fig)
            ax = {}
            print(i)
            fig.subplots_adjust(top=0.9)
            pylab.subplots_adjust(hspace=0.65)
            for j in numpy.arange(0, 6):
                ax[j] = fig.add_subplot(gs[(2*j):(2*(j+1)), :8])
                ax[j].set_xscale('log')
                #for j,k in enumerate(c[i]):
                if j < len(c[i]):
                    k = c[i][j]
                    m = load_all_results.read('movie', k, spot_type).zfill(4)
                    #col = colorFader(color1,color2, mix = j / float(n))
                    nparticles = len(results[k])
                    Ds_now = results[k].D_app.to_numpy(dtype = numpy.float64)
                    Ds_now = Ds_now[~numpy.isnan(Ds_now)]
                    nparticles_notnan = len(Ds_now)
                    Ds_now = Ds_now[numpy.greater(Ds_now, numpy.zeros_like(Ds_now))]
                    nparticles_neither_negative = len(Ds_now)
                    Ds_now_median = round(numpy.median(Ds_now),10)
                    Ds_now_std = numpy.std(Ds_now)
                    Ds_now_sem = round(Ds_now_std / numpy.sqrt(nparticles_neither_negative), 10)
                    percent_included = 100 * round(float(nparticles_neither_negative) / nparticles, 10)
                    weights = 100 * numpy.ones_like(Ds_now) / float(len(Ds_now))
                    h = ax[j].hist(Ds_now, bins = bins_now, #density = True,
                                   color = 'b', weights = weights)
                    pylab.axvline(numpy.median(Ds_now), color = 'k')
                    ax[j].text(bins_now[22], 12, 'N = ' + str(nparticles_neither_negative))
                    ax[j].text(bins_now[::-1][0] + 5e-3, 17, 'movie ' + m, fontsize = 12)
                    ax[j].text(bins_now[::-1][0] + 5e-3, 2, 'D = ' + format(Ds_now_median, '.1e')                                                     + '\n+/-' + format(Ds_now_sem, '.1e'), fontsize = 12, color = 'b')
                    print(str(percent_included) + '% of all tracked particles')
                    ax[j].set_ylim(0, 20)
                    #ax[j].set_xlim(1e-4, 1e-1)
                    pylab.yticks([0, 10])
                    if j != len(c[i]) - 1:
                        pylab.xticks([])

                else:
                    #ax[j].hist(results[c[i][0]].alpha.to_numpy(dtype = numpy.float64), bins = bins_now, normed = True, color = 'b')
                    ax[j].axis('off')

            if isinstance(label, str):
                filename = 'Dapp_' + label + '_' + i + '.png'
            else:
                filename = 'Dapp_' + i + '.png'

            pylab.savefig(basic_directory + central_directory + 'plots/Ds_per_cat_per_day/histograms/' + filename)

    return h

def collect_Dapp_vs_movienumber(spot_type, results, avoid = ['1s'], label = '', days = 'all_days', logy = False):
    '''
    Collect - Make one figure per condition per day, each containing one histogram of alpha per movie. The point is to see whether alpha changes during the course of movie-taking.

    INPUT
    -----
    results : dictionary of DataFrames
        A dictionary of DataFrames of the results per particle. These are files saved as 'results_per_particle.pkl' that contain, for each particle, its D, alpha, diagonal size and starting snr.

    avoid : list of strings, defaults to ['1s']
        A list of strings describing features that you are not interested in, found in the movie filenames.

    OUTPUT
    ------
    Graphs as described above, saved in the general folder for this spot_type (muNS or origins).
    '''

    c = numpy.load(basic_directory + '_' + spot_type + '/' + 'pooled_directories_by_day.npy', allow_pickle=True).item()
    c2 = numpy.load(basic_directory + '_' + spot_type + '/' + 'pooled_directories.npy', allow_pickle=True).item()

    c2 = list(c2.keys())
    if isinstance(avoid, list):
        for l in avoid:
            c2 = [x for x in c2 if l not in x]

    keys_of_interest = [x for x in c.keys() if not any([y in x for y in avoid])]

    if isinstance(days, list):
        keys_of_interest = [x for x in keys_of_interest if any([d in x for d in days])]

    print('keys_of_interest:')
    print(keys_of_interest)

    if isinstance(avoid, list):
        for l in avoid:
            keys_of_interest = [x for x in keys_of_interest if l not in x]

    all_days = load_all_results.select_specifications(spot_type)['all_days']
    central_directory = load_all_results.select_specifications(spot_type)['central_directory']

    answer = {}
    for i in c2:
        answer[i] = {}
        for d in all_days:
            answer[i][d] = pandas.DataFrame(columns = ['movie', 'D', 'D_std', 'D_sem'])

    f = open(basic_directory + central_directory + 'Dapps_percent_of_unphysical_particles.txt', 'w')

    now = datetime.datetime.now()
    string_now = now.strftime("%Y%m%d")

    for i in keys_of_interest[:]:
        n = len(c[i])
        if n > 0:
            print(i)
            for k in c[i]:
                #print(k)
                f.write(k + '\n')
                movie = int(load_all_results.read('movie', k, spot_type))
                strain = load_all_results.read('strain', k, spot_type)
                time_between_frames = load_all_results.read('time_between_frames', k, spot_type).zfill(4) + 'ms'
                starvation_time = load_all_results.read('starvation_time', k, spot_type) + 'h'
                con = load_all_results.read('condition', k, spot_type)
                if len(con) > 0:
                    condition = strain + '_' + starvation_time + '_' + time_between_frames + '_' + con
                else:
                    condition = strain + '_' + starvation_time + '_' + time_between_frames
                day = load_all_results.read('day', k, spot_type)

                nparticles = len(results[k])
                Ds_now = results[k].D_app.to_numpy(dtype = numpy.float64)
                Ds_now = Ds_now[~numpy.isnan(Ds_now)]
                Ds_now = Ds_now[numpy.greater(Ds_now, numpy.zeros_like(Ds_now))]
                nparticles_neither_negative = len(Ds_now)
                f.write(str(round(100 - 100 * float(nparticles_neither_negative) / nparticles, 1)) + '% of particles have D = nan or < 0.\n')
                #alphas_greater_1p2 = alphas_now[numpy.greater(alphas_now, 1.2 * numpy.ones_like(alphas_now))]
                #f.write(str(round(100 * float(len(alphas_greater_1p2)) / nparticles, 1)) + '% of particles have α > 1.2.\n\n')
                #alphas_now = alphas_now[numpy.less(alphas_now, 1.2 * numpy.ones_like(alphas_now))]

                Ds_now_median = numpy.median(Ds_now)
                Ds_now_std = numpy.std(Ds_now)
                Ds_now_sem = Ds_now_std / numpy.sqrt(nparticles_neither_negative)
                percent_included = 100 * round(float(nparticles_neither_negative) / nparticles, 2)

                little_answer =  pandas.DataFrame(data = numpy.array([[movie, Ds_now_median, Ds_now_std, Ds_now_sem]]), columns = ['movie', 'D_app', 'D_app_std', 'D_app_sem'])

                answer[condition][day] = answer[condition][day].append(little_answer)

    if isinstance(label, str):
        filename = 'Dapp_' + label + '_vs_movienumber.npy'
    else:
        filename = 'Dapp_vs_movienumber.npy'

    numpy.save(basic_directory + central_directory + '_' + string_now + '_' + label + '_' + filename, answer)

    f.close()

    colors_now = {}
#    colors_now['muNS'] = {}
#    colors_now['muNS']['210323'] = '#33B8FF'
#    colors_now['muNS']['201226'] = '#2002FF'
#    colors_now['muNS']['201009'] = '#FFA702'
#    colors_now['muNS']['201006'] = '#01B10B'
#    colors_now['muNS']['201004'] = '#B10101'

    colors_now['origins'] = {}
    colors_now['origins']['210227'] = 'r' #'#33B8FF' #'#2002FF'
    colors_now['origins']['210330'] = 'g' #'#2002FF' #'#FFA702'
    colors_now['origins']['210518'] =  'b' #'#FFA702' #'#01B10B'

    legend_elements = []
    g1 = Line2D([0], [0], linewidth = 0, marker=markershapes['0h'], color=colors['bLR1']['0h'], label='WT 0h', markersize=ms, markeredgecolor = 'k', markeredgewidth = 1,alpha = 0.9)
    g2 = Line2D([0], [0], linewidth = 0, marker=markershapes['6hlowN'], color=colors['bLR1']['6hlowN'], label='WT 6h', markersize=ms, markeredgecolor = 'k', markeredgewidth = 1,alpha = 0.9)
    g3 = Line2D([0], [0], linewidth = 0, marker=markershapes['0h'], color=colors['bLR2']['0h'], label='ΔpolyP 0h', markersize=ms, markeredgecolor = 'k', markeredgewidth = 1,alpha = 0.9)
    g4 = Line2D([0], [0], linewidth = 0, marker=markershapes['6hlowN'], color=colors['bLR2']['6hlowN'], label='ΔpolyP 6h', markersize=ms,markeredgecolor = 'k', markeredgewidth = 1, alpha = 0.9)
    legend_elements = [g1, g2, g3, g4]

#    for i in colors_now[spot_type].keys():
#        g = Line2D([0], [0], marker='o', color=colors_now[spot_type][i], label=i, markersize=10, alpha = 1)
#        legend_elements.append(g)

    for c in list(answer.keys())[:]:
        print(c)
        time_between_frames = load_all_results.read('time_between_frames', c, spot_type)
        strain = load_all_results.read('strain', c, spot_type)
        starvation_time = load_all_results.read('starvation_time', c, spot_type) + 'h'
        condition = load_all_results.read('condition', c, spot_type)
        if strain == 'bLR1':
            strain_name = 'WT'
        elif strain == 'bLR2':
            strain_name = 'ΔpolyP'
        fig_label = 'origins, D_app per movienumber'
#        fig_label = strain_name + ' ' + starvation_time + ' ' + condition
        fig_label = fig_label + ' ' + label
        pylab.figure(fig_label, figsize = (6,6)) #(9.5,5.5)
        ax = pylab.gca()
#        ax.ticklabel_format(style = 'sci')
        if logy:
            ax.set_yscale('log', nonposy='clip')
        pylab.title(fig_label)
        pylab.xlabel('movie number\n(shifted by the first number of each day)')
        pylab.ylabel(r'$D_{app}$' + ' ' + r'$(\mu m^{2}/s^{\alpha})$')
        for d in list(answer[c].keys())[:]:
            movies = answer[c][d].movie.to_numpy(dtype = numpy.float64)
            if len(movies) > 0:  # you can change 0 to 1 if you are only interested in the dependence of D_app on movienumber, in which case it only makes sense to look at days with more than 1 movie
                min_movie = movies.min()
                new_movienumbers = movies - min_movie
                new_movienumbers = [x-20000 if x>=20000 else x for x in new_movienumbers]
                print(new_movienumbers)
#                pylab.errorbar(new_movienumbers, answer[c][d].D_app.to_numpy(dtype = numpy.float64), fmt = 'o-', yerr = answer[c][d].D_app_sem.to_numpy(dtype = numpy.float64), label = d, color = colors_now[spot_type][d], capsize = 5, elinewidth = 1, linewidth = 2, markersize = 10, alpha = 1)
                pylab.plot(new_movienumbers, answer[c][d].D_app.to_numpy(dtype = numpy.float64), markershapes[starvation_time + condition], label = d, color = colors[strain][starvation_time + condition],  markersize = ms, markeredgecolor = 'k', markeredgewidth = 1, alpha = 0.8)
#                eb = pylab.errorbar(new_movienumbers, answer[c][d].D_app.to_numpy(dtype = numpy.float64), fmt = 'o-', yerr = answer[c][d].D_app_std.to_numpy(dtype = numpy.float64), color = colors_now[d], capsize = 7, elinewidth = 1, linewidth = 2)
#                eb[-1][0].set_linestyle('--')
                #pylab.plot(new_movienumbers, answer[c][d].D_app.to_numpy(dtype = numpy.float64), 'o--', label = d)
        pylab.legend(handles=legend_elements, frameon = True, loc = 5, handletextpad = 0, borderaxespad = 0.2, bbox_to_anchor=(1.4, 0.85))

        if spot_type == 'origins':
            pylab.xlim(-0.5, 6.5)
            pylab.ylim(1e-4, 4e-4)
#            pylab.ylim(5e-5, 3e-3)

        elif spot_type == 'muNS':
            pylab.xlim(-1, 25)
            pylab.ylim(1e-4, 1e-3)

        elif spot_type == 'muNS_lowC':
            pylab.xlim(-1, 25)
            pylab.ylim(1e-4, 1e-3)

#        pylab.ylim(-0.0001, 0.025)

#        filename = strain_name + '_' + starvation_time + '_' + condition + '_' + label
#
#        if isinstance(label, str):
#            filename = string_now + '_' + filename + '_Dapp_' + label + '_vs_time'
#        else:
#            filename = string_now + '_' + filename + '_Dapp_vs_time'
#
#        pylab.savefig(basic_directory + central_directory + 'plots/D_apps_per_movienumber/' + filename + '.png', bbox_inches = 'tight')
#
#        pylab.savefig(basic_directory_paper_figs + filename + '.png', bbox_inches = 'tight')
#        pylab.savefig(basic_directory_paper_figs + filename + '.svg', bbox_inches = 'tight')

    filename = string_now + '_' + spot_type + '_Dapp_per_movienumber' + label

    pylab.savefig(basic_directory + central_directory + 'plots/D_apps_per_movienumber/' + filename + '.png', bbox_inches = 'tight')

    pylab.savefig(basic_directory_paper_figs + filename + '.png', bbox_inches = 'tight')
    pylab.savefig(basic_directory_paper_figs + filename + '.svg', bbox_inches = 'tight')

    return answer

def bin_dataframes(df, quantity, below_size_limit_only = True, avoid = ['lowC'], quantiles = False):
    '''
    Consider a subset of the DataFrame, binned per the quantity you have chosen.
    '''

    results = {}

    keys_of_interest = [x for x in df.keys() if not any([y in x for y in avoid])]

    if quantity == 'particle_size':
        bin_limits = numpy.linspace(7, 14, 7)

    if quantiles:
        bin_limits_now = {}
    else:
        bin_limits_now = bin_limits

    for j, k in enumerate(bin_limits_now):
        results[str(j)] = {}


    for i in keys_of_interest:
        print(i)

        if quantity == 'particle_size':
            df[i]['particle_size'] = df[i]['average_starting_magnitude']**(1./3)

        if below_size_limit_only:
#            l = df[i].below_diffraction_limit
#            l = df[i][df[i].below_diffraction_limit == 1.0]
#            l = list(l)
#            a = numpy.where(~numpy.isnan(l))[0] # all entries where the value for 'below_size_limit' is either True or False, but not nan
#            r_now = df[i].iloc[a, :]
#            r_now = r_now[r_now.below_diffraction_limit] # all entries where the spots are smaller than the size limit (previously the diffraction limit but judged as too strict)
            r_now = df[i][df[i].below_diffraction_limit == 1.0] # all entries where the spots are smaller than the size limit (previously the diffraction limit but judged as too strict)

            r_now = r_now[r_now.average_starting_magnitude!=0]  # omit the 0 entries THIS SHOULD BECOME OBSOLETE ONCE I HAVE FITTED ALL MOVIES
#        results[i] = r_now.copy(deep = True)
        else:
            r_now = df[i]

#        if quantiles:  # the complication with quantiles is that you will not end up with the same bin limits across all conditions, unless you then adjust them
#            bin_limits_now[i] = r_now.loc[:, quantity].quantile(bin_limits).to_numpy(dtype = numpy.float64)

        for j, k in enumerate(bin_limits_now):
            if j < len(bin_limits_now) - 1:
                results[str(j)][i] = r_now[(bin_limits_now[j] <= r_now.loc[:,quantity]) & (r_now.loc[:,quantity] < bin_limits_now[j+1])]
            elif j == len(bin_limits_now) - 1:
                results[str(j)][i] = r_now[r_now.loc[:,quantity] > bin_limits_now[j]]
            print('bin ' + str(j) + ': N = ' + str(len(results[str(j)][i])))

    return results




desired_order_list = {}
desired_order_list['muNS'] = ['bLR31_0h', 'bLR31_6h_lowN', #'bLR31_6h_lowC',
                                  'bLR32_0h', 'bLR32_6h_lowN',  #'bLR32_6h_lowC'
                                  ]
desired_order_list['origins'] = ['bLR1_0h',
                                  #                                  'bLR1_6h_lowN',
                                  'bLR1_6h_lowC',
                                  'bLR2_0h',
                                  #'bLR2_6h_lowN',
                                  'bLR2_6h_lowC'
                                  ]

def plot_medians_vs_condition(spot_type, quantity, results, msd_timepoint = None, bin_label = '', file_with_mode_occupants = '/Volumes/GlennGouldMac/PolyP/data/_muNS/diameter7_minmass110_search_range4_memory20_stub_length20/results/20210726_mode_occupants_from_msd_at_2p1s.npy', avoid = [],
                              plot_params={}, zoom_low=False):
    '''
    Plot the median of the distribution for the different conditions, separating the mobile from the immobile mode using the value of the msd at a certain time point as the divider.

    INPUT
    -----
    spot_type : str
        This can be 'origins', 'muNS', 'fixed_origins', 'fixed_muNS', depending on the object you have tracked.

    quantity : str
        The quantity to plot. I can be any of the columns in the results pandas DataFrame: 'D_app', 'alpha', 'starting_snr',... TO BE COMPLETED WITH LIST OF COLUMNS

    results : dictionary of pandas DataFrames, typically pooled
        This dictionary is the result of load_all_results.load_all_results() subsequently pooled with the concatenate function.

    msd_timepoint : float or None (defaults). When float, typically 2 (s)
        The timepoint at which you have evaluated the msd, in order to separate particles by its value.

    OUTPUT
    ------
    The results DataFrame, and a dot plot of the median values per condition, for each subpopulation of mobility.
    '''

    now = datetime.datetime.now()
    string_now = now.strftime("%Y%m%d")


    if ((isinstance(msd_timepoint, float)) and (msd_timepoint == 2)):
        mode_occupants = numpy.load(file_with_mode_occupants, allow_pickle=True).item()
        keep_track_of_mobilities = True
        results_classified = {}
        results_classified['1'] = {}
        results_classified['2'] = {}

    elif ((isinstance(msd_timepoint, float)) and (msd_timepoint != 2)):
        raise ValueError('I do not have the mode occupants for t = ' + str(msd_timepoint) + '.')

    else:
        keep_track_of_mobilities = False

    print(keep_track_of_mobilities)

    keys_of_interest = [x for x in results.keys() if not any([y in x for y in avoid])]
    print(keys_of_interest)
    for i in keys_of_interest:
        print(i)
        total_N = len(results[i])

        if keep_track_of_mobilities:

            m1 = [x for x in mode_occupants['1'][i] if x in results[i].index]
            m2 = [x for x in mode_occupants['2'][i] if x in results[i].index]
            print('mode 1: ' + str(100 * len(m1) / total_N))
            print('mode 2: ' + str(100 * len(m2) / total_N))

            results_classified['1'][i] = results[i].loc[m1]
            results_classified['2'][i] = results[i].loc[m2]
            final_results = results_classified
        else:
            final_results = results

    ft = quantity + ' median values'
    if len(bin_label) > 0:
        ft = ft + ', bin ' + str(bin_label)
    pylab.figure(ft, figsize = (6, 5))
    pylab.title(ft)
    legend_entries = []

    all_results_mobility = {}

    for i in keys_of_interest:
        print('key: ', i)
        strain = load_all_results.read('strain', i, spot_type)
        starvation_time = load_all_results.read('starvation_time', i, spot_type) + 'h'
        time_between_frames = load_all_results.read('time_between_frames', i, spot_type)
        condition = load_all_results.read('condition', i, spot_type)
        if keep_track_of_mobilities:
            data1 = results_classified['1'][i].loc[:,quantity].to_numpy(dtype = numpy.float64)
            err1 = results_classified['1'][i].loc[:,quantity + "_err"].to_numpy(dtype = numpy.float64)
            n1 = len(data1)
            data2 = results_classified['2'][i].loc[:,quantity].to_numpy(dtype = numpy.float64)
            err2 = results_classified['2'][i].loc[:,quantity + "_err"].to_numpy(dtype = numpy.float64)
            n2 = len(data2)
            data = results[i].loc[:,quantity].to_numpy(dtype = numpy.float64)
            err = results[i].loc[:,quantity + "_err"].to_numpy(dtype = numpy.float64)
            n = len(data)
        else:
            data = results[i].loc[:,quantity].to_numpy(dtype = numpy.float64)
            err = results[i].loc[:,quantity + "_err"].to_numpy(dtype = numpy.float64)
            n = len(data)
        legend_entries.append(strain + ' ' + starvation_time)
        pylab.figure(ft)

        if keep_track_of_mobilities:
            folder = basic_directory_paper_figs + '_' + spot_type + '/' + string_now + '_medians_per_population_' + quantity + '_data/'
            os.makedirs(folder, exist_ok=True)
            file_name = folder + f"{quantity}_{strain}_{starvation_time}_{condition}"
            std = 0 if numpy.isnan(data.std()) else data.std()
            std1 = 0 if numpy.isnan(data1.std()) else data1.std()
            std2 = 0 if numpy.isnan(data2.std()) else data2.std()
            if strain + '_' + starvation_time == 'bLR31_0h':
                print(strain+ "_" + starvation_time, "HERE")
                pylab.errorbar(positions[strain][starvation_time + condition], numpy.nanmedian(data), xerr = 0,
                               yerr = numpy.sqrt((std / numpy.sqrt(n))**2 + numpy.nanmedian(err)**2),
                               fmt = markershapes[starvation_time + condition], markersize = ms, markeredgecolor = colors[strain][starvation_time + condition], ecolor = colors[strain][starvation_time + condition], color = 'w', alpha = 1, markeredgewidth = 2, capsize = 10)
                numpy.savetxt(file_name + '.csv', data, delimiter=',')
                print("data_mean", data.mean())
            else:
                print("data1_mean", data1.mean())
                print("data2_mean", data2.mean())
                pylab.errorbar(positions[strain][starvation_time + condition], numpy.nanmedian(data1), xerr = 0,
                               yerr = numpy.sqrt((std1 / numpy.sqrt(n1))**2 + numpy.nanmean(err1)**2),
                               fmt = markershapes[starvation_time + condition], markersize = ms, markeredgecolor = colors[strain][starvation_time + condition], color = 'w', ecolor = colors[strain][starvation_time + condition], alpha = 1, markeredgewidth = 2, capsize = 10)
                numpy.savetxt(file_name + '_immobile.csv', data1, delimiter=',')
                numpy.savetxt(file_name + '_mobile.csv', data2, delimiter=',')

                pylab.errorbar(positions[strain][starvation_time + condition], numpy.nanmedian(data2), xerr = 0,
                               yerr = numpy.sqrt((std2 / numpy.sqrt(n2))**2 + numpy.nanmean(err2)**2),
                               fmt = markershapes[starvation_time + condition], markersize = ms, markeredgecolor = 'k', color = colors[strain][starvation_time + condition], alpha = 1, markeredgewidth = 2, capsize = 10)
                pylab.plot(positions[strain][starvation_time + condition], numpy.nanmedian(data2), markershapes[starvation_time + condition], markersize = ms, markeredgecolor = 'k', color = 'w', alpha = 0.4, markeredgewidth = 2)

        else:
            print('here')
            print(starvation_time + condition)
            std = 0 if numpy.isnan(data.std()) else data.std()
            std = numpy.nanstd(data)
            if 'bLR' in strain:
                print("std", std)
                pylab.errorbar(positions[strain][starvation_time + condition], numpy.nanmedian(data), xerr = 0,
                               yerr = numpy.sqrt((std / numpy.sqrt(n))**2 + numpy.nanmedian(err)**2),
                            #    numpy.sqrt((numpy.nanstd(data) / numpy.sqrt(n))**2 + numpy.nanmean(err)**2),
                               fmt = markershapes[starvation_time + condition], markersize = ms, markeredgecolor = 'k', ecolor = colors[strain][starvation_time + condition], color = colors[strain][starvation_time + condition], alpha = 1, markeredgewidth = 2, capsize = 10)
            else:
                pylab.errorbar(positions[strain][starvation_time + condition], numpy.nanmedian(data), xerr = 0,
                               yerr = numpy.sqrt((std / numpy.sqrt(n))**2 + numpy.nanmean(err)**2),
                               fmt = markershapes[starvation_time + condition], markersize = ms, markeredgecolor = colors[strain][starvation_time + condition], ecolor = colors[strain][starvation_time + condition], color = 'w', alpha = 1, markeredgewidth = 2, capsize = 10)

    pylab.legend(legend_entries)
    if spot_type == 'muNS_lowC':
        pylab.xticks([1, 1.2, 1.4, 1.6], ['WT\n0h', '\n6h lowC', 'ΔpolyP\n0h', '\n6h lowC'])
    elif spot_type == 'muNS':
        pylab.xticks([1, 1.2, 1.4, 1.6], ['WT\n0h', '\n6h lowN', 'ΔpolyP\n0h', '\n6h lowN'])


    if quantity == 'D_app':
        pylab.ylabel(r'$D_{app} (\mu m ^{2} / s^{\alpha}$)', labelpad = 0)
        if spot_type == 'muNS':
            if zoom_low:
                pylab.ylim(9e-4, 3.5e-3)
            else:
                pylab.ylim(0, 2.1e-2)
                pylab.yticks([1e-3, 5e-3, 1e-2, 1.5e-2, 2e-2])
        elif 'origins_' in spot_type:
            pylab.ylim([1e-4, 4.5e-4])  # For consistency with Fig1
            pylab.yticks([1e-4, 2e-4, 3e-4, 4e-4])
        elif spot_type == 'origins':
            pylab.ylim(1e-4, 4.2e-4)
#            pylab.yticks([1e-3, 1e-2, 2e-2, 3e-2])
        pylab.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    elif quantity == "msd_xs":
        if spot_type == 'muNS':
            pylab.ylabel(r'MSD at t = 2s', labelpad = 0)
            if zoom_low:
                print("zoom on low mode")
                pylab.ylim(3e-3, 1.3e-2)
                pylab.yticks([4e-3, 6e-3, 8e-3, 10e-3, 12e-3])
                pylab.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        elif spot_type == "origins":
            pylab.ylabel(r'MSD at t = 30s', labelpad = 0)

            pylab.ylim(1.5e-3, 5e-3)
            pylab.yticks([2e-3, 3e-3, 4e-3, 5e-3])

            pylab.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    elif quantity == 'alpha':
        pylab.ylabel(r'$\alpha$', rotation = 0, labelpad = 10)
        if spot_type == 'muNS':
            pylab.ylim(0.3, 0.9)
            # pylab.yticks([0.4, 0.5, 0.6, 0.7])
        elif 'origins_' in spot_type:
            pylab.ylim(0.25, 0.7)
            pylab.yticks([.3, .4, .5, .6, .7])
        elif 'origins' in spot_type:
            pylab.ylim(0.23, .7)
            # pylab.yticks(numpy.arange(0.35, 0.47, 0.025))

    os.makedirs(basic_directory_paper_figs + '_' + spot_type + '/' + string_now, exist_ok=True)
    low = "_low" if zoom_low else ""
    pylab.savefig(basic_directory_paper_figs + '_' + spot_type + '/' + string_now + '_medians_per_population_' + quantity + '0_05' + low + '.svg')
    pylab.savefig(basic_directory_paper_figs + '_' + spot_type + '/' + string_now + '_medians_per_population_' + quantity + '0_05' + low + '.png')
    print('Saved at', basic_directory_paper_figs + '_' + spot_type + '/' + string_now + '_medians_per_population_' + quantity + '0_05' + low + '.png')

    return final_results, data

# number of spots in the final DFs #
n = {}
n['bLR31_0h_0210ms'] = 2904
n['bLR32_0h_0210ms'] = 3036
n['bLR31_6h_0210ms_lowN'] = 3593
n['bLR32_6h_0210ms_lowN'] = 5023


def histogram_all(spot_type, quantity, results, avoid = ['1s'], compare = False, time_label = 'tstart0p21sec_tend0p84sec', plot_quantity_vs_position = False, show_particles_with_alpha_neg = False, show_median = False, bin_label = '', ignore_timelag = False, below_size_limit_only = True, size_limit = 0.450, per_day = False, n_subset = None, D_app_mode = '', msd_mode = '', msd_timepoint = 2, file_with_mode_occupants = '/Volumes/GlennGouldMac/PolyP/data/_muNS/diameter7_minmass110_search_range4_memory20_stub_length20/results/20210726_mode_occupants_from_msd_at_2p1s.npy', particle_size_constraint = []):
    '''
    Create and plot histograms for a quantity in the results file.

    INPUT
    -----
    spot_type : str
        This can be 'origins', 'muNS', 'fixed_origins', 'fixed_muNS', depending on the object you have tracked.

    quantity : str
        The quantity to plot. I can be any of the columns in the results pandas DataFrame: 'D_app', 'alpha', 'starting_snr',... TO BE COMPLETED WITH LIST OF COLUMNS

    results : dictionary of pandas DataFrames, typically pooled
        This dictionary is the result of load_all_results.load_all_results() subsequently pooled with the concatenate function.

    avoid : list of strings, defaults to ['1s']
        Movies with this string in their filename will be ignored.

    compare : boolean, defaults to False
        If True, then you will make comparative plots, where you compare dataA to dataB. To use this feature, first histogram_all(dataA) with compare = False, then histogram_all(dataB) with compare = True and new_figure = False. You will obtain a fully colored histogram of the quantity of dataA, and over it superimposed a transparent histogram of the quantity of dataB. All histograms are normalized such that the y axis represents the % of particles in each bin.

        #    new_figure : boolean, defaults to True
        #When True, you will create a new figure. When False, you will plot histograms on top of existing # figures with the same name.

    OUTPUT
    ------
    Two outputs: 1. the results dictionary of pandas DataFrames that you used to create the histograms and 2. a 2d numpy array representing the histograms, where one column has the bin midpoints and the other column has the bin heights - in other words, a representation of the histograms. You can use this later on to fit curves on the histograms.

    This function also saves the 2d numpy array version of the histograms as well as the figures, if you have chosen plot = True.
    '''

    ## useful labels and info for figures and filenames ##
    now = datetime.datetime.now()
    string_now = now.strftime("%Y%m%d")

    if per_day:
        pd = '_per_day_'
    else:
        pd = '_'

    if len(bin_label) > 0:
        string_now = string_now + '_bin' + bin_label

    central_directory = load_all_results.select_specifications(spot_type)['central_directory']

    ##
    if ignore_timelag:
        r = {k: results[k] for k in desired_order_list[spot_type]}   # ordered dictionary, to keep the subplots consistent later on
    else:
        r = results

    if particle_size_constraint:
        constraint_str = 'size_range_' + str(particle_size_constraint[0]) + 'to' + str(particle_size_constraint[1])
    else:
        constraint_str = 'full_size_range'

    ## select which parts of the input data to use ##
    keys_of_interest = [x for x in r.keys() if not any([y in x for y in avoid])]

    h = {}  ### all the histograms and bins that will be plotted will occupy this dictionary
    midpoints_now = {}  # useful for fitting later
    f = {} ### histogram midpoints and histogram values at that bin

    data_medians = {}
    data_sems = {}

    #    limiting_D_app = {} # I get these values from the point of intersection between bimodal fit curves to D_app. Then I use them to split populations in D_app or α.
    #    limiting_D_app['bLR31_0h_0210ms'] = numpy.nan
    #    limiting_D_app['bLR31_6h_0210ms_lowN'] = 0.003828
    #    limiting_D_app['bLR32_0h_0210ms'] = 0.000557
    #    limiting_D_app['bLR32_6h_0210ms_lowN'] = 0.008328
    mode_occupants2 = {}
    mode_occupants2['1'] = {}
    mode_occupants2['2'] = {}

    if len(msd_mode) > 0:
        mode_occupants = numpy.load(file_with_mode_occupants, allow_pickle=True).item()

    for i in keys_of_interest[:]:
        print(i)
        r_now = r[i].copy(deep = True)
        if len(particle_size_constraint) > 0:
            r_now = r_now[r_now.average_starting_magnitude > (particle_size_constraint[0])**3]
            r_now = r_now[r_now.average_starting_magnitude < (particle_size_constraint[1])**3]
        nparticles = len(r_now)
        print('There are ' + str(len(r_now)) + ' measurements, total.')

        if below_size_limit_only:  # if you only want to consider points with width below the size limit
    #            r_now = r_now[r_now.below_diffraction_limit]
            r_now = r_now[r_now.below_diffraction_limit == 1]
            print('Here you see ' + str(len(r_now)) + ' of these particles, the ones below the size limit.')

        if isinstance(n_subset, int):
            particles = set(numpy.array(r_now.index))
            particles = random.sample(particles, n_subset)
            r_now = r_now.loc[particles, :]

        strain = load_all_results.read('strain', i, spot_type)
        starvation_time = load_all_results.read('starvation_time', i, spot_type) + 'h'
        condition = load_all_results.read('condition', i, spot_type)
        label = strain + '_' + starvation_time
        midpoints_now[i] = []

        if starvation_time == '6h':
            label = label + '_' + condition
        #print(label)

        if strain == 'bLR1' or strain == 'bLR31':
            label2 = label.replace(strain,'WT')
        elif strain == 'bLR2' or strain == 'bLR32':
            label2 = label.replace(strain, 'ΔpolyP')
        elif strain == 'LR262':
            label2 = label.replace(strain, 'WT_Ppk2B')
        elif strain == 'LR264':
            label2 = label.replace(strain, 'ΔpolyP_PpkB')
        label2 = label2.replace('_', ', ')

        ft = spot_type + ', ' + quantity   # figure name
        stitle = spot_type + ', ' + quantity  # figure title
    #        print('1478: ' + ft)
        if per_day:
            print('PERDAY')
            day = i.split('_')[::-1][0]
            print(day)
            ft = ft + ', ' + day
            stitle = stitle + ', ' + day
    #            print('1485: ' + ft)

        time_between_frames = ''
        if len(bin_label) > 0:
            ft = ft + ' bin ' + bin_label
    #            print('1490: ' + ft)

        if not ignore_timelag:
            time_between_frames = load_all_results.read('time_between_frames', i, spot_type)
            ft = ft + ', time between frames: ' + str(time_between_frames)
            stitle = stitle + ', time between frames: ' + str(time_between_frames) + 'ms'
    #            print('1496: ' + ft)

        if below_size_limit_only:
    #            if not compare:
            ft = ft + ', below diff. limit'
            stitle = stitle + '\nbelow size limit'

    #        print('1503: ' + ft)

        if not compare:
            if quantity == 'D_app':
                fig = pylab.figure(ft, figsize = (11, 7))  #(11, 5) # D_app figure is larger
            else:
                fig = pylab.figure(ft, figsize = (7.5, 7))  #(11, 5)
        else:
            fig = pylab.gcf()

        fig.suptitle(stitle)
    #        pylab.ylabel('% of particles')
        gs = GridSpec(len(keys_of_interest), 10, figure=fig)

    #        gs = GridSpec(4, 10, figure=fig)
    #        gs = GridSpec(6, 10, figure=fig)
        ax = {}
        fig.subplots_adjust(top=0.9)
        pylab.subplots_adjust(hspace=0.08) #0.5, 0.08

    #        if spot_type == 'muNS':
    #            pylab.subplots_adjust(hspace=0.08) #0.5, 0.08
    #        elif spot_type == 'origins':
    #            pylab.subplots_adjust(hspace=0.08)

        if quantity == 'alpha':
            nparticles = len(r_now)
            print(nparticles)
            if starvation_time == '6h':
                if D_app_mode == 'low':
                    r_now = r_now[r_now.D_app <= limiting_D_app[i]]
                elif D_app_mode == 'high':
                    r_now = r_now[r_now.D_app > limiting_D_app[i]]

            if len(msd_mode) > 0:
                mode_occupants2['1'][i] = [x for x in mode_occupants['1'][i] if x in list(r_now.index)]
                mode_occupants2['2'][i] = [x for x in mode_occupants['2'][i] if x in list(r_now.index)]

                if msd_mode == 'low':
                    r_now = r_now.loc[mode_occupants2['1'][i], :]
                elif msd_mode == 'high':
                    r_now = r_now.loc[mode_occupants2['2'][i], :]

            data = r_now.alpha
            data = data.to_numpy(dtype = numpy.float64)
            xmax = 1.25
            if len(msd_mode) > 0:
                ymax = 22
            else:
                ymax = 22#15 20 19 #14 #15#27#22 #25 #15 when only lowN,  22 when only lowC
            xtext_N = 0.9#0.8
            ymin = 0
            xmin = -0.05
    #            if show_particles_with_alpha_neg:
    #                xmin = -0.5
    #            else:
    #                xmin = -0.05
            bins_now = numpy.arange(xmin, xmax, 0.05)
            if not compare:
                ytext_N = 12#9 8#14
                color_text_now = colors[strain][starvation_time + condition]
                t = 'α = '
                value_x_location = bins_now[::-1][0] + 0.06
            else:
                if msd_mode == 'low':
                    ytext_N = 100#10#9
                elif msd_mode == 'high':
                    ytext_N = 100#7
                else:
                    ytext_N = 6
    #            ytext_N = 6
                color_text_now = '#000000'
                t = '/ '
                value_x_location = bins_now[::-1][0] + 0.32
            xtext_s = xmax + 1e-2
            ytext_s = ymax - 6#20 #
            for k, l in enumerate(bins_now):
                if k < len(bins_now) - 1:
                    c = numpy.linspace(bins_now[k], bins_now[k+1], num=3)
                    midpoints_now[i].append(c[1])
            label2_x = bins_now[::-1][0] + 0.06  # location of strain label (or xtext_s?)
            label2_y = 12
            value_y_location = 5

        elif quantity == 'D_app':
            nparticles = len(r_now)
    #            if starvation_time == '6h':
    #                if D_app_mode == 'low':
    #                    r_now = r_now[r_now.D_app <= limiting_D_app[i]]
    #                elif D_app_mode == 'high':
    #                    r_now = r_now[r_now.D_app > limiting_D_app[i]]

            if len(msd_mode) > 0:
                mode_occupants2['1'][i] = [x for x in mode_occupants['1'][i] if x in list(r_now.index)]
                mode_occupants2['2'][i] = [x for x in mode_occupants['2'][i] if x in list(r_now.index)]

                if msd_mode == 'low':
                    r_now = r_now.loc[mode_occupants2['1'][i], :]
                elif msd_mode == 'high':
                    r_now = r_now.loc[mode_occupants2['2'][i], :]

            data = r_now.D_app
            data = data.to_numpy(dtype = numpy.float64)

            if spot_type == 'muNS':
                if len(bin_label) > 0:
                    ymax = 27
                else:
                    ymax = 18 #22 #26  #30 # 25 when lowN only
                xmin = 1e-5 #3e-5 # 5e-5 when lowN only #TODO change here for different settings
                xmax = 1e-1 #5e-1 # when lowN only, 4e-2  when lowC?
                xtext_N =  1.5e-2 #8e-3 # 2e-2 when lowN only
                xtext_s = xmax + 2e-3
                ytext_s = ymax - 4
                value_y_location = ytext_s - 10 #15

            elif spot_type == 'muNS_lowC':
                if len(bin_label) > 0:
                    ymax = 27
                else:
                    ymax = 15# 22 #18 #26  #30 # 25 when lowN only
                xmin = 2e-5 #3e-5 # 5e-5 when lowN only
                xmax = 2e-2 #5e-2 #5e-1 # when lowN only, 4e-2  when lowC?
                xtext_N =  5e-3#1e-2 #8e-3 # 2e-2 when lowN only
                xtext_s = xmax + 2e-3
                ytext_s = ymax - 4
                value_y_location = ytext_s - 10 #15
                ytext_N = 10

            elif 'origins' in spot_type:
                print(spot_type)
                ymax = 20
                xmin = 1e-5 #5e-6
                xmax = 1e-2 #5e-2
                xtext_N = 2e-3 #8e-3 # 1e-2
                xtext_s = xmax + 1e-3 #xmax + 5e-3
                ytext_s = ymax - 4
                value_y_location = ytext_s - 10 #15

            elif spot_type == 'origins_comp':
                print("HERE", spot_type)
                ymax = 20
                xmin = 1e-5 #5e-6
                xmax = 1e-2 #5e-2
                xtext_N = 2e-3 #8e-3 # 1e-2
                xtext_s = xmax + 1e-3 #xmax + 5e-3
                ytext_s = ymax - 4
                value_y_location = ytext_s - 10 #15


            ymin = 0
            bins_now = numpy.logspace(-6, 0, num = 50)
            if not compare:
                t = r'$D_{app} = $'
                tend = r'$\mu m ^{2} / s^{\alpha}$'
                ytext_N = 10 #13#12
                color_text_now = colors[strain][starvation_time + condition]
                value_x_location = xtext_s
            else:
                t = '| '
                tend = ''
                if spot_type == 'muNS':
                    if msd_mode == 'low':
                        ytext_N = 100#10#9
                    elif msd_mode == 'high':
                        ytext_N = 100#7
                    else:
                        ytext_N = 10
                elif 'origins' in spot_type:
                    ytext_N = 8


                color_text_now = '#000000'
                if spot_type == 'muNS':
                    value_x_location = xtext_s + 4e0
                elif 'origins' in spot_type:
                    value_x_location = xtext_s + 4e-1

                elif spot_type == 'muNS_lowC':
                    value_x_location = xtext_s + 4e-1
            for k, l in enumerate(bins_now):
                if k < len(bins_now) - 1:
                    c = numpy.logspace(numpy.log10(bins_now[k]), numpy.log10(bins_now[k+1]), num=3)
                    midpoints_now[i].append(c[1])

        elif quantity == 'starting_snr':
            nparticles = len(r_now)
            data = r_now.starting_snr
            data = data.to_numpy(dtype = numpy.float64)

            xmin = 0.9
            xmax = 4
            ymax = 25
            bins_now = numpy.arange(1, 10, 0.1)
            xtext_s = xmax + 0.05
            ytext_s = ymax - 8
            value_y_location = ytext_s - 17
            xtext_N = xmax - 1
            if not compare:
                t = 'starting snr = '
                ytext_N = 14
                color_text_now = colors[strain][starvation_time + condition]
                value_x_location = xtext_s
            else:
                t = '| '
                ytext_N = 9
                color_text_now = '#000000'
                value_x_location = xtext_s + 4e0
            for k, l in enumerate(bins_now):
                if k < len(bins_now) - 1:
                    c = numpy.linspace(bins_now[k], bins_now[k+1], num=3)
                    midpoints_now[i].append(c[1])

        elif quantity == 'average_starting_magnitude':
            nparticles = len(r_now)
            if starvation_time == '6h':
                if D_app_mode == 'low':
                    r_now = r_now[r_now.D_app <= limiting_D_app[i]]
                elif D_app_mode == 'high':
                    r_now = r_now[r_now.D_app > limiting_D_app[i]]

            fig.suptitle(spot_type + ', magnitude, time between frames: ' + str(time_between_frames) + 'ms')
            data = r_now.average_starting_magnitude
            data = data.to_numpy(dtype = numpy.float64)

            xmin = 0
            xmax = 2500
            ymax = 25
            bins_now = numpy.arange(1, 5e3, 100)
            xtext_s = xmax + 12
            ytext_s = ymax - 6
            value_y_location = ytext_s - 17
            xtext_N = xmax - 600
            if not compare:
                t = 'mag. = '
                ytext_N = 14
                color_text_now = colors[strain][starvation_time + condition]
                value_x_location = xtext_s
            else:
                t = '| '
                ytext_N = 9
                color_text_now = '#000000'
                value_x_location = xtext_s + 4e0
            for k, l in enumerate(bins_now):
                if k < len(bins_now) - 1:
                    c = numpy.linspace(bins_now[k], bins_now[k+1], num=3)
                    midpoints_now[i].append(c[1])

        elif quantity == 'average_starting_sigma':
            nparticles = len(r_now)
            if starvation_time == '6h':
                if D_app_mode == 'low':
                    r_now = r_now[r_now.D_app <= limiting_D_app[i]]
                elif D_app_mode == 'high':
                    r_now = r_now[r_now.D_app > limiting_D_app[i]]

            fig.suptitle(spot_type + ', ave. starting σ, time between frames: ' + str(time_between_frames) + 'ms')
            data = r_now.average_starting_sigma
            data = data.to_numpy(dtype = numpy.float64)

            xmin = 0 #0
            xmax = 3
            ymin = 0
            ymax = 25 #60
            bins_now = numpy.arange(xmin, 10, 0.1)
            xtext_s = xmax + 0.1 # strain label
            ytext_s = ymax - 4   # strain label
            value_y_location = ytext_s - 15 # label for median and sem
            xtext_N = xmax - 1.5 # N label
            if not compare:
                t = 'σ = '
                ytext_N = 12
                color_text_now = colors[strain][starvation_time + condition]
                value_x_location = xtext_s
            else:
                t = '| '
                ytext_N = 9
                color_text_now = '#000000'
                value_x_location = xtext_s + 4e0
            for k, l in enumerate(bins_now):
                if k < len(bins_now) - 1:
                    c = numpy.linspace(bins_now[k], bins_now[k+1], num=3)
                    midpoints_now[i].append(c[1])

        elif quantity == 'particle_size':
            nparticles = len(r_now)
    #            if starvation_time == '6h':
    #                if D_app_mode == 'low':
    #                    r_now = r_now[r_now.D_app <= limiting_D_app[i]]
    #                elif D_app_mode == 'high':
    #                    r_now = r_now[r_now.D_app > limiting_D_app[i]]

            fig.suptitle(spot_type + ', ' + quantity + ', time between frames: ' + str(time_between_frames) + 'ms')
            if below_size_limit_only:
    #                r_now = r_now[r_now.below_diffraction_limit]
                r_now = r_now[r_now.below_diffraction_limit == 1]
                print(len(r_now))
                fig.suptitle(spot_type + ', ' + quantity + ', time between frames: ' + str(time_between_frames) + 'ms,\nspots with <w_o> < ' + str(size_limit) + ' mum')
            data = (r_now.average_starting_magnitude)**(1./3)
            data = data.to_numpy(dtype = numpy.float64)

            xmin = 6#5 with the old size limit at 0.225 mum
            xmax = 18#13 #25 - go up to 25 if you want to see more of the lowC ones.
            xtext_s = xmax + 0.15
            ymin = 0
            if len(bin_label) > 0:
                ymax = 25
            else:
                ymax = 8.5#50 #8.5 #15 with the old size limit at 0.225 mum #30
            ytext_s = ymax - 1 # 5, 10
            value_y_location = ytext_s - 3  # -10, - 20
            bins_now = numpy.arange(xmin, xmax, 0.25)  # let's keep the same size bins throughout the analysis, iterate. To see all particles, use bin width 0.5.
    #            bins_now = numpy.arange(1, xmax, 1)
            xtext_N = 14#10  #xmax - 8
            if not compare:
                t = 'mag. = '
                ytext_N = ymax - 2 #- 10
                color_text_now = colors[strain][starvation_time + condition]
                value_x_location = xtext_s
            else:
                t = '| '
                ytext_N = 9
                color_text_now = '#000000'
                value_x_location = xtext_s + 5.5
            for k, l in enumerate(bins_now):
                if k < len(bins_now) - 1:
                    c = numpy.linspace(bins_now[k], bins_now[k+1], num=3)
                    midpoints_now[i].append(c[1])

        elif quantity == 'previous_step_size':
            nparticles = len(r_now)
            if starvation_time == '6h':
                if D_app_mode == 'low':
                    r_now = r_now[r_now.D_app <= limiting_D_app[i]]
                elif D_app_mode == 'high':
                    r_now = r_now[r_now.D_app > limiting_D_app[i]]

            fig.suptitle(spot_type + ', ' + quantity + ', time between frames: ' + str(time_between_frames) + 'ms')
            if below_size_limit_only:
    #                r_now = r_now[r_now.below_diffraction_limit]
                r_now = r_now[r_now.below_diffraction_limit == 1.0]
                fig.suptitle(spot_type + ', ' + quantity + ', time between frames: ' + str(time_between_frames) + 'ms,\nspots with <w_o> < ' + str(size_limit) + ' mum')
            data = r_now.previous_step_size
            data = data.to_numpy(dtype = numpy.float64)
            data = data[~numpy.isnan(data)]
            data = data * px_to_micron * 1000
            xmin = 0
            xmax = 2.5 * px_to_micron * 1000 #3 previously # 25 - go up to 25 if you want to see more of the lowC ones.
            xtext_s = (xmax + 0.05) * px_to_micron * 1000
            ymin = 0
            ymax = 35#20#33 #when binned 20
            ytext_s = ymax - 10
            value_y_location = ytext_s - 21
            bins_now = numpy.arange(xmin, xmax, 10)  # let's keep the same size bins throughout the analysis, iterate. To see all particles, use bin width 0.5.
            #            bins_now = numpy.arange(1, xmax, 1)
            xtext_N = 1.7 * px_to_micron * 1000

            if not compare:
                t = 'mag. = '
                ytext_N = ymax - 7
                color_text_now = colors[strain][starvation_time + condition]
                value_x_location = xtext_s
            else:
                t = '| '
                ytext_N = 9
                color_text_now = '#000000'
                value_x_location = xtext_s + 5.5

            for k, l in enumerate(bins_now):
                if k < len(bins_now) - 1:
                    c = numpy.linspace(bins_now[k], bins_now[k+1], num=3)
                    midpoints_now[i].append(c[1])

        elif quantity == 'msd_xs':
            nparticles = len(r_now)
            data = r_now.msd_xs
            data = data.to_numpy(dtype = numpy.float64)

            if spot_type == 'origins':
                ymax = 27
                xmin = 3e-4 #5e-6
                xmax = 7e-2 #5e-2
                xtext_N = 2e-3 #8e-3 # 1e-2
                xtext_s = xmax + 1e-3 #xmax + 5e-3
                ytext_s = ymax - 4
                value_y_location = ytext_s - 10 #15
            if 'origins' in spot_type:
                ymax = 27
                xmin = 3e-4 #5e-6
                xmax = 7e-2 #5e-2
                xtext_N = 2e-3 #8e-3 # 1e-2
                xtext_s = xmax + 1e-3 #xmax + 5e-3
                ytext_s = ymax - 4
                value_y_location = ytext_s - 10 #15

            ymin = 0
            bins_now = numpy.logspace(-6, 0, num = 50)
            if not compare:
                t = r'$D_{app} = $'
                tend = r'$\mu m ^{2} / s^{\alpha}$'
                ytext_N = 10 #13#12
                color_text_now = colors[strain][starvation_time + condition]
                value_x_location = xtext_s
            else:
                t = '| '
                tend = ''
                if 'origins' in spot_type:
                    ytext_N = 8


                color_text_now = '#000000'

                if 'origins' in spot_type:
                    value_x_location = xtext_s + 4e-1
            for k, l in enumerate(bins_now):
                if k < len(bins_now) - 1:
                    c = numpy.logspace(numpy.log10(bins_now[k]), numpy.log10(bins_now[k+1]), num=3)
                    midpoints_now[i].append(c[1])

        else:
            raise ValueError('I do not know where to find the quantity you have asked me to plot.')

        print('N = ' + str(nparticles))

        data_now = data[~numpy.isnan(data)]
        nparticles_notnan = len(data_now)
        print('N not nan = ' + str(nparticles_notnan))
        data_now = data_now[numpy.greater(data_now, numpy.zeros_like(data_now))]
        nparticles_neither_negative = len(data_now)
        print('N neither negative = ' + str(nparticles_neither_negative))

        data_now_median = round(numpy.median(data_now), 11)
        data_medians[i] = data_now_median
        data_now_std = numpy.std(data_now)
        data_now_sem = round(data_now_std / numpy.sqrt(nparticles_neither_negative),11)
        data_sems[i] = data_now_sem
        percent_included = 100 * round(float(nparticles_neither_negative) / nparticles, 1)

    #        weights = 100 * numpy.ones_like(data_now) / float(nparticles_neither_negative)
    #        if len(bin_label) > 0:
    #            nparticles = n[i] # keep track of % over total N particles, not just of the bin itself. makes more sense I think.
    #        if len(bin_label) > 0:
    #            weights = 100 * numpy.ones_like(data_now) / n[i]
    #        else:
        weights = 100 * numpy.ones_like(data_now) / float(nparticles)

        j = position_subplot[spot_type][label]

        if not compare:
            ax[j] = fig.add_subplot(gs[j:j+1, :8])
            if quantity == 'D_app':
                ax[j].set_xscale('log')

            if 'bLR' in strain:
                h[i] = ax[j].hist(data_now, bins = bins_now, weights = weights, color = colors[strain][starvation_time + condition], hatch=htc[starvation_time], alpha = 1)
                ax[j].step(h[i][1], [0] + list(h[i][0]), color = colors[strain][starvation_time + condition])
            else:
                h[i] = ax[j].hist(data_now, bins = bins_now, weights = weights, color = 'w', edgecolor = colors[strain][starvation_time + condition], alpha = 1, lw=2)

            print(strain)
            if starvation_time == '0h' and 'LR' not in strain:
                ax[j].hist(data_now, bins = bins_now, weights = weights, histtype='step', color = colors[strain][starvation_time + condition], alpha = 1, lw = 2)

    #            ax[j].hist(data_now, bins = bins_now, weights = weights, facecolor = None, edgecolor = colors[strain][starvation_time + condition])
            if j < len(keys_of_interest) - 1:
                ax[j].set_xticks([])
            if quantity == 'average_starting_sigma':
                ax[j].axvline((size_limit / px_to_micron) / numpy.sqrt(2), color = 'k')
        elif compare:
            fig = pylab.gcf()
            ax = fig.axes
            print(len(ax))
            if j < len(keys_of_interest) - 1:
                ax[j].set_xticks([])

            if quantity == 'D_app':
                ax[j].set_xscale('log')

    #            h[i] = ax[j].hist(data_now, bins = bins_now, weights = weights, histtype='step', color = colors[strain][starvation_time + condition], alpha = 1, lw = 2)
            if ((len(msd_mode) > 0) and (starvation_time == '0h') and (strain == 'bLR31')):
                h[i] = numpy.zeros((len(data_now), 2))
            elif msd_mode == 'high':
                h[i] = ax[j].hist(data_now, bins = bins_now, weights = weights, #histtype='step',
                                  color = 'w', alpha = 0.3, lw = 2)
                h[i] = ax[j].hist(data_now, bins = bins_now, weights = weights, histtype='step',
                                    color = 'k', alpha = 1, lw = 2)
                if j < len(keys_of_interest) - 1:
                    ax[j].set_xticks([])

            elif msd_mode == 'low':
                h[i] = ax[j].hist(data_now, bins = bins_now, weights = weights, histtype='step',
                                  color = 'k', alpha = 1, lw = 2)
    #                h[i] = ax[j].hist(data_now, bins = bins_now, weights = weights, #histtype='step',
    #                      color = 'k', alpha = 0.5, lw = 2)
                if j < len(keys_of_interest) - 1:
                    ax[j].set_xticks([])



    #       pylab.plot(midpoints_now[i], h[i][0], 'k.')

        if show_median:
            ax[j].axvline(numpy.median(data_now), color = 'k', linestyle = '-')

    #        if show_particles_with_alpha_neg:
    #            neg_particles = r_now[r_now.alpha < 0].index
    #            data_alpha_neg = r_noa.loc[neg_particles, quantity].to_numpy(dtype = numpy.float64)
    #            weights_neg = 100 * numpy.ones_like(data_alpha_neg) / float(nparticles_neither_negative)
    #            ax[j].hist(data_alpha_neg, bins = bins_now, weights = weights_neg, color = 'k')

        print(ytext_N)
        if not compare:
            ax[j].text(xtext_N, ytext_N, 'N = ' + str(nparticles_neither_negative), color = color_text_now)
        ax[j].text(xtext_s, ytext_s, label2, color = colors[strain][starvation_time + condition])

        ### for D_app ###
        if quantity == 'D_app':
            if len(msd_mode) == 0:
                ax[j].text(value_x_location, value_y_location, t + format(data_now_median, '.1e') + '\n+/-' + format(data_now_sem, '.1e') + ' ' + tend, color = color_text_now, fontsize='small')
            ax[j].set_yticks([0, 10, 20]) # 10 when lowN only
            if j < len(keys_of_interest) - 1:
                ax[j].set_xticks([])
            else:
                pylab.xlabel(r'$D_{app}$' + ' ' + r'$(\mu m ^{2} / s^{\alpha})$')

        ## for α ###
        elif quantity == 'alpha':
            if len(msd_mode) == 0:
                ax[j].text(value_x_location, value_y_location, t + format(data_now_median, '.2f') + '+/-' + format(data_now_sem, '.3f'), color = color_text_now, fontsize='small') # median and standard deviation
            pylab.xlabel(r'$\alpha$')
            pylab.yticks([0, 10, 20])
            if j < len(keys_of_interest) - 1:
                ax[j].set_xticks([])


        ## for starting_snr ###
        elif quantity == 'starting_snr':
            ax[j].text(value_x_location, value_y_location, t + format(data_now_median, '.1f') + '\n+/-' + format(data_now_sem, '.2f'), color = color_text_now) # median and standard deviation
            if j < len(keys_of_interest) - 1:
                ax[j].set_xticks([])

        ## for average_starting_magnitude ###
        elif quantity == 'average_starting_magnitude':
            ax[j].text(value_x_location, value_y_location, t + format(data_now_median, '.1f') + '\n+/-' + format(data_now_sem, '.2f'), color = color_text_now) # median and standard deviation

        ## for average_starting_sigma ###
        elif quantity == 'average_starting_sigma':
            ax[j].text(value_x_location, value_y_location, t + format(data_now_median, '.1f') + '\n+/-' + format(data_now_sem, '.2f'), color = color_text_now) # median and standard deviation

        ## for particle_size ###
        elif quantity == 'particle_size':
            ax[j].text(value_x_location, value_y_location, t + format(data_now_median, '.1f') + '\n+/-' + format(data_now_sem, '.2f'), color = color_text_now) # median and standard deviation
            if j == len(keys_of_interest):
                pylab.xlabel('particle size (arb)')
        ## for previous_step_size ###
        elif quantity == 'previous_step_size':
            ax[j].text(value_x_location, value_y_location, t + format(data_now_median, '.1f') + '\n+/-' + format(data_now_sem, '.2f'), color = color_text_now) # median and standard deviation
            pylab.yticks([0, 10])
            if j < len(keys_of_interest) - 1:
                ax[j].set_xticks([])
            elif j == len(keys_of_interest):
                pylab.xlabel('step size (nm)')

        elif quantity == 'msd_xs':
            ax[j].set_xscale('log')
            ax[j].set_yticks([0, 10, 20]) # 10 when lowN only
            if j < len(keys_of_interest) - 1:
                ax[j].set_xticks([])
            else:
                pylab.xlabel(f"MSD at 30 s")

        ### for alpha origins size  ###
    #        ax[j].text(0.75, 10, 'N = ' + str(nparticles_neither_negative))
    #        ax[j].text(1.25, 11, label2, color = colors[strain][starvation_time + condition])
    #        ax[j].text(1.25, 1, 'α = ' + format(data_now_median, '.2f') + '\n+/-' + format(data_now_sem, '.3f'))
    #        pylab.xlim(0, xmax)
    #        ax[j].set_ylim(0, 15)

        pylab.xlim(xmin, xmax)
        ax[j].set_ylim(ymin, ymax)
        fig.text(0.04, 0.5, '% of particles', va='center', rotation='vertical')
        if j < len(keys_of_interest) - 1:
            ax[j].set_xticks([])

        print(str(percent_included) + '% of all tracked particles')

        os.makedirs(basic_directory + central_directory + 'plots/' + constraint_str + '/' + quantity + 's_per_cat_' + time_label + '/', exist_ok=True)
        if per_day:
            pylab.savefig(basic_directory + central_directory + 'plots/' + constraint_str + '/' + quantity + 's_per_cat_' + time_label + '/' + quantity + '_histograms_' + str(time_between_frames) + 'ms' + '_' + day + '_' + string_now + '.png')
        else:
            pylab.savefig(basic_directory + central_directory + 'plots/' + constraint_str + '/' + quantity + 's_per_cat_' + time_label + '/' + quantity + '_histograms_' + str(time_between_frames) + 'ms' + '_' + string_now + '.png')

        os.makedirs(basic_directory_paper_figs + '_' + spot_type + '/' + constraint_str + '/' + quantity + '_histograms_per_cat/', exist_ok=True)
        pylab.savefig(basic_directory_paper_figs + '_' + spot_type + '/' + constraint_str + '/' + quantity + '_histograms_per_cat/' + string_now + '_histograms_' + quantity + 's_per_cat' + pd + time_label + str(time_between_frames) + 'ms.svg')
        pylab.savefig(basic_directory_paper_figs + '_' + spot_type + '/' + constraint_str + '/' + quantity + '_histograms_per_cat/' + string_now + '_histograms_' + quantity + 's_per_cat' + pd +  time_label + str(time_between_frames) + 'ms.png')

        os.makedirs(basic_directory_paper_data + '_' + spot_type + '/' + constraint_str + '/' + quantity + '_histograms_per_cat/', exist_ok=True)
        numpy.save(basic_directory_paper_data + '_' + spot_type + '/' + constraint_str + '/' + quantity + '_histograms_per_cat/' + string_now + '_input_to_histograms_' + quantity + 's_per_cat' + pd + time_label + str(time_between_frames) + 'ms.npy', r)

        f[i] = numpy.zeros([len(midpoints_now[i]),2])
        f[i][:,0] = midpoints_now[i]
        if len(msd_mode) > 0:
            if starvation_time == '0h':
                if strain != 'bLR31':
                    f[i][:,1] = numpy.zeros(len(midpoints_now[i]))
        else:
            f[i][:,1] = h[i][0]

    if plot_quantity_vs_position:
        fg = spot_type + ', ' + quantity + ', time between frames: ' + str(time_between_frames) + '\nsummary'
        if below_size_limit_only:
            fg = fg + ', below diff. limit'
        if len(bin_label) > 0:
            fg = fg + 'bin ' + bin_label

        if quantity == 'D_app':
            fig = pylab.figure(fg, figsize = (6, 6))
            ax = fig.gca()
            ax.set_yscale('log')
            ax.set_ylabel('D ' + r'$(\mu m^{2}/s^{\alpha})$')
            if 'origins' in spot_type:
                ymin = 1e-4 #3e-5
                ymax = 4e-4 #5e-3
            elif spot_type == 'muNS':
                ymin = 1e-4 #5e-5
                ymax = 2e-2 #5e-2
        elif quantity == 'alpha':
            fig = pylab.figure(fg, figsize = (6, 6))
            pylab.ylabel('α', rotation = 0, fontsize = 18, labelpad = 10)
            if 'origins' in spot_type:
                ymin = 0.3 #0
                ymax = 0.55
            elif spot_type == 'muNS':
                ymin = 0 #0.3 when lowN only #0
                ymax = 0.5 #0.7 when lowN only
                pylab.yticks(numpy.arange(ymin, ymax + 0.1, 0.1))
            pylab.ylim(ymin, ymax)

        elif quantity == 'previous_step_size':
            pylab.ylabel('step size (px)')
            ymin = 0 #0
            ymax = 1

        data_median_of_medians = {}
        if 'origins' in spot_type:
            data_median_of_medians['bLR1_0h'] = []
            data_median_of_medians['bLR2_0h'] = []
            data_median_of_medians['bLR1_6h'] = []
            data_median_of_medians['bLR2_6h'] = []
        elif spot_type == 'muNS':
            data_median_of_medians['bLR31_0h'] = []
            data_median_of_medians['bLR32_0h'] = []
            data_median_of_medians['bLR31_6hlowN'] = []
            data_median_of_medians['bLR32_6hlowN'] = []
    #            data_median_of_medians['bLR31_6hlowC'] = []
    #            data_median_of_medians['bLR32_6hlowC'] = []

        for i in keys_of_interest[:]: #list(results.keys())
            strain = load_all_results.read('strain', i, spot_type)
            starvation_time = load_all_results.read('starvation_time', i, spot_type) + 'h'
            condition = load_all_results.read('condition', i, spot_type)
            print(i)
            data_median_of_medians[strain + '_' +  starvation_time + condition].append(data_medians[i])
    #              pylab.errorbar(positions[strain][starvation_time + condition], data_medians[i], xerr = 0, yerr = data_sems[i], fmt = 'o', markersize = 15, color = colors[strain][starvation_time + condition])
            pylab.plot(positions[strain][starvation_time + condition], data_medians[i],  markershapes[starvation_time + condition], markersize = ms, markeredgecolor = 'k', color = colors[strain][starvation_time + condition], alpha = 0.7)

        for i in data_median_of_medians.keys():
            data_median_of_medians[i] = numpy.array(data_median_of_medians[i])
            strain = load_all_results.read('strain', i, spot_type)
            starvation_time = load_all_results.read('starvation_time', i, spot_type) + 'h'
            condition = load_all_results.read('condition', i, spot_type)

            pylab.hlines(data_median_of_medians[i].mean(), positions[strain][starvation_time + condition] - 0.04, positions[strain][starvation_time + condition] + 0.04, color = colors[strain][starvation_time + condition], linestyles='solid', label='', linewidth = 4)

    #        pylab.xticks([1, 1.2, 1.4, 1.6, 1.8, 2.0], ['WT\n0h', '\n6h lowN', '\n6h lowC', 'ΔpolyP\n0h', '\n6h lowN', '\n6h lowC'])

    #format(data_now_median, '.1f')

        ft = spot_type + ', ' + quantity + ', time between frames: ' + str(time_between_frames) + '\nsummary'
        if below_size_limit_only:
            ft = ft + ' below diff. limit'
        if len(bin_label) > 0:
            ft = ft + ', bin ' + bin_label
    #        pylab.yticks(ticks = [ymin, ymax], labels = [r'$10^{-4}$', r'$10^{-3}$'])
    #        ax.set_yticks([])
    #        ax.set_yticks(numpy.arange(ymin, ymax + 1e-4, 1e-4))
    #        ax.set_yticklabels([r'$10^{-4}$', '', '', '', '', '', '', '', '', r'$10^{-3}$'])
        pylab.title(ft)
        pylab.xticks([1, 1.2, 1.4, 1.6], ['WT\n0h', '\n6h lowC', 'ΔpolyP\n0h', '\n6h lowC'])

        pylab.savefig(basic_directory + central_directory + 'plots/' + constraint_str + '/' + quantity + 's_per_cat_' + time_label + '/' + string_now + '_' + quantity + str(time_between_frames) + 'ms' + '_summary_median' + pd + '.png')


        pylab.savefig(basic_directory_paper_figs + '_' + spot_type + '/' + constraint_str + '/' + quantity + 's_per_cat_summary_' + time_label + '/' + string_now + '_' + quantity + '_' + str(time_between_frames) + 'ms' + '_summary_median' + pd + '.svg')
        pylab.savefig(basic_directory_paper_figs + '_' + spot_type + '/' + constraint_str + '/' + quantity + 's_per_cat_summary_' + time_label + '/' + string_now + '_' + quantity + '_' + str(time_between_frames) + 'ms' + '_summary_median' + pd + '.png')
        numpy.save(basic_directory_paper_data + '_' + spot_type + '/' + constraint_str + '/' + quantity + 's_per_cat_summary_' + time_label + '/' + string_now + '_' + quantity + '_' +  str(time_between_frames) + 'ms' + '_summary_median' + pd + '.npy', data_medians)
        numpy.save(basic_directory_paper_data + '_' + spot_type + '/' + constraint_str + '/' + quantity + 's_per_cat_summary_' + time_label + '/' + string_now  + '_results_' +  str(time_between_frames) + 'ms' + pd + '.npy', results)
    pylab.savefig(basic_directory_paper_figs + '_' + spot_type + '/' + quantity + '_histograms_all.svg')
    pylab.savefig(basic_directory_paper_figs + '_' + spot_type + '/' + quantity +'_histograms_all.png')
    print(basic_directory_paper_figs + '_' + spot_type + '/' + constraint_str + '/' + quantity + 's_per_cat_summary_' )
    return results, f, data_medians, r_now

def histogram_per_size_bin(spot_type, binned_results, quantity, size_bins = [1, 3, 5], compare = False, msd_mode = '', file_with_mode_occupants = '/Volumes/GlennGouldMac/PolyP/data/_muNS/diameter7_minmass110_search_range4_memory20_stub_length20/results/20210726_mode_occupants_from_msd_at_2p1s.npy'):
    '''
    '''
    now = datetime.datetime.now()
    string_now = now.strftime("%Y%m%d")

    if len(msd_mode) > 0:
        mode_occupants = numpy.load(file_with_mode_occupants, allow_pickle=True).item()

    mode_occupants2 = {}
    mode_occupants2['1'] = {}
    mode_occupants2['2'] = {}

    populations = pandas.DataFrame(columns = ['condition', 'size_bin', 'mode', 'fraction'])

    if quantity == 'alpha':
        xmin = 0
        xmax = 1.25
        xtext_bin = 1.27
        ytext_bin = 16
        ymin = 0
        ymax = 21
        xtext_N = xtext_bin
        ytext_N = 7
        bins_now = numpy.arange(xmin, xmax, 0.05)
        yl = '% particles'
    elif quantity == 'D_app':
        xmin = 1e-5 #TODO Change this on condition
        xmax = 1e-1
        xtext_bin = 1.1e-1
        ytext_bin = 16
        ymin = 0
        ymax = 27#24
        xtext_N = xtext_bin #1.1e-4
        ytext_N = 7#14
        bins_now = numpy.logspace(-6, 0, num = 50)
        units = r'$ \mu m^2 / s^\alpha$'
        yl = '% particles'
    elif quantity == 'previous_step_size':
        xmin = 0
        xmax = 4
        xtext_bin = 165
        ytext_bin = 18
        ymin = 0
        ymax = 31 #30
        xtext_N = xtext_bin #1.1e-4
        ytext_N = 7#14
        bins_now = numpy.arange(0, xmax, 0.1) * px_to_micron * 1000
        units = 'nm'
        yl = '% steps'

    for i in list(binned_results['0'].keys())[:]:
        print(i)
        strain = load_all_results.read('strain', i, spot_type)
        starvation_time = load_all_results.read('starvation_time', i, spot_type) + 'h'
        condition = load_all_results.read('condition', i, spot_type)
 #        time_between_frames = load_all_results.read('time_between_frames', i, spot_type)

        fig = pylab.figure(i + ' ' + quantity, figsize = (7.5, 1.5 * len(size_bins)))
        gs = GridSpec(len(size_bins), 10, figure=fig)

        if compare:
            fig = pylab.gcf()
            ax = fig.axes

        else:
            fig.suptitle(i + ': ' + quantity)
            ax = {}

        for j, k in enumerate(size_bins):
            print(j)
            r_now = binned_results[str(k)][i]
            nparticles = len(r_now)
            if not compare:
                ax[j] = fig.add_subplot(gs[j:j+1, :8])
            if quantity == 'D_app':
                ax[j].set_xscale('log')
            if len(msd_mode) > 0:
                if quantity == 'previous_step_size':
                    if msd_mode == 'low':
                        mode_occupants2['1'][i] = [x for x in mode_occupants['1'][i] if x in list(set(r_now.particle))]
                        r_now = r_now.loc[r_now['particle'].isin(mode_occupants2['1'][i])]
                    elif msd_mode == 'high':
                        mode_occupants2['2'][i] = [x for x in mode_occupants['2'][i] if x in list(set(r_now.particle))]
                        r_now = r_now.loc[r_now['particle'].isin(mode_occupants2['2'][i])]
                else:
                    if msd_mode == 'low':
                        mode_occupants2['1'][i] = [x for x in mode_occupants['1'][i] if x in list(r_now.index)]
                        r_now = r_now.loc[mode_occupants2['1'][i], :]
                    elif msd_mode == 'high':
                        mode_occupants2['2'][i] = [x for x in mode_occupants['2'][i] if x in list(r_now.index)]
                        r_now = r_now.loc[mode_occupants2['2'][i], :]

            nparticles2 = len(r_now)
            d = [[i, k, msd_mode, 100 * nparticles2 / nparticles]]
            little_answer =  pandas.DataFrame(data = d, columns = ['condition', 'size_bin', 'mode', 'fraction'])
 #            print(little_answer)
            populations = populations.append(little_answer)

            data = r_now.loc[:, quantity].to_numpy(dtype = numpy.float64)
            if quantity == 'previous_step_size':
                data = data * px_to_micron * 1000
            weights = 100 * numpy.ones_like(data) / nparticles
            if ((msd_mode == 'low') and (i != 'bLR31_0h_0210ms')):
                ax[j].hist(data, bins = bins_now, weights = weights, histtype = 'step', color = 'k', lw = 2, alpha = 1)
            elif ((msd_mode == 'high') and (i != 'bLR31_0h_0210ms')):
                ax[j].hist(data, bins = bins_now, weights = weights, color = 'white', alpha = 0.3)
                ax[j].hist(data, bins = bins_now, weights = weights, histtype = 'step', color = '#000000', alpha = 1, lw = 2)
            elif len(msd_mode) == 0:
                ax[j].hist(data, bins = bins_now, weights = weights, color = colors[strain][starvation_time + condition], hatch = htc[starvation_time])
                ax[j].hist(data, bins = bins_now, weights = weights, histtype = 'step', color = colors[strain][starvation_time + condition], lw = 2)


            ax[j].set_xlim(xmin * px_to_micron * 1000, xmax * px_to_micron * 1000)
            if quantity == 'previous_step_size':
                ax[j].set_xlim(xmin * px_to_micron * 1000, 1.5 * px_to_micron * 1000)
                ax[j].set_yticks(ticks = [0, 15])
            else:
                ax[j].set_yticks([0, 10])
                ax[j].set_xlim(xmin, xmax)
            ax[j].set_ylim(ymin, ymax)
            if j < len(size_bins) - 1:
                ax[j].set_xticks([])
            if j == len(size_bins) - 1:
                if quantity == 'D_app':
                    ax[j].set_xlabel(quantity + r' $(\mu m^2 / s^\alpha)$')
                elif quantity == 'alpha':
                    ax[j].set_xlabel(quantity)
                elif quantity == 'previous_step_size':
                    ax[j].set_xlabel('step size (nm)')
            if len(msd_mode) == 0:
                ax[j].text(xtext_N, ytext_N, 'N = ' + str(len(data)))
                ax[j].text(xtext_bin, ytext_bin, 'bin ' + str(k))


        fig.subplots_adjust(top=0.9)
        fig.subplots_adjust(hspace=0.08) #0.5, 0.08
        fig.text(0.0, 0.5, yl, va='center', rotation='vertical')
        os.makedirs(basic_directory_paper_figs + '_' + spot_type + '/' + quantity + '_per_size_bin/', exist_ok=True)
        pylab.savefig(basic_directory_paper_figs + '_' + spot_type + '/' + quantity + '_per_size_bin/' + string_now + '_' + i.strip('/') +'_' + str(len(size_bins)) + '_bins.svg')
        pylab.savefig(basic_directory_paper_figs + '_' + spot_type + '/' + quantity + '_per_size_bin/' + string_now + '_' + i.strip('/') +'_' + str(len(size_bins)) + '_bins.png')

 #    populations.to_pickle(basic_directory_paper_data + '_' + spot_type + '/'  + string_now + quantity + '_populations_per_size_bin/' + str(len(size_bins)) + '_bins.pkl')

    return binned_results, populations

def stacked_histograms_per_day(spot_type, quantity, results_per_day, avoid = ['1s'], time_label = 'tstart0p21_tend0p84sec', show_particles_with_alpha_neg = False, bin_label = '', ignore_timelag = False, below_size_limit_only = True, size_limit = 0.200):
    ### !!! confirm the days have consistent colors - checked, days appear in the same order for all conditions.

    now = datetime.datetime.now()
    string_now = now.strftime("%Y%m%d")

    if quantity == 'alpha':
        if spot_type == 'muNS':
            xmax = 1.25
            ymax = 15 #27#22 #15 #25 #15 when only lowN,  22 when only lowC
            xtext_N = 0.8
            ymin = 0
            if show_particles_with_alpha_neg:
                xmin = -0.5
            else:
                xmin = -0.05
            bins_now = numpy.arange(xmin, xmax, 0.05)
            ytext_N = 12#8#14
            t = 'α = '
            value_x_location = bins_now[::-1][0] + 0.06
            for k, l in enumerate(bins_now):
                if k < len(bins_now) - 1:
                    c = numpy.linspace(bins_now[k], bins_now[k+1], num=3)
            label2_x = bins_now[::-1][0] + 0.06  # location of strain label (or xtext_s?)
            label2_y = 10
            value_y_location = 5
        elif 'origins' in spot_type:
            xmax = 1.25
            ymax = 15 #27#22 #15 #25 #15 when only lowN,  22 when only lowC
            xtext_N = 0.8
            ymin = 0
            if show_particles_with_alpha_neg:
                xmin = -0.5
            else:
                xmin = -0.05
            bins_now = numpy.arange(xmin, xmax, 0.05)
            ytext_N = 12#8#14
            t = 'α = '
            value_x_location = bins_now[::-1][0] + 0.06
            for k, l in enumerate(bins_now):
                if k < len(bins_now) - 1:
                    c = numpy.linspace(bins_now[k], bins_now[k+1], num=3)
            label2_x = bins_now[::-1][0] + 0.06  # location of strain label (or xtext_s?)
            label2_y = 10
            value_y_location = 5

        xlab = 'α'
    elif quantity == 'D_app':
        if spot_type == 'muNS':
            ymax = 16#18 #20 #30 # 25 when lowN only
            xmin = 1e-4#3e-5 # 5e-5 when lowN only
            xmax = 1e-1#5e-1 # when lowN only, 4e-2  when lowC?
            xtext_N = 1.01e-4 #3e-2 #when lowN only; 8e-3
            xtext_s = xmax + 2e-3
            ytext_s = ymax - 4
            value_y_location = ytext_s - 10 #15
        elif spot_type == 'origins':
            ymax = 20
            xmin = 1e-5#5e-6
            xmax = 1e-2#5e-2
            xtext_N = 2e-3# 8e-3 # 1e-2
            xtext_s = xmax + 1e-3 #xmax + 5e-3
            ytext_s = ymax - 4
            value_y_location = ytext_s - 10 #15
        ymin = 0
        bins_now = numpy.logspace(-6, 0, num = 50)
        t = r'$D_{app} = $'
        tend = r'$\mu m ^{2} / s^{\alpha}$'
        ytext_N = 12
        value_x_location = xtext_s
        xlab = r'$ D_{app} (\mu m^{2} / s^{\alpha})$'

    groups = {}
    groups['muNS'] = {}
    groups['muNS']['bLR31_0h_0210ms'] = []
    groups['muNS']['bLR31_6h_0210ms_lowN'] = []
    groups['muNS']['bLR32_0h_0210ms'] = []
    groups['muNS']['bLR32_6h_0210ms_lowN'] = []

    groups['origins'] = {}
    groups['origins']['bLR1_0h_5000ms'] = []
    groups['origins']['bLR1_6h_5000ms_lowN'] = []
    groups['origins']['bLR2_0h_5000ms'] = []
    groups['origins']['bLR2_6h_5000ms_lowN'] = []

    colorlist = {}
    colorlist['muNS'] = ['#000000', '#444444', '#888888', '#CCCCCC', '#FFFFFF']
    colorlist['origins'] = ['#888888', '#CCCCCC', '#FFFFFF']

    data = {}
    data['all'] = {}
    nparticles = {}
    weights = {}

    for i in groups[spot_type].keys():
        data['all'][i] = []
        nparticles[i] = {}
        nparticles[i]['all'] = 0
        weights[i] = []

    for i in results_per_day.keys():
        strain = load_all_results.read('strain', i, spot_type)
        starvation_time = load_all_results.read('starvation_time', i, spot_type) + 'h'
        time_between_frames = load_all_results.read('time_between_frames', i, spot_type) + 'ms'
        condition = load_all_results.read('condition', i, spot_type)
        day = i.split('_')[::-1][0]
        j = strain + '_' + starvation_time + '_' + time_between_frames
        if starvation_time != '0h':
            j = j + '_' + condition
        groups[spot_type][j].append(i)
        data_piece = results_per_day[i].loc[:,quantity].to_numpy(dtype = numpy.float64)
        data_piece = data_piece[numpy.greater(data_piece, numpy.zeros_like(data))]

        data['all'][j].append(data_piece)
        nparticles[j][i] = len(data_piece)
        nparticles[j]['all'] = nparticles[j]['all'] + nparticles[j][i]

    for i in nparticles.keys():
        for j in nparticles[i].keys():
            if j != 'all':
                weights[i].append(100 * numpy.ones(nparticles[i][j]) / nparticles[i]['all'])

        strain = load_all_results.read('strain', i, spot_type)
        starvation_time = load_all_results.read('starvation_time', i, spot_type) + 'h'
        time_between_frames = load_all_results.read('time_between_frames', i, spot_type) + 'ms'
        condition = load_all_results.read('condition', i, spot_type)

        figtitle = quantity + ', day to day variability'

        if quantity == 'D_app':
            fig = pylab.figure(figtitle, figsize = (11, 7))  #(11, 5) # D_app figure is larger
        else:
            fig = pylab.figure(figtitle, figsize = (7.5, 7))  #(11, 5)

        fig.suptitle(figtitle)
        gs = GridSpec(4, 10, figure=fig)
        ax = {}
        fig.subplots_adjust(top=0.9)
        k = strain + '_' + starvation_time
        if starvation_time != '0h':
            k = k + '_' + condition
        j = position_subplot[spot_type][k]
        color_text_now = colors[strain][starvation_time + condition]

        if spot_type == 'muNS':
            pylab.subplots_adjust(hspace=0.08) #0.5, 0.08
        elif spot_type == 'origins':
            pylab.subplots_adjust(hspace=0.08)

        ax[j] = fig.add_subplot(gs[j:j+1, :8])
        if quantity == 'D_app':
            ax[j].set_xscale('log')

        ax[j].hist(data['all'][i], bins = bins_now, histtype = 'bar', stacked = True, color = colorlist[spot_type], edgecolor = colors[strain][starvation_time + condition], linewidth = 1, weights = weights[i])
        ax[j].hist(data['all'][i], bins = bins_now, histtype = 'step', stacked = True, fill = True, color = len(colorlist[spot_type]) * [colors[strain][starvation_time + condition]], weights = weights[i], alpha = 0.1)
        ax[j].set_xlim(xmin, xmax)
        ax[j].set_ylim(ymin, ymax)
        ax[j].text(xtext_N, ytext_N, 'N = ' + str(nparticles[i]['all']), color = color_text_now)

        if j < 3:
            ax[j].set_xticks([])
        fig.text(0.0, 0.5, '% of particles', va='center', rotation='vertical')
        pylab.xlabel(xlab)

    pylab.savefig(basic_directory_paper_figs + '_' + spot_type + '/' + string_now + '_' + quantity + '_day_to_day_stacked_histograms.svg')
    numpy.save(basic_directory_paper_figs + '_' + spot_type + '/' + string_now + '_imsd_results_pooled_per_day.npy', results_per_day)

    return data, groups

def fit_histograms_statnorm(spot_type, quantity, results, avoid = ['1s'], plot = True, log_option = False, time_label = 'tstart0p21sec_tend0p84sec', plot_fit_results = True, show_mean_fit = True, per_day = False, ignore_timelag = False):
    '''
    '''
    central_directory = load_all_results.select_specifications(spot_type)['central_directory']

    pylab.close('all')

    keys_of_interest = list(results.keys())[:]
    results2 = results.copy()
    for i in results.keys():
        if i not in keys_of_interest:
            del results2[i]

    h = histogram_all(spot_type, quantity, results2, avoid = avoid, time_label = time_label, show_median = False, below_size_limit_only = False, per_day = per_day)[1]


    now = datetime.datetime.now()
    string_now = now.strftime("%Y%m%d")

    fits = {}
    fits['statnorm'] = {}

    results = pandas.DataFrame(columns = ['condition', 'function', 'mean', 'sigma', 'amplitude', 'mean_fit_unc', 'sigma_fit_unc', 'amplitude_fit_unc'])

    for i in keys_of_interest:
 #        if quantity == 'alpha':
 #            h[i] = h[i][1:]
        time_between_frames = load_all_results.read('time_between_frames', i, spot_type)

        if log_option:
            fits['statnorm'][i] = scipy.optimize.curve_fit(statnorm_log, h[i][:,0], h[i][:,1])#, p0 = [1.9e-4, 1e-4, 8])
        else:
            fits['statnorm'][i] = scipy.optimize.curve_fit(statnorm, h[i][:,0], h[i][:,1])

        d = [[i, 'statnorm', fits['statnorm'][i][0][0], fits['statnorm'][i][0][1], fits['statnorm'][i][0][2], numpy.sqrt(numpy.diag(fits['statnorm'][i][1]))[0], numpy.sqrt(numpy.diag(fits['statnorm'][i][1]))[1], numpy.sqrt(numpy.diag(fits['statnorm'][i][1]))[2]]]
        little_answer =  pandas.DataFrame(data = d, columns = ['condition', 'function', 'mean', 'sigma', 'amplitude', 'mean_fit_unc', 'sigma_fit_unc', 'amplitude_fit_unc'])
        results = results.append(little_answer)
        results.to_pickle(basic_directory + central_directory + 'results/fits_to_' + quantity + 's_per_cat_' + time_label + '/' + quantity + '_histograms_' + str(time_between_frames) + 'ms_normal_fit_' + string_now + '.pkl')
        results.to_csv(basic_directory + central_directory + 'results/fits_to_' + quantity + 's_per_cat_' + time_label + '/' + quantity + '_histograms_' + str(time_between_frames) + 'ms_normal_fit_' + string_now + '.csv')

    if plot:

        for i in keys_of_interest:
            print(i)
            time_between_frames = load_all_results.read('time_between_frames', i, spot_type)
            strain = load_all_results.read('strain', i, spot_type)
            starvation_time = load_all_results.read('starvation_time', i, spot_type) + 'h'
            condition = load_all_results.read('condition', i, spot_type)
            j = i.split('_')
            k = i.split('_')[0] + '_' + i.split('_')[1]
            if len(j) > 3:
                k = k + '_' + i.split('_')[3]
            print(k)
            k2 = strain + '_' + starvation_time
            if starvation_time != '0h':
                k2 = k2 + '_' + condition
            ft = spot_type + ', ' + quantity   # figure name
            stitle = spot_type + ', ' + quantity  # figure title
            if per_day:
                day = i.split('_')[::-1][0]
                ft = ft + ', ' + day
                stitle = stitle + ', ' + day
            if not ignore_timelag:
                time_between_frames = load_all_results.read('time_between_frames', i, spot_type)
                ft = ft + ', time between frames: ' + str(time_between_frames)
                stitle = stitle + ', time between frames: ' + str(time_between_frames) + 'ms'

            fig = pylab.figure(ft)
            axes = fig.get_axes()
            axes[position_subplot[spot_type][k2]].plot(h[i][:,0], h[i][:,1], 'k.')

            if log_option:
                axes[position_subplot[spot_type][k2]].plot(h[i][:,0], statnorm_log(h[i][:,0], fits['statnorm'][i][0][0], fits['statnorm'][i][0][1], fits['statnorm'][i][0][2]), 'k', linewidth = 2)
            else:
                axes[position_subplot[spot_type][k2]].plot(h[i][:,0], statnorm(h[i][:,0], fits['statnorm'][i][0][0], fits['statnorm'][i][0][1], fits['statnorm'][i][0][2]), 'k', linewidth = 2)

            if show_mean_fit:
                axes[position_subplot[spot_type][k2]].axvline(fits['statnorm'][i][0][0], color = 'k', linestyle = '--')

            ft = ft.replace(', ', '_').replace(': ', '').replace(' ', '').replace('timebetweenframes', '')

            pylab.savefig(basic_directory + central_directory + 'plots/fits_to_' + quantity + 's_per_cat_' + time_label + '/' + string_now + '_' + quantity + '_histograms_' + ft + 'ms_normal_fit.png')
 #
    if plot_fit_results:
        ft = ft.replace(', ', '_').replace(': ', '').replace(' ', '').replace('timebetweenframes', '')

        pylab.figure('fit results, mean ' + spot_type + ', ' + quantity + ', time between frames: ' + str(time_between_frames))
        pylab.title('fit results, mean\n' + spot_type + ', ' + quantity + ', time between frames: ' + str(time_between_frames))
        for i in fits['statnorm'].keys():
            strain = load_all_results.read('strain', i, spot_type)
            starvation_time = load_all_results.read('starvation_time', i, spot_type) + 'h'
            condition = load_all_results.read('condition', i, spot_type)
            pylab.errorbar(positions[strain][starvation_time + condition], fits['statnorm'][i][0][0], xerr = 0, yerr = numpy.sqrt(numpy.diag(fits['statnorm'][i][1]))[0], color = colors[strain][starvation_time + condition], fmt = markershapes[starvation_time + condition], markersize = ms, alpha = 0.7, markeredgecolor = 'k')
        if spot_type == 'muNS':
            pylab.xticks([1, 1.2, 1.4, 1.6], ['WT\n0h', '\n6h lowN', 'ΔpolyP\n0h', '\n6h lowN'])
        elif spot_type == 'origins':
            pylab.xticks([1, 1.2, 1.4, 1.6], ['WT\n0h', '\n6h lowN', 'ΔpolyP\n0h', '\n6h lowN'])
        pylab.ylabel('mean')
 #        pylab.ylim(0, 1)
        pylab.savefig(basic_directory + central_directory + 'plots/fits_to_' + quantity + 's_per_cat_' + time_label + '/' + string_now + '_fit_results_mean_' + quantity + '_histograms_' + ft + 'ms_normal_fit.png')

        pylab.figure('fit results, width ' + spot_type + ', ' + quantity + ', time between frames: ' + str(time_between_frames))
        pylab.title('fit results, width\n' + spot_type + ', ' + quantity + ', time between frames: ' + str(time_between_frames))
        for i in fits['statnorm'].keys():
            strain = load_all_results.read('strain', i, spot_type)
            starvation_time = load_all_results.read('starvation_time', i, spot_type) + 'h'
            condition = load_all_results.read('condition', i, spot_type)
            pylab.errorbar(positions[strain][starvation_time + condition], numpy.abs(fits['statnorm'][i][0][1]), xerr = 0, yerr = numpy.sqrt(numpy.diag(fits['statnorm'][i][1]))[1], color = colors[strain][starvation_time + condition], fmt = markershapes[starvation_time + condition], markersize = ms, alpha = 0.7, markeredgecolor = 'k')
        if spot_type == 'muNS':
            pylab.xticks([1, 1.2, 1.4, 1.6], ['WT\n0h', '\n6h lowN', 'ΔpolyP\n0h', '\n6h lowN'])

 #            pylab.xticks([1, 1.2, 1.4, 1.6, 1.8, 2.0], ['WT\n0h', '\n6h lowN', '\n6h lowC', 'ΔpolyP\n0h', '\n6h lowN', '\n6h lowC'])
        elif spot_type == 'origins':
            pylab.xticks([1, 1.2, 1.4, 1.6], ['WT\n0h', '\n6h lowN', 'ΔpolyP\n0h', '\n6h lowN'])
        pylab.ylabel('abs(σ)')
 #        pylab.ylim(0, 1)
        pylab.savefig(basic_directory + central_directory + 'plots/fits_to_' + quantity + 's_per_cat_' + time_label + '/' + string_now + '_fit_results_width_' + quantity + '_histograms_' + ft + 'ms_normal_fit.png')

        pylab.figure('fit results, amplitude ' + spot_type + ', ' + quantity + ', time between frames: ' + str(time_between_frames))
        pylab.title('fit results, amplitude\n' + spot_type + ', ' + quantity + ', time between frames: ' + str(time_between_frames))
        for i in fits['statnorm'].keys():
            strain = load_all_results.read('strain', i, spot_type)
            starvation_time = load_all_results.read('starvation_time', i, spot_type) + 'h'
            condition = load_all_results.read('condition', i, spot_type)
            pylab.errorbar(positions[strain][starvation_time + condition], fits['statnorm'][i][0][2], xerr = 0, yerr = numpy.sqrt(numpy.diag(fits['statnorm'][i][1]))[2], color = colors[strain][starvation_time + condition], fmt = markershapes[starvation_time + condition], markersize = ms, alpha = 0.7, markeredgecolor = 'k')
        if spot_type == 'muNS':
            pylab.xticks([1, 1.2, 1.4, 1.6], ['WT\n0h', '\n6h lowN', 'ΔpolyP\n0h', '\n6h lowN'])

 #            pylab.xticks([1, 1.2, 1.4, 1.6, 1.8, 2.0], ['WT\n0h', '\n6h lowN', '\n6h lowC', 'ΔpolyP\n0h', '\n6h lowN', '\n6h lowC'])
        elif spot_type == 'origins':
            pylab.xticks([1, 1.2, 1.4, 1.6], ['WT\n0h', '\n6h lowN', 'ΔpolyP\n0h', '\n6h lowN'])
        pylab.ylabel('amplitude')
 #        pylab.ylim(0, 50)

        pylab.savefig(basic_directory + central_directory + 'plots/fits_to_' + quantity + 's_per_cat_' + time_label + '/' + string_now + '_fit_results_amplitude_' + quantity + '_histograms_' + ft + 'ms_normal_fit.png')

    return fits['statnorm'], results, h

def fit_histograms_bistatnorm(spot_type, quantity, results, avoid = ['1s'], plot = True, log_option = False, time_label = 'tstart0p21sec_tend0p84sec', plot_fit_results = True, per_day = False, ignore_timelag = False, starved_only_bistatnorm = True):
    '''
    '''
    now = datetime.datetime.now()
    string_now = now.strftime("%Y%m%d")

    central_directory = load_all_results.select_specifications(spot_type)['central_directory']

    if per_day:
        pd = '_per_day_'
    else:
        pd = '_'

    keys_of_interest = list(results.keys())[:]
    results2 = results.copy()
    for i in results.keys():
        if i not in keys_of_interest:
            del results2[i]

    h = histogram_all(spot_type, quantity, results2, avoid = avoid, time_label = time_label, show_median = False, below_size_limit_only = False, per_day = per_day)[1]

    #USE SAME GUESS FOR ALL #
    now = datetime.datetime.now()
    string_now = now.strftime("%Y%m%d")

    fits = {}
    fits['statnorm'] = fit_histograms_statnorm(spot_type, quantity, results, avoid = avoid, plot = False, log_option = log_option, time_label = time_label, plot_fit_results = False, show_mean_fit = False)[0]

    fits['bistatnorm'] = {}
    y_curve_1 = {} # the curve that describes the first mode, after bimodal fitting
    y_curve_2 = {} # the curve that describes the second mode, after bimodal fitting
    intersection = {} # the points of intersection between the two curves

    results = pandas.DataFrame(columns = ['condition', 'function', 'mean_1', 'mean_2', 'sigma_1','sigma_2', 'amplitude_1', 'amplitude_2', 'mean_fit_unc_1',  'mean_fit_unc_2', 'sigma_fit_unc_1', 'sigma_fit_unc_2', 'amplitude_fit_unc_1',   'amplitude_fit_unc_2', 'percent_population_1', 'percent_population_2', 'mode_intersection_point'])

    initial_guess = {}
    initial_guess['bistatnorm'] = {}
    if quantity == 'D_app':
        global_guess = pandas.read_pickle('/Volumes/GlennGouldMac/PolyP/data/_muNS/diameter7_minmass110_search_range4_memory20_stub_length20/results/fits_to_D_apps_per_cat_tstart0p21sec_tend0p84sec/D_app_histograms_0210ms_binormal_fit_ordered_20210615.pkl')
    elif quantity == 'alpha':
        global_guess = pandas.read_pickle('/Volumes/GlennGouldMac/PolyP/data/_muNS/diameter7_minmass110_search_range4_memory20_stub_length20/results/fits_to_alphas_per_cat_tstart0p21sec_tend0p84sec/210622/alpha_histograms_0210ms_binormal_fit_ordered_20210622.pkl')
 #        global_guess = pandas.read_pickle('/Volumes/GlennGouldMac/PolyP/data/_muNS/diameter7_minmass110_search_range4_memory20_stub_length20/results/fits_to_alphas_per_cat_tstart0p21sec_tend0p84sec/210506/alpha_histograms_0210ms_binormal_fit_ordered_20210506.pkl')

    for i in global_guess.condition:
        g = global_guess[global_guess.condition==i]
        initial_guess['bistatnorm'][i] = []
        for j in g.columns[2:8]:
            initial_guess['bistatnorm'][i].append(numpy.float(g.loc[:, j]))
 #    if quantity == 'alpha':  # to force the second mode of WT 6h lowN to be like for 6h lowN ΔpolyP
 #        initial_guess['bistatnorm']['bLR31_6h_0210ms_lowN'] = initial_guess['bistatnorm']['bLR32_6h_0210ms_lowN']

 #    if quantity == 'alpha':
 #        initial_guess['bistatnorm']['bLR31_6h_0210ms_lowN'] = initial_guess['bistatnorm']['bLR31_0h_0210ms']
    ## using the results of the fit to WT 0h 210 ms to statnorm

 #    if quantity == 'alpha':
 #
 #        initial_guess['bistatnorm']['bLR31_0h_0210ms'] = [fits['statnorm']['bLR31_0h_0210ms'][0][0], fits['statnorm']['bLR31_0h_0210ms'][0][0], fits['statnorm']['bLR31_0h_0210ms'][0][1], fits['statnorm']['bLR31_0h_0210ms'][0][1], fits['statnorm']['bLR31_0h_0210ms'][0][2], fits['statnorm']['bLR31_0h_0210ms'][0][2]]
    #
    #        initial_guess['bistatnorm']['bLR32_0h_0210ms'] = [fits['statnorm']['bLR31_0h_0210ms'][0][0], fits['statnorm']['bLR31_0h_0210ms'][0][0], fits['statnorm']['bLR31_0h_0210ms'][0][1], fits['statnorm']['bLR31_0h_0210ms'][0][1], fits['statnorm']['bLR31_0h_0210ms'][0][2], fits['statnorm']['bLR31_0h_0210ms'][0][2]]
    #
    #        initial_guess['bistatnorm']['bLR31_6h_0210ms_lowN'] = [fits['statnorm']['bLR31_6h_0210ms_lowN'][0][0], fits['statnorm']['bLR31_6h_0210ms_lowN'][0][0], fits['statnorm']['bLR31_6h_0210ms_lowN'][0][1], fits['statnorm']['bLR31_6h_0210ms_lowN'][0][1], fits['statnorm']['bLR31_6h_0210ms_lowN'][0][2], fits['statnorm']['bLR31_6h_0210ms_lowN'][0][2]]
    #
    #        initial_guess['bistatnorm']['bLR32_6h_0210ms_lowN'] = [0.32, 0.72, fits['statnorm']['bLR31_0h_0210ms'][0][1], 3e-1, 16.5, 20]

    #        initial_guess['bistatnorm']['bLR31_6h_0210ms_lowC'] = [0.29146533,  0.06734266,  0.19601818,  0.04822481, 17.11572685, 32.85294952]

    ###        initial_guess['bistatnorm']['bLR31_6h_0210ms_lowC'] = [fits['statnorm']['bLR31_6h_0210ms_lowC'][0][0], fits['statnorm']['bLR31_6h_0210ms_lowC'][0][0], fits['statnorm']['bLR31_6h_0210ms_lowC'][0][1], fits['statnorm']['bLR31_6h_0210ms_lowC'][0][1], fits['statnorm']['bLR31_6h_0210ms_lowC'][0][2], fits['statnorm']['bLR31_6h_0210ms_lowC'][0][2]]

    #        initial_guess['bistatnorm']['bLR32_6h_0210ms_lowC'] = [0.28037551, 0.0655877 ,  0.18688838, 0.05184822, 14.61206503, 41.40400566] # got these by fitting first myself... it's a little bit of cheating but not much.

    #    elif quantity == 'D_app':

    #        initial_guess['bistatnorm']['bLR31_0h_0210ms'] =  [fits['statnorm']['bLR31_0h_0210ms'][0][0], fits['statnorm']['bLR31_0h_0210ms'][0][0], fits['statnorm']['bLR31_0h_0210ms'][0][1], 3e-1, 30, 30]#[fits['statnorm']['bLR31_0h_0210ms'][0][0], 1e-3, fits['statnorm']['bLR31_0h_0210ms'][0][1], 3e-1, 30, 30]
    #        initial_guess['bistatnorm']['bLR32_0h_0210ms'] = [2.32214192e-03, 1.13500855e-03, 4.80300338e-01, 2.76742062e-01,1.26964226e+01, 2.24879491e+01]#[fits['statnorm']['bLR32_0h_0210ms'][0][0], 6e-3, fits['statnorm']['bLR32_0h_0210ms'][0][1], 3e-1, 17, 11]
    #        initial_guess['bistatnorm']['bLR31_6h_0210ms_lowN'] = [6.65811227e-03,  1.18552645e-03, -2.82635511e-01, -3.30784561e-01, 1.34905093e+01,  2.49915665e+01]#[fits['statnorm']['bLR31_0h_0210ms'][0][0], 5e-3, fits['statnorm']['bLR31_0h_0210ms'][0][1], 3e-1, 50, 12] #[fits['statnorm']['bLR31_0h_0210ms'][0][0], 3e-3, fits['statnorm']['bLR31_0h_0210ms'][0][1], 3e-1, 50, 12]
    #        initial_guess['bistatnorm']['bLR32_6h_0210ms_lowN'] = [3.57486287e-03, 1.80680961e-02, 4.67057548e-01, 2.36275975e-01, 1.28723816e+01, 2.65246771e+01] #[2e-3, 1.8e-2, fits['statnorm']['bLR31_0h_0210ms'][0][1], 3e-1, 16.5, 20]#[fits['statnorm']['bLR31_0h_0210ms'][0][0], 1.8e-2, fits['statnorm']['bLR31_0h_0210ms'][0][1], 3e-1, 16.5, 20]

    #        initial_guess['bistatnorm']['bLR31_6h_0210ms_lowC'] = [fits['statnorm']['bLR31_6h_0210ms_lowC'][0][0], 1e-4, fits['statnorm']['bLR31_6h_0210ms_lowC'][0][1], 3e-1, 50, 12]
    #        initial_guess['bistatnorm']['bLR32_6h_0210ms_lowC'] = [fits['statnorm']['bLR32_6h_0210ms_lowC'][0][0], 1.8e-2, fits['statnorm']['bLR32_6h_0210ms_lowC'][0][1], 3e-1, 16.5, 20]

    #    for i in list(fits['statnorm'].keys())[:]:

    ### start here ###
    for i in keys_of_interest[:]:
        print(i)
        time_between_frames = load_all_results.read('time_between_frames', i, spot_type)
        strain = load_all_results.read('strain', i, spot_type)
        condition = load_all_results.read('condition', i, spot_type)
        starvation_time = load_all_results.read('starvation_time', i, spot_type) + 'h'
        j = strain + '_' + starvation_time + '_' + time_between_frames + 'ms'
        if starvation_time != '0h':
            j = j  + '_' + condition
    ##        print('initial guess\n')
    ##        print(initial_guess['bistatnorm'][i])
    ##        try:
        if log_option:
            fits['bistatnorm'][i] = scipy.optimize.curve_fit(bistatnorm_log, h[i][:,0], h[i][:,1], p0 = initial_guess['bistatnorm'][j])#, p0 = initial_guess['bistatnorm'][i], bounds = ((-numpy.inf, -numpy.inf, -numpy.inf, -numpy.inf, 0, 0),(+numpy.inf,+numpy.inf, +numpy.inf, +numpy.inf, +numpy.inf, +numpy.inf)))
            y_curve_1[i] = statnorm_log(h[i][:,0], fits['bistatnorm'][i][0][0], fits['bistatnorm'][i][0][2], fits['bistatnorm'][i][0][4])
            y_curve_2[i] = statnorm_log(h[i][:,0], fits['bistatnorm'][i][0][1], fits['bistatnorm'][i][0][3], fits['bistatnorm'][i][0][5])

        else:
            fits['bistatnorm'][i] = scipy.optimize.curve_fit(bistatnorm, h[i][:,0], h[i][:,1], p0 = initial_guess['bistatnorm'][j])#, bounds = ((-numpy.inf, -numpy.inf, -numpy.inf, -numpy.inf, 0, 0),(+numpy.inf,+numpy.inf, +numpy.inf, +numpy.inf, +numpy.inf, +numpy.inf)))
            y_curve_1[i] = statnorm(h[i][:,0], fits['bistatnorm'][i][0][0], fits['bistatnorm'][i][0][2], fits['bistatnorm'][i][0][4])
            y_curve_2[i] = statnorm(h[i][:,0], fits['bistatnorm'][i][0][1], fits['bistatnorm'][i][0][3], fits['bistatnorm'][i][0][5])

        intersection[i] = intersect.intersection(h[i][:,0], y_curve_1[i], h[i][:,0], y_curve_2[i])
    #        print('INTERSECTION')
    #        print(intersection[i])
    #        print(len(intersection[i]))
    #        print(len(intersection[i][0]))
        if len(intersection[i][0]) > 0:
            ip = intersection[i][0].min()
            occupancy_1 = len(numpy.where(results2[i].loc[:,quantity] <= ip)[0])
            occupancy_2 = len(numpy.where(results2[i].loc[:,quantity] > ip)[0])
        else:
            ip = numpy.nan
            occupancy_1 = len(results2[i])
            occupancy_2 = 0

        occupancy_total = len(results2[i])

    #        occupancy_1 = numpy.trapz(y_curve_1[i], x = h[i][:,0])
    #        occupancy_2 = numpy.trapz(y_curve_2[i], x = h[i][:,0])
    #        occupancy_total = occupancy_1 + occupancy_2
        occupancy_1 = 100 * occupancy_1 / occupancy_total
        occupancy_2 = 100 * occupancy_2 / occupancy_total

        d_bi = [[i, 'bistatnorm', fits['bistatnorm'][i][0][0], fits['bistatnorm'][i][0][1], numpy.abs(fits['bistatnorm'][i][0][2]), numpy.abs(fits['bistatnorm'][i][0][3]), fits['bistatnorm'][i][0][4], fits['bistatnorm'][i][0][5], numpy.sqrt(numpy.diag(fits['bistatnorm'][i][1]))[0], numpy.sqrt(numpy.diag(fits['bistatnorm'][i][1]))[1], numpy.sqrt(numpy.diag(fits['bistatnorm'][i][1]))[2], numpy.sqrt(numpy.diag(fits['bistatnorm'][i][1]))[3], numpy.sqrt(numpy.diag(fits['bistatnorm'][i][1]))[4], numpy.sqrt(numpy.diag(fits['bistatnorm'][i][1]))[5], occupancy_1, occupancy_2, ip]]

        little_answer_bistatnorm =  pandas.DataFrame(data = d_bi, columns = ['condition', 'function', 'mean_1', 'mean_2', 'sigma_1','sigma_2', 'amplitude_1', 'amplitude_2', 'mean_fit_unc_1',  'mean_fit_unc_2', 'sigma_fit_unc_1', 'sigma_fit_unc_2', 'amplitude_fit_unc_1',   'amplitude_fit_unc_2', 'percent_population_1', 'percent_population_2', 'mode_intersection_point'])
        results = results.append(little_answer_bistatnorm)

        d_s = [[i, 'statnorm', fits['statnorm'][i][0][0], numpy.nan,  numpy.abs(fits['statnorm'][i][0][1]), numpy.nan, fits['statnorm'][i][0][2], numpy.nan, numpy.sqrt(numpy.diag(fits['statnorm'][i][1]))[0], numpy.nan,  numpy.sqrt(numpy.diag(fits['statnorm'][i][1]))[1], numpy.nan,  numpy.sqrt(numpy.diag(fits['statnorm'][i][1]))[2], numpy.nan, 100, 0, numpy.nan]]

        little_answer_statnorm =  pandas.DataFrame(data = d_s, columns = ['condition', 'function', 'mean_1', 'mean_2', 'sigma_1','sigma_2', 'amplitude_1', 'amplitude_2', 'mean_fit_unc_1',  'mean_fit_unc_2', 'sigma_fit_unc_1', 'sigma_fit_unc_2', 'amplitude_fit_unc_1',   'amplitude_fit_unc_2', 'percent_population_1', 'percent_population_2', 'mode_intersection_point'])
        results = results.append(little_answer_statnorm)

    ##        results.to_pickle(basic_directory + central_directory + 'results/fits_to_' + quantity + 's_per_cat_' + time_label + '/' + quantity + '_histograms_' + str(time_between_frames) + 'ms_binormal_fit' + '.pkl')
    ##        results.to_csv(basic_directory + central_directory + 'results/fits_to_' + quantity + 's_per_cat_' + time_label + '/' + quantity + '_histograms_' + str(time_between_frames) + 'ms_binormal_fit' + '.csv')
    #
        j = i.split('_')
        k = i.split('_')[0] + '_' + i.split('_')[1]
        if len(j) > 3:
            k = k + '_' + i.split('_')[3]

        if plot:
    ##               fig = pylab.figure(spot_type + ', ' + quantity + ', time between frames: ' + str(time_between_frames))
            strain = load_all_results.read('strain', i, spot_type)
            starvation_time = load_all_results.read('starvation_time', i, spot_type) + 'h'
            condition = load_all_results.read('condition', i, spot_type)
            k2 = strain + '_' + starvation_time
            if starvation_time != '0h':
                k2 = k2 + '_' + condition
            ft = spot_type + ', ' + quantity   # figure name
            stitle = spot_type + ', ' + quantity  # figure title
            if per_day:
                day = i.split('_')[::-1][0]
                ft = ft + ', ' + day
                stitle = stitle + ', ' + day
            if not ignore_timelag:
                time_between_frames = load_all_results.read('time_between_frames', i, spot_type)
                ft = ft + ', time between frames: ' + str(time_between_frames)
                stitle = stitle + ', time between frames: ' + str(time_between_frames) + 'ms'

            fig = pylab.figure(ft)
            axes = fig.get_axes()

            if log_option:
                print('LOG')
                if starved_only_bistatnorm:
                    if '6h' in k2:
                        axes[position_subplot[spot_type][k2]].plot(h[i][:,0], bistatnorm_log(h[i][:,0], fits['bistatnorm'][i][0][0], fits['bistatnorm'][i][0][1], fits['bistatnorm'][i][0][2], fits['bistatnorm'][i][0][3], fits['bistatnorm'][i][0][4], fits['bistatnorm'][i][0][5]), 'k', linewidth = 2)
                        axes[position_subplot[spot_type][k2]].plot(h[i][:,0], y_curve_1[i], 'k--', linewidth = 2)
                        axes[position_subplot[spot_type][k2]].plot(h[i][:,0], y_curve_2[i], 'k--', linewidth = 2)

    #                        axes[position_subplot[spot_type][k2]].plot(h[i][:,0], statnorm_log(h[i][:,0], fits['bistatnorm'][i][0][0], fits['bistatnorm'][i][0][2], fits['bistatnorm'][i][0][4]), 'k--', linewidth = 2)
    #                        axes[position_subplot[spot_type][k2]].plot(h[i][:,0], statnorm_log(h[i][:,0], fits['bistatnorm'][i][0][1], fits['bistatnorm'][i][0][3], fits['bistatnorm'][i][0][5]), 'k--', linewidth = 2)

                    elif '0h' in k2:
                        axes[position_subplot[spot_type][k2]].plot(h[i][:,0], statnorm_log(h[i][:,0], fits['statnorm'][i][0][0], fits['statnorm'][i][0][1], fits['statnorm'][i][0][2]), 'k', linewidth = 2)
                else:
                    axes[position_subplot[spot_type][k2]].plot(h[i][:,0], bistatnorm_log(h[i][:,0], fits['bistatnorm'][i][0][0], fits['bistatnorm'][i][0][1], fits['bistatnorm'][i][0][2], fits['bistatnorm'][i][0][3], fits['bistatnorm'][i][0][4], fits['bistatnorm'][i][0][5]), 'k', linewidth = 2)
                    axes[position_subplot[spot_type][k2]].plot(h[i][:,0], y_curve_1[i], 'k--', linewidth = 2)
                    axes[position_subplot[spot_type][k2]].plot(h[i][:,0], y_curve_2[i], 'k--', linewidth = 2)

    #                    axes[position_subplot[spot_type][k2]].plot(h[i][:,0], statnorm_log(h[i][:,0], fits['bistatnorm'][i][0][0], fits['bistatnorm'][i][0][2], fits['bistatnorm'][i][0][4]), 'k--', linewidth = 2)
    #                    axes[position_subplot[spot_type][k2]].plot(h[i][:,0], statnorm_log(h[i][:,0], fits['bistatnorm'][i][0][1], fits['bistatnorm'][i][0][3], fits['bistatnorm'][i][0][5]), 'k--', linewidth = 2)


            else:
                if starved_only_bistatnorm:
                    if '6h' in k2:
                        axes[position_subplot[spot_type][k2]].plot(h[i][:,0], bistatnorm(h[i][:,0], fits['bistatnorm'][i][0][0], fits['bistatnorm'][i][0][1], fits['bistatnorm'][i][0][2], fits['bistatnorm'][i][0][3], fits['bistatnorm'][i][0][4], fits['bistatnorm'][i][0][5]), 'k', linewidth = 2)
                        axes[position_subplot[spot_type][k2]].plot(h[i][:,0], y_curve_1[i], 'k--', linewidth = 2)
                        axes[position_subplot[spot_type][k2]].plot(h[i][:,0], y_curve_2[i], 'k--', linewidth = 2)

    #                        axes[position_subplot[spot_type][k2]].plot(h[i][:,0], statnorm(h[i][:,0], fits['bistatnorm'][i][0][0], fits['bistatnorm'][i][0][2], fits['bistatnorm'][i][0][4]), 'k--', linewidth = 2)
    #                        axes[position_subplot[spot_type][k2]].plot(h[i][:,0], statnorm(h[i][:,0], fits['bistatnorm'][i][0][1], fits['bistatnorm'][i][0][3], fits['bistatnorm'][i][0][5]), 'k--', linewidth = 2)

                    elif '0h' in k2:
                        axes[position_subplot[spot_type][k2]].plot(h[i][:,0], statnorm(h[i][:,0], fits['statnorm'][i][0][0], fits['statnorm'][i][0][1], fits['statnorm'][i][0][2]), 'k', linewidth = 2)
                else:
                    axes[position_subplot[spot_type][k2]].plot(h[i][:,0], bistatnorm_log(h[i][:,0], fits['bistatnorm'][i][0][0], fits['bistatnorm'][i][0][1], fits['bistatnorm'][i][0][2], fits['bistatnorm'][i][0][3], fits['bistatnorm'][i][0][4], fits['bistatnorm'][i][0][5]), 'k', linewidth = 2)
                    axes[position_subplot[spot_type][k2]].plot(h[i][:,0], y_curve_1[i], 'k--', linewidth = 2)
                    axes[position_subplot[spot_type][k2]].plot(h[i][:,0], y_curve_2[i], 'k--', linewidth = 2)

    #                    axes[position_subplot[spot_type][k2]].plot(h[i][:,0], statnorm(h[i][:,0], fits['bistatnorm'][i][0][0], fits['bistatnorm'][i][0][2], fits['bistatnorm'][i][0][4]), 'k--', linewidth = 2)
    #                    axes[position_subplot[spot_type][k2]].plot(h[i][:,0], statnorm(h[i][:,0], fits['bistatnorm'][i][0][1], fits['bistatnorm'][i][0][3], fits['bistatnorm'][i][0][5]), 'k--', linewidth = 2)


    ##                axes[position_sublot[spot_type][k2]].axvline(fits['bistatnorm'][i][0][0], color = 'k', linestyle = '--')
    ##                axes[position_sublot[spot_type][k2]].axvline(fits['bistatnorm'][i][0][1], color = 'k', linestyle = '--')
    #
    ###        axes[position_sublot[spot_type][k2]].plot(h[i][:,0], bistatnorm_fix_first(h[i][:,0], fits['bistatnorm_fix_first'][i][0][0], fits['bistatnorm_fix_first'][i][0][1], fits['bistatnorm_fix_first'][i][0][2], fits['bistatnorm_fix_first'][i][0][3]), 'k', linewidth = 2)
    #
    ##            pylab.savefig(basic_directory + central_directory + 'plots/fits_to_' + quantity + 's_per_cat_' + time_label + '/' + quantity + '_histograms_' + str(time_between_frames) + 'ms_binormal_fit_' + string_now + '.png')
    ##
    ##        except:
    ##            print('skipping ' + str(i))
    #
    ### The first mode in the DataFrame does not always correspond to the lowest mode. Therefore, I need to order the modes, so that I can keep track of the populations of mobility and correctly distinguish the less mobile from the more mobile groups. ##
    #
        results_ordered = results.copy(deep = True) # deep = True to create a new copy and keep the original intact, i.e. to not mirror the two DataFrames
        results.set_index(numpy.arange(0, len(results)), inplace = True) # to facilitate finding the rows where the columns need to be swapped
        results_ordered.set_index(numpy.arange(0, len(results)), inplace = True)
        w = numpy.where(results.mean_1 > results.mean_2)[0]  # the rows where the columns need to be swapped
        results_ordered.loc[w, 'mean_1'] = results.loc[w, 'mean_2']
        results_ordered.loc[w, 'mean_2'] = results.loc[w, 'mean_1']
        results_ordered.loc[w, 'sigma_1'] = results.loc[w, 'sigma_2']
        results_ordered.loc[w, 'sigma_2'] = results.loc[w, 'sigma_1']
        results_ordered.loc[w, 'amplitude_1'] = results.loc[w, 'amplitude_2']
        results_ordered.loc[w, 'amplitude_2'] = results.loc[w, 'amplitude_1']
        results_ordered.loc[w, 'mean_fit_unc_1'] = results.loc[w, 'mean_fit_unc_2']
        results_ordered.loc[w, 'mean_fit_unc_2'] = results.loc[w, 'mean_fit_unc_1']
        results_ordered.loc[w, 'sigma_fit_unc_1'] = results.loc[w, 'sigma_fit_unc_2']
        results_ordered.loc[w, 'sigma_fit_unc_2'] = results.loc[w, 'sigma_fit_unc_1']
        results_ordered.loc[w, 'amplitude_fit_unc_1'] = results.loc[w, 'amplitude_fit_unc_2']
        results_ordered.loc[w, 'amplitude_fit_unc_2'] = results.loc[w, 'amplitude_fit_unc_1']
    # don't flip the percent_population results because by definition I have ordered them already
    #        results_ordered.loc[w, 'percent_population_1'] = results.loc[w, 'percent_population_2']
    #        results_ordered.loc[w, 'percent_population_2'] = results.loc[w, 'percent_population_1']

    ## WHEN IT SKIPS IT DOES NOT KNOW THE TIME BETWEEN FRAMES
        results_ordered.to_pickle(basic_directory + central_directory + 'results/fits_to_' + quantity + 's_per_cat_' + time_label + '/' + string_now + '_' + quantity + '_histograms_' + str(time_between_frames) + 'ms_binormal_fit' + pd + 'ordered.pkl')
        results_ordered.to_csv(basic_directory + central_directory + 'results/fits_to_' + quantity + 's_per_cat_' + time_label + '/' + string_now + '_' + quantity + '_histograms_' + str(time_between_frames) + 'ms_binormal_fit' + pd + 'ordered.csv')


        ### doing it by hand now, do it better later ###
    #    intermediate_results = {}
    #    for j in ['bLR31_0h_0210ms', 'bLR31_6h_0210ms_lowN','bLR32_0h_0210ms','bLR32_6h_0210ms_lowN']:
    #        intermediate_results[j] = {}
    #        intermediate_results[j]['mean_1'] = []
    #        intermediate_results[j]['mean_2'] = []
    #        intermediate_results[j]['sigma_1'] = []
    #        intermediate_results[j]['sigma_2'] = []
    #        intermediate_results[j]['amplitude_1'] = []
    #        intermediate_results[j]['amplitude_2'] = []

    ### to show horizontal lines for mean, if there are multiple points per condition, uncomment lines 2398 - 2418 and anything referring to intermediate_results below ###
    #    for i in results_ordered.condition:
    #        strain = load_all_results.read('strain', i, 'muNS')
    #        starvation_time = load_all_results.read('starvation_time', i, 'muNS') + 'h'
    #        condition = load_all_results.read('condition', i, 'muNS')
    #        j = strain + '_' + starvation_time + '_' + '0210ms'
    #        if starvation_time != '0h':
    #            j = j + '_' + condition
    #        if j in i:
    #            intermediate_results[j]['mean_1'].append(numpy.float(results_ordered[results_ordered.condition==i].mean_1))
    #            intermediate_results[j]['mean_2'].append(numpy.float(results_ordered[results_ordered.condition==i].mean_2))
    #            intermediate_results[j]['sigma_1'].append(numpy.float(results_ordered[results_ordered.condition==i].sigma_1))
    #            intermediate_results[j]['sigma_2'].append(numpy.float(results_ordered[results_ordered.condition==i].sigma_2))
    #            intermediate_results[j]['amplitude_1'].append(numpy.float(results_ordered[results_ordered.condition==i].amplitude_1))
    #            intermediate_results[j]['amplitude_2'].append(numpy.float(results_ordered[results_ordered.condition==i].amplitude_2))
    #
    #            intermediate_results[j]['mean_mean_1'] = numpy.array(intermediate_results[j]['mean_1']).mean()
    #            intermediate_results[j]['mean_mean_2'] = numpy.array(intermediate_results[j]['mean_2']).mean()
    #            intermediate_results[j]['mean_sigma_1'] = numpy.array(intermediate_results[j]['sigma_1']).mean()
    #            intermediate_results[j]['mean_sigma_2'] = numpy.array(intermediate_results[j]['sigma_2']).mean()
    #            intermediate_results[j]['mean_amplitude_1'] = numpy.array(intermediate_results[j]['amplitude_1']).mean()
    #            intermediate_results[j]['mean_amplitude_2'] = numpy.array(intermediate_results[j]['amplitude_2']).mean()


    if plot_fit_results:

        legend_elements = [Line2D([0], [0], marker = markershapes[starvation_time + condition], color = 'k', label = 'mobile mode', markersize = ms, alpha = 0.7, markeredgewidth = 2), Line2D([0], [0], marker = 'o', markerfacecolor = 'w', color = 'k', label = 'immobile mode', markersize = ms, alpha = 0.7, markeredgewidth = 2)]

        for i in results_ordered.condition.values:
            strain = load_all_results.read('strain', i, spot_type)
            starvation_time = load_all_results.read('starvation_time', i, spot_type) + 'h'
            condition = load_all_results.read('condition', i, spot_type)
            time_between_frames = load_all_results.read('time_between_frames', i, spot_type)

            ### MEAN ###
            fig = pylab.figure('fit results, means ' + spot_type + ', ' + quantity + ', time between frames: ' + str(time_between_frames))
            pylab.title('fit results, means\n' + spot_type + ', ' + quantity + ', time between frames: ' + str(time_between_frames))

            if quantity == 'D_app':
                ax = fig.gca()
                ax.set_yscale('log')
                ymin_means, ymax_means = 5e-4, 3e-2

            elif quantity == 'alpha':
                ymin_means, ymax_means = 0.25, 0.8

            if starved_only_bistatnorm:
                if starvation_time == '0h':
                    r_now = results_ordered[results_ordered.function == 'statnorm']
                elif starvation_time == '6h':
                    r_now = results_ordered[results_ordered.function == 'bistatnorm']
            else:
                r_now = results_ordered[results_ordered.function == 'bistatnorm']

            r_now = r_now[r_now.condition==i]
            j = strain + '_' + starvation_time + '_' + '0210ms'
            if starvation_time != '0h':
                j = j + '_' + condition

            pylab.errorbar(positions[strain][starvation_time + condition] - 0.025, r_now.mean_1, xerr = 0, yerr = r_now.mean_fit_unc_1, color = colors[strain][starvation_time + condition], fmt = markershapes[starvation_time + condition], markersize = ms, markerfacecolor = 'w', alpha = 0.7, markeredgecolor = colors[strain][starvation_time + condition], markeredgewidth = 2)
        #            pylab.hlines(intermediate_results[j]['mean_mean_1'], positions[strain][starvation_time + condition] - 0.025 - 0.03, positions[strain][starvation_time + condition] - 0.025 + 0.03, color = colors[strain][starvation_time + condition], linewidth = 3)

            pylab.errorbar(positions[strain][starvation_time + condition] + 0.025, r_now.mean_2, xerr = 0, yerr = r_now.mean_fit_unc_2, color = colors[strain][starvation_time + condition], fmt = markershapes[starvation_time + condition], markersize = ms, alpha = 0.7, markeredgecolor = colors[strain][starvation_time + condition], markeredgewidth = 2)
    #            pylab.hlines(intermediate_results[j]['mean_mean_2'], positions[strain][starvation_time + condition] + 0.025 - 0.03, positions[strain][starvation_time + condition] + 0.025 + 0.03, color = colors[strain][starvation_time + condition], linewidth = 3)

            if spot_type == 'muNS':
                pylab.xticks([1, 1.2, 1.4, 1.6], ['WT\n0h', '\n6h lowN', 'ΔpolyP\n0h', '\n6h lowN'])

    #                pylab.xticks([1, 1.2, 1.4, 1.6, 1.8, 2.0], ['WT\n0h', '\n6h lowN', '\n6h lowC', 'ΔpolyP\n0h', '\n6h lowN', '\n6h lowC'])
            elif spot_type == 'origins':
                pylab.xticks([1, 1.2, 1.4, 1.6], ['WT\n0h', '\n6h lowN', 'ΔpolyP\n0h', '\n6h lowN'])

            if quantity == 'D_app':
                pylab.ylabel('mean ' + r'$ D_{app} (\mu m^{2} / s^{\alpha})$')
            elif quantity == 'alpha':
                pylab.ylabel('mean α')
            pylab.ylim(ymin_means, ymax_means)
            pylab.legend(handles=legend_elements, frameon = False)
            pylab.savefig(basic_directory + central_directory + 'plots/fits_to_' + quantity + 's_per_cat_' + time_label + '/fit_results_vs_condition_meanvalues_' + quantity + '_histograms_' + str(time_between_frames) + 'ms_binormal_fit' + pd + string_now + '.png')

            pylab.savefig(basic_directory_paper_figs + '_' + spot_type + '/' + string_now + '_' + quantity +  '_summary_fit_results' + pd + 'means.svg')
            pylab.savefig(basic_directory_paper_figs + '_' + spot_type + '/' + string_now + '_' + quantity +  '_summary_fit_results' + pd + 'means.png')

            ### width ###
            pylab.figure('fit results, widths ' + spot_type + ', ' + quantity + ', time between frames: ' + str(time_between_frames))
            pylab.title('fit results, widths\n' + spot_type + ', ' + quantity + ', time between frames: ' + str(time_between_frames))
            pylab.errorbar(positions[strain][starvation_time + condition] - 0.025, numpy.abs(r_now.sigma_1), xerr = 0, yerr = r_now.sigma_fit_unc_1, color = colors[strain][starvation_time + condition], fmt = markershapes[starvation_time + condition], markersize = ms, markerfacecolor = 'w', alpha = 0.7, markeredgecolor = colors[strain][starvation_time + condition], markeredgewidth = 2)
    #            pylab.hlines(numpy.abs(intermediate_results[j]['mean_sigma_1']), positions[strain][starvation_time + condition] - 0.025 - 0.03, positions[strain][starvation_time + condition] - 0.025 + 0.03, color = colors[strain][starvation_time + condition], linewidth = 3)

            pylab.errorbar(positions[strain][starvation_time + condition] + 0.025, numpy.abs(r_now.sigma_2), xerr = 0, yerr = r_now.sigma_fit_unc_2, color = colors[strain][starvation_time + condition], fmt =  markershapes[starvation_time + condition], markersize = ms, alpha = 0.7, markeredgecolor = colors[strain][starvation_time + condition], markeredgewidth = 2)
    #            pylab.hlines(numpy.abs(intermediate_results[j]['mean_sigma_2']), positions[strain][starvation_time + condition] + 0.025 - 0.03, positions[strain][starvation_time + condition] + 0.025 + 0.03, color = colors[strain][starvation_time + condition], linewidth = 3)

            if spot_type == 'muNS':
                pylab.xticks([1, 1.2, 1.4, 1.6], ['WT\n0h', '\n6h lowN', 'ΔpolyP\n0h', '\n6h lowN'])
    #                pylab.xticks([1, 1.2, 1.4, 1.6, 1.8, 2.0], ['WT\n0h', '\n6h lowN', '\n6h lowC', 'ΔpolyP\n0h', '\n6h lowN', '\n6h lowC'])
                pylab.ylim(-0.025, 0.8)
            elif spot_type == 'origins':
                pylab.xticks([1, 1.2, 1.4, 1.6], ['WT\n0h', '\n6h lowN', 'ΔpolyP\n0h', '\n6h lowN'])

            pylab.ylabel('abs(σ)')
            pylab.legend(handles=legend_elements, frameon = False)
            pylab.savefig(basic_directory + central_directory + 'plots/fits_to_' + quantity + 's_per_cat_' + time_label + '/fit_results_vs_condition_widths_' + quantity + '_histograms_' + str(time_between_frames) + 'ms_normal_fit' + pd + string_now + '.png')
            pylab.savefig(basic_directory_paper_figs + '_' + spot_type + '/' + string_now + '_' + quantity +  '_summary_fit_results' + pd + 'widths.svg')
            pylab.savefig(basic_directory_paper_figs + '_' + spot_type + '/' + string_now + '_' + quantity + '_summary_fit_results' +  pd + 'widths.png')


            ### amplitude ###
            pylab.figure('fit results, amplitudes ' + spot_type + ', ' + quantity + ', time between frames: ' + str(time_between_frames))
            pylab.title('fit results, amplitudes\n' + spot_type + ', ' + quantity + ', time between frames: ' + str(time_between_frames))
            pylab.errorbar(positions[strain][starvation_time + condition] - 0.025, r_now.amplitude_1, xerr = 0, yerr = r_now.amplitude_fit_unc_1, color = colors[strain][starvation_time + condition], fmt =  markershapes[starvation_time + condition], markersize = ms, markerfacecolor = 'w', alpha = 0.7, markeredgecolor = colors[strain][starvation_time + condition], markeredgewidth = 2)
    #            pylab.hlines(intermediate_results[j]['mean_amplitude_1'], positions[strain][starvation_time + condition] - 0.025 - 0.03, positions[strain][starvation_time + condition] - 0.025 + 0.03, color = colors[strain][starvation_time + condition], linewidth = 3)

            pylab.errorbar(positions[strain][starvation_time + condition] + 0.025, r_now.amplitude_2, xerr = 0, yerr = r_now.amplitude_fit_unc_2, color = colors[strain][starvation_time + condition], fmt =  markershapes[starvation_time + condition], markersize = ms, alpha = 0.7, markeredgecolor = colors[strain][starvation_time + condition], markeredgewidth = 2)
    #            pylab.hlines(intermediate_results[j]['mean_amplitude_2'], positions[strain][starvation_time + condition] + 0.025 - 0.03, positions[strain][starvation_time + condition] + 0.025 + 0.03, color = colors[strain][starvation_time + condition], linewidth = 3)

            if spot_type == 'muNS':
                pylab.xticks([1, 1.2, 1.4, 1.6], ['WT\n0h', '\n6h lowN', 'ΔpolyP\n0h', '\n6h lowN'])
    #              pylab.xticks([1, 1.2, 1.4, 1.6, 1.8, 2.0], ['WT\n0h', '\n6h lowN', '\n6h lowC', 'ΔpolyP\n0h', '\n6h lowN', '\n6h lowC'])
            elif spot_type == 'origins':
              pylab.xticks([1, 1.2, 1.4, 1.6], ['WT\n0h', '\n6h lowN', 'ΔpolyP\n0h', '\n6h lowN'])
            pylab.ylabel('amplitude')
    #            pylab.ylim(0, 55)
            pylab.legend(handles=legend_elements, frameon = False)
            pylab.savefig(basic_directory + central_directory + 'plots/fits_to_' + quantity + 's_per_cat_' + time_label + '/fit_results_vs_condition_amplitudes_' + quantity + '_histograms_' + str(time_between_frames) + 'ms_normal_fit' + pd + string_now + '.png')
            pylab.savefig(basic_directory_paper_figs + '_' + spot_type + '/' + string_now + '_' + quantity +  '_summary_fit_results' + pd + 'amplitudes.svg')
            pylab.savefig(basic_directory_paper_figs + '_' + spot_type + '/' + string_now + '_' + quantity + '_summary_fit_results' + pd + 'amplitudes.png')

    results_ordered.to_pickle(basic_directory_paper_data + '_' + spot_type + '/' + string_now + '_' + quantity + '_fit_results' + pd + 'ordered.pkl')
    results_ordered.to_csv(basic_directory_paper_data + '_' + spot_type + '/' + string_now + '_' + quantity + '_fit_results' + pd + 'ordered.csv')

    numpy.save(basic_directory_paper_data + '_' + spot_type + '/' + string_now + '_histogram_outlines' + pd + quantity + '.npy', h)
    #    numpy.save(basic_directory_paper_data + '_' + spot_type + '/' + string_now + '_means_of_fit_values' + pd + quantity + '.npy', intermediate_results)

    return fits['bistatnorm'], fits['statnorm'], w, results_ordered, h, results, y_curve_1, y_curve_2, intersection, initial_guess

def calculate_population_percentages(spot_type, quantity, results, avoid = ['1s'], plot = False, log_option = True, time_label = 'tstart0p21sec_tend0p84sec', per_day = False, ignore_timelag = False, starved_only_bistatnorm = True):
    '''
    '''
    central_directory = load_all_results.select_specifications(spot_type)['central_directory']

    now = datetime.datetime.now()
    string_now = now.strftime("%Y%m%d")

    if per_day:
        pd = '_per_day_'
    else:
        pd = '_'

    keys_of_interest = list(results.keys())[:]
    results2 = results.copy()
    for i in results.keys():
        if i not in keys_of_interest:
            del results2[i]

    ### First, the total area of the original curve. ###
    #    h = histogram_all(spot_type, quantity, results2, avoid = avoid, show_median = False, time_label = time_label, below_size_limit_only = False, per_day = per_day)[1]

    pylab.close('all')

    use = fit_histograms_bistatnorm(spot_type, quantity, results2, avoid = avoid, plot = plot, log_option = log_option, per_day = per_day, starved_only_bistatnorm = starved_only_bistatnorm)
    fits = {}
    fits['bistatnorm'] = use[0]
    fits['statnorm'] = use[1]
    rows_to_swap = use[2]
    h = use[4]

    populations = {}
    populations['statnorm'] = {}
    populations['bistatnorm'] = {}

    area = pandas.DataFrame(columns = ['condition', 'population_1_percent', 'population_2_percent'])

    for i in fits['bistatnorm'].keys():
        time_between_frames = load_all_results.read('time_between_frames', i, spot_type)
        strain = load_all_results.read('strain', i, spot_type)
        starvation_time = load_all_results.read('starvation_time', i, spot_type) + 'h'
        condition = load_all_results.read('condition', i, spot_type)
        populations['bistatnorm'][i] = numpy.zeros([len(h[i]),3])
        populations['statnorm'][i] = numpy.zeros([len(h[i]),2])
        populations['bistatnorm'][i][:,0] = h[i][:,0]
        populations['statnorm'][i][:,0] = h[i][:,0]

        if log_option:
            populations['bistatnorm'][i][:,1] = statnorm_log(populations['bistatnorm'][i][:,0], fits['bistatnorm'][i][0][0], fits['bistatnorm'][i][0][2], fits['bistatnorm'][i][0][4])
            populations['bistatnorm'][i][:,2] = statnorm_log(populations['bistatnorm'][i][:,0], fits['bistatnorm'][i][0][1], fits['bistatnorm'][i][0][3], fits['bistatnorm'][i][0][5])
            populations['statnorm'][i][:,1] = statnorm_log(populations['statnorm'][i][:,0], fits['statnorm'][i][0][0], fits['statnorm'][i][0][1], fits['statnorm'][i][0][2])
        else:
            populations['bistatnorm'][i][:,1] = statnorm(populations['bistatnorm'][i][:,0], fits['bistatnorm'][i][0][0], fits['bistatnorm'][i][0][2], fits['bistatnorm'][i][0][4])
            populations['bistatnorm'][i][:,2] = statnorm(populations['bistatnorm'][i][:,0], fits['bistatnorm'][i][0][1], fits['bistatnorm'][i][0][3], fits['bistatnorm'][i][0][5])
            populations['statnorm'][i][:,1] = statnorm(populations['statnorm'][i][:,0], fits['statnorm'][i][0][0], fits['statnorm'][i][0][1], fits['statnorm'][i][0][2])

    #        a_total = numpy.trapz(h[i][:,1], h[i][:,0])
        a_0 = numpy.trapz(populations['bistatnorm'][i][:,1], populations['bistatnorm'][i][:,0])
        a_1 = numpy.trapz(populations['bistatnorm'][i][:,2], populations['bistatnorm'][i][:,0])
        a_total = a_0 + a_1
        d = [[i, 100 * a_0 / a_total, 100 * a_1 / a_total]]
        little_answer =  pandas.DataFrame(data = d, columns =  ['condition', 'population_1_percent', 'population_2_percent'])
        area = area.append(little_answer)
        area.to_pickle(basic_directory + central_directory + 'results/fits_to_' + quantity + 's_per_cat_' + time_label + '/' + quantity + '_histograms_' + str(time_between_frames) + 'ms_binormal_fits_three_curves' + pd + string_now + '.pkl')
        area.to_csv(basic_directory + central_directory + 'results/fits_to_' + quantity + 's_per_cat_' + time_label + '/' + quantity + '_histograms_' + str(time_between_frames) + 'ms_binormal_fits_three_curves' + pd + string_now + '.csv')

        if plot:
            j = i.split('_')
            k = i.split('_')[0] + '_' + i.split('_')[1]
            if len(j) > 3:
                k = k + '_' + i.split('_')[3]
            k2 = strain + '_' + starvation_time
            if starvation_time != '0h':
                k2 = k2 + '_' + condition
            ft = spot_type + ', ' + quantity   # figure name
            stitle = spot_type + ', ' + quantity  # figure title
            if per_day:
                day = i.split('_')[::-1][0]
                ft = ft + ', ' + day
                stitle = stitle + ', ' + day
            else:
                day = ''
            if not ignore_timelag:
                time_between_frames = load_all_results.read('time_between_frames', i, spot_type)
                ft = ft + ', time between frames: ' + str(time_between_frames)
                stitle = stitle + ', time between frames: ' + str(time_between_frames) + 'ms'

    #            fig = pylab.figure(spot_type + ', ' + quantity + ', time between frames: ' + str(time_between_frames))
            fig = pylab.figure(ft)
            axes = fig.get_axes()
            if starved_only_bistatnorm:
                if '6h' in k2:
                    axes[position_subplot[spot_type][k2]].plot(populations['bistatnorm'][i][:,0], populations['bistatnorm'][i][:,1], 'k', linewidth = 2, linestyle = '--')
                    axes[position_subplot[spot_type][k2]].plot(populations['bistatnorm'][i][:,0], populations['bistatnorm'][i][:,2], 'k', linewidth = 2, linestyle = '--')
                elif '0h' in k2:
                    axes[position_subplot[spot_type][k2]].plot(populations['statnorm'][i][:,0], populations['statnorm'][i][:,1], 'k', linewidth = 2, linestyle = '--')
            else:
                axes[position_subplot[spot_type][k2]].plot(populations['bistatnorm'][i][:,0], populations['bistatnorm'][i][:,1], 'k', linewidth = 2, linestyle = '--')
                axes[position_subplot[spot_type][k2]].plot(populations['bistatnorm'][i][:,0], populations['bistatnorm'][i][:,2], 'k', linewidth = 2, linestyle = '--')

            pylab.savefig(basic_directory + central_directory + 'plots/fits_to_' + quantity + 's_per_cat_' + time_label + '/' + quantity + '_histograms_' + str(time_between_frames) + 'ms_binormal_fits_three_curves' + pd + day + '_' + string_now + '.png')

            pylab.savefig(basic_directory_paper_figs + '_' + spot_type + '/' + string_now + '_fits_to_' + quantity + 's_per_cat_histograms_' + str(time_between_frames) + 'ms_binormal_fits_three_curves' + pd + day + '.png')
            pylab.savefig(basic_directory_paper_figs + '_' + spot_type + '/' + string_now + '_fits_to_' + quantity + 's_per_cat_histograms_' + str(time_between_frames) + 'ms_binormal_fits_three_curves' + pd + day + '.svg')


    area_ordered = area.copy(deep = True)
    area.set_index(numpy.arange(0, len(area)), inplace = True) # to facilitate finding the rows where the columns need to be swapped
    area_ordered.set_index(numpy.arange(0, len(area)), inplace = True)
    area_ordered.loc[rows_to_swap, 'population_1_percent'] = area.loc[rows_to_swap, 'population_2_percent']
    area_ordered.loc[rows_to_swap, 'population_2_percent'] = area.loc[rows_to_swap, 'population_1_percent']

    area_ordered.to_pickle(basic_directory + central_directory + 'results/fits_to_' + quantity + 's_per_cat_' + time_label + '/' + quantity + '_histograms_' + str(time_between_frames) + 'ms_binormal_fits_three_curves' + '_ordered' + pd + string_now + '.pkl')
    area_ordered.to_csv(basic_directory + central_directory + 'results/fits_to_' + quantity + 's_per_cat_' + time_label + '/' + quantity + '_histograms_' + str(time_between_frames) + 'ms_binormal_fits_three_curves' + '_ordered' + pd + string_now + '.csv')

    numpy.save(basic_directory_paper_data + '_' + spot_type + '/' + string_now + '_imsd_fit_results_' + str(time_between_frames) + 'ms' + pd + day + '.npy', results2)

    return area, fits, area_ordered

def plot_populations(spot_type, quantity, time_label = 'tstart0p21sec_tend0p84sec', per_day = False):
    '''

    '''
    central_directory = load_all_results.select_specifications(spot_type)['central_directory']

    now = datetime.datetime.now()
    string_now = now.strftime("%Y%m%d")

    if per_day:
        pd = '_per_day_'
    else:
        pd = '_'
    area_ordered = pandas.read_pickle(basic_directory + central_directory + 'results/fits_to_' + quantity + 's_per_cat_' + time_label + '/' + quantity + '_histograms_0210ms_binormal_fits_three_curves_ordered' + pd + string_now + '.pkl')

    legend_elements = [Line2D([0], [0], marker = 'o', color = 'k', label = 'mobile mode', markersize = ms), Line2D([0], [0], marker = 'o', markerfacecolor = 'w', color = 'k', label = 'immobile mode', markersize = ms)]

    for i in area_ordered.index:
        t = area_ordered.loc[i, :]
        c = t.condition
        strain = load_all_results.read('strain', c, spot_type)
        starvation_time = load_all_results.read('starvation_time', c, spot_type) + 'h'
        time_between_frames = load_all_results.read('time_between_frames', c, spot_type)
        condition = load_all_results.read('condition', c, spot_type)
        figtitle = spot_type + '_' + quantity + '_mobility_populations_' + str(time_between_frames) + 'ms_' + quantity
        pylab.figure(figtitle, figsize = (7, 5))
        pylab.plot(positions[strain][starvation_time + condition], t.population_1_percent, 'o', color = colors[strain][starvation_time + condition], markerfacecolor = 'none', markersize = ms)
        pylab.plot(positions[strain][starvation_time + condition], t.population_2_percent, 'o', color = colors[strain][starvation_time + condition], markersize = ms)
        pylab.ylim(0, 100)
        pylab.xticks([1, 1.3, 1.8, 2.1], ['WT\n0h', 'WT\n6h', 'ΔpolyP\n0h', 'ΔpolyP\n6h'])
        pylab.xlim(0.9, 2.2)
        pylab.title(spot_type + ', ' + quantity + ',\nmobility_populations_' + str(time_between_frames) + 'ms_' + quantity)
        pylab.ylabel('%')

        pylab.legend(handles=legend_elements, loc=6, frameon = False)
        pylab.xticks([1, 1.2, 1.4, 1.6, 1.8, 2.0], ['WT\n0h', '\n6h lowN', '\n6h lowC', 'ΔpolyP\n0h', '\n6h lowN', '\n6h lowC'])

        pylab.savefig(basic_directory + central_directory + 'plots/fits_to_' + quantity + 's_per_cat_' + time_label + '/populations_from_' + quantity + '_histograms_' + str(time_between_frames) + 'ms_binormal_fits_three_curves' + pd + '_ordered_' + string_now + '.png', bbox_inches = 'tight')

    return area_ordered

def plot_bimodal_mean_values(spot_type = 'muNS', quantity = 'D_app'):

    f = pandas.read_pickle('/Volumes/GlennGouldMac/PolyP/data/_muNS/diameter7_minmass110_search_range4_memory20_stub_length20/results/fits_to_' + quantity + 's_per_cat_tstart0p21sec_tend0p84sec/' + quantity + '_histograms_0210ms_binormal_fit_ordered_20210615.pkl') #'_histograms_0210ms_binormal_fit.pkl')

    fig = pylab.figure(quantity + ', bimodal fit results')
    pylab.title(quantity + ', bimodal fit results')
    if quantity == 'D_app':
        ax = fig.gca()
        ax.set_yscale('log')
        ymin = 7e-4#5e-5
        ymax = 3e-2#5e-2
        yl = 'D_app (mum^2/s^α)'
    elif quantity == 'alpha':
        ymin = 0
        ymax = 0.8
        yl = 'α'
    for i in f.condition:
        if '210' in i:
            strain = load_all_results.read('strain', i, spot_type)
            starvation_time = load_all_results.read('starvation_time', i, spot_type) + 'h'
            condition = load_all_results.read('condition', i, spot_type)
            g = f[f.condition==i]
            if strain + starvation_time != 'bLR310h':
                pylab.plot(positions[strain][starvation_time+condition], g.mean_1, markershapes[starvation_time + condition], color = colors[strain][starvation_time + condition], markersize = ms, markeredgecolor = colors[strain][starvation_time + condition], markeredgewidth = 2, markerfacecolor = 'w')
            pylab.plot(positions[strain][starvation_time+condition], g.mean_2, markershapes[starvation_time + condition], color = colors[strain][starvation_time + condition], markersize = ms, markeredgecolor = colors[strain][starvation_time + condition], markeredgewidth = 2)

    #    pylab.xticks([1, 1.2, 1.4, 1.6, 1.8, 2.0], ['WT\n0h', 'WT\n6h lowN', 'WT\n6h lowC', 'ΔpolyP\n0h', 'ΔpolyP\n6h lowN', 'ΔpolyP\n6h lowC'])
    pylab.xticks([1, 1.2, 1.4, 1.6], ['WT\n0h', 'WT\n6h lowN', 'ΔpolyP\n0h', 'ΔpolyP\n6h lowN'])

    pylab.ylim(ymin,ymax)
    pylab.ylabel(yl)

    return f

def plot_all_steps(trajectories, spot_type, avoid = ['1000'], label = ''):

    #    typically_avoid = load_all_results.select_specifications(spot_type)['typically_avoid']
    search_range = load_all_results.select_specifications(spot_type)['search_range']
    central_directory = load_all_results.select_specifications(spot_type)['central_directory']

    now = datetime.datetime.now()
    string_now = now.strftime("%Y%m%d")
    string_now = label + string_now

    bins_steps = numpy.arange(0, 5, 0.1) * px_to_micron
    bins_loc_unc = numpy.arange(0, 0.11, 0.0005) * px_to_micron

    #    keys_of_interest = [x for x in trajectories.keys() if typically_avoid not in x]
    keys_of_interest = [x for x in trajectories.keys() if not any([y in x for y in avoid])]

    for i in keys_of_interest[:]:
        print(i)
        strain = load_all_results.read('strain', i, spot_type)
        starvation_time = load_all_results.read('starvation_time', i, spot_type) + 'h'
        condition = load_all_results.read('condition', i, spot_type)
        pylab.figure(i)
        pylab.title(i)
        steps = trajectories[i].previous_step_size.to_numpy(dtype = numpy.float64) * px_to_micron
        nparticles = len(set(trajectories[i].particle))
        print(len(steps))
        #steps = steps[~numpy.isnan(steps)]
        print(loc_uncertainties.max())
        weights = 100 * numpy.ones_like(steps) / float(len(steps))
        h_steps = pylab.hist(steps, bins = bins_steps, #density = True,
                             color = colors[strain][starvation_time + condition], alpha = 1,
                             weights = weights, label = 'step sizes')

        med_loc_unc = numpy.nanmedian(loc_uncertainties)
        med_step_size = numpy.nanmedian(steps)
        pylab.axvline(med_loc_unc, color = 'k', label = 'median loc. uncertainty:\n' + "{:.3f}".format(med_loc_unc) + ' mum (theoretical)')
        pylab.axvline(med_step_size, color = 'b', label = 'median step size: ' + "{:.3f}".format(med_step_size) + ' mum')
        pylab.axvline(search_range * px_to_micron, color = 'r', label = 'search range: ' +  "{:.3f}".format(search_range * px_to_micron) + ' mum')
        pylab.text(0.27, 1, 'steps: ' + str(len(steps)))
        pylab.text(0.27, 2.2, 'particles: ' + str(nparticles))
        pylab.ylim(0, 25)
        pylab.xlim(-0.01, 0.45)
        pylab.ylabel('% of steps')
        pylab.xlabel('distance (mum)')
        pylab.legend(frameon = False)

        if 'step_size_distributions' not in os.listdir(basic_directory + central_directory + 'plots/'):
            os.mkdir(basic_directory + '_' + spot_type + 'plots/step_size_distributions/')

        pylab.savefig(basic_directory + central_directory + 'plots/step_size_distributions/' + i + '_step_size_distribution_histogram_' + string_now + '.png', bbox_inches = 'tight')

    return trajectories

def plot_histogram_traj_quantity(spot_type, pooled_trajectories, traj_quantity, label = None, frames = None):

    now = datetime.datetime.now()
    string_now = now.strftime("%Y%m%d")

    central_directory = load_all_results.select_specifications(spot_type)['central_directory']

    for i in list(pooled_trajectories.keys())[:]:
        s = load_all_results.read('strain', i, spot_type)
        st = load_all_results.read('starvation_time', i, spot_type) + 'h'
        c = load_all_results.read('condition', i, spot_type)
        t = pooled_trajectories[i]
        fig = pylab.figure(i, linewidth=2)
        pylab.title(i)
        if spot_type == 'origins':
            if traj_quantity == 'snr':
                data = t.snr.to_numpy(dtype = numpy.float64)
                max_x = 60  # 60
                max_x_show = None #20 # max_x
                max_y_show = 20 # 35
                bins_now = numpy.arange(0, max_x, 1)
                pylab.xlabel('snr')

            elif traj_quantity == 'previous_step_size':
                data = t.previous_step_size.to_numpy(dtype = numpy.float64)
                data = data * px_to_micron * 1000
                max_x = 4.5 * px_to_micron * 1000#60  # 60
                max_x_show = 0.2 * 1000 #None #20 # max_x
                max_y_show = 30
                bins_now = numpy.arange(0, max_x, 0.1) * px_to_micron * 1000
    #            pylab.axvline(4 * px_to_micron, color = 'k', linestyle = '--', label = 'search range')
                pylab.xlabel('step size (nm)')

        elif spot_type == 'muNS':
            if traj_quantity == 'previous_step_size':
                data = t.previous_step_size.to_numpy(dtype = numpy.float64)
                data = data * px_to_micron * 1000
                max_x = 4.5 * px_to_micron * 1000#60  # 60
                max_x_show = 0.3 * 1000 #None #20 # max_x
                max_y_show = 15
                bins_now = numpy.arange(0, max_x, 0.1) * px_to_micron * 1000
                        #            pylab.axvline(4 * px_to_micron, color = 'k', linestyle = '--', label = 'search range')
                pylab.xlabel('step size (nm)')


        m = numpy.nanmedian(data)
        weights = 100 * numpy.ones_like(data) / float(len(data))
        N = str(len(data))
        pylab.axvline(m, color = 'k', alpha = 1, label = 'median: ' + str(round(m, 0)) + ' nm', linestyle = '--')
        pylab.hist(data, bins = bins_now, weights = weights, alpha = 1, label = 'N = ' + N, color = colors[s][st + c], hatch=htc[st])

        if st == '0h':
            pylab.hist(data, bins = bins_now, weights = weights, histtype = 'step', alpha = 1, color = colors[s][st + c], lw = 2)


        pylab.ylim(0, max_y_show)
        pylab.xlim(0, max_x_show)
        pylab.legend(loc=1, frameon = False)
        pylab.ylabel('% of particles')

        handles, labels = pylab.gca().get_legend_handles_labels()

        i = i.strip('/')

        if not isinstance(frames, list):
            framelabel = '_all_frames_'
        figure_filename = basic_directory + central_directory + 'plots/' + traj_quantity + '/' + string_now + '_' + i + '_' + traj_quantity + framelabel + 'histogram_'
        if isinstance(label, str):
            figure_filename = figure_filename + label

        pylab.savefig(figure_filename + '.png', bbox_inches = 'tight')

        pylab.savefig(basic_directory_paper_figs + '_' + spot_type + '/' + traj_quantity + '/' + string_now + '_' + traj_quantity + '_' + i + '.svg')
        pylab.savefig(basic_directory_paper_figs + '_' + spot_type + '/' + traj_quantity + '/' + string_now + '_' + traj_quantity + '_' + i + '.png')

        print(figure_filename)

    numpy.save(basic_directory_paper_data + '_' + spot_type + '/' + traj_quantity + '/' + string_now + '_trajectories.npy', pooled_trajectories)

    return pooled_trajectories

def plot_mean_quantity1_vs_binned_quantity2(spot_type, results, quantity1, quantity2, bin_limits = numpy.arange(0, 1.5, 0.25), avoid = ['0030', '0090'], ignore_timelag = False, below_size_limit = True):
    '''
    '''
    central_directory = load_all_results.select_specifications(spot_type)['central_directory']

    keys_of_interest = [x for x in results.keys() if not any([y in x for y in avoid])]

    now = datetime.datetime.now()
    string_now = now.strftime("%Y%m%d")

    r = {}
    bin_info = {}
    if quantity2 == 'particle_size':
        bin_limits = numpy.linspace(7, 14, 9)

    for i in keys_of_interest[:]:
        print(i)
        bin_info[i] = {}
        figtitle = spot_type + '_' + quantity1 + '_vs_' + quantity2
        if not ignore_timelag:
            time_between_frames = load_all_results.read('time_between_frames', i, spot_type) + 'ms'
            figtitle = time_between_frames + '_' + figtitle
        if below_size_limit:
            figtitle = figtitle + '\n width < 450 nm'
        pylab.figure(figtitle, figsize = (8,5))
        pylab.title(figtitle)

        strain = load_all_results.read('strain', i, spot_type)
        starvation_time = load_all_results.read('starvation_time', i, spot_type) + 'h'
        condition = load_all_results.read('condition', i, spot_type)

        r[i] = results[i].copy(deep = True)
        if below_size_limit:
    #            r_now = r[i][r[i].below_diffraction_limit]
            r_now = r[i][r[i].below_diffraction_limit == 1.0]
        else:
            r_now = r[i]

        if quantity2 == 'particle_size':
            r_now['particle_size'] = r_now['average_starting_magnitude']**(1./3)
    #            l = r[i].below_diffraction_limit
    #            l = list(l)
    #            a = numpy.where(~numpy.isnan(l))[0] # all entries where the value for 'below_size_limit' is either True or False, but not nan
    #            r_now = r[i].iloc[a, :]
    #            r_now = r_now[r_now.below_diffraction_limit] # all entries where the spots are smaller than the diffraction limit (? how is that possible and why are we excluding these?) -> indeed criterion judged too strict, using a higher value now
            r_now = r_now[r_now.particle_size!=0]  # omit the 0 entries THIS SHOULD BECOME OBSOLETE ONCE I HAVE FITTED ALL MOVIES
            r[i] = r_now.copy(deep = True)

        for j, k in enumerate(bin_limits):
            bin_info[i][j] = {}
            if j < len(bin_limits) - 1:
                results_now = r[i][(bin_limits[j] <= r[i].loc[:,quantity2]) & (r[i].loc[:,quantity2] < bin_limits[j+1])]
                bin_info[i][j]['limits'] = [bin_limits[j], bin_limits[j+1]]

            elif j == len(bin_limits) - 1:
                results_now = r[i][r[i].loc[:,quantity2] > bin_limits[j]]
                bin_info[i][j]['limits'] = [bin_limits[j], numpy.inf]

            data1 = results_now.loc[:,quantity1].to_numpy(dtype = numpy.float64)
            data1 = data1[~numpy.isnan(data1)]
    #            pylab.plot(j, numpy.median(data1), '--', color = colors[strain][starvation_time + condition], linewidth = 50)
            print('bin' + str(j))
            print(len(data1))
            bin_info[i][j]['N'] = len(data1)
            pylab.errorbar(k, numpy.median(data1), xerr = 0, yerr = numpy.std(data1) / numpy.sqrt(len(data1)), fmt =  markershapes[starvation_time + condition], ls = '--', color = colors[strain][starvation_time + condition], alpha = 0.7, markersize = ms, markeredgecolor = 'k', markeredgewidth = 1)
    #            pylab.text(k, numpy.median(data1) + 0.0015, str(bin_occupants[i][j]))

    #            print(len(data1))

        pylab.xticks(bin_limits[::2])

        if quantity1 == 'alpha':
            pylab.ylim(0.3, 0.7)

        if quantity2 == 'particle_size':
            pylab.xlabel(quantity2 + ' (arb)')
        else:
            pylab.xlabel(quantity2)

        if quantity1 == 'D_app':
            pylab.ylabel(quantity1 + ' median (mum^2/s^α)')
        else:
            pylab.ylabel(quantity1 + ' median')

        figure_filename = basic_directory + central_directory + 'plots/' + figtitle + '_' + string_now + '.png'

        figure_filename = basic_directory_paper_figs + '_' + spot_type + '/' + quantity1 + '_vs_' + quantity2 + '_bins/' + string_now + '_' + figtitle + '.png'

        figure_filename = basic_directory_paper_figs + '_' + spot_type + '/' + quantity1 + '_vs_' + quantity2 + '_bins/' + string_now + '_' + figtitle + '.svg'

    #        if isinstance(label, str):
    #            figure_filename = figure_filename + '_' + label
    #            figure_filename = figure_filename + '.png'
        pylab.savefig(figure_filename, bbox_inches = 'tight')

        numpy.save(basic_directory_paper_data + '_' + spot_type + '/' + quantity1 + '_vs_' + quantity2 + '_bins/' + string_now + '_' + quantity2 + '_bin_info' + '.npy', bin_info)

        numpy.save(basic_directory_paper_data + '_' + spot_type + '/' + quantity1 + '_vs_' + quantity2 + '_bins/' + string_now + '_input_to_' + figtitle + '.npy', r)

    return bin_info

def plot_quantity1_vs_quantity2(spot_type, results, quantity1, quantity2, avoid = ['lowC'], ignore_timelag = False, below_size_limit = True, loglog = True):
    '''
    '''
    central_directory = load_all_results.select_specifications(spot_type)['central_directory']

    now = datetime.datetime.now()
    string_now = now.strftime("%Y%m%d")
    if below_size_limit:
        string_now = 'below_size_limit_' + string_now

    for i in results.keys():
        r = results[i].copy()
        if below_size_limit:
    #            r = r[r.below_diffraction_limit]
            r = r[r.below_diffraction_limit == 1]

        fig = pylab.figure(i)
        if loglog:
            ax = fig.gca()
            ax.set_xscale('log')
            ax.set_yscale('log')
        if quantity1 == 'particle_size':
            datay = r.loc[:, 'average_starting_magnitude'].to_numpy(dtype = numpy.float64)
            datay[numpy.greater(datay, numpy.zeros_like(datay))]
            datay = datay**(1./3)
        else:
            datay = r.loc[:, quantity1].to_numpy(dtype = numpy.float64)

        if quantity2 == 'particle_size':
            datax = r.loc[:, 'average_starting_magnitude'].to_numpy(dtype = numpy.float64)
            datax[numpy.greater(datax, numpy.zeros_like(datax))]
            datax = datax**(1./3)

        else:
            datax = r.loc[:, quantity2].to_numpy(dtype = numpy.float64)

        pylab.plot(datax, datay, 'o', alpha = 0.025)

        if quantity1 == 'D_app':
            pylab.text(2e1, 5e-2, 'N = ' + str(len(datax)))
            pylab.ylim(5e-5, 5e-1)

        elif quantity1 == 'alpha':
            pylab.text(2e1, 0.7, 'N = ' + str(len(datax)))
            pylab.ylim(0, 1.2)

        pylab.xlim(5e0, 2e1)
        pylab.xlabel(quantity2)
        pylab.ylabel(quantity1)

        pylab.title(i)
        pylab.savefig(basic_directory + '_muNS/diameter7_minmass110_search_range4_memory20_stub_length20/plots/' + quantity1 + '_vs_' + quantity2 + '/' + i + string_now + '.png')

    return results

def plot_Dapp_vs_alpha(spot_type, results, per_day=False, load = False, pool = False, avoid = ['lowC', '1000ms'], ignore_timelag = False):
    '''

    '''

    if load:
        r = load_all_results.load_all_results('muNS', 'results_per_particle')

    if pool:
        results = concatenate('muNS', r, per_day=False, load=False, ax=0)

    keys_of_interest = [x for x in results.keys() if not any([y in x for y in avoid])]

    central_directory = load_all_results.select_specifications(spot_type)['central_directory']

    nparticles = {}
    nparticles['bLR31_0h'] = 0
    nparticles['bLR32_0h'] = 0
    nparticles['bLR31_6h'] = 0
    nparticles['bLR32_6h'] = 0

    ### first, find the indicative line fits ###

    f = {}

    results_now = {}

    for i in keys_of_interest[:]:
        results_now[i] = results[i][results[i].alpha > 0]
        results_now[i] = results[i][results[i].alpha > 0]
        strain = load_all_results.read('strain', i, spot_type)
        st = load_all_results.read('starvation_time', i, spot_type) + 'h'
        if not ignore_timelag:
            time_between_frames = load_all_results.read('time_between_frames', i, spot_type)
        else:
            time_between_frames = ''
        alphas = results_now[i].alpha.to_numpy(dtype = numpy.float64)
        D_apps = results_now[i].D_app.to_numpy(dtype = numpy.float64)
        f[i] = numpy.polyfit(alphas, numpy.log10(D_apps), 1)

    ### then plot all indicative lines, and each condition one by one ###
    xs = numpy.linspace(0, 1.25, 100)

    for i in keys_of_interest[:]:
        results_now[i] = results[i][results[i].alpha > 0]
        strain = load_all_results.read('strain', i, spot_type)
        st = load_all_results.read('starvation_time', i, spot_type) + 'h'
        condition = load_all_results.read('condition', i, spot_type)
        if not ignore_timelag:
            time_between_frames = load_all_results.read('time_between_frames', i, spot_type)
        else:
            time_between_frames = ''
    #        n = len(results[i])
        figtitle = i #time_between_frames #strain + '_' + st #'all data'
    #        nparticles[figtitle] = nparticles[figtitle] + n
        nparticles = len(results[i])
        fig = pylab.figure(figtitle)
        ax = fig.gca()
        pylab.title(figtitle)
        ax.set_yscale('log')
        pylab.plot(results_now[i].alpha, results_now[i].D_app, 'o', alpha = 0.05, color = colors[strain][st + condition], label = '')  #alpha = 500./nparticles
        pylab.plot(xs, 10**(f[i][0] * xs + f[i][1]), '-', color = 'white')
        pylab.plot(xs, 10**(f[i][0] * xs + f[i][1]), '--', color = colors[strain][st + condition], label = i)
        pylab.text(0.85, 4e-1, 'N = ' + str(nparticles))
        alphas = results_now[i].alpha.to_numpy(dtype = numpy.float64)
        D_apps = results_now[i].D_app.to_numpy(dtype = numpy.float64)
        pylab.xlabel('alpha')
        pylab.ylabel('D_app')
        pylab.xlim(-0.05, 1.25)
        weights = 100 * numpy.ones_like(alphas) / float(len(alphas))
        fig_h = pylab.figure(figtitle + '_heatmap')
        heatmap, xedges, yedges, z = pylab.hist2d(results_now[i].alpha.to_numpy(dtype = numpy.float64), results_now[i].D_app.to_numpy(dtype = numpy.float64), bins=[numpy.linspace(0, 1.25, 51), numpy.logspace(-5, 0, num = 51)], weights = weights)
        pylab.close(figtitle + '_heatmap')
        pylab.figure(figtitle + '_heatmap')
        pylab.imshow(heatmap.T, origin = 'lower', cmap = 'plasma')
        if spot_type == 'muNS':
            pylab.ylim(1e-5, 1e0)  # (1e-5, 1e0)
        elif spot_type == 'origins':
            pylab.ylim(1e-6, 1e-1)
        xl = [str(x) for x in xedges]
        xl = xl[::10]
        pylab.xticks(numpy.arange(0, 51, 10), labels = xl)
        yl = ["{:.0e}".format(y) for y in yedges]
        yl = yl[::10]
        pylab.yticks(numpy.arange(0, 51, 10), labels = yl)
        pylab.title(figtitle + '\nN = ' + str(nparticles))
        pylab.ylabel('D_app (mum^2/s^α)')
        pylab.xlabel('α')

        now = datetime.datetime.now()
        string_now = now.strftime("%Y%m%d")

        pylab.savefig(basic_directory + central_directory + 'plots/D_app_vs_alpha/' + string_now + '_heatmap_' + i + '.png')

        pylab.savefig(basic_directory_paper_figs + '_' + spot_type + '/D_app_vs_alpha/' + string_now + '_heatmap_' + i + '.png')
        pylab.savefig(basic_directory_paper_figs + '_' + spot_type + '/D_app_vs_alpha/' + string_now + '_heatmap_' + i + '.svg')

    numpy.save(basic_directory_paper_data + '_' + spot_type + '/D_app_vs_alpha/' + string_now + '_heatmap.npy', heatmap)
    numpy.save(basic_directory_paper_data + '_' + spot_type + '/D_app_vs_alpha/' + string_now + '_results.npy', results_now)

    return f, heatmap, xedges, yedges, alphas, D_apps

units = {}
units['alpha'] = ''
units['D_app'] = ' (' + r'$\mu m ^{2} / s^{\alpha}$' + ')'
units['particle_size'] = ' (arb)'
units['average_starting_offset'] = ''

def heatmap_quantity1_vs_quantity2(spot_type, results, quantity1, quantity2, per_day=False, avoid = [], below_size_limit_only=True, ignore_timelag = False, loglog=False):
    '''

    '''

    number_of_bins = 51 # (same for both quantities but is it necessary?)
    mins = {}
    maxs = {}
    mins['alpha'] = 0
    mins['particle_size'] = 6
    mins['D_app'] = 1e-4 #1e-5
    maxs['alpha'] = 1.25
    maxs['particle_size'] = 16
    maxs['D_app'] = 1e-1#1e0
    mins['average_starting_offset'] = 50
    maxs['average_starting_offset'] = 200

    keys_of_interest = [x for x in results.keys() if not any([y in x for y in avoid])]

    central_directory = load_all_results.select_specifications(spot_type)['central_directory']

    nparticles = {}
    nparticles['bLR31_0h'] = 0
    nparticles['bLR32_0h'] = 0
    nparticles['bLR31_6h'] = 0
    nparticles['bLR32_6h'] = 0

    ### first, find the indicative line fits ###

    f = {}

    results_now = {}

    for i in keys_of_interest[:]:
        if below_size_limit_only:
            results_now[i] = results[i][results[i].below_diffraction_limit > 0]
        if quantity1 == 'alpha':
            results_now[i] = results[i][results[i].alpha > 0]
            yvalues = results_now[i].alpha.to_numpy(dtype = numpy.float64)
            ymin = mins['alpha']
            ymax = maxs['alpha']
            bins_y = numpy.linspace(ymin, ymax, number_of_bins)

        elif quantity1 == 'D_app':
            results_now[i] = results[i][results[i].D_app > 0]
            yvalues = results_now[i].D_app.to_numpy(dtype = numpy.float64)
            ymin = mins['D_app']
            ymax = maxs['D_app']
            bins_y = numpy.logspace(numpy.log10(ymin), numpy.log10(ymax), num = number_of_bins)

        elif quantity1 == 'particle_size':
            results_now[i] = results[i][results[i].average_starting_magnitude > 0]
            if spot_type == 'muNS_lowC':
                results_now[i]['avg_start_mag_cbrt'] = numpy.cbrt(results_now[i].average_starting_magnitude)
                results_now[i] = results_now[i][results_now[i].avg_start_mag_cbrt > 6]
                results_now[i] = results_now[i][results_now[i].avg_start_mag_cbrt < 18]
                yvalues = results_now[i].avg_start_mag_cbrt.to_numpy(dtype = numpy.float64)
            else:
                yvalues = results_now[i].average_starting_magnitude.to_numpy(dtype = numpy.float64)
                yvalues = yvalues ** (1./3)

            ymin = mins['particle_size']
            ymax = maxs['particle_size']
            bins_y = numpy.linspace(ymin, ymax, number_of_bins)

        if quantity2 == 'alpha':
    #            results_now[i] = results[i][results[i].alpha > 0]
            xvalues = results_now[i].alpha.to_numpy(dtype = numpy.float64)
            xmin = mins['alpha']
            xmax = maxs['alpha']
            xs = numpy.linspace(xmin, xmax, number_of_bins) # the continuous vars
            bins_x = numpy.linspace(xmin, xmax, number_of_bins)

        elif quantity2 == 'D_app':
    #            results_now[i] = results[i][results[i].D_app > 0]
            xvalues = results_now[i].D_app.to_numpy(dtype = numpy.float64)
            if spot_type == 'muNS_lowC':
                xmin = 10**(-5)
            else:
                xmin = mins['D_app']
            xmax = maxs['D_app']
            xs = numpy.logspace(numpy.log10(xmin), numpy.log10(xmax), number_of_bins) # the continuous vars
            bins_x = numpy.logspace(numpy.log10(xmin), numpy.log10(xmax), number_of_bins)

        elif quantity2 == 'particle_size':
    #            results_now[i] = results[i][results[i].magnitude > 0]
            xvalues = results_now[i].average_starting_magnitude.to_numpy(dtype = numpy.float64)
            xvalues = xvalues ** (1./3)
            xmin = mins['particle_size']
            xmax = maxs['particle_size']
            xs = numpy.linspace(xmin, xmax, number_of_bins) # the continuous vars
            print(xs)
            bins_x = numpy.linspace(xmin, xmax, number_of_bins)

        elif quantity2 == 'average_starting_offset':
            xvalues = results_now[i].average_starting_offset.to_numpy(dtype = numpy.float64)
            xmin = mins['average_starting_offset']
            xmax = maxs['average_starting_offset']
            xs = numpy.linspace(xmin, xmax, number_of_bins) # the continuous vars
            bins_x = numpy.linspace(xmin, xmax, number_of_bins)

        nparticles[i] = len(results_now[i])
        strain = load_all_results.read('strain', i, spot_type)
        st = load_all_results.read('starvation_time', i, spot_type) + 'h'
        condition = load_all_results.read('condition', i, spot_type)
        if not ignore_timelag:
            time_between_frames = load_all_results.read('time_between_frames', i, spot_type)
        else:
            time_between_frames = ''

    #        if loglog:
    #            f[i] = numpy.polyfit(numpy.log(xvalues), numpy.log10(yvalues), 1)
    #        elif quantity2 == 'D_app':
    #            f[i] = numpy.polyfit(xvalues, numpy.log10(yvalues), 1)
    #        elif quantity1 == 'D_app':
    #            f[i] = numpy.polyfit(numpy.log10(xvalues), yvalues, 1)

    ### then plot all indicative lines, and each condition one by one ###

    #    for i in keys_of_interest[:]:
    #        results_now[i] = results[i][results[i].alpha > 0]
    #        strain = load_all_results.read('strain', i, spot_type)
    #        st = load_all_results.read('starvation_time', i, spot_type) + 'h'
    #        condition = load_all_results.read('condition', i, spot_type)
    #        time_between_frames = load_all_results.read('time_between_frames', i, spot_type)
    #        #        n = len(results[i])
    #        figtitle = i #time_between_frames #strain + '_' + st #'all data'
                #        nparticles[figtitle] = nparticles[figtitle] + n
    #        nparticles = len(results_now[i])

    #        fig = pylab.figure(figtitle)
    #        ax = fig.gca()
    #        pylab.title(figtitle)
    #        if quantity1 == 'D_app':
    #            ax.set_yscale('log')
    #        pylab.plot(xvalues, yvalues, 'o', alpha = 0.05, color = colors[strain][st + condition], label = '')  #alpha = 500./nparticles
    #        pylab.plot(xs, 10**(f[i][0] * xs + f[i][1]), '-', color = 'white')
    #        pylab.plot(xs, 10**(f[i][0] * xs + f[i][1]), '--', color = colors[strain][st + condition], label = i)
    #        pylab.text(0.85, 4e-1, 'N = ' + str(nparticles))
    #        alphas = results_now[i].alpha.to_numpy(dtype = numpy.float64)
    #        D_apps = results_now[i].D_app.to_numpy(dtype = numpy.float64)
    #        pylab.xlabel(quantity2)
    #        pylab.ylabel(quantity1)
    #        pylab.xlim(xmin, xmax)

        weights = 100 * numpy.ones_like(xvalues) / float(len(xvalues))
        fig = pylab.gcf()
        # print(fig.canvas.get_window_title())

    #        fig_h = pylab.figure(i + '_heatmap_' + quantity1 + '_vs_' + quantity2, figsize = (5, 5))
        pylab.close(i + '_heatmap_' + quantity1 + '_vs_' + quantity2)
        pylab.figure(i + '_heatmap_' + quantity1 + '_vs_' + quantity2, figsize = (5,5))
        fig = pylab.gcf()
        # print(fig.canvas.get_window_title())

        X,Y = numpy.meshgrid(bins_x, bins_y, indexing='ij')
        fig = pylab.gcf()
        # print(fig.canvas.get_window_title())

        heatmap, xedges, yedges, z = pylab.hist2d(xvalues, yvalues, bins=[bins_x, bins_y], weights = weights)
        fig = pylab.gcf()
        # print(fig.canvas.get_window_title())
        im = pylab.pcolormesh(X,Y,heatmap, cmap = 'plasma')
        fig = pylab.gcf()
        # print(fig.canvas.get_window_title())

        if quantity1 == 'D_app':
            pylab.yscale('log')
        elif quantity2 == 'D_app':
            pylab.xscale('log')


    #        pylab.figure(i)
    #        pylab.imshow(heatmap.T, origin = 'lower', cmap = 'plasma')

    #        if spot_type == 'muNS':
    #            pylab.ylim(1e-3, 1e0)  # (1e-5, 1e0)
    #        elif spot_type == 'origins':
    #            pylab.ylim(1e-6, 1e-1)
    #        xl = [str(x) for x in xedges]

    #        if quantity2 == 'D_app':
    #            pylab.xticks(numpy.arange(0, number_of_bins, 10), labels = ["{:.0e}".format(x) for x in  bins_x[::10]])
    #        else:
    #            pylab.xticks(numpy.arange(0, number_of_bins, 10), labels = [str(x) for x in  bins_x[::10]])

    #        print(bins_x)
    #        yl = [str(y) for y in yedges]
    #        yl = ["{:.1e}".format(y) for y in yedges]

    #        if quantity1 == 'D_app':
    #            pylab.yticks(numpy.arange(0, number_of_bins, 10), labels = ["{:.0e}".format(y) for y in  bins_y[::10]])
    #        else:
    #            pylab.yticks(numpy.arange(0, number_of_bins, 10), labels = [str(y) for y in  bins_y[::10]])

    #        yl = yl[::10]
    #        pylab.yticks(numpy.arange(ymin, ymax, 10), labels = yl)
        pylab.title(i + ': ' + quantity1 + ' vs ' + quantity2 + '\nN = ' + str(nparticles[i]))

        pylab.ylabel(quantity1 + units[quantity1])
        pylab.xlabel(quantity2 + units[quantity2])

    #        pylab.yscale('log')

        now = datetime.datetime.now()
        string_now = now.strftime("%Y%m%d")

        os.makedirs(basic_directory + central_directory + 'plots/' + quantity1 + '_vs_' + quantity2, exist_ok=True)
        pylab.savefig(basic_directory + central_directory + 'plots/' + quantity1 + '_vs_' + quantity2 + '/' + string_now + '_heatmap_' + i.strip('/') + '.png')

        os.makedirs(basic_directory_paper_figs + '_' + spot_type + '/' + quantity1 + '_vs_' + quantity2, exist_ok=True)
        pylab.savefig(basic_directory_paper_figs + '_' + spot_type + '/' + quantity1 + '_vs_' + quantity2 + '/' + string_now + '_heatmap_' + i.strip('/') + '.png')
        pylab.savefig(basic_directory_paper_figs + '_' + spot_type + '/' + quantity1 + '_vs_' + quantity2 + '/' + string_now + '_heatmap_' + i.strip('/') + '.svg')

    os.makedirs(basic_directory_paper_data + '_' + spot_type + '/' + quantity1 + '_vs_' + quantity2, exist_ok=True)
    numpy.save(basic_directory_paper_data + '_' + spot_type + '/' + quantity1 + '_vs_' + quantity2 + '/' + string_now + '_heatmap.npy', heatmap)
    numpy.save(basic_directory_paper_data + '_' + spot_type + '/' + quantity1 + '_vs_' + quantity2 + '/' + string_now + '_results.npy', results_now)

    return xvalues, yvalues, xedges, yedges

def histogram_imsd_value(spot_type, imsds, timepoint, avoid = None, ignore_timelag = False, label = '0210ms', fit = True, color_by_2s = False, omit_in_plot_fit = [], fit_unimodal = ['WT_0h_0210ms']):

    '''
    '''

    if isinstance(avoid, list):
        keys_of_interest = [x for x in imsds.keys() if not any([y in x for y in avoid])]
    else:
        keys_of_interest = [x for x in imsds.keys()]

    # if color_by_2s:
    #     mode_occupants = numpy.load('/Volumes/GlennGouldMac/PolyP/data/_muNS/diameter7_minmass110_search_range4_memory20_stub_length20/results/20210726_mode_occupants_from_msd_at_2p1s.npy', allow_pickle=True).item()

    central_directory = load_all_results.select_specifications(spot_type)['central_directory']

    imsd_values = {}
    imsd_values[str(timepoint)] = {}
    midpoints_now = {}
    f = {}
    intersection = {}
    y_curve_1 = {}
    y_curve_2 = {}
    occupants = {}
    occupants['1'] = {}
    occupants['2'] = {}
    mode_occupants2 = {}
    mode_occupants2['1'] = {}
    mode_occupants2['2'] = {}

    for i in keys_of_interest[:]:
        imsd_values[str(timepoint)][i] = {}

    xmin = 2e-4#5e-5
    xmax = 1e0#2e1
    bins_now = numpy.logspace(numpy.log10(xmin), numpy.log10(xmax), 50)

    figtitle = 'msd at ' + str(timepoint) + 's' + ', ' + label

    fig = pylab.figure(figtitle, figsize = (7.5, 7))
    fig.suptitle(figtitle)
    gs = GridSpec(4, 10, figure=fig)
    ax = {}
    fig.subplots_adjust(top=0.9)
    pylab.subplots_adjust(hspace=0.08) #0.5, 0.08

    h = {} # this dictionary will include the histograms for each condition

    if fit:
        fits = {} # if fit = True, this dictionary will include the fit results. I am prepared to do normal and binormal fits.
        fits['statnorm'] = {}
        fits['bistatnorm'] = {}
        initial_guess = {}
        initial_guess['statnorm'] = {}
        initial_guess['statnorm']['bLR31_0h_0210ms'] = [5.04814335e-03,
                                                        2.09465631e-01,
                                                        1.40776453e+01]

        initial_guess['bistatnorm'] = {}
        initial_guess['bistatnorm']['bLR31_0h_0210ms'] = [5.04814335e-03, 2e-2,
                                                          #6.66553900e-03,
                                                          2.09465631e-01, 4.77246568e-01,
         1.40776453e+01, 9.69407573e+00]
        initial_guess['bistatnorm']['bLR31_6h_0210ms_lowN'] = [5e-3, 4e-2, 4.3e-1, 2.9e-1, 6.7e+0, 1.6e1]
        initial_guess['bistatnorm']['bLR31_6h_0210ms_lowC'] = [5e-3, 4e-2, 4.3e-1, 2.9e-1, 6.7e+0, 1.6e1]
        initial_guess['bistatnorm']['bLR32_0h_0210ms'] = [5.95105685e-03, 3.88602447e-02, 3.31515599e-01, 3.20277638e-01,
         1.83796557e+01, 4.53339812e+00]
        initial_guess['bistatnorm']['bLR32_6h_0210ms_lowN'] = [1.2e-2, 1e-1, 4.3e-1, 2.9e-1, 6.7e+0, 1.6e1]
        initial_guess['bistatnorm']['bLR32_6h_0210ms_lowC'] = [1.2e-2, 1e-1, 4.3e-1, 2.9e-1, 6.7e+0, 1.6e1]

        fit_results = pandas.DataFrame(columns = ['condition', 'function', 'mean_1', 'mean_2', 'sigma_1','sigma_2', 'amplitude_1', 'amplitude_2', 'mean_fit_unc_1',  'mean_fit_unc_2', 'sigma_fit_unc_1', 'sigma_fit_unc_2', 'amplitude_fit_unc_1',   'amplitude_fit_unc_2'])

    for i in keys_of_interest[:]:
        midpoints_now[i] = []
        strain = load_all_results.read('strain', i, spot_type)
        st = load_all_results.read('starvation_time', i, spot_type) + 'h'
        condition = load_all_results.read('condition', i, spot_type)
        label = strain + '_' + st + '_' + condition

        k = strain + '_' + st
        if st != '0h':
            k = k + '_' + condition
        j = position_subplot[spot_type][k]

        imsd_values[str(timepoint)][i] = imsds[i][(imsds[i].index >= timepoint)].iloc[0]
        actual_time = imsd_values[str(timepoint)][i].name
        print(i)
        print(actual_time)

        nparticles = len(imsds[i].columns)
        weights = 100 * numpy.ones_like(imsd_values[str(timepoint)][i]) / nparticles

        for k, l in enumerate(bins_now):
            if k < len(bins_now) - 1:
                c = numpy.linspace(bins_now[k], bins_now[k+1], num=3)
                midpoints_now[i].append(c[1])

    #     ax[j] = fig.add_subplot(gs[j:j+1, :8])
    #     ax[j].set_xscale('log')
        h[i] = pylab.hist(imsd_values[str(timepoint)][i], bins = bins_now, weights = weights, color = colors[strain][st + condition], hatch=htc[st])
        if color_by_2s:
    #         mode_occupants2['1'][i] = [x for x in mode_occupants['1'][i] if x in list(imsd_values[str(timepoint)][i].index)]

    #         r_now = imsd_values[str(timepoint)][i].loc[mode_occupants2['1'][i]]
    #         weights2 = 100 * numpy.ones_like(r_now) / nparticles
    #         ax[j].hist(r_now, bins = bins_now, weights = weights2, color = 'k', hatch=htc[st], alpha = 0.5)
    # #        ax[j].plot(midpoints_now[i], h[i][0], 'k.')
            f[i] = numpy.zeros([len(midpoints_now[i]),2])
            f[i][:,0] = midpoints_now[i]
            f[i][:,1] = h[i][0]

    #         if st == '0h':
    #             ax[j].hist(imsd_values[str(timepoint)][i], bins = bins_now, weights = weights, histtype='step', color = colors[strain][st + condition], alpha = 1, lw = 2)

    #         if j < 3:
    #             ax[j].set_xticks([])

    #         pylab.xlim(xmin, xmax)
    #         pylab.ylim(0, 10)
    #         pylab.xlabel('msd at t = ' + str(timepoint) + 's ' + r'$(\mu m ^{2})$')
    #         fig.text(0.04, 0.5, '% of particles', va='center', rotation='vertical')
    #         if i == 'bLR32_6h_0210ms_lowC':
    #             ax[j].set_yticks([0, 2])
    #             ax[j].set_ylim(0, 3)
    #             ax[j].text(0.1, 3*3/5, 'N = ' + str(nparticles))
    #         else:
    #             ax[j].set_yticks([0, 3])
    #             ax[j].set_ylim(0, 5)
    #             ax[j].text(0.1, 3, 'N = ' + str(nparticles))

            if fit:
                fits['bistatnorm'][i] = scipy.optimize.curve_fit(bistatnorm_log, f[i][:,0], f[i][:,1], p0 = initial_guess['bistatnorm'][i])

                y_curve_1[i] = statnorm_log(f[i][:,0], fits['bistatnorm'][i][0][0], fits['bistatnorm'][i][0][2], fits['bistatnorm'][i][0][4])
                y_curve_2[i] = statnorm_log(f[i][:,0], fits['bistatnorm'][i][0][1], fits['bistatnorm'][i][0][3], fits['bistatnorm'][i][0][5])
                intersection[i] = intersect.intersection(f[i][:,0], y_curve_1[i], f[i][:,0], y_curve_2[i])

                if len(intersection[i][0]) > 0:
                    print("I", i)
                    ip = intersection[i][0].min()
                    occupants['1'][i] = list(imsd_values[str(timepoint)][i][imsd_values[str(timepoint)][i] <= ip].index)
                    occupants['2'][i] = list(imsd_values[str(timepoint)][i][imsd_values[str(timepoint)][i] > ip].index)
                    occupancy_1 = len(occupants['1'][i])
                    occupancy_2 = len(occupants['2'][i])

    #                occupancy_1 = len(numpy.where(imsd_values[str(timepoint)][i] <= ip)[0])
    #                occupancy_2 = len(numpy.where(imsd_values[str(timepoint)][i] > ip)[0])
            else:
                ip = numpy.nan
                occupants['1'][i] = list(imsd_values[str(timepoint)][i].index)
                occupants['2'][i] = []
                occupancy_1 = len(imsd_values[str(timepoint)])
                occupancy_2 = 0

    #         occupancy_total = len(imsd_values[str(timepoint)][i])
    #         occupancy_1 = 100 * occupancy_1 / occupancy_total
    #         occupancy_2 = 100 * occupancy_2 / occupancy_total

    #         d_bi = [[i, 'bistatnorm', fits['bistatnorm'][i][0][0], fits['bistatnorm'][i][0][1], fits['bistatnorm'][i][0][2], fits['bistatnorm'][i][0][3], fits['bistatnorm'][i][0][4], fits['bistatnorm'][i][0][5], numpy.sqrt(numpy.diag(fits['bistatnorm'][i][1]))[0], numpy.sqrt(numpy.diag(fits['bistatnorm'][i][1]))[1], numpy.sqrt(numpy.diag(fits['bistatnorm'][i][1]))[2], numpy.sqrt(numpy.diag(fits['bistatnorm'][i][1]))[3], numpy.sqrt(numpy.diag(fits['bistatnorm'][i][1]))[4], numpy.sqrt(numpy.diag(fits['bistatnorm'][i][1]))[5], occupancy_1, occupancy_2, ip]]

    #         little_answer_bistatnorm =  pandas.DataFrame(data = d_bi, columns = ['condition', 'function', 'mean_1', 'mean_2', 'sigma_1','sigma_2', 'amplitude_1', 'amplitude_2', 'mean_fit_unc_1',  'mean_fit_unc_2', 'sigma_fit_unc_1', 'sigma_fit_unc_2', 'amplitude_fit_unc_1',   'amplitude_fit_unc_2', 'percent_population_1', 'percent_population_2', 'mode_intersection_point'])

    #         fit_results = fit_results.append(little_answer_bistatnorm)

    #         if i in fit_unimodal:
    #             fits['statnorm'][i] = scipy.optimize.curve_fit(statnorm_log, f[i][:,0], f[i][:,1], p0 = initial_guess['statnorm'][i])

    #             d_s = [[i, 'statnorm', fits['statnorm'][i][0][0], numpy.nan,  fits['statnorm'][i][0][1], numpy.nan, fits['statnorm'][i][0][2], numpy.nan, numpy.sqrt(numpy.diag(fits['statnorm'][i][1]))[0], numpy.nan,  numpy.sqrt(numpy.diag(fits['statnorm'][i][1]))[1], numpy.nan,  numpy.sqrt(numpy.diag(fits['statnorm'][i][1]))[2], numpy.nan]]

    #             little_answer_statnorm =  pandas.DataFrame(data = d_s, columns = ['condition', 'function', 'mean_1', 'mean_2', 'sigma_1','sigma_2', 'amplitude_1', 'amplitude_2', 'mean_fit_unc_1',  'mean_fit_unc_2', 'sigma_fit_unc_1', 'sigma_fit_unc_2', 'amplitude_fit_unc_1',   'amplitude_fit_unc_2'])

    #             fit_results = fit_results.append(little_answer_statnorm)

    #         fig = pylab.figure(figtitle)

    #         if i in fit_unimodal:
    #             print(j)
    #             print(i + ' in unimodal')
    #             ax[j].plot(f[i][:,0], statnorm_log(f[i][:,0], fits['statnorm'][i][0][0], fits['statnorm'][i][0][1], fits['statnorm'][i][0][2]), 'k', linewidth = 2)

    #         if i not in omit_in_plot_fit:
    #             print('plotting_fits')
    #             ax[j].plot(f[i][:,0], bistatnorm_log(f[i][:,0], fits['bistatnorm'][i][0][0], fits['bistatnorm'][i][0][1], fits['bistatnorm'][i][0][2], fits['bistatnorm'][i][0][3], fits['bistatnorm'][i][0][4], fits['bistatnorm'][i][0][5]), 'k', linewidth = 2)
    #             ax[j].plot(f[i][:,0], statnorm_log(f[i][:,0], fits['bistatnorm'][i][0][0], fits['bistatnorm'][i][0][2], fits['bistatnorm'][i][0][4]), 'k--', linewidth = 2)
    #             ax[j].plot(f[i][:,0], statnorm_log(f[i][:,0], fits['bistatnorm'][i][0][1], fits['bistatnorm'][i][0][3], fits['bistatnorm'][i][0][5]), 'k--', linewidth = 2)


    figname = 'msd_at_' + str(actual_time) + 's'
    figname = figname.replace('.', 'p')
    now = datetime.datetime.now()
    string_now = now.strftime("%Y%m%d")

    pylab.savefig(basic_directory + central_directory + 'plots/' + figname + '_' + i.strip('/') + '_' + string_now + '.png', bbox_inches = 'tight')

    os.makedirs(basic_directory + central_directory + 'results/', exist_ok=True)
    print(occupants['1'].keys())
    numpy.save(basic_directory + central_directory + 'results/' + string_now + '_mode_occupants_from_msd_at_' + str(timepoint).replace('.', 'p') + 's.npy', occupants)
    print(basic_directory + central_directory + 'results/' + string_now + '_mode_occupants_from_msd_at_' + str(timepoint).replace('.', 'p') + 's.npy')

    os.makedirs(basic_directory_paper_figs + '_' + spot_type + '/imsds/msd_at_' + str(timepoint).replace('.', 'p') + 's/', exist_ok=True)
    pylab.savefig(basic_directory_paper_figs + '_' + spot_type + '/imsds/msd_at_' + str(timepoint).replace('.', 'p') + 's/' + string_now + '_' + figname + '.png', bbox_inches = 'tight')
    pylab.savefig(basic_directory_paper_figs + '_' + spot_type + '/imsds/msd_at_' + str(timepoint).replace('.', 'p') + 's/' + string_now + '_' + figname + '.svg', bbox_inches = 'tight')

    fn = string_now + '_msd_at_' + str(actual_time).replace('.', 'p') + 's'

    os.makedirs(basic_directory_paper_data + '_' + spot_type + '/', exist_ok=True)
    numpy.save(basic_directory_paper_data + '_' + spot_type + '/' + fn + '.npy', imsd_values)

    fit_results.to_pickle(basic_directory_paper_data + '_' + spot_type + '/' + fn + '_fit_results.pkl')

    numpy.save(basic_directory_paper_data + '_' + spot_type + '/' + fn + '.npy', imsd_values)

    # return imsd_values, fit_results, occupants

def combine_msd_at_timepoint_with_previous_results(spot_type, results, msd_at_timepoint, timepoint):
    '''
    '''

    now = datetime.datetime.now()
    string_now = now.strftime("%Y%m%d")
    string_now = string_now[2:]

    r = {}

    for i in results.keys():
        r[i] = pandas.concat([results[i], msd_at_timepoint[str(timepoint)][i]], axis = 1)
        r[i] = r[i].rename(columns = {timepoint :'msd_at_' + str(timepoint).replace('.', 'p') + 's'})

    numpy.save(basic_directory_paper_data + '_' + spot_type + '/' + string_now + '_pooled_filtered_combined_results_with_msd_' + str(timepoint).replace('.', 'p') + 's_per_particle.npy', r)

    return r

def plot_imsd_vs_result_quantity(spot_type, results, imsds, quantity, timepoint, per_day=False, load = True, pool = True, avoid = ['lowC', '1000ms']):
    '''

    '''

    if load:
        r = load_all_results.load_all_results('muNS', 'results_per_particle')
        ims = load_all_results.load_all_results('muNS', 'imsds_all_renamed')

    if pool:
        results = concatenate('muNS', r, per_day=False, load=False, ax=0)
        imsds = concatenate('muNS', ims, per_day=False, load=False, ax=1)

    keys_of_interest = [x for x in results.keys() if not any([y in x for y in avoid])]

    central_directory = load_all_results.select_specifications(spot_type)['central_directory']

    nparticles = {}
    nparticles['bLR31_0h'] = 0
    nparticles['bLR32_0h'] = 0
    nparticles['bLR31_6h'] = 0
    nparticles['bLR32_6h'] = 0

    #    ### first, find the indicative line fits ###
    #
    #    f = {}
    #
    #    for i in keys_of_interest[:]:
    #        results_now = results[i][results[i].alpha > 0]
    #        results_now = results[i][results[i].alpha > 0]
    #        strain = load_all_results.read('strain', i, spot_type)
    #        st = load_all_results.read('starvation_time', i, spot_type) + 'h'
    #        time_between_frames = load_all_results.read('time_between_frames', i, spot_type)
    #        alphas = results_now.alpha.to_numpy(dtype = numpy.float64)
    #        D_apps = results_now.D_app.to_numpy(dtype = numpy.float64)
    #        f[i] = numpy.polyfit(alphas, numpy.log10(D_apps), 1)

    ### then plot all indicative lines, and each condition one by one ###
    #    if quantity == 'alpha':
    #        xs = numpy.linspace(0, 1.25, 100)

    for i in keys_of_interest[:]:
        results_now = results[i].loc[:, quantity]
        results_now = results_now[results_now > 0]
        imsds_now = imsds[i][(imsds[i].index > timepoint)].iloc[0]
        actual_time = imsds_now.name
        strain = load_all_results.read('strain', i, spot_type)
        st = load_all_results.read('starvation_time', i, spot_type) + 'h'
        time_between_frames = load_all_results.read('time_between_frames', i, spot_type)
    #        n = len(results[i])
        nparticles = len(results[i])
        figtitle = i + '\nN = ' + str(nparticles) #time_between_frames #strain + '_' + st #'all data'
        #        nparticles[figtitle] = nparticles[figtitle] + n

        particles = results_now.index
        fig_hist = pylab.figure(figtitle)
        ax_hist = fig_hist.gca()
        pylab.title(figtitle)
        c = pandas.concat((results_now, imsds_now), axis = 1)
        c = c.dropna()
        xvalues = numpy.array(c.iloc[:,0], dtype = numpy.float64)
        yvalues = numpy.array(c.iloc[:,1], dtype = numpy.float64)
        weights = 100 * numpy.ones_like(xvalues) / float(len(xvalues))
        if quantity == 'D_app':
            heatmap, xedges, yedges, z = pylab.hist2d(c.iloc[:,0], c.iloc[:,1], bins=[numpy.logspace(-5, 0, num = 101), numpy.logspace(-4, 1, num = 101)], weights = weights) #, norm=matplotlib.colors.PowerNorm(0.1)
            xl = ["{:.0e}".format(x) for x in xedges]
            xl = xl[::20]
            pylab.xticks(numpy.arange(0, 101, 20), labels = xl)
        elif quantity == 'alpha':
            heatmap, xedges, yedges, z = pylab.hist2d(c.iloc[:,0], c.iloc[:,1], bins=[numpy.linspace(0, 1.25, 101), numpy.logspace(-4, 1, num = 101)], weights = weights) #, norm=matplotlib.colors.PowerNorm(0.1)
            xl = [str(x) for x in xedges]
            xl = xl[::20]
            pylab.xticks(numpy.arange(0, 101, 20), labels = xl)

        elif quantity == 'average_starting_magnitude':
            heatmap, xedges, yedges, z = pylab.hist2d(c.iloc[:,0], c.iloc[:,1], bins=[numpy.linspace(0, 2500, 101), numpy.logspace(-4, 1, num = 101)], weights = weights) #, norm=matplotlib.colors.PowerNorm(0.1)
            xl = [str(x) for x in xedges]
            xl = xl[::20]
    #            pylab.xticks(numpy.arange(0, 101, 20), labels = xl)

        pylab.imshow(heatmap.T, origin = 'lower', cmap = 'plasma')#, vmin = 0, vmax = 0.5)

        pylab.xlabel(quantity)
        pylab.ylabel('msd at ' + str(timepoint) + 's')
        yl = ["{:.0e}".format(y) for y in yedges]
        yl = yl[::20]
        pylab.yticks(numpy.arange(0, 101, 20), labels = yl)

    #        fig = pylab.figure(figtitle + '_scatter')
    #        ax = fig.gca()
    #        ax.set_yscale('log')
    #        if quantity == 'D_app':
    #            ax.set_xscale('log')
    #        pylab.plot(c.iloc[:,0], c.iloc[:,1], 'o', alpha = 0.025, color = colors[strain][st + condition], label = '')  #alpha = 500./nparticles
    #        pylab.plot(xs, 10**(f[i][0] * xs + f[i][1]), '-', color = 'white')
    #        pylab.plot(xs, 10**(f[i][0] * xs + f[i][1]), '--', color = colors[strain][st + condition], label = i)
    #        pylab.text(4e-1, 4e-1, 'N = ' + str(nparticles))  #alpha: (0.9, 3e0), D_app: (4e-1, 4e-1)
    #        alphas = results_now.alpha.to_numpy(dtype = numpy.float64)
    #        D_apps = results_now.D_app.to_numpy(dtype = numpy.float64)
    #        pylab.xlabel(quantity)
    #        pylab.ylabel('msd at ' + str(timepoint) + 's')
    #        pylab.xlim(-0.05, 1.2) # alpha
    #        pylab.xlim(1e-5, 1e0) # D_app
    #        pylab.xlim(0, 6) # starting_snr
    #        pylab.ylim(5e-6, 1e1)
        pylab.savefig(basic_directory + central_directory + 'plots/imsd_' + str(timepoint).replace('.', 'p') + 's_vs_' + quantity + '/heatmap_' + i + '.png')

    #    for i in ['0030ms', '0090ms', '0210ms']:
    #        pylab.figure(i)
    #        pylab.xlim(-0.05, 1.2)
    #        pylab.ylim(1e-5, 1e0)
    #        pylab.xlabel('alpha')
    #        pylab.ylabel('D_app')
    #        for j in keys_of_interest:
    #            if i in j:
    #                print(j)
    #                fig = pylab.figure(i)
    #                ax = fig.gca()
    #                ax.set_yscale('log')
    #                strain = load_all_results.read('strain', j, spot_type)
    #                st = load_all_results.read('starvation_time', j, spot_type) + 'h'
    #                pylab.plot(xs, 10**(f[j][0] * xs + f[j][1]), '-', color = 'white')
    #                pylab.plot(xs, 10**(f[j][0] * xs + f[j][1]), '--', color = colors[strain][st + condition], label = strain + '_' + st)
    #        pylab.legend()
    #        pylab.savefig(basic_directory + central_directory + 'plots/Dapp_vs_alpha/' + '_all_fits_' + i + '.png')


    #    for i in ['bLR31_0h','bLR31_6h','bLR32_0h','bLR32_6h']:
    #        pylab.figure(i)
    #        pylab.text(0.9, 4e-1, 'N = ' + str(nparticles[i]))
    #        pylab.savefig(basic_directory + central_directory + 'plots/Dapp_vs_alpha/' + i + '.png')

    return c, xedges, yedges

def histogram_quantity(spot_type, results, quantity, avoid = ['lowC'], below_size_limit = True):

    central_directory = load_all_results.select_specifications(spot_type)['central_directory']

    keys_of_interest = [x for x in results.keys() if not any([y in x for y in avoid])]

    now = datetime.datetime.now()
    string_now = now.strftime("%Y%m%d")

    if below_size_limit:
        string_now = 'below_size_limit_' + string_now

    for i in keys_of_interest:
        print(i)
        pylab.figure(i)
        pylab.title(i)
        r = results[i].copy(deep = True)
        if below_size_limit:
            r = r[r.below_diffraction_limit]
        if quantity == 'average_starting_magnitude':
            data = r.average_starting_magnitude.to_numpy(dtype = numpy.float64)
        elif quantity == 'average_starting_sigma':
            data = r.average_starting_sigma.to_numpy(dtype = numpy.float64)
        elif quantity == 'particle_size':
            data = r.average_starting_magnitude.to_numpy(dtype = numpy.float64)
            data = data**(1./3)
        elif quantity == 'average_starting_amplitude':
            data = r.average_starting_amplitude.to_numpy(dtype = numpy.float64)

        data_now = data[numpy.greater(data, numpy.zeros_like(data))]
        s = load_all_results.read('strain', i, 'muNS')
        d = i.split('_')[::-1][0]
        st = load_all_results.read('starvation_time', i, 'muNS') + 'h'
        if st == '6h':
            k = st + 'lowN'
        else:
            k = st
        p = positions[s][k]
        sym = symbols['muNS'][d]
        weights = 100 * numpy.ones_like(data_now) / float(len(data_now))
        pylab.hist(data_now, bins = numpy.arange(0, 250, 10), color = colors[s][k], weights = weights)
        pylab.text(200, 30, 'N = ' + str(len(data_now)), color = colors[s][k])
        pylab.xlabel(quantity)
        pylab.xlim(0, 260)
        pylab.ylim(0, 35)
        pylab.savefig(basic_directory + central_directory + 'plots/average_starting_amplitudes_across_days/' + i + '_' + string_now + '.png')

    #figure('summary of sizes, below_size_limit')
    #       pylab.errorbar(p, numpy.median(data_now), xerr = 0, yerr = data_now.std() / numpy.sqrt(len(data_now)), fmt = sym, color = colors[s][k], markersize = 7, markeredgecolor = 'k', alpha = 0.7, capsize = 5)
    #       pylab.xticks([1, 1.2, 1.4, 1.6], ['WT\n0h', '\n6h', 'ΔpolyP\n0h', '\n6h'])
            #ylim(500, 2000)

### auxiliary mathematical functions ###

def statnorm(x, xo, s, a):
    '''
    '''

    z = x-xo

    A = 1 / (numpy.sqrt(2 * numpy.pi))
    B = - z**2
    C = 2 * s**2
    value = a * A * numpy.exp(B/C)

    return value

def statnorm_log(x, xo, s, a):
    '''
        '''

    z = numpy.log10(x) - numpy.log10(xo)

    A = 1 / (numpy.sqrt(2 * numpy.pi))
    B = - z**2
    C = 2 * s**2
    value = a * A * numpy.exp(B/C)

    return value

def bistatnorm(x, x0o, x1o, s0, s1, a0, a1):
    '''
    '''

    z0 = x-x0o
    z1 = x-x1o

    A0 = 1 / (numpy.sqrt(2 * numpy.pi))
    B0 = - z0**2
    C0 = 2 * s0**2
    value0 = a0 * A0 * numpy.exp(B0/C0)
    A1 = 1 / (numpy.sqrt(2 * numpy.pi))
    B1 = - z1**2
    C1 = 2 * s1**2
    value1 = a1 * A1 * numpy.exp(B1/C1)
    value = value0 + value1

    return value

def bistatnorm_log(x, x0o, x1o, s0, s1, a0, a1):
    '''
    '''

    z0 = numpy.log10(x) - numpy.log10(x0o)
    z1 = numpy.log10(x) - numpy.log10(x1o)

    A0 = 1 / (numpy.sqrt(2 * numpy.pi))
    B0 = - z0**2
    C0 = 2 * s0**2
    value0 = a0 * A0 * numpy.exp(B0/C0)
    A1 = 1 / (numpy.sqrt(2 * numpy.pi))
    B1 = - z1**2
    C1 = 2 * s1**2
    value1 = a1 * A1 * numpy.exp(B1/C1)
    value = value0 + value1

    return value

def bistatnorm_fix_first(x, x1o, s0, s1, a0, a1, x0o = 1.10618448e-03):  # s0 = 3.05317220e-01
    '''
    '''
    z0 = x-x0o
    z1 = x-x1o

    A0 = 1 / (numpy.sqrt(2 * numpy.pi))
    B0 = - z0**2
    C0 = 2 * s0**2
    value0 = a0 * A0 * numpy.exp(B0/C0)
    A1 = 1 / (numpy.sqrt(2 * numpy.pi))
    B1 = - z1**2
    C1 = 2 * s1**2
    value1 = a1 * A1 * numpy.exp(B1/C1)
    value = value0 + value1

    return value

def bistatnorm_log_fix_first(x, x1o, s0, s1, a0, a1, x0o = 1.10618448e-03):  # s0 = 3.05317220e-01
    '''
        '''
    z0 = numpy.log10(x) - numpy.log10(x0o)
    z1 = numpy.log10(x) - numpy.log10(x1o)
    A0 = 1 / (numpy.sqrt(2 * numpy.pi))
    B0 = - z0**2
    C0 = 2 * s0**2
    value0 = a0 * A0 * numpy.exp(B0/C0)
    A1 = 1 / (numpy.sqrt(2 * numpy.pi))
    B1 = - z1**2
    C1 = 2 * s1**2
    value1 = a1 * A1 * numpy.exp(B1/C1)
    value = value0 + value1

    return value

def lognorm(x, s):

    A = 1 / (s * x * numpy.sqrt(2 * numpy.pi))
    B = - numpy.log10(x)**2
    C = 2 * s**2
    value = A * numpy.exp(B/C)

    return value

### Finally, some useful snippets of code I typed in the terminal recently. ###
'''
    for i in dirs_now:
    ...:     print(i)
    ...:     subloc = '/analysis/diameter7_minmass110_percentile65/search
    ...: _range4_memory20/'
    ...:     loc = 'C://Users/magkiria/Documents/RNPs/data/201226/muNS/6h/
    ...: ' + i + subloc
    ...:     s = 'simple_Gauss'
    ...:     fits = [x for x in os.listdir(loc) if re.search(s, x)]
    ...:     for j in fits:
    ...:         print(j)
    ...:         shutil.copy(loc + j, i + subloc + j)

2021.05.27: to get for how many frames σ > size limit

n = {}

for i in numpy.arange(0, 6):
    n[str(i)] = 0

for j,i in enumerate(list(test.index)[:]):
    sigmas = []
    sigmas.append(test.loc[i, 'sigma_frame0'])
    sigmas.append(test.loc[i, 'sigma_frame1'])
    sigmas.append(test.loc[i, 'sigma_frame2'])
    sigmas.append(test.loc[i, 'sigma_frame3'])
    sigmas.append(test.loc[i, 'sigma_frame4'])
    sigmas = numpy.array(sigmas)
    if not numpy.isnan(sigmas).any():
        m = len(numpy.where(sigmas > 1.315)[0])
        n[str(m)] = n[str(m)] + 1
    print(str(j * 100/(len(test))))

    pp = {}  # plot parameters for each quantity
    pp['alpha'] = {}
    pp['Dapp'] = {}

    pp['alpha']['data'] = r_now.alpha
    if show_particles_with_alpha_neg:
    pp['alpha']['xmin'] = -0.5
    else:
    pp['alpha']['xmin'] = -0.05
    pp['alpha']['xmax'] = 1.25
    pp['alpha']['ymax'] = 25  #15 when only lowN
    pp['alpha']['xtext_N'] = 0.8
    pp['alpha']['bins_now'] = numpy.arange(pp['alpha']['xmin'], pp['alpha']['xmax'], 0.05)
    if not compare:
    pp['alpha']['ytext_N'] = 12
    pp['alpha']['color_text'] = colors[strain][starvation_time + condition]
    pp['alpha']['t'] = 'α = '
    pp['alpha']['value_x_loc'] = pp['alpha']['bins_now'][::-1][0] + 0.06
    else:
    pp['alpha']['ytext_N'] = 6
    pp['alpha']['color_text'] = '#000000'
    pp['alpha']['t'] = '/ '
    pp['alpha']['value_x_loc'] = bins_now[::-1][0] + 0.32
    pp['alpha']['xtext_s'] = pp['xmax'] + 1e-1
    pp['alpha']['ytext_s'] = 20 #ymax - 2
    for k, l in enumerate(pp['alpha']['bins_now']):
    if k < len(pp['alpha']['bins_now']) - 1:
    c = numpy.linspace(pp['alpha']['bins_now'][k], pp['alpha']['bins_now'][k+1], num=3)
    pp['alpha']['midpoints_now'][i].append(c[1])
    pp['alpha']['label2_x'] = pp['alpha']['bins_now'][::-1][0] + 0.06  # location of strain label (or xtext_s?)
    pp['alpha']['label2_y'] = 10
    pp['alpha']['value_y_location'] = 1
    '''

#        initial_guess['bistatnorm']['bLR31_0h_0030ms'] = [fits['statnorm']['bLR31_0h_0030ms'][0][0], 1e-3, fits['statnorm']['bLR31_0h_0030ms'][0][1], 3e-1, 20, 20]
#        initial_guess['bistatnorm']['bLR32_0h_0030ms'] = [fits['statnorm']['bLR32_0h_0030ms'][0][0], 1e-3, fits['statnorm']['bLR32_0h_0030ms'][0][1], 3e-1, 17, 11]
#        initial_guess['bistatnorm']['bLR31_6h_0030ms_lowN'] = [fits['statnorm']['bLR31_6h_0030ms_lowN'][0][0], 1e-3, fits['statnorm']['bLR31_6h_0030ms_lowN'][0][1], 3e-1, 50, 12]
#        initial_guess['bistatnorm']['bLR32_6h_0030ms_lowN'] = [fits['statnorm']['bLR31_0h_0030ms'][0][0], 1.8e-2, fits['statnorm']['bLR31_0h_0030ms'][0][1], 3e-1, 16.5, 20]
#
#        initial_guess['bistatnorm']['bLR31_0h_0090ms'] = [fits['statnorm']['bLR31_0h_0090ms'][0][0], fits['statnorm']['bLR31_0h_0090ms'][0][0], fits['statnorm']['bLR31_0h_0090ms'][0][1], 3e-1, 30, 30]
#        initial_guess['bistatnorm']['bLR32_0h_0090ms'] = [fits['statnorm']['bLR31_0h_0090ms'][0][0], fits['statnorm']['bLR31_0h_0090ms'][0][0], 3e-1, 3e-1, 10, 20]
#        initial_guess['bistatnorm']['bLR31_6h_0090ms_lowN'] = [fits['statnorm']['bLR31_0h_0090ms'][0][0], 3e-3, fits['statnorm']['bLR31_0h_0090ms'][0][1], 3e-1, 50, 12]
#        initial_guess['bistatnorm']['bLR32_6h_0090ms_lowN'] = [fits['statnorm']['bLR31_0h_0090ms'][0][0], 1.8e-2, fits['statnorm']['bLR31_0h_0090ms'][0][1], 3e-1, 16.5, 20]


#        initial_guess['bistatnorm']['bLR31_0h_0030ms'] = [fits['statnorm']['bLR31_0h_0030ms'][0][0], fits['statnorm']['bLR31_0h_0030ms'][0][0], fits['statnorm']['bLR31_0h_0030ms'][0][1], fits['statnorm']['bLR31_0h_0030ms'][0][1], fits['statnorm']['bLR31_0h_0030ms'][0][2], fits['statnorm']['bLR31_0h_0030ms'][0][2]]
#        initial_guess['bistatnorm']['bLR32_0h_0030ms'] = [fits['statnorm']['bLR31_0h_0030ms'][0][0], fits['statnorm']['bLR31_0h_0030ms'][0][0], fits['statnorm']['bLR31_0h_0030ms'][0][1], fits['statnorm']['bLR31_0h_0030ms'][0][1], fits['statnorm']['bLR31_0h_0030ms'][0][2], fits['statnorm']['bLR31_0h_0030ms'][0][2]]
#        initial_guess['bistatnorm']['bLR31_6h_0030ms_lowN'] = [fits['statnorm']['bLR31_0h_0030ms'][0][0], fits['statnorm']['bLR31_0h_0030ms'][0][0], fits['statnorm']['bLR31_0h_0030ms'][0][1], fits['statnorm']['bLR31_0h_0030ms'][0][1], fits['statnorm']['bLR31_0h_0030ms'][0][2], fits['statnorm']['bLR31_0h_0030ms'][0][2]]
#        initial_guess['bistatnorm']['bLR32_6h_0030ms_lowN'] = [0.32, 0.72, fits['statnorm']['bLR31_0h_0030ms'][0][1], 3e-1, 16.5, 20]

#        initial_guess['bistatnorm']['bLR31_0h_0090ms'] = [fits['statnorm']['bLR31_0h_0030ms'][0][0], fits['statnorm']['bLR31_0h_0030ms'][0][0], fits['statnorm']['bLR31_0h_0030ms'][0][1], fits['statnorm']['bLR31_0h_0030ms'][0][1], fits['statnorm']['bLR31_0h_0030ms'][0][2], fits['statnorm']['bLR31_0h_0030ms'][0][2]]
#        initial_guess['bistatnorm']['bLR32_0h_0090ms'] = [0.05, 0.5, 1, 1, 1,1]
#        initial_guess['bistatnorm']['bLR31_6h_0090ms_lowN'] = [fits['statnorm']['bLR31_0h_0030ms'][0][0], fits['statnorm']['bLR31_0h_0030ms'][0][0], fits['statnorm']['bLR31_0h_0030ms'][0][1], fits['statnorm']['bLR31_0h_0030ms'][0][1], fits['statnorm']['bLR31_0h_0030ms'][0][2], fits['statnorm']['bLR31_0h_0030ms'][0][2]]
#        initial_guess['bistatnorm']['bLR32_6h_0090ms_lowN'] = [0.32, 0.72, fits['statnorm']['bLR31_0h_0030ms'][0][1], 3e-1, 16.5, 20]
