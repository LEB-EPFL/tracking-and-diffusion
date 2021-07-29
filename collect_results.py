# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
import pandas
import numpy
import scipy
import os
import re
import sys
from IPython.display import clear_output
import time
import json
import datetime
import load_all_results

##### SETUP #####

with open('general_info.json') as jf:
    data = json.load(jf)

px_to_micron = data['px_to_micron']# μm per pixel
basic_directory = data['basic_directory']
all_starvation_times = data['all_starvation_times']

##### POOLING FUNCTIONS ###

def combine_particle_results(trajectories, fit_results_alpha_D_app_individual, spot_type, label = None, avoid = None, transfer_columns = ['starting_snr', 'average_starting_magnitude', 'average_starting_offset', 'below_diffraction_limit']):
    '''
    Combine per-particle information included in the trajectories DataFrames with information in the DataFrames you obtained when you fit to the msds (for instance using fit_imsds_within_time_range() from process_msds.py).
    This pertains to information that characterizes an entire particle, i.e. not information that characterizes a particle in a certain frame.
    
    INPUT
    -----
    trajectories : a dictionary of pandas DataFrames
        This contains the particle trajectories. Each entry in the dictionary corresponds to a movie.
    
    fit_results_alpha_D_app_individual : a dictionary of pandas DataFrames
        Each entry corresponds to a movie and contains a DataFrame. This DataFrame contains the fit results to the imsds for each particle, i.e. the estimated D_app and α. Typically, it is the outcome of one of the fit functions from process_msds.py.
        
    spot_type : str
        This can be 'origins', 'muNS', 'fixed_origins', 'fixed_muNS', depending on the object you are interested in.
        
    label : str or None (default)
        If you want to attach a label to the filename of the final result, you can specify here.
        
    avoid : list of strings or None (default)
        If you want to avoid certain datasets, here you can specify characteristic strings in their filename - for example 'lowC' if you are focusing on lowN.
        
    transfer_columns : list of strings, defaults to ['starting_snr', 'average_starting_magnitude', 'average_starting_offset', 'below_diffraction_limit']
        A list of the quantities from the trajectories DataFrame that you want to append to the results DataFrame. These strings need to be names of columns in the trajectories DataFrame. Note that it only makes sense to transfer them to the results DataFrame if they are the same for all frames, i.e. if they characterize entire particles and not just spots in one frame.

    OUTPUT
    ------
    A dictionary of DataFrames where each entry corresponds to a movie. The DataFrame in each entry contains information about every particle found in that movie: the index contains the particle id and each column corresponds to a global quantity for that particle, such as its D_app, α, or starting_snr value.
    The function also saves each DataFrame entry in the analysis folder that corresponds to the movie where it belongs.
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

def define_pool_categories(spot_type, time_between_frames = ['5000'], strains = ['bLR1', 'bLR2'], starvation_times = all_starvation_times, conditions = ['lowN', 'lowC']):
    '''
    Here you define all relevant categories for your data, such that you can pool data later on.
    There are two levels of pooling and four :
    Level 1: all movies of the same condition and time lag within a day. Keep track of days. You may also keep track of the time between frames (level 1.1) or not, in which case you group together movies taken with different time lags (level 1.2).
    Level 2: all movies of the same condition and time lag across all days. Again, you may also keep track of the time between frames (level 2.1) or not (level 2.2).
    In what follows you will create all reasonable and possible combinations of your experimental parameters, which will define all reasonable and possible pool categories for your data.
    
    INPUT
    -----
    spot_type : str
        This can be 'origins', 'muNS', 'fixed_origins', 'fixed_muNS', depending on the object you are interested in.
    
    time_between_frames : list of strings, defaults to ['5000']
        Each entry in the list corresponds to a chosen time between frames during acquisition, in msec.
        
    strains : list of strings, defaults to ['bLR1', 'bLR2']
        Each entry in the list corresponds to all the strains you are working with.
        
    starvation_times : list of strings, defaults to all_starvation_times defined at the beginning of the script
        Each entry in the list corresponds to a starvation time (e.x. '0h' or '6h').
        
    conditions : list of strings, defaults to ['lowN', 'lowC']
        Each entry in the list corresponds to a condition of your experiments.
        
    OUTPUT
    ------
    The function returns four lists, corresponding to the categories for the two levels of pooling described above.
    IMPORTANTLY, it also sorts movies into their categories for each type of grouping, and it saves these categorizations in files that are used further on in the functions that do the actual pooling of data.
    Finally, if also saves a human-readable text file with all the categories, so that you can check them for yourself and keep track of them for clarity.
    NOTE that under all this rely naming conventions that we used in our experiments so far, as relevant information is read from the filenames. If you follow different conventions, you might need to adapt the function read() in load_all_results.py.
    '''

    days = specs[spot_type]['all_days']
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
                        #print(k)
                        categories_with_day.append(k)
                        categories_with_day_without_timelags.append(s + '_' + st + '_' + str(d))
                elif st == '6h':
                    for c in conditions:
                        k = s + '_' + st + '_' + str(t).zfill(4) + 'ms' + '_' + c
                        categories.append(k)
                        #print(k)
                        categories_without_timelags.append(s + '_' + st + '_' + c)
                        for d in days:
                            k = s + '_' + st + '_' + str(t).zfill(4) + 'ms' + '_' + c + '_' + str(d)
                            categories_with_day.append(k)
                            categories_with_day_without_timelags.append(s + '_' + st + '_' + c + '_' + str(d))

    categories_without_timelags = list(set(categories_without_timelags))
    categories_with_day_without_timelags = list(set(categories_with_day_without_timelags))
    
    location = basic_directory + '_' + spot_type + '/'

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

    f.close()

    return categories_with_day, categories, categories_without_timelags, categories_with_day_without_timelags

def allocate_movies_to_categories(spot_type, movies = 'all_movies', time_between_frames = ['5000'], strains = ['bLR1', 'bLR2'], starvation_times = all_starvation_times, conditions = ['lowN', 'lowC']):
    '''
    Here you sort movies into the categories that they belong to.
    
    INPUT
    -----
    spot_type : str
        This can be 'origins', 'muNS', 'fixed_origins', 'fixed_muNS', depending on the object you are interested in.
    
    movies : list of strings or 'all_movies' (default)
        When a list of strings, each entry in the list should be the name of a movie. When 'all_movies', the function will categorize all movies you have. It knows to find them from the information you have specified in the file with general info, general_info.json.
    
    time_between_frames : list of strings, defaults to ['5000']
        Each entry in the list corresponds to a chosen time between frames during acquisition, in msec.
    
    strains : list of strings, defaults to ['bLR1', 'bLR2']
        Each entry in the list corresponds to all the strains you are working with.
    
    starvation_times : list of strings, defaults to all_starvation_times defined at the beginning of the script
    Each entry in the list corresponds to a starvation time (e.x. '0h' or '6h').
    
    conditions : list of strings, defaults to ['lowN', 'lowC']
        Each entry in the list corresponds to a condition of your experiments.
        
    OUTPUT
    ------
    The function outputs four dictionaries, corresponding to the four different types of groupings:
    Level 1: all movies of the same condition and time lag within a day. Keep track of days. You may also keep track of the time between frames (level 1.1) or not, in which case you group together movies taken with different time lags (level 1.2).
    Level 2: all movies of the same condition and time lag across all days. Again, you may also keep track of the time between frames (level 2.1) or not (level 2.2).
    In each one of the four dictionaries, each entry corresponds to a category in that grouping (for example, for level 2.1 we have a category called "bLR1_0h_0210ms"). Finally, in each entry there is a list of the movies that belong to that category.
    The function also saves these dictionaries, which will be read later on when you pool the data. It also saves a text file with the categorization of all movies, such that you can check for yourself that it makes sense.
    Note that, to avoid accidental re-writing, every time you run this function and save these files they will have the date and time appended to their filename. Before you move on to pooling, you need finalize which files you want to use as a reference and remove this part of the filename from them, such that they can be found by the function that pools the data (pool() below).
    '''

    avoid = specs[spot_type]['typically_avoid']
    categories = define_pool_categories(spot_type, time_between_frames = time_between_frames, strains = strains, starvation_times = starvation_times, conditions = conditions)
    
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

    f = open(basic_directory + '/_' + spot_type + '/' + 'pooled_movies_' + string_now + '.txt', 'w')

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

    numpy.save(basic_directory + '/_' + spot_type + '/' + 'pooled_directories_by_day_' + string_now + '.npy', pooled_directories_by_day)
    numpy.save(basic_directory + '/_' + spot_type + '/' + 'pooled_directories_by_day_without_timelag_' + string_now + '.npy', pooled_directories_by_day_without_timelag)
    numpy.save(basic_directory + '/_' + spot_type + '/' + 'pooled_directories_' + string_now + '.npy', pooled_directories)
    numpy.save(basic_directory + '/_' + spot_type + '/' + 'pooled_directories_without_timelag_' + string_now + '.npy', pooled_directories_without_timelag)

    return pooled_directories_by_day, pooled_directories, pooled_directories_without_timelag, pooled_directories_by_day_without_timelag

def pool(spot_type, file_type, ax, per_day = False, ignore_timelag = False, days = 'all_days', starvation_times = 'all', load = False, avoid = None):
    '''
    Here we pool data. This is essentially a simple concatenation of pandas DataFrames, but heavily embelished to cover our specific needs for the types of experiments that we do.
    
    INPUT
    -----
    spot_type : str
        This can be 'origins', 'muNS', 'fixed_origins', 'fixed_muNS', depending on the object you are interested in.

    file_type : dictionary of DataFrames, or str
        If a dictionary of pandas DataFrames, this is the data that you will pool. For example, you can load all trajectories you have ever had using load_all_results() in load_all_results.py, and then input them here.
        If a string, this describes the type of input to load_all_results(), such that the function goes and loads all data itself.
        I have typically used a dictionary of pandas DataFrames as input.
        
    ax : integer, 0 or 1
        THIS IS IMPORTANT. Choose 1 if you are combining DataFrames of msds, and 0 otherwise. This number specifies the axis along which the DataFrames will be concatenated, and this is why it depends on what these DataFrames are.
        
    per_day : boolean, defaults to False
        When True, you will combine the data keeping track of each day separately (i.e. level 1 explained in allocate_movies_to_categories()).
        When False, you will combine the data from all days together (i.e. level 2 explained in allocate_movies_to_categories()).
        
    ignore_timelag : boolean, defaults to False
        When False, you will continue keeping track of the time between frames in the pooling (levels x.1 explained in allocate_movies_to_categories()).
        When True, you will ignore the time between frames in the pooling (levels x.2 explained in allocate_movies_to_categories()).
        
    days : list of strings or 'all_days' (default)
        If a list of strings, here you specify which days to include in the pooling. If 'all_days', you will include all days, as you have specified them in general_info.json .
        
    starvation_times : list of strings or all_starvation_times (default, defined at the beginning of the script)
        Each entry in the list corresponds to a starvation time (e.x. '0h' or '6h'). If you type ['0h'], for instance, you will only work with 0h data.
        
    load : boolean, defaults to False
        If the input to "file_type" is a DataFrame, set this to False. If it is a str, set it to True, such that the function loads the data itself.
        
    avoid : list of strings or None (default)
        If you want to avoid certain datasets, here you can specify characteristic strings in their filename - for example 'lowC' if you are focusing on lowN.
        
    OUTPUT
    ------
    A dictionary of DataFrames where each entry corresponds to a category of data (e.x. 'bLR1_6h_0210ms_lowN') and contains all data described by that category, combined.
    NOTE that for now I do not save these pooled DataFrames, as they are quick to generate and we might still add data to our pool. I found it confusing to have to keep track of different pooled DataFrames and what they contained, so I prefer to generate them on the spot as I work. This way I know exactly what I am including each time.
    That said, once the full set of data has been finalized it will be convenient to save these "master dictionaries of DataFrames" and always refer to those, instead of generating them every time.
    '''

    if load:
        if isinstance(file_type, str):
            data = load_all_results.load_all_results(spot_type, file_type, days = days, starvation_times = starvation_times)
        else:
            print('You need to specify the filename of the files I should load.')
    else:
        data = file_type.copy()

    if isinstance(avoid, list):
        keys_of_interest = [x for x in data.keys() if not any([y in x for y in avoid])]
    else:
        keys_of_interest = [x for x in data.keys()]

    for i in file_type.keys():
        if i not in keys_of_interest:
            del data[i]

    if per_day:
        if not ignore_timelag:
            categories = numpy.load(basic_directory + '_' + spot_type + '/' + 'pooled_directories_by_day.npy', allow_pickle = True).item()
        else:
            categories = numpy.load(basic_directory + '_' + spot_type + '/' + 'pooled_directories_by_day_without_timelag.npy', allow_pickle = True).item()
    else:
        if not ignore_timelag:
            categories = numpy.load(basic_directory + '_' + spot_type + '/' + 'pooled_directories.npy', allow_pickle = True).item()
        else:
            categories = numpy.load(basic_directory + '_' + spot_type + '/' + 'pooled_directories_without_timelag.npy', allow_pickle = True).item()

    if isinstance(file_type, str):
        if 'imsd' in file_type:
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
    
    below_size_limit_only : boolean, defaults to True
        When True, you will filter out particles that do not satisfy the size limit. This is a limit imposed on the value of sigma found by the Gaussian fit results. See also filter_by_width() in process_trajectories.py. In that function we keep those particles but label them as large; here we can choose to remove them.
    
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
    
    filtered_results = numpy.load(reference_results_file, allow_pickle = True).item()
    
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

### under construction ###
def filter_traj(spot_type, trajectories, reference_results_file):
    '''
    Filter out particles with unphysical results, from the trajectories. The particles that will remain are the ones that have passed the filtering done by filter_results() on the DataFrames that contain the results per particle (D_app, α, and particle size).
    
    INPUT
    -----
    spot_type : str
    This can be 'origins', 'muNS', 'fixed_origins', 'fixed_muNS', depending on    the object you have tracked.
    
    trajectories : dictionary of pandas DataFrames
    Each entry in this dictionary contains the trajectories dataframe for a given condition; in other words, the input is the pooled trajectories. Note that you need to use the renamed trajectories, where the particle id contains its date and movie number.
    
    reference_results_file : str
    The full path + filename to the file that contains the filtered results, generated with the function filter_results() above. Particles not present in this DataFrame will be removed from the msds.
    
    OUTPUT
    ------
    A dictionary of pandas DataFrames, similar to the input trajectories, where all particles with unphysical properties have been removed.
        
        '''
    
    filtered_results = numpy.load(reference_results_file, allow_pickle = True).item()
    
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


def bin_dataframe(df, quantity, below_size_limit_only = False, avoid = ['lowC'], quantiles = False):
    '''
    Consider a subset of the DataFrame, binned per the quantity you have chosen. This currently only accepts particle size as the quantity by which to bin.
    
    NOTE that you need to define bin limits after line 565 below.
    
    INPUT
    -----
    df : a dictionary of DataFrames
        This will typically be a dictionary of DataFrames with the per-particle results, pooled. The quantity by which you bin needs to be a column of the DataFrames.
        
    quantity : str
        The quantity by which you bin. Currently the only option is 'particle_size'. This must also be a column of the DataFrames.
    
    below_size_limit_only : boolean, defaults to True
        When True, you will ignore particles that do not satisfy the size limit. This is a limit imposed on the value of sigma found by the Gaussian fit results. See also filter_by_width() in process_trajectories.py, where we label particles according to their size compared to a user-defined limit.

    avoid : list of strings or None (default)
        If you want to avoid certain datasets, here you can specify characteristic strings in their filename - for example 'lowC' if you are focusing on lowN.
        
    quantiles : boolean, defaults to False
        This is under development. You can choose to define quantiles instead of regularly-spaced bins; since it has not been fully implemented it defaults to False.
        
    OUTPUT
    ------
    A dictionary of dictionaries of DataFrames, structured as follows: the higher-level keys denote the bin number; the following keys denote the condition; and finally you have the DataFrame for that bin and condition. For example, if you name your output df_binned, this will have keys ['0'], ['1'], etc, for bins 0, 1, etc: df['0'], df['1'], ... Each of those will then have keys for each condition, for instance df['0']['bLR31_0h'] contains all data for WT 0h that belongs to bin 0. 
    '''
    
    results = {}
    
    keys_of_interest = [x for x in df.keys() if not any([y in x for y in avoid])]
    
    if quantity == 'particle_size':
        bin_limits = numpy.linspace(7, 14, 8)
    
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
            r_now = df[i][df[i].below_diffraction_limit == 1.0] # all entries where the spots are smaller than the size limit (previously the diffraction limit but judged as too strict)
            
            r_now = r_now[r_now.average_starting_magnitude!=0]  # omit the 0 entries THIS SHOULD BECOME OBSOLETE ONCE I HAVE FITTED ALL MOVIES
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




