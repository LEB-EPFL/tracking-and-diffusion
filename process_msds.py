# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
import pylab
import matplotlib
import pandas
import numpy
import scipy
import trackpy
import os
import re
import sys
from IPython.display import clear_output
import time
import load_all_results
import json

##### NOTE #####
# In this script we often use dictionaries of DataFrames as input.
# For information on python dictionaries, see here:
# https://docs.python.org/3/tutorial/datastructures.html
# For information on python DataFrames, see here:
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
# For present purposes, remember that a dictionary is a container of data that can have multiple entries, each entry having a name called a "key". Here, these keys will often be the name of a movie. Later on when we pool data together, these keys will be the names of the experimental conditions.
# You will naturally obtain data in this format using the loading function from load_all_results. For instance, if I am working on chromosome origins, I typically obtain a dictionary of imsds using the function load_all_results() from load_all_results.py, like so: trajectories = load_all_results('origins', 'imsds_all_renamed', days = 'all_days', starvation_times = 'all', avoid = []). For more info see the documentation in that script.

##### SETUP #####
with open('general_info.json') as jf:
    data = json.load(jf)

px_to_micron = data['px_to_micron']# μm per pixel
basic_directory = data['basic_directory']
all_starvation_times = data['all_starvation_times']

##### FUNCTIONS #####

### Rename columns (i.e. particles) in imsds. ###

def rename_particles_in_imsds(imsds, spot_type):
    '''
    Rename particles in preparation for pooling. Their new names will contain information on the day and movie number.
    This function is similar to rename_particles_in_traj() in process_trajectories.py, however it differs in its inner workings because the structure of the msd DataFrames is different from the structure of the trajectories DataFrames.
    
    INPUT
    -----
    imsds : a dictionary of DataFrames with imsds
    
    spot_type : str
        This can be 'origins', 'muNS', 'fixed_origins', 'fixed_muNS', depending on the object you have tracked.

    OUTPUT
    ------
    A dictionary with imsds, where each particle (i.e. column) has been renamed following the format yymmddnnnnpppp where yy is the year the movie was taken, mm the month, dd the day, nnnn is the number of the movie to which the particle belongs, and pppp is the particle id within that movie. For example, 20100400040005 is the id of particle 5 from movie 4 taken on 201004.
    '''
    
    imsds_renamed = {}
    analysis_directories = load_all_results.define_analysis_directories(spot_type)
    
    number_of_movies = len(list(imsds.keys()))
    
    for j,k in enumerate(list(imsds.keys())):
        sys.stdout.write(str(j) + ' out of ' + str(number_of_movies) + '\n')
        sys.stdout.flush()
        #print(k)
        imsds_renamed[k] = imsds[k].copy()
        m = load_all_results.read('movie', k, spot_type)
        d = load_all_results.read('day', k, spot_type)
        s = load_all_results.read('starvation_time', k, spot_type) + 'h'
        for p in list(set(imsds_renamed[k].columns)):
            new_p = d + m.zfill(4) + str(p).zfill(4)
            new_p = int(new_p)
            imsds_renamed[k].rename(columns = {p: new_p}, inplace=True)
        
        location = basic_directory + d + '/' + spot_type + '/' + s + '/' + k + analysis_directories[d][k]
        imsds_renamed[k].to_pickle(location + 'imsds_all_renamed.pkl')
    
    return imsds_renamed

# It takes me 2.5 minutes to go rename all imsds from four days.

def fit_imsds_within_time_range(spot_type, imsds, t_start = 10, t_end = 4 * 10, avoid = ['1s']):
    '''
    Fit a line to a specified time range of each imsd in log log, to get individual values for α, D_app. ###
        

    INPUT
    -----
    spot_type : str
        This can be 'origins', 'muNS', 'fixed_origins', 'fixed_muNS', depending on the object you have tracked.

    imsds : a dictionary of DataFrames
        Contains the imsds you want to fit within a certain time range.
        
    t_start: float or int
        The start of the time frame over which you want to apply a linear fit to the log - log of the msds, in seconds.

    t_end : float or int
        The end of the time frame over which you want to apply a linear fit to the log - log of the msds, in seconds.
        
    avoid : list of str
        A list containing strings that characterize movies that you want to avoid. For example, the dictionary of imsds might contain entries corresponding to movies whose name includes '1s', if, for instance, they were taken at 1 frame per second. You might not be interested in these movies for present purposes, and here you can exclude them from this part of the analysis (and save some time).
    OUTPUT
    ------
    A dictionary of DataFrames, where each entry corresponds to an entry in the input imsds, and each DataFrame has a row per particle with its values for the resulting D_app and α.
    This dictionary is also saved in a specified location downstream of the basic_directory you have defined above.
    '''
    
    results = {}
    
    analysis_directories = load_all_results.define_analysis_directories(spot_type)

    keys_of_interest = list(imsds.keys())
    
    if isinstance(avoid, list):
        keys_of_interest = [x for x in keys_of_interest if not any([y in x for y in avoid])]

    number_of_movies = len(keys_of_interest)

    for h,k in enumerate(keys_of_interest[:]):
        print(k)
        sys.stdout.write(str(h) + ' out of ' + str(number_of_movies) + '\n')
        start = numpy.where(imsds[k].index >= t_start)[0][0]
        print('starting index: ' + str(start))
        end = numpy.where(imsds[k].index <= t_end)[0].max()
        print('ending index: ' + str(end))
        time_between_frames = load_all_results.read('time_between_frames', k, spot_type)
        # you can uncomment the lines below if you want to double-check that you are considering the correct time points, by looking at the indices it chooses for indices of the msds (indices show time lags).
#        print('time between frames: ' + str(time_between_frames))
#        print(start)
#        print(end)
        imsd_now = imsds[k].iloc[start:end, :]
        m = load_all_results.read('movie', k, spot_type)
        d = load_all_results.read('day', k, spot_type)
        s = load_all_results.read('starvation_time', k, spot_type) + 'h'

        results[k] = pandas.DataFrame(columns=['D_app', 'alpha'], index = imsd_now.columns)

        for i in imsd_now.columns:
            chosen_particle = i
            focus = imsd_now.loc[:,chosen_particle]
            test = numpy.zeros([len(focus), 2])
            test[:,0] = focus.index
            test[:,1] = focus.to_numpy()
            alpha, D = numpy.polyfit(numpy.log10(test[:,0]), numpy.log10(test[:,1]), 1)
            D_app = (10**D) / 4.
            results[k].loc[chosen_particle, 'alpha'] = alpha
            results[k].loc[chosen_particle, 'D_app'] = D_app

        location = basic_directory + d + '/' + spot_type + '/' + s + '/' + k + analysis_directories[d][k]
        results[k].to_pickle(location + 'fit_results_alpha_Dapp_individual_tstart' + str(t_start).replace('.', 'p') + '_tend' + str(t_end).replace('.', 'p') + 'sec.pkl')

    return results

##### TO BE DEVELOPED IF THERE IS INTEREST AND TIME #####

def remove_unphysical_results(results, imsds, trajectories):
    '''
    Remove particles with α < 0, α or D_app nan. That way you will always consider the same particles throughout the analysis, regardless of the quantity that you are looking at. Currently there are small discrepancies in the number of particles included in different graphs, on the order of a few particles over a couple of thousand.
    This function can include various filters that are relevant at different stages of the analysis. 
    
    INPUT
    -----
    
    OUTPUT
    ------
    A numpy array with the ids of particles with α < 0, α or D_app nan, and the input DataFrames with those particles removed.
    '''
    print('function will be revisited')




