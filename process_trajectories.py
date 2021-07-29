# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
import tracking
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
import datetime
import json

##### NOTE #####
# In this script we often use dictionaries of DataFrames as input.
# For information on python dictionaries, see here:
# https://docs.python.org/3/tutorial/datastructures.html
# For information on python DataFrames, see here:
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
# For present purposes, remember that a dictionary is a container of data that can have multiple entries, each entry having a name called a "key". Here, these keys will often be the name of a movie. Later on when we pool data together, these keys will be the names of the experimental conditions.
# You will naturally obtain data in this format using the loading function from load_all_results. For instance, if I am working on chromosome origins, I typically obtain a dictionary of trajectories using the function load_all_results() from load_all_results.py, like so: trajectories = load_all_results('origins', 'filtered_trajectories_all_renamed', days = 'all_days', starvation_times = 'all', avoid = []). For more info see the documentation in that script.

##### SETUP #####
with open('general_info.json') as jf:
    data = json.load(jf)

px_to_micron = data['px_to_micron']# μm per pixel
basic_directory = data['basic_directory']
all_starvation_times = data['all_starvation_times']

##### USEFUL FOR PLOTS #####

# function to generate color gradients, copied on 2021.02.09 from https://stackoverflow.com/questions/25668828/how-to-create-colour-gradient-in-python #

def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=numpy.array(matplotlib.colors.to_rgb(c1))
    c2=numpy.array(matplotlib.colors.to_rgb(c2))
    
    return matplotlib.colors.to_hex((1-mix)*c1 + mix*c2)

##### FUNCTIONS #####

def append_starting_snr_to_traj(spot_type, trajectories):
    '''
    To each trajectory DataFrame, append a column with the starting SNR of each particle. Here starting SNR is defined as the average SNR of the first five frames of a particles' appearence.
    
    INPUT
    -----
    spot_type : str
        A string that describes the type of spot you are interested in. It can be 'origins', 'muNS', 'fixed_origins', or 'fixed_muNS'.

    trajectories : dictionary of DataFrames
        A dictionary of trajectories DataFrames, where each key corresponds to the name of the movie the trajectories belong to.
        
    OUTPUT
    ------
    A similar dictionary where each DataFrame has an extra column, 'starting_snr'.
    This dictionary is also saved in a specified location downstream of the basic_directory you have defined above.
    '''
    
    analysis_directories = load_all_results.define_analysis_directories(spot_type)
    
    for i in list(trajectories.keys()):
        print(i)
        trajectories[i]['starting_snr'] = 0
        trajectories[i]['first_frame'] = 0
        for p in set(trajectories[i].particle):
            a = trajectories[i][trajectories[i].particle==p]
            first_five_frames = a.frame[:5].to_numpy()
            average_starting_snr = a.loc[a['frame'] <= first_five_frames[4]].snr.mean()
            trajectories[i].loc[trajectories[i].particle==p,'starting_snr'] = average_starting_snr
            trajectories[i].loc[trajectories[i].particle==p,'first_frame'] = first_five_frames[0]
        d = load_all_results.read('day', i, spot_type)
        s = load_all_results.read('starvation_time', i, spot_type) + 'h'
        location = basic_directory + d + '/' + spot_type + '/' + s + '/' + i + analysis_directories[d][i]
        trajectories[i].to_pickle(location + 'filtered_trajectories_all_with_starting_snr.pkl')

    return trajectories

def calculate_intensity_magnitude(spot_type, trajectories):
    '''
    Calculate the magnitude of total intensity from the fit results to the Gaussian fits for the intensity. The formula goes as follows: magnitude = pi * w^2 * h where h is the amplitude and w is the width of the Gaussian (w = sqrt(2) * sigma).
    
    INPUT
    -----
    spot_type : str
        This can be 'origins', 'muNS', 'fixed_origins', 'fixed_muNS', depending on the object you have tracked.
        
    trajectories : dictionary of DataFrames
        A dictionary of trajectories DataFrames, where each key corresponds to the name of the movie the trajectories belong to.

    OUTPUT
    ------
    The dictionary of trajectories DataFrames you entered as input, with an extra column called "magnitude" and the corresponding values for the intensity magnitude.
    This dictionary is also saved in a specified location downstream of the basic_directory you have defined above.
    '''
    analysis_directories = load_all_results.define_analysis_directories(spot_type)

    now = datetime.datetime.now()
    string_now = now.strftime("%Y%m%d")

    results = {}
    
    for i in trajectories.keys():
        print(i)
        t = trajectories[i]
        if 'amplitude' not in t.columns:
            print('You need to calculate the amplitudes.')
        else:
            results[i] = t.copy(deep = True)
            results[i]['magnitude'] =  numpy.pi * t.amplitude * t.sigma**2 * 2 # factor of sqrt(2) with the sigma because of how I define the Gaussian compared to microbetracker, which they used in Parry. sigma_parry = numpy.sqrt(2) * sigma_sofia.

            d = load_all_results.read('day', i, spot_type)
            s = load_all_results.read('starvation_time', i, spot_type) + 'h'
            location = basic_directory + d + '/' + spot_type + '/' + s + '/' + i + analysis_directories[d][i]
            results[i].to_pickle(location + 'filtered_trajectories_all_with_magnitude_from_simple_Gauss_' + string_now + '.pkl')

    return results

def append_starting_fit_values_to_traj(spot_type, trajectories, label = 'simple'):
    '''
    To each trajectory DataFrame, append a column with the starting value of quantities obtained in fitting a Gaussian on top of the intensity profile of each particle. Here "starting" is defined as the average value of the quantity in the first five frames of a particles' appearence.
    
    INPUT
    -----
    spot_type : str
        This can be 'origins', 'muNS', 'fixed_origins', 'fixed_muNS', depending on the object you have tracked.
        
    trajectories : dictionary of DataFrames
        A dictionary of trajectories DataFrames, where each key corresponds to the name of the movie the trajectories belong to.
        
    label : str, defaults to 'simple'
        A string that describes the fit you have performed, and which will be appended to the filename of the output DataFrame. Here we use 'simple' to denote that we fit the spots to a 'simple Gaussian' (as opposed to one that could be asymmetric, which I initially considered).
    
    OUTPUT
    ------
    A similar dictionary where each DataFrame has extra columns:
    - 'average_starting_offset' where offset corresponds to the local background
    - 'average_starting_magnitude' where magnitude corresponds to the integrated intensity of the spot (see calculate_intensity_magnitude() above)
    - 'average_starting_sigma' where sigma corresponds to the width of the Gaussian
    - 'average_starting_amplitude' where amplitude is the amplitude of the Gaussian, corresponding to its max intensity
    - 'sigma_framexx', for xx from 0 to 4, is the width of the particle during frames 0-4 (rarely used, considered using it for filtering out very large particles, TBD)
    This dictionary is also saved in a specified location downstream of the basic_directory you have defined above.
    '''
    
    analysis_directories = load_all_results.define_analysis_directories(spot_type)
    
    now = datetime.datetime.now()
    string_now = now.strftime("%Y%m%d")

    results = {}
    
    for i in list(trajectories.keys())[:]:
        print(i)
        results[i] = trajectories[i].copy(deep=True)
        results[i]['average_starting_offset'] = 0  # initialize new columns
        results[i]['average_starting_magnitude'] = 0
        results[i]['average_starting_sigma'] = 0
        results[i]['average_starting_amplitude'] = 0
        for f in numpy.arange(0, 5):
            results[i]['sigma_frame' + str(f)] = 0 # I want to keep track of sigma per frame for the first 5 frames of appearence of each particle
        results[i]['first_frame'] = 0              # I also keep track of the first frame of appearence, it might be useful later
        
        for p in list(set(results[i].particle))[:]:
            a = trajectories[i][trajectories[i].particle==p]  # focus on the part of the original DataFrame that describes one particle at a time
            first_five_frames = a.frame[:5].to_numpy()        # get its first five frames of appearence
            average_starting_magnitude = a.loc[a['frame'] <= first_five_frames[4]].magnitude.mean()
            average_starting_sigma = a.loc[a['frame'] <= first_five_frames[4]].sigma.mean()
            average_starting_offset = a.loc[a['frame'] <= first_five_frames[4]].offset.mean()
            average_starting_amplitude = a.loc[a['frame'] <= first_five_frames[4]].amplitude.mean()
            sigma_frame0 = numpy.float(a.loc[a['frame'] == first_five_frames[0]].sigma)
            sigma_frame1 = numpy.float(a.loc[a['frame'] == first_five_frames[1]].sigma)
            sigma_frame2 = numpy.float(a.loc[a['frame'] == first_five_frames[2]].sigma)
            sigma_frame3 = numpy.float(a.loc[a['frame'] == first_five_frames[3]].sigma)
            sigma_frame4 = numpy.float(a.loc[a['frame'] == first_five_frames[4]].sigma)

            results[i].loc[results[i].particle==p,'average_starting_magnitude'] = average_starting_magnitude
            results[i].loc[results[i].particle==p,'average_starting_sigma'] = average_starting_sigma
            results[i].loc[results[i].particle==p,'sigma_frame0'] = sigma_frame0
            results[i].loc[results[i].particle==p,'sigma_frame1'] = sigma_frame1
            results[i].loc[results[i].particle==p,'sigma_frame2'] = sigma_frame2
            results[i].loc[results[i].particle==p,'sigma_frame3'] = sigma_frame3
            results[i].loc[results[i].particle==p,'sigma_frame4'] = sigma_frame4
            results[i].loc[results[i].particle==p,'average_starting_offset'] = average_starting_offset
            results[i].loc[results[i].particle==p,'average_starting_amplitude'] = average_starting_amplitude
            results[i].loc[results[i].particle==p,'first_frame'] = first_five_frames[0]
                
        d = load_all_results.read('day', i, spot_type)
        s = load_all_results.read('starvation_time', i, spot_type) + 'h'
        location = basic_directory + d + '/' + spot_type + '/' + s + '/' + i + analysis_directories[d][i]
        results[i].to_pickle(location + 'filtered_trajectories_all_with_starting_values_from_' + label + '_Gauss_fit_' + string_now + '.pkl')
    
    return results

def filter_by_width(spot_type, trajectories, width_limit = 0.450, label1 = 'simple', label2 = 'by_ave_sigma'):
    '''
    INPUT
    -----
    spot_type : str
        This can be 'origins', 'muNS', 'fixed_origins', 'fixed_muNS', depending on the object you have tracked.

    trajectories : dictionary of DataFrames
        A dictionary of trajectories DataFrames, where each key corresponds to the name of the movie the trajectories belong to.
        In the case of muNS, usually I will first calculate the magnitude of all entries, then filter particles by width; so these DataFrames will already have a column named "magnitude", even though it is not used in this filter.

    width_limit : float
        The maximum width allowed for the spots, in μm. Typically it is the diffraction limit, following Parry et al, 2014 (see SI on "Measurement of GFP-mNS Particle Fluorescence Intensity").
        
    label1 : str, defaults to 'simple'
        A string that describes the fit you have performed, and which will be appended to the filename of the output DataFrame. Here we use 'simple' to denote that we fit the spots to a 'simple Gaussian' (as opposed to one that could be asymmetric, which I initially considered).

    label2 : str, defaults to 'by_ave_sigma'
        A string that describes the quantity by which you filtered the spots. Here, 'by_ave_sigma' indicates that I have classified them based on their average starting width, where the width comes from a fit to a simple Gaussian.

    OUTPUT
    ------
    The dictionary of trajectories DataFrames you entered as input, where each DataFrame has an extra column called 'below_diffraction_limit'. Particles with width larger than the set limit have False as entry, all other particles have True. Note that False may be marked with a 0 and True with a 1.
    This dictionary is also saved in a specified location downstream of the basic_directory you have defined above.
    '''

    width_limit_in_pixels = width_limit / px_to_micron
    print('Upper cutoff for width: ' + str(width_limit_in_pixels) + ' pixels, corresponding to ' + str(width_limit * 1000) + ' nm.')
    
    label2 = label2 + '_' + str(width_limit * 1000).replace('.', 'p') + 'nm'
    
    analysis_directories = load_all_results.define_analysis_directories(spot_type)

    now = datetime.datetime.now()
    string_now = now.strftime("%Y%m%d")

    results = {}

    for i in list(trajectories.keys())[:]:
        print(i)
        results[i] = trajectories[i].copy(deep = True)
        results[i]['below_diffraction_limit'] = 0
#        for p in [21032300020107]:
        for p in list(set(results[i].particle))[:]:
            tp = results[i][results[i].particle == p]
            sigmas = []
            sigmas.append(tp.sigma_frame0.mean())
            sigmas.append(tp.sigma_frame1.mean())
            sigmas.append(tp.sigma_frame2.mean())
            sigmas.append(tp.sigma_frame3.mean())
            sigmas.append(tp.sigma_frame4.mean())
            sigmas_parry = numpy.sqrt(2) * numpy.array(sigmas)
            sigmas_parry = [numpy.abs(x) for x in sigmas_parry]  # new I have not yet ran the code with this
            sigmas_parry = numpy.array(sigmas_parry)
#            if any(x > width_limit_in_pixels for x in sigmas_parry):
#                print('At least one sigma is above the limit. Entry: False.')
#                print(numpy.where(sigmas > width_limit_in_pixels))
            if sigmas_parry.mean() > width_limit_in_pixels:
                results[i].loc[results[i].particle==p, 'below_diffraction_limit'] = False
            else:
                results[i].loc[results[i].particle==p, 'below_diffraction_limit'] = True
#                print('No sigma is above the limit. Entry: True.')
        d = load_all_results.read('day', i, spot_type)
        s = load_all_results.read('starvation_time', i, spot_type) + 'h'
        location = basic_directory + d + '/' + spot_type + '/' + s + '/' + i + analysis_directories[d][i]
        results[i].to_pickle(location + 'filtered_trajectories_all_with_starting_values_from_' + label1 + '_Gauss_fit_classified_' + label2 + '_' + string_now + '.pkl')

    return results


def rename_particles_in_traj(trajectories, spot_type):
    '''
    Rename particles in preparation for pooling. Their new names will contain information on the day and movie number.
    
    INPUT
    -----
    trajectories : dictionary of DataFrames
        A dictionary of trajectories DataFrames, where each key corresponds to the name of the movie the trajectories belong to.
    
    spot_type : str
        This can be 'origins', 'muNS', 'fixed_origins', 'fixed_muNS', depending on the object you have tracked.
    
    OUTPUT
    ------
    A dictionary with trajectories, where each particle has been renamed following the format yymmddnnnnpppp where yy is the year the movie was taken, mm the month, dd the day, nnnn is the number of the movie to which the particle belongs, and pppp is the particle id within that movie. For example, 20100400040005 is the id of particle 5 from movie 4 taken on 201004.
    This dictionary is also saved in a specified location downstream of the basic_directory you have defined above.
    '''

    trajectories_renamed = {}
    analysis_directories = load_all_results.define_analysis_directories(spot_type)

    number_of_movies = len(list(trajectories.keys()))

    for j,k in enumerate(list(trajectories.keys())):
        sys.stdout.write(str(j) + ' out of ' + str(number_of_movies) + '\n')
        sys.stdout.flush()
        #print(k)
        trajectories_renamed[k] = trajectories[k].copy()
        m = load_all_results.read('movie', k, spot_type)
        d = load_all_results.read('day', k, spot_type)
        s = load_all_results.read('starvation_time', k, spot_type) + 'h'
        for p in list(set(trajectories_renamed[k].particle)):
            new_p = d + m.zfill(4) + str(p).zfill(4)
            new_p = int(new_p)
            locs = list(trajectories_renamed[k].loc[trajectories_renamed[k]['particle'] == p].index)
            trajectories_renamed[k].loc[locs,'particle'] = new_p

        location = basic_directory + d + '/' + spot_type + '/' + s + '/' + k + analysis_directories[d][k]
        trajectories_renamed[k].to_pickle(location + 'filtered_trajectories_all_renamed.pkl')
    
    return trajectories_renamed

# It takes me about 4 minutes to rename particles in the trajectories DataFrames of four days.

def count_absences(spot_type, trajectories):
    '''
    Count how long particles are absent from their trajectories. This can help judge, post-tracking, the mamory you used in tracking, over all many movies en masse. For example, if you see that particles are absent for far fewer frames than the memory you chose, you may want to decrease the memory and thus 1. speed up tracking and 2. distance yourself from the danger of connecting particles that should not belong to the same track.
        
    INPUT
    -----
        
    spot_type : str
        This can be 'origins', 'muNS', 'fixed_origins', 'fixed_muNS', depending on the object you have tracked.
        
    trajectories : a dictionary of DataFrames
        A dictionary of trajectories DataFrames, where each key corresponds to the name of the movie the trajectories belong to.
        
    OUTPUT
    ------
    A dictionary with all the absences, per key of the trajectories dictionary.
    This dictionary is also saved in a specified location downstream of the basic_directory you have defined above.
    '''
    
    absences = {}
    analysis_directories = load_all_results.define_analysis_directories(spot_type)
    
    for i in trajectories.keys():
        print(i)
        d = load_all_results.read('day', i, spot_type)
        print(d)
        s = load_all_results.read('starvation_time', i, spot_type) + 'h'
        print(s)
        result = tracking.count_absences(trajectories[i])
        location = basic_directory + d + '/' + spot_type + '/' + s + '/' +  i + analysis_directories[d][i]
        numpy.save(location + 'absences_dictionary.npy', result[0])
        numpy.save(location + 'absences_total.npy', result[1])
        absences[i] = result[1]
    
    return absences

##### PLOTS #####

def plot_all_absences(absences, spot_type, search_range = 4, memory = 10):
    '''
    Plot a histogram of the number of empty frames per movie per trajectory. In other words, the number of frames during which a particle was absent.
    
    INPUT
    -----
    absences : dictionary of lists
        A dictionary of the absences per item, where each item corresponds to a movie.
        
    memory : int, defaults to 10
        The memory you chose when you calculated the trajectories for which the absences have been calculated.
    
    OUTPUT
    ------
    A normalized histogram with a vertical line marking the memory, plotted and saved in the directory indicated by the keys in absences.
    The function will also show you a figure per dictionary key, containing a histogram of the absences for that entry. These figures are also saved in a specified location downstream of the basic_directory you have defined above.
    
    '''
    analysis_directories = load_all_results.define_analysis_directories(spot_type)
    
    for i in filtered_trajectories_all.keys():
        pylab.figure(i)
        pylab.hist(absences[i], bins = numpy.arange(0.5, 11.5, 1), normed = True)
        pylab.axvline(memory)
        pylab.xlabel('empty frames')
        pylab.ylabel('occurences, normalized')
        pylab.ylim(0, 1.0)
        d = load_all_results.read('day', i, spot_type)
        s = load_all_results.read('starvation_time', i, spot_type) + 'h'
        location = basic_directory + d + '/' + spot_type + '/' + s + '/' +  i + analysis_directories[d][i]
        pylab.title(i)
        pylab.savefig(location + 'total_absences.png')

    return absences

################################
##### NOT CHECKED RECENTLY #####

def plot_trajectories_by_starting_snr(spot_type, movie, superimpose = 'phase', starting_snr_bins = numpy.array([1, 1.5, 2, 2.5, 3, 3.5, 4]), label = False):
    '''
    '''
    
    diameter = specs[spot_type]['diameter']
    minmass = specs[spot_type]['minmass']
    percentile = specs[spot_type]['percentile']
    search_range = specs[spot_type]['search_range']
    memory = specs[spot_type]['memory']
    stub_length = specs[spot_type]['stub_length']
    
    analysis_directories = load_all_results.define_analysis_directories(spot_type)
    
    d = load_all_results.read('day', movie, spot_type)
    s = load_all_results.read('starvation_time', movie, spot_type) + 'h'
    
    location = basic_directory + d + '/' + spot_type + '/' + s + '/' +  movie + analysis_directories[d][movie]
    trajectories = pandas.read_pickle(location + 'filtered_trajectories_all_with_starting_snr.pkl')
    t = {}
    
    if 'starting_snr' not in trajectories.columns:
        print(movie + ':')
        print('You need to calculate the starting snr before you can color trajectories by it!')
    else:
        
        location = basic_directory + d + '/' + spot_type + '/' + s + '/' + movie
        if superimpose == 'phase':
            images = tracking.load_images(location + 'phase_images/')
        elif superimpose == 'fluorescence':
            images = tracking.load_images(location + 'images/')
        pylab.figure(movie)
        number_of_particles = len(set(trajectories.particle))
        pylab.title(movie + ' ' + str(number_of_particles) + ' particles\ndiameter {!r}, minmass {!r}, search range {!r}, memory {!r}, stub length {!r}'.format(diameter, minmass, search_range, memory, stub_length))
        for j, k in enumerate(starting_snr_bins):
            if j < len(starting_snr_bins) - 1:
                traj_now = trajectories.loc[(trajectories.starting_snr > starting_snr_bins[j]) & (trajectories.starting_snr < starting_snr_bins[j+1])]
                t[j] = traj_now
                number_of_particles_now = len(set(traj_now.particle))
                percent_now = 100 * float(number_of_particles_now) / number_of_particles
                percent_now = str(int(round(percent_now, 0)))
                snr_text = str(round(starting_snr_bins[j],1)).zfill(2) + ' < starting snr < ' + str(round(starting_snr_bins[j+1],1))
                snr_text = snr_text + ' ' + percent_now + '%'
            elif j == len(starting_snr_bins) - 1:
                traj_now = trajectories.loc[trajectories.starting_snr > starting_snr_bins[j]]
                t[j] = traj_now
                number_of_particles_now = len(set(traj_now.particle))
                percent_now = 100 * float(number_of_particles_now) / number_of_particles
                percent_now = str(int(round(percent_now, 0)))
                snr_text = str(round(starting_snr_bins[j],1)) + ' < starting snr' + '          '
                snr_text = snr_text + ' ' + percent_now + '%'
            color_snr = colorFader('#FF0000','#00FF00',mix=(float(j)/len(starting_snr_bins)))
            ax = trackpy.plot_traj(traj_now, superimpose = images[0], plot_style = {'color':color_snr}, label = False)
            ax.text(1300, 60 * (j), snr_text, color = color_snr, fontsize=14)

    return trajectories, t

def plot_trajectories_by_quantity(spot_type, movie, quantity, superimpose = 'phase', bins = numpy.array([1, 1.5, 2, 2.5, 3, 3.5, 4]), label = False):
    '''
    '''
    
    diameter = specs[spot_type]['diameter']
    minmass = specs[spot_type]['minmass']
    percentile = specs[spot_type]['percentile']
    search_range = specs[spot_type]['search_range']
    memory = specs[spot_type]['memory']
    stub_length = specs[spot_type]['stub_length']
    
    analysis_directories = load_all_results.define_analysis_directories(spot_type)
    
    d = load_all_results.read('day', movie, spot_type)
    s = load_all_results.read('starvation_time', movie, spot_type) + 'h'
    
    location = basic_directory + d + '/' + spot_type + '/' + s + '/' +  movie + analysis_directories[d][movie]
    trajectories = pandas.read_pickle(location + 'filtered_trajectories_all_with_starting_snr.pkl')
    t = {}
    
    if quantity not in trajectories.columns:
        print(movie + ':')
        print('You need to calculate the ' + quantity + ' before you can color trajectories by it!')
    else:
        
        location = basic_directory + d + '/' + spot_type + '/' + s + '/' + movie
        if superimpose == 'phase':
            images = tracking.load_images(location + 'phase_images/')
        elif superimpose == 'fluorescence':
            images = tracking.load_images(location + 'images/')
        pylab.figure(movie)
        number_of_particles = len(set(trajectories.particle))
        pylab.title(movie + ' ' + str(number_of_particles) + ' particles\ndiameter {!r}, minmass {!r}, search range {!r}, memory {!r}, stub length {!r}'.format(diameter, minmass, search_range, memory, stub_length))
        for j, k in enumerate(bins):
            if j < len(bins) - 1:
                traj_now = trajectories.loc[(trajectories.loc[:,quantity] > bins[j]) & (trajectories.loc[:,quantity] < bins[j+1])]
                t[j] = traj_now
                number_of_particles_now = len(set(traj_now.particle))
                percent_now = 100 * float(number_of_particles_now) / number_of_particles
                percent_now = str(int(round(percent_now, 0)))
                texto = str(round(bins[j],1)).zfill(2) + ' < ' + quantity + ' < ' + str(round(bins[j+1],1))
                texto = texto + ' ' + percent_now + '%'
            elif j == len(bins) - 1:
                traj_now = trajectories.loc[trajectories.loc[:,quantity] > bins[j]]
                t[j] = traj_now
                number_of_particles_now = len(set(traj_now.particle))
                percent_now = 100 * float(number_of_particles_now) / number_of_particles
                percent_now = str(int(round(percent_now, 0)))
                texto = str(round(bins[j],1)) + ' < ' + quantity + '    '
                texto = texto + ' ' + percent_now + '%'
            color_q = colorFader('#FF0000','#00FF00',mix=(float(j)/len(bins)))
            ax = trackpy.plot_traj(traj_now, superimpose = images[0], plot_style = {'color':color_q}, label = False)
            ax.text(1300, 60 * (j), texto, color = color_q, fontsize=14)

    return trajectories, t

##### NOT USED AT PRESENT  #####

def calculate_emsd_from_pooled_traj(spot_type, trajectories_pooled, final_frame = None, label = None):

    emsd = {}
    
    keys_of_interest = list(trajectories_pooled.keys())
    for j,k in enumerate(keys_of_interest[1:]):
        strain = load_all_results.read('strain', k, spot_type)
        starvation_time = load_all_results.read('starvation_time', k, spot_type) + 'h'
        condition = load_all_results.read('condition', k, spot_type)
        time_between_frames = float(load_all_results.read('time_between_frames', k, spot_type))
        
        trajectories = trajectories_pooled[k]
        
        if not isinstance(final_frame, int):
            final_frame = trajectories.frame.to_numpy(dtype = numpy.float64).max()
            final_frame = int(final_frame)
            print('final frame: ' + str(final_frame))

        trajectories.index.names = ['framenumber']  # this is to bypass a current bug in trackpy. the filtered_trajectories have an index named 'frame' and an identical column named 'frame', which confuses trackpy. So here I rename the index.
        drift = trackpy.compute_drift(trajectories)
        corrected_trajectories = trackpy.subtract_drift(trajectories.copy(), drift)

        to = time.time()
        
        emsd[k] = trackpy.emsd(corrected_trajectories, px_to_micron, 1./ time_between_frames, detail = True, max_lagtime = final_frame)
        
        t1 = time.time()
        print(k)
        print('This took ' + str(t1-to) + ' s.')
        
        #        emsd_error_simple = []  # omitting this for now to save time as I have not used it yet. If I comment this back in, I need to input the imsds into the function.
#        for i in imsds_pooled[k].index:
#            value = numpy.std(imsds.loc[i])/numpy.sqrt(len(imsds.loc[i]))  # error due to standard deviation among all particles at that dt
#            emsd_error_simple.append(value)
#        emsd_error_simple = pandas.Series(emsd_error_simple, index = emsd.index) # from list to Series
        emsd[k] = pandas.DataFrame(data = emsd[k], index = emsd[k].index)   # from Series to DataFrame
#        emsd[k]['error_simple'] = emsd_error_simple
        emsd[k]['error_finiteness'] = emsd[k]['msd'] / numpy.sqrt(emsd[k]['N'])
        emsd[k] = emsd[k].set_index('lagt')
        emsd[k].index.name = 'lagt'
        
        emsd[k].to_pickle(basic_directory + '_' + spot_type + '/emsd_' + k + '.pkl')
        
        file = open(basic_directory + '_' + spot_type + '/msd_info.txt', 'a')
        file.write('\n\nParameters used to calculate the ensemble mean-square displacement for ' + str(k) + ':\npx_to_micron: ' + str(px_to_micron) + '\ntime between frames: ' + str(time_between_frames) + ' msec.\n')
        file.close()

        print(str(j) + ' out of ' + str(len(list(trajectories_pooled.keys()))) + ' categories')

    return emsd



