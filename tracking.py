# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
import matplotlib
#matplotlib.use('Agg')
#matplotlib.use('TkAgg')
import matplotlib.patches
import os
import pims
import pylab
import trackpy
import datetime
import pandas
import numpy
import warnings

### BEFORE YOU RUN THE SCRIPT ###
# This script relies on a particular folder architecture. As things are right now, the script should be in the same directory as the data. Each dataset - i.e. each movie - has its own folder. With this folder, there is a directory called 'images' that has all the 
# fluorescent images; a directory called 'phase_images', that has all the phase images; and also one or two phase images.

# I have been using the hexadecimal color code, which lets you define colors in the format '#RRGGBB' where R, G, B are hex symbols. For more information and colors, see for ex. https://htmlcolorcodes.com/fr/

def load_images(image_location, extension = '.tif'):
    '''
    Load the images found in image location.
    
    INPUT
    -----
    image_location : string
        The full path name to the folder that contains the images you want to load. 
    
    OUTPUT
    ------
    A sequence containing the loaded images (see pims documentation for more details).
    '''
    
    frames = pims.ImageSequence(image_location + '*' + extension)
    #print('Loaded ' + str(len(frames)) +  ' frames.')
    return frames


def bandpass(frame, diameter):
    '''
    Passes a Gaussian filter over the image. Essentially a function from trackpy.
        
    INPUT
    -----
    frame : n x n array
        The image for which you wish to calculate the noise level.
        
    diameter : odd integer
        The diameter of the features you wish to track later on, i.e. the input to trackpy.locate.
        
    OUTPUT
    ------
    The image convolved with a Gaussian.
    '''
    
    if numpy.issubdtype(frame.dtype, numpy.integer):
        threshold = 1
    else:
        threshold = 1./256

    bandpassed_frame = trackpy.preprocessing.bandpass(frame, 1, diameter, threshold)

    return bandpassed_frame

def calculate_noise(frame, diameter):
    '''
    Calculates the noise in a frame from the dark areas, in the method described by Savin & Doyle, as implemented in trackpy.
        
    INPUT
    -----
    frame : n x n array
        The image for which you wish to calculate the noise level.
        
    diameter : odd integer
        The diameter of the features you wish to track later on, i.e. the input to trackpy.locate.
        
    OUTPUT
    ------
        The mean and standard deviation of the noise in the image. See also trackpy.uncertainty.measure_noise - this is essentially a reproduction of that function.
    '''
    
    bandpassed_frame = bandpass(frame, diameter)
    
    noise = trackpy.uncertainty.measure_noise(bandpassed_frame, frame, diameter/2.)
    
    return noise

def test_parameters(dataset_name, framenumber, diameter, minmass, percentile = 65, invert = False, xlims = None, ylims = None, markersize = 1, color = '#FF0000', superimpose_phase = True, extension = '*.tif', vmin = None, vmax = None):
    '''
    Test tracking parameters. This function uses trackpy to look for the bright spots in a frame. The aim of this function is to allow you to tune the tracking parameters on a frame of your choice, before tracking a whole movie.

    INPUT
    -----
    dataset_name : string
        The full path name to the folder that contains the images you want to analyse.

    framenumber : integer
        The number of the frame you want to try your tracking parameters on.
        
    diameter, minmass, percentile : parameters for trackpy.locate 
        Look at the trackpy documentation: https://soft-matter.github.io/trackpy/
        
    invert : boolean, defaults to False
        Option to invert the image, if the spots of interest are darker than the background - like in phase movies. See also trackpy.locate.
        
    xlims, ylims : lists, default to None
            This allows for cropping. When a list of integers, they will be the limits of the cropped image.
            
    markersize : int, defaults to 1
        The size of the marker that annotates the found points. Useful for comparing two results on the same plot, to decide on the optimal choice. 

    color : str, defaults to '#FF0000'
        The color of the markers that annotate the found spots. I like to specify it in hexadecimal. Useful for comparing the results of different tracking parameters on the same image.
        
    superimpose_phase : boolean, defaults to True
        Option to superimpose the phase image on top of the fluorescent image when it plots the result.
        
    extension : string, defaults to '*.tif'
        The file extension of the image files that contain the images that you will be tracking.
        
    OUTPUT
    ------
    A python DataFrame with the coordinates of all the found bright spots, along with information on their size, brigthness, shape, and an identification number (again, see also trackpy documentation).
        
    ''' 
    #pylab.gray()    # sets all subsequent plots to be in grayscale
    
    fl_frames = load_images(dataset_name + 'images/', extension = extension)  # load all the fluorescent images (see initial comments for directory architecture)
    if superimpose_phase:
        ph_frames = load_images(dataset_name + 'phase_images/', extension = extension)  # load all the fluorescent images (see initial comments for directory architecture)

    if isinstance(xlims, list):
        fl_frames = [x[:, xlims[0]:xlims[1]] for x in fl_frames]
        if superimpose_phase:
            ph_frames = [x[:, xlims[0]:xlims[1]] for x in ph_frames]

    if isinstance(ylims, list):
        fl_frames = [x[ylims[0]:ylims[1], :] for x in fl_frames]
        if superimpose_phase:
            ph_frames = [x[ylims[0]:ylims[1], :] for x in ph_frames]

    coordinates = trackpy.locate(fl_frames[framenumber], diameter, minmass=minmass, percentile = percentile, invert = invert)  # now find all the bright spots for the chosen frame

    pylab.figure('testing spot parameters')

    if superimpose_phase:
#pylab.imshow(ph_frames[framenumber], alpha = 1, vmin = ph_frames[framenumber].min(), vmax = ph_frames[framenumber].max())
        f = ph_frames[framenumber]
    else:
        f = fl_frames[framenumber]

    if isinstance(vmin, int) or isinstance(vmax, int):
        trackpy.annotate(coordinates[coordinates.frame == framenumber], f, plot_style = {'markersize': markersize, 'markeredgewidth' : 1}, imshow_style = {'vmin' : vmin, 'vmax' : vmax, 'cmap' : 'Greens_r', 'alpha' : 0.5}, color = color)   # and show what you've found
    else:
        trackpy.annotate(coordinates[coordinates.frame == framenumber], f, plot_style = {'markersize': markersize, 'markeredgewidth' : 1}, imshow_style = {'vmin' : vmin, 'vmax' : vmax, 'cmap' : 'Greens_r', 'alpha' : 0.75}, color = color)   # and show what you've found

    print('diameter = ' + str(diameter) + ', minmass = ' + str(minmass) + ', found ' + str(len(coordinates)) + ' particles')

    return coordinates

def find_spots_in_movie(dataset_name, diameter, minmass, percentile = 65, invert = False, xlims = None, ylims = None, final_frame = None, superimpose_phase = True, extension = '.tif', use_if_existing_coordinates = False):
    '''
    Find the bright spots in a collection of frames. Essentially a glorified version of trackpy.batch.
            
    INPUT
    -----
    dataset_name : string
        The full path name to the folder that contains the images you want to analyse.
            
    diameter, minmass, percentile : parameters for trackpy.locate
        See trackpy documentation.
        
    invert : boolean, defaults to False
        Option to invert the image, if the spots of interest are darker than the background - like in phase movies. See also trackpy.batch.
        
    xlims, ylims : default to None
        If not None, they are lists of x- and y-limits, if you want to only track a subset of the images.
        
    final_frame : int, defaults to None
        If you don't want to track the whole movie, you can specify here the last frame up to which you will track it. If you do not specify, it will take the last frame from the trajectories DataFrame.
        
    superimpose_phase : boolean, defaults to True
      Option to superimpose the phase image on top of the fluorescent image when it plots the result.
      
    extension : string, defaults to '*.tif'
        The file extension of the image files that contain the images that you will be tracking.
    '''
    
    analysis_folder = 'analysis'
    print('dataset name:')
    print(dataset_name)
    
    if analysis_folder not in os.listdir(dataset_name):
        #print('no folder')
       os.mkdir(dataset_name + analysis_folder) # If the analysis folder does not yet exist, make it.
    
    # In the analysis folder, you will also make a new folder for each new set of tracking parameters that you will choose.
    
    # First you construct the name of this folder, which contains all the parameters.
    
    label = 'diameter'+ str(diameter) + '_minmass' + str(minmass) + '_percentile' + str(percentile)

    # Then you make the folder, if it does not exist already. If it does, its contents will be overwritten. In principle this should not matter, since the function inputs should be the same, and thus so should the results.

    coordinate_directory = dataset_name + analysis_folder + '/' + label

    if label not in os.listdir(dataset_name + analysis_folder):
        os.mkdir(coordinate_directory)

    fl_frames = load_images(dataset_name +'images/', extension = extension) # load all the fluorescent images, which you will track
    if superimpose_phase:
        ph_frames = load_images(dataset_name +'phase_images/', extension = extension) # load all the phase images
    print(len(fl_frames))  # print the number of loaded frames (a sanity-check)

    if isinstance(xlims, list):  # if you chose to crop the images and only look at a subset, the cropping happens here
        fl_frames = [x[xlims[0]:xlims[1], ylims[0]:ylims[1]] for x in fl_frames]
        if superimpose_phase:
            ph_frames = [x[xlims[0]:xlims[1], ylims[0]:ylims[1]] for x in ph_frames]

    if superimpose_phase:
        ph_start = ph_frames[0]     # first phase image
        ph_end = ph_frames[len(ph_frames)-1]  # last phase image

    if 'coordinates.pkl' not in os.listdir(coordinate_directory):  # if you have not tracked and saved the coordinates already, do it now
        coordinates = trackpy.batch(fl_frames, diameter, minmass = minmass, percentile = percentile, invert = invert)  # find the bright spots
        for i in list(set(coordinates.frame)):
            coordinates.loc[coordinates.frame==i, 'noise'] = calculate_noise(fl_frames[i], diameter)[0]  # calculate the noise in each frame (after Savin and Doyle) - [0] is the mean, [1] would be the std
        if 'noise' in coordinates.columns:
            coordinates['snr'] = coordinates.mass/coordinates.noise
        else:
            print('Note that I am not calculating the SNR because I do not have a noise estimate.')

        coordinates.to_pickle(coordinate_directory + '/coordinates.pkl')  # save their coordinates as a pandas DataFrame
    else:
        if use_if_existing_coordinates:
            answer = 'y'
        else:
            answer = input('Shall I use the coordinates already saved for this set of tracking parameters? Type y if yes, or n if you want me to track the particles from scratch.')
        if answer == 'y':
            coordinates = pandas.read_pickle(coordinate_directory + '/coordinates.pkl')  # if the coordinates already exist, just load them. Saves time.
        elif answer == 'n':
            coordinates = trackpy.batch(fl_frames, diameter, minmass = minmass, percentile = percentile, invert = invert)  # find the bright spots
            for i in list(set(coordinates.frame)):
                coordinates.loc[coordinates.frame==i, 'noise'] = calculate_noise(fl_frames[i], diameter)[0]  # calculate the noise in each frame (after Savin and Doyle)
            if 'noise' in coordinates.columns:
                coordinates['snr'] = coordinates.mass/coordinates.noise
            else:
                print('Note that I am not calculating the SNR because I do not have a noise estimate.')

            coordinates.to_pickle(coordinate_directory + '/coordinates.pkl')  # save their coordinates as a pandas DataFrame
        else:
            raise ValueError('I do not understand your answer. Please type y or n.')

#        warnings.warn('I am using the coordinates found previously for this set of tracknig parameters. \nIf you do not want that, please erase the file that contains the DataFrame of coordinates, and try again.')


    fig = pylab.figure('annotated frame 00')  # plot the static coordinates on the first frame
    pylab.gray()
    trackpy.annotate(coordinates[coordinates.frame == 1], fl_frames[0], plot_style = {'markersize': 1}, legend = False)
    if superimpose_phase:
        pylab.imshow(ph_frames[0], alpha = 0.5)
    ax = fig.add_subplot(111)
    ax.text(fl_frames[0].shape[1] + 25, fl_frames[0].shape[0] - 125, 'diameter: ' + str(diameter) + 'px\nminmass: ' + str(minmass) + '\npercentile: ' + str(percentile), fontsize = 11)
    pylab.title('frame 00')
    pylab.ylim(0, fl_frames[0].shape[0])
    pylab.xlim(0, fl_frames[0].shape[1])
    pylab.savefig(dataset_name + analysis_folder + '/' + label + '/coordinates_start.png', dpi = 300)

    fig = pylab.figure('annotated frame ' + str(len(fl_frames)-1))  # plot the static coordinates on the last frame
    trackpy.annotate(coordinates[coordinates.frame == len(fl_frames)-1], fl_frames[len(fl_frames)-1], plot_style = {'markersize': 1}, legend = False)
    ax = fig.add_subplot(111)
    if superimpose_phase:
        pylab.imshow(ph_frames[len(ph_frames)-1], alpha = 0.5)
    ax.text(fl_frames[0].shape[1] + 25, fl_frames[0].shape[0] - 125, 'diameter: ' + str(diameter) + 'px\nminmass: ' + str(minmass) + '\npercentile: ' + str(percentile), fontsize = 11)
    pylab.title('annotated frame ' + str(len(fl_frames)-1))
    pylab.ylim(0, fl_frames[len(fl_frames)-1].shape[0])
    pylab.xlim(0, fl_frames[len(fl_frames)-1].shape[1])
    pylab.savefig(dataset_name + analysis_folder + '/' + label + '/coordinates_end.png', dpi = 300)

#   noise = calculate_noise(fl_frames[framenumber], diameter)
#   average_signal = coordinates.mass.mean()
#   average_snr = average_signal/noise[0]

#   print('Average S/N: ' + str(average_snr) + '.')
#   print('Noise level: ' + str(noise[0]) + '.')
#   print('Noise standard deviation: ' + str(noise[1]) + '.')

    return coordinates

def translate_selection_rule(selection_rule):
    '''
    Creates a string that describes the selection rule. Useful for creating representative filenames and looking for them afterwards.
    
    INPUT
    -----
    selection_rule : list of lists of the form ['quantity', 'relation', value]; defaults to None
    Here you specify the selection criteria. For instance, to only include spots with SNR > 10, you would write selection_rule = [['snr', 'greater', 10]]. To add a second rule on size, you would write selection_rule = [['snr', 'greater', 10], ['size','lesser', 100]]. Etc.
    Note the following rules: the first item of the list must be a column of the coordinates DataFrame. The second column must be one of ['greater', 'lesser', 'inbetween', 'equal']. If 'inbetween', then the value entry must be a list of two numbers.
    
    
    OUTPUT
    ------
    A string that contains the information of the rule.
    '''
    rules = ''  # initialize string of all the rule descriptions
    
    if isinstance(selection_rule, list):
        for i in numpy.arange(len(selection_rule)):
            quantity = selection_rule[i][0]
            relation = selection_rule[i][1]
            selection_value = selection_rule[i][2]
            #print(quantity)
            #print(relation)
            #print(selection_value)
            if relation == 'inbetween':
                value_string = [0,0]
                value_string[0] = str(round(float(selection_value[0]),2))
                print(value_string[0])
                value_string[0] = value_string[0].replace('.', 'p')
                value_string[1] = str(round(float(selection_value[1]),2))
                value_string[1] = value_string[1].replace('.', 'p')
                rule_in_words = quantity + '_inbetween_' + value_string[0] + '_' + value_string[1]
            elif relation == 'greater':
                value_string = str(round(float(selection_value),2))
                value_string = value_string.replace('.', 'p')
                rule_in_words = quantity + '_greater_than_' + value_string
            elif relation == 'lesser':
                value_string = str(round(float(selection_value),2))
                value_string = value_string.replace('.', 'p')
                rule_in_words = quantity + '_smaller_than_' + value_string
            elif relation == 'equal':
                value_string = str(round(float(selection_value),2))
                value_string = value_string.replace('.', 'p')
                rule_in_words = quantity + '_equals_' + value_string

            rules = rules + '_' + rule_in_words
    
    return rules

def select_coordinates(coordinates, selection_rule = None, frame = None):
    '''
    Selects coordinates based on some value, for example the SNR.
    
    INPUT
    -----
    coordinates : DataFrame
        The DataFrame with the coordinates you want to select from.
    
    selection_rule : list of lists of the form ['quantity', 'relation', value]; defaults to None
        Here you specify the selection criteria. For instance, to only include spots with SNR > 10, you would write selection_rule = [['snr', 'greater', 10]]. To add a second rule on size, you would write selection_rule = [['snr', 'greater', 10], ['size','lesser', 100]]. Etc.
        Note the following rules: the first item of the list must be a column of the coordinates DataFrame. The second column must be one of ['greater', 'lesser', 'inbetween', 'equal']. If 'inbetween', then the value entry must be a list of two numbers.

    frame : integer, defaults to None
        If you want the selection rule to only be valid for a specific frame, here you speficy this frame.
        
    OUTPUT
    ------
    A new coordinates DataFrame, where points have been removed if they do not comply with the selection rule.
    '''
    
    if isinstance(coordinates, pandas.DataFrame):
        selected_coordinates = coordinates.copy()
    else:
        selected_coordinates = None

    if isinstance(selection_rule, list):
        for i in numpy.arange(len(selection_rule)):
            rule_in_words = translate_selection_rule(selection_rule)  # the characterizing string that will go into the filenames of the resulting files
            print(rule_in_words)
            quantity = selection_rule[i][0]
            relation = selection_rule[i][1]
            selection_value = selection_rule[i][2]
            if relation == 'inbetween':
                if isinstance(coordinates, pandas.DataFrame):
                    selected_coordinates = selected_coordinates.loc[(selected_coordinates[quantity] > selection_value[0]) & (selected_coordinates[quantity] < selection_value[1])]
                else:
                    selected_coordinates = coordinates
            elif relation == 'greater':
                if isinstance(coordinates, pandas.DataFrame):
                    selected_coordinates = selected_coordinates.loc[selected_coordinates[quantity] > selection_value]
                else:
                    selected_coordinates = coordinates
            elif relation == 'lesser':
                if isinstance(coordinates, pandas.DataFrame):
                    selected_coordinates = selected_coordinates.loc[selected_coordinates[quantity] < selection_value]
                else:
                    selected_coordinates = coordinates
            elif relation == 'equal':
                if isinstance(coordinates, pandas.DataFrame):
                    selected_coordinates = selected_coordinates.loc[selected_coordinates[quantity] == selection_value]
                else:
                    selected_coordinates = coordinates

    return selected_coordinates, rule_in_words

def link_movie_spots(dataset_name, diameter, minmass, search_range, memory, stub_length, percentile = 65, xlims = None, ylims = None, superimpose_phase = True, extension = '.tif', selection_rule = None, plot = True):
    '''
    Make tracks our of the (already located) bright spots in a collection of frames.
        
    INPUT
    -----
    dataset_name : string
        The full path name to the folder that contains the images you want to analyse.
        
    diameter, minmass, percentile : parameters for trackpy.locate
        See trackpy documentation.
        
    search_range, memory : parameters for trackpy.link_df (links spots into trajectories)
        See trackpy documentation.
        
    stub_length : parameter for trackpy.filter_stubs (filters trajectories)
        See trackpy documentation.
        
    xlims, ylims : default to None
        If not None, they are lists of x- and y-limits, if you want to only track a subset of the images.
        
    superimpose_phase : boolean, defaults to True
        Option to superimpose the phase image on top of the fluorescent image when it plots the result.
        
    extension : string, defaults to '*.tif'
        The file extension of the image files that contain the images that you will be tracking.
        
    selection_rule : list of lists of the form ['quantity', 'relation', value]; defaults to None
        Here you specify the selection criteria. For instance, to only include spots with SNR > 10, you would write selection_rule = [['snr', 'greater', 10]]. To add a second rule on size, you would write selection_rule = [['snr', 'greater', 10], ['size','lesser', 100]]. Etc.
        Note the following rules: the first item of the list must be a column of the coordinates DataFrame. The second column must be one of ['greater', 'lesser', 'inbetween', 'equal']. If 'inbetween', then the value entry must be a list of two numbers.

        
    OUTPUT
    ------
    Two DataFrames: one of all the trajectories, and one of all the filtered trajectories.
    '''
    now = datetime.datetime.now()
    string_now = '_' + now.strftime("%Y%m%d")

    analysis_folder = 'analysis'  # Makes a new folder every day, where it will save the results. Here you define the folder name, based on today's date.

    fl_frames = load_images(dataset_name +'images/', extension = extension) # load all the fluorescent images, which you will track
    if superimpose_phase:
        ph_frames = load_images(dataset_name +'phase_images/', extension = extension) # load all the phase images, which you will superimpose over the fluorescent ones on the plots
        ph_start = ph_frames[0]
        ph_end = ph_frames[len(ph_frames)-1]
    
    label1 = 'diameter'+ str(diameter) + '_minmass' + str(minmass) + '_percentile' + str(percentile)
    
    coordinate_directory = dataset_name + analysis_folder + '/' + label1 + '/'
    
    # If the coordinates have not already been found, raise an error that says so. Otherwise, load them.
    
    if 'coordinates.pkl' not in os.listdir(coordinate_directory):
        raise ValueError('I do not have the coordinates for this movie and set of parameters.')
    else:
        coordinates = pandas.read_pickle(coordinate_directory + 'coordinates.pkl')

    # Within the directory of the coordinates, make a new directory for the trajectories, named after the linking parameters.
    label2 = 'search_range' + str(search_range) + '_memory' + str(memory).zfill(2)
    trajectory_directory = label2

    if trajectory_directory not in os.listdir(coordinate_directory):
        os.mkdir(coordinate_directory + trajectory_directory)
    
    if selection_rule:
        coordinates, rule_in_words = select_coordinates(coordinates, selection_rule = selection_rule)

    trajectories = trackpy.link_df(coordinates, search_range = search_range, memory = memory)  # link the bright spots into trajectories
    if 'noise' in trajectories.columns:
        trajectories['snr'] = trajectories.mass/trajectories.noise
    else:
        print('Note that I am not calculating the SNR because I do not have a noise estimate.')

    if selection_rule is None:
        trajectories.to_pickle(coordinate_directory + trajectory_directory + '/trajectories.pkl')  # save the trajectories as a DataFrame
    elif isinstance(selection_rule, list):
        trajectories.to_pickle(coordinate_directory + trajectory_directory + '/trajectories' + '_' + rule_in_words + '.pkl')  # save the trajectories as a DataFrame

    filtered_trajectories = trackpy.filter_stubs(trajectories, threshold = stub_length) # filter the trajectories according to their length
    label3 = '_stub_length' + str(stub_length).zfill(3)
    if isinstance(selection_rule, list):
        label3 = label3 + '_' + rule_in_words
    l = coordinate_directory + trajectory_directory + '/filtered_trajectories' + label3
    filtered_trajectories.to_pickle(l + '.pkl') # save the filtered trajectories as a DataFrame

    if plot:
#        pylab.close('filtered trajectories, beginning')
#        fig = pylab.figure('filtered trajectories, beginning', figsize = (6, 6))  # plot the filtered trajectories up to frame 1, just at the beginning of the movie
#        pylab.gray()
#        pylab.title('filtered trajectories, beginning')
#        ax = fig.add_subplot(111)
#        ax.text(fl_frames[0].shape[1] + 25, fl_frames[0].shape[0] - 215, 'diameter: ' + str(diameter) + ' px\nminmass: ' + str(minmass) + '\npercentile: ' + str(percentile) + '\nsearch range: ' + str(search_range) + '\nmemory: ' + str(memory) + '\nstub length: ' + str(stub_length) , fontsize = 11)
#        trackpy.plot_traj(filtered_trajectories[filtered_trajectories.frame<=1], superimpose = fl_frames[0])
#        if superimpose_phase:
#            pylab.imshow(ph_frames[0], alpha = 0.5)  # and superimpose the phase image, with some transparency so you can also see the fluorescent and the trajectory points
#        pylab.ylim(0, fl_frames[0].shape[0])
#        pylab.xlim(0, fl_frames[0].shape[1])
#        pylab.savefig(coordinate_directory +  trajectory_directory + '/filtered-trajectories_start' + label3 + '.png', dpi = 300)
#
#        pylab.close('filtered trajectories, end')
#        fig = pylab.figure('filtered trajectories, end', figsize = (6, 6))  # plot the filtered trajectories at the end of the movie only
#        ax = fig.add_subplot(111)
#        ax.text(fl_frames[0].shape[1] + 25, fl_frames[0].shape[0] - 215, 'diameter: ' + str(diameter) + ' px\nminmass: ' + str(minmass) + '\npercentile: ' + str(percentile) + '\nsearch range: ' + str(search_range) + '\nmemory: ' + str(memory) + '\nstub length: ' + str(stub_length) , fontsize = 11)
#        if len(filtered_trajectories[filtered_trajectories.frame>=len(fl_frames)-2]) > 0:  # It may be that no particles are left at the final frame, in which case plot_traj will give an error saying that the DataFrame is empty.
#            trackpy.plot_traj(filtered_trajectories[filtered_trajectories.frame>=len(fl_frames)-2], superimpose = fl_frames[len(fl_frames)-1])
#        elif len(filtered_trajectories[filtered_trajectories.frame>=len(fl_frames)-2]) == 0:
#            pylab.imshow(fl_frames[len(fl_frames)-1])
#        if superimpose_phase:
#            pylab.imshow(ph_frames[len(ph_frames)-1], alpha = 0.5)  # again superimpose the last phase frame with some transparency, as above
#        pylab.title('filtered trajectories, end')
#        pylab.ylim(0, fl_frames[len(fl_frames)-1].shape[0])
#        pylab.xlim(0, fl_frames[len(fl_frames)-1].shape[1])
#        pylab.savefig(coordinate_directory + trajectory_directory + '/filtered-trajectories_end' + label3 + '.png', dpi = 300)

        pylab.close('filtered trajectories, complete movie')
        fig = pylab.figure('filtered trajectories, complete movie', figsize = (6, 6))  # plot the filtered trajectories for the whole movie
        pylab.gray()
        pylab.title('filtered trajectories, complete movie')
        ax = fig.add_subplot(111)
        ax.text(fl_frames[0].shape[1] + 25, fl_frames[0].shape[0] - 215, 'diameter: ' + str(diameter) + ' px\nminmass: ' + str(minmass) + '\npercentile: ' + str(percentile) + '\nsearch range: ' + str(search_range) + '\nmemory: ' + str(memory) + '\nstub length: ' + str(stub_length) , fontsize = 13)
        trackpy.plot_traj(filtered_trajectories, superimpose = fl_frames[len(fl_frames)-1])
        if superimpose_phase:
            pylab.imshow(ph_frames[0], alpha = 0.5)  # and superimpose the first phase image, with some transparency so you can also see the fluorescent image and the trajectory points
        pylab.ylim(0, fl_frames[0].shape[0])
        pylab.xlim(0, fl_frames[0].shape[1])
        pylab.savefig(coordinate_directory +  trajectory_directory + '/filtered_trajectories_start' + label3 + string_now + '.png', dpi = 300)

    print('I found ' + str(len(set(filtered_trajectories.particle))) + ' trajectories.')
    
    return trajectories, filtered_trajectories, l
  
def track(dataset_name, diameter, minmass, search_range, memory, stub_length, percentile = 65, xlims = None, ylims = None, superimpose_phase = True, extension = '.tif', selection_rule = None, plot = True, use_if_existing_coordinates = True):
    '''
    Track the bright spots in a collection of frames.
    
    INPUT
    -----
    dataset_name : string
        The full path name to the folder that contains the images you want to analyse.

    diameter, minmass, percentile : parameters for trackpy.locate 
        See trackpy documentation.    
    
    search_range, memory : parameters for trackpy.link_df (links spots into trajectories)
        See trackpy documentation.    
    
    stub_length : parameter for trackpy.filter_stubs (filters trajectories)
        See trackpy documentation.
        
    xlims, ylims : default to None
        If not None, they are lists of x- and y-limits, if you want to only track a subset of the images.
        
    superimpose_phase : boolean, defaults to True
        Option to superimpose the phase image on top of the fluorescent image when it plots the result.
        
    phase : string, defaults to 'BF'
        An identifies string that needs to be part of the filenames of the two phase images, in order for the code to pick them up.
        
    extension : string, defaults to '*.tif'
        The file extension of the image files that contain the images that you will be tracking.
        
    selection_rule : list of lists of the form ['quantity', 'relation', value]; defaults to None
        Here you specify the selection criteria. For instance, to only include spots with SNR > 10, you would write selection_rule = [['snr', 'greater', 10]]. To add a second rule on size, you would write selection_rule = [['snr', 'greater', 10], ['size','lesser', 100]]. Etc.
        Note the following rules: the first item of the list must be a column of the coordinates DataFrame. The second column must be one of ['greater', 'lesser', 'inbetween', 'equal']. If 'inbetween', then the value entry must be a list of two numbers.
        
    OUTPUT
    ------
    Five DataFrames: the coordinates, all the trajectories, the filtered trajectories filtered according to their length, the trajectorie s. 
    
    '''
  
    current_folder = os.getcwd()
    analysis_folder = 'analysis'  # Makes a new analysis folder
  
    label1 = 'diameter'+ str(diameter) + '_minmass' + str(minmass) + '_percentile' + str(percentile)
    coordinate_directory = dataset_name + analysis_folder + '/' + label1 + '/'

    label2 = 'search_range' + str(search_range) + '_memory' + str(memory).zfill(2)
    trajectory_directory = label2
    
    coordinates = find_spots_in_movie(dataset_name, diameter, minmass, percentile = percentile, xlims = xlims, ylims = ylims, superimpose_phase = superimpose_phase, extension = extension, use_if_existing_coordinates = use_if_existing_coordinates)
    
    trajectories, filtered_trajectories, filtered_trajectories_filename = link_movie_spots(dataset_name, diameter, minmass, search_range, memory, stub_length, percentile = percentile, xlims = xlims, ylims = ylims, superimpose_phase = superimpose_phase, extension = extension, selection_rule = selection_rule, plot = plot)
    
    os.chdir(coordinate_directory + trajectory_directory)
  
    os.chdir(current_folder)
    
    return coordinates, trajectories, filtered_trajectories, filtered_trajectories_filename

#def select_trajectories(trajectories, selection_rule, frame):
#    '''
#    trajectories : pandas DataFrame
#        
#    selection_rule : list of lists of the form ['quantity', 'relation', value]; defaults to None
#        Here you specify the selection criteria. For instance, to only include spots with SNR > 10, you would write selection_rule = [['snr', 'greater', 10]]. To add a second rule on size, you would write selection_rule = [['snr', 'greater', 10], ['size','lesser', 100]]. Etc.
#        Note the following rules: the first item of the list must be a column of the coordinates DataFrame. The second column must be one of ['greater', 'lesser', 'inbetween', 'equal']. If 'inbetween', then the value entry must be a list of two numbers.
#    frame :
#    '''
#    result = trajectories[]

def trajectory_movie(dataset_name, analysis_folder, trajectory_specifier, particle_id = None, selection_rule = None, xlimits = [None, None], ylimits = [None, None], superimpose_phase = True, extension = '.tif', vmin = None, vmax = None):
    '''
    Make a movie where the evolution of the trajectory is shown on the images, for a specific particle.
    It is useful not only for sharing the results, but also for sanity-checking what you are doing. I recommend making and watching these movies as you optimize the tracking parameters.
    
    IMPORTANT: Right now it assumes you want to see the filtered trajectories, with stub_length as specified in the function input.
    
    INPUT
    -----
    dataset_name : string
        The full path name to the folder that contains the images you want to analyse.
    
    analysis_folder : string
        The name of the folder that contains the tracking results that you want to show. For each dataset, you will likely try various parameters; by choosing the analysis folder, you are choosing which result to show. The analysis folder should have the following format: yyyymmdd-analysis/diameterx_minmassx_percentilex_search_rangex_memoryx_stub_lengthx/ where 'x' are values of the corresponding parameters.

    stub_length : integer
        The length of the trajectories you want to plot. It is helpful to specifiy it since you may have tried filtering the trajectories to different lengths (see trackpy.filter_stubs for more).
        
    particle_id : integer, defaults to None
        If an integer, it is the id of the particle of interest: the movie will only include the area around that particle, and its trajectory alone.
        
    directory_specifier : string, defaults to 'stub_length'
        The subdirectory in which the movie frames will be saved. It is useful to make these subdirectories, because you may want to take a look at trajectories filtered with different stub lengths, which - in the current organization - are in the same folder.
        
    selection_rule : list of lists of the form ['quantity', 'relation', value]; defaults to None
        Here you specify the selection criteria. For instance, to only include spots with SNR > 10, you would write selection_rule = [['snr', 'greater', 10]]. To add a second rule on size, you would write selection_rule = [['snr', 'greater', 10], ['size','lesser', 100]]. Etc.
        Note the following rules: the first item of the list must be a column of the coordinates DataFrame. The second column must be one of ['greater', 'lesser', 'inbetween', 'equal']. If 'inbetween', then the value entry must be a list of two numbers.

    superimpose_phase : boolean, defaults to True
        Option to superimpose the phase image on top of the fluorescent image when it plots the result.

    extension : string, defaults to '*.tif'
        The file extension of the image files that contain the images that you will be saving.
        
    OUTPUT
    ------
    The trajectories DataFrame that you are plotting.
    '''
    
    working_directory = dataset_name + analysis_folder
    #trajectory_specifier = select_coordinates(None, selection_rule = selection_rule)[1]
    # Make a new directory for the frames of this movie, if it does not yet exist.
    trajectory_filenames = [x for x in os.listdir(working_directory) if 'filtered_trajectories' in x]  # find the files that contain filtered trajectories from the tracking analysis. There may be more than one filtered trajectories, with different stub lengths.
    if isinstance(trajectory_specifier, str):
        trajectory_filename = [x for x in trajectory_filenames if trajectory_specifier in x]  # from all the filtered trajectory files, pick the one that was generated according to the selection rule you have specified.
        if len(trajectory_filename) > 1:
            raise ValueError('There are more than one trajectory files with these specifications.')
        elif len(trajectory_filename) == 0:
            raise ValueError('There are no trajectory files with these specifications.')
        elif len(trajectory_filename) == 1:
            trajectory_filename = trajectory_filename[0]
            print('I have chosen this file: ')
            print(trajectory_filename)
        else:
            raise ValueError('Something is off with the trajectory filenames, check the script.')

#    if len(trajectory_filename) > 1:
#        if 'stub_length' in directory_specifier:
#            stub_length = directory_specifier.split('stub_length')[1][:3]
#        else:
#            stub_length = input('What stub length should I pick?')
#        stub_length_label = '-stub_length' + str(stub_length).zfill(3)
#        trajectory_filename = 'filtered_trajectories' + stub_length_label
#    else:
#        trajectory_filename = trajectory_filename[0]


    trajectories = pandas.read_pickle(working_directory + trajectory_filename)  # and load it. These are the trajectories you will show.
    print(working_directory + trajectory_filename)
    all_framenumbers = set(trajectories.frame)  # Extract all framenumbers - it is useful for matching correctly each frame with the trajectories up to that frame.

    if isinstance(particle_id, int):
        trajectories = trajectories[trajectories.particle==particle_id]
        xo, yo = trajectories.x.mean(), trajectories.y.mean()
        print('I have chosen particle ' + str(particle_id) + '.')
    if isinstance(xlimits[0], int):
        trajectories = trajectories[(trajectories.x > xlimits[0]) & (trajectories.x < xlimits[1])]
    occupied_framenumbers = list(set(trajectories.frame))  # Extract framenumbers of participating frames.
    # we might use the line below at a later stage of analysis
    #discarded_trajectories = pandas.read_pickle(working_directory + 'discarded_trajectories-stub_length' + str(stub_length).zfill(3) + '-ratio0.5')
#new_directory = 'trajectory_movie_frames/' + directory_specifier
    new_directory = 'trajectory_movie_frames/' + trajectory_specifier

    if isinstance(particle_id, int):
        new_directory = new_directory + '/particle' + str(particle_id).zfill(4)
    else:
        new_directory = new_directory + 'all_particles'

    if superimpose_phase:
        new_directory = new_directory + '_with_ph/'

    if new_directory not in os.listdir(working_directory):
        os.makedirs(working_directory + '/' + new_directory)

    fl_frames = load_images(dataset_name +'images/', extension = extension)  # load the fluorescence images
    total_framenumber = len(fl_frames)
    if superimpose_phase:
        ph_frames = load_images(dataset_name +'phase_images/', extension = extension)  # load the phase images

#   In what follows, find the minima and maxima of intensity in each image, and then the minima of all the minima and the maxima of all the maxima. You will use these to set the grayscale limits in the frames of the new movie, as 'vmin = ' and 'vmax = ' in the plots below. Without this, the default would adapt the grayscale to the maximum and minimum intensity of each frame, and the resulting movie would look like it flickers since not all frames have the same max and min intensities.

    minima = []  # List of minima and maxima of intensity, per image.
    maxima = []

    for i in fl_frames:
        minima.append(i.min())
        maxima.append(i.max())

    minima = numpy.array(minima)
    maxima = numpy.array(maxima)

    global_min = minima.min()
    global_max = maxima.max()

    if superimpose_phase:
        minima_ph = []  # List of minima and maxima of intensity, per image.
        maxima_ph = []

        for i in ph_frames:
            minima_ph.append(i.min())
            maxima_ph.append(i.max())

        minima_ph = numpy.array(minima_ph)
        maxima_ph = numpy.array(maxima_ph)

        global_min_ph = minima_ph.min()
        global_max_ph = maxima_ph.max()

    pylab.gray()

    for i in list(all_framenumbers):
        #print(i)
        fig = pylab.figure('current figure', figsize = (6,6))
        plotstyle = {}
        plotstyle_ann = {'color' : '#FF0000', 'markeredgewidth' : 1, 'markersize' : 30}
        #if isinstance(particle_id, int):
            # we may use the following three lines at a later stage
            #if i in discarded_trajectories.frame:
            #   plotstyle_traj = {'color' : '#FF0000', 'markeredgewidth' : 1, 'markersize' : 30}
            #else:
        plotstyle_traj = {'color' : '#009900', 'markeredgewidth' : 1, 'markersize' : 30}

        if i == occupied_framenumbers[0]:
            trackpy.plot_traj(trajectories[trajectories.frame <= i],
                              superimpose = fl_frames[i],
                              legend = False, plot_style = {'color' : '#00FF00'})
            trackpy.annotate(trajectories[trajectories.frame==i], fl_frames[i], plot_style = plotstyle_ann, imshow_style = {'vmin' : global_min, 'vmax' : global_max, 'cmap' : 'gray'})
        elif i == occupied_framenumbers[1]:
            trackpy.plot_traj(trajectories[trajectories.frame <= (i-1)],
                              superimpose = fl_frames[i],
                              legend = False, plot_style = plotstyle_traj)
            trackpy.plot_traj(trajectories[(trajectories.frame >= i-1) & (trajectories.frame <= i)],
                                  superimpose = fl_frames[i],
                                  legend = False, plot_style = {'color' : '#00FF00'})
            trackpy.annotate(trajectories[trajectories.frame==i], fl_frames[i], plot_style = plotstyle_ann, imshow_style = {'vmin' : global_min, 'vmax' : global_max, 'cmap' : 'gray'})
        elif i in occupied_framenumbers:
            trackpy.plot_traj(trajectories[trajectories.frame <= i - 1],
                              superimpose = fl_frames[i],
                              legend = False, plot_style = plotstyle_traj)
                #trackpy.plot_traj(trajectories[(trajectories.frame >= i - 1) & (trajectories.frame <= i-1)],
                #             superimpose = fl_frames[i],
                #             legend = False, plot_style = plotstyle_traj)
            trackpy.plot_traj(trajectories[(trajectories.frame >= i-1) & (trajectories.frame <= i)],
                                                superimpose = fl_frames[i],
                                                legend = False, plot_style = {'color' : '#00FF00'})
            trackpy.annotate(trajectories[trajectories.frame==i], fl_frames[i], plot_style = plotstyle_ann, imshow_style = {'vmin' : global_min, 'vmax' : global_max, 'cmap' : 'gray'})

        else:
            pylab.imshow(fl_frames[i], vmin = global_min, vmax = global_max, cmap = 'gray')
            pylab.xlabel('x [px]')
            pylab.ylabel('y [px]')

        pylab.imshow(fl_frames[i], vmin = global_min, vmax = global_max, cmap = 'gray')
        pylab.xlabel('x [px]')
        pylab.ylabel('y [px]')

        #if isinstance(particle_id, int) and i in discarded_trajectories.frame:
        #   trackpy.annotate(trajectories[trajectories.frame==i], fl_frames[i], plot_style = plotstyle_ann)

        if superimpose_phase:
            pylab.imshow(ph_frames[i], alpha = 0.5, vmin = global_min_ph, vmax = global_max_ph, cmap = 'gray')
        if any(xlimits) or any(ylimits):
            pylab.ylim(ylimits[0], ylimits[1])
            pylab.xlim(xlimits[0], xlimits[1])
        elif isinstance(particle_id, int):
            pylab.ylim(yo + 30, yo - 30)
            pylab.xlim(xo - 30, xo + 30)
#fig.patch.set_facecolor('red')
        pylab.savefig(working_directory + new_directory + '/frame' + str(i).zfill(3) + string_now + '.png'
                      #, transparent = True
                      )
        pylab.close()
#print('frame ' + str(i))

    # The function returns the loaded trajectories, just in case you want to take a look at something directly.
    return trajectories

def measure_step_size(trajectories):
    '''
    '''
    t = trajectories
    
    if t.index.name == 'frame':
        t.index.names = ['framenumber']  # This is because trackpy names the index of filtered trajectories as 'frame', while maintaining the 'frame' column. So, to remove ambiguity, if you want to clean a trajectory that has been filtered, just remane the index.


    t.sort_values(['particle', 'frame'], inplace = True) # will sort the DataFrame rows first by particle, then by frame
    t = t.set_index(numpy.arange(len(t))) # make reasonable indices that correspond to the row number (trackpy keeps the indices from the coordinates DataFrame, which are no longer meaningful in the trajectories)

    t_previous = t.copy()  # make two copies of the original trajectories DataFrame. These are auxiliary.
    t_next = t.copy()
    t_previous = t_previous.set_index(numpy.arange(1, len(t)+1))  # DataFrame t_previous is a copy of t, except the indices have all been shifted by +1.
    t_next = t_next.set_index(numpy.arange(-1, len(t)-1))  # DataFrame t_next is a copy of t, except the indices have all been shifted by -1.
    t.insert(1, 'yprevious', t_previous.y)  # Initialize a new column of floats in the original DataFrame t. This column will be populated by the y position of each particle in the previous frame.
    t.insert(2, 'ynext', t_next.y)  # This new column of floats will be populated by the y position of each particle in the next frame.
    t.insert(4, 'xprevious', t_previous.x)  # same for x
    t.insert(5, 'xnext', t_next.x)
    t.insert(8, 'previous_step_size', 0.0)  # Initialize a new column of floats that will be populated by the size of the step that was just taken by the particle.
    t.insert(9, 'next_step_size', 0.0)  # Initialize a new column of floats that will be populated by the size of the step that will next be taken by the particle.
    t.insert(6, 'previous_particle_id', t_previous.particle)  # Auxiliary column: it contains the value of the particle id of the previous row.
    t.insert(7, 'next_particle_id', t_next.particle)  # Auxiliary column: it contains the value of the particle id of the next row. These will be used to check that we are calculating step sizes from pairs of coordinates that correspond to the same particle.

    change_from_previous = t.previous_particle_id != t.particle  # this yields all the rows where there has been a change in particle id from the previous row
    t.loc[change_from_previous, 'yprevious'] = numpy.NaN  # wherever this happens, insert NaN in the entries for yprevious, xprevious - since the previous row corresponds to a different particle, and there is no value for the previous y, x positions of the particle of this row
    t.loc[change_from_previous, 'xprevious'] = numpy.NaN  # as above

    change_to_next = t.next_particle_id != t.particle  # same as previous three lines, but for rows where there is a change between the particle id in this row and the next
    t.loc[change_to_next, 'ynext'] = numpy.NaN
    t.loc[change_to_next, 'xnext'] = numpy.NaN

    t.pop('previous_particle_id')  # remove these auxiliary columns
    t.pop('next_particle_id')

    t.previous_step_size = numpy.sqrt((t.yprevious-t.y)**2+(t.xprevious-t.x)**2)  # calculate the step size the particle took to arrive at the current frame
    t.next_step_size = numpy.sqrt((t.ynext-t.y)**2+(t.xnext-t.x)**2)              # calculate the step size the particle will take to get to the next frame

    t.insert(12, 'ep_over_previous_step_size', 0.0)  # initialize new colums for the criterion: the ratio of uncertainty (ep) over the previous/next step size
    t.insert(13, 'ep_over_next_step_size', 0.0)
    
    t['ep_over_previous_step_size'] = t['ep'] / t['previous_step_size']           # Calculate the ratio of localization uncertainty to the previous step size. Ratio must be smaller than what specified in the function input.
    t['ep_over_next_step_size'] = t['ep'] / t['next_step_size']                   # Calculate the ratio of localization uncertainty to the next step size. Ratio must be smaller than what specified in the function input.

    t.pop('yprevious')
    t.pop('xprevious')
    t.pop('xnext')
    t.pop('ynext')
    t.pop('next_step_size')
    t.pop('ep_over_next_step_size')

    return t

def count_absences(trajectories):
    '''
    Count the number of frames for which a particle is absent from the movie. This is useful for checking that the choice of memory makes sense.
    
    INPUT
    -----
    trajectories : pandas DataFrame
        A pandas DataFrame with particle trajectories.
        
    OUTPUT
    ------
    Two items. The first, absences, is a dictionary of lists, where each entry of the dictionary corresponds to a particle of the input trajectories DataFrame. The second, total_absences, is a concatenated list of all absences, regardless of particle or frame.
    '''
    absences = {}
    total_absences = []
    
    for p in set(trajectories.particle):
        frames = trajectories[trajectories.particle==p].frame
        absences[p] = frames - frames.shift(periods = 1)
        absences[p] = absences[p] - 1
        total_absences = total_absences + list(absences[p])
    
    return absences, total_absences

def calculate_msds(trajectories, stub_length, px_to_micron, fps, final_frame = None, trajectory_type = 'filtered', selection_rule = None, starting_snr_range = None, file_location = None, other_label = None, recalculate_msds = True):
    '''
    Given trajectories, calculate the individual mean-square-displacements and the ensemble mean-square-displacement. 
    
    INPUT
    -----
    trajectories : string or pandas DataFrame
        If string, the directory that contains the tracked particle trajectories.
        If pandas DataFrame, the trajectories dataframe.
        
    stub_length : int
        The stub length of the trajectories, when you filtered them. This need be specified in case you want to examine the effect of different trajectory lengths, for instance; all files for trajectories will be in the same folder.
        
    selection_rule : list of lists of the form ['quantity', 'relation', value]; defaults to None
        Here you specify the selection criteria. For instance, to only include spots with SNR > 10, you would write selection_rule = [['snr', 'greater', 10]]. To add a second rule on size, you would write selection_rule = [['snr', 'greater', 10], ['size','lesser', 100]]. Etc.
        Note the following rules: the first item of the list must be a column of the coordinates DataFrame. The second column must be one of ['greater', 'lesser', 'inbetween', 'equal']. If 'inbetween', then the value entry must be a list of two numbers.
        
    final_frame : int, defaults to None
        When an integer and not None, this is the maximum frame that will be considered in the msd calculation. This is useful in cases where the movie obviously bleaches, so you don't want to consider frames beyond a certain point.
    
    uncertainty_to_step_size_ratio : float, defaults to 0.5
        Noe currently used.
        
    px_to_micron : float
        The pixel-to-micron conversion.
        
    fps : float
        The frames per second during acquisition.
        
    trajectory_type : string
        The type of trajectory to load: 'filtered' (for length) is the only type we have for now.
        
    starting_snr_range : list of floats, defaults to None
        If not None, the range of starting snr values of the trajectories you want to include in the calculation of the msd.
        
    file_location : str
        If trajectories is not a string that denotes the location of the trajectories, you will need to specify the directory where you work, with this variable.
    
    other_label : str
        If there is a specific characteristic of this msd, you can add a label here to append to its filename when it gets saved.
        
    OUTPUT
    ------
       The individual msd (imsd) of each particle, as well as the ensemble msd (emsd) of all of them, with the associated standard error for each timestep dt (starting from the smallest dt at 1./fps, up to the largest dt allowed by the movie duration). Both the imsd and the emsd are saved as pandas DataFrames.
    '''
    
    emsd_error = {}
    
    if trajectory_type == 'filtered':
        label_trajectory = '_stub_length' + str(stub_length).zfill(3)
    if isinstance(selection_rule, list):
        trajectory_specifier = select_coordinates(None, selection_rule = selection_rule)[1]
        #print(trajectory_specifier)
        label_trajectory =  label_trajectory + '_' + trajectory_specifier
    #print(label_trajectory)
    elif selection_rule is None:
        label_trajectory = label_trajectory
    else:
        raise ValueError('I do not recognize this type of trajectory.')

    if isinstance(trajectories, str):
        file_location = trajectories
        trajectories = pandas.read_pickle(file_location + trajectory_type + '_trajectories' + label_trajectory + '.pkl')
        print('Loaded trajectories: ' + file_location + trajectory_type + '_trajectories' + label_trajectory + '.pkl')
    elif isinstance(trajectories, pandas.DataFrame):
        trajectories = trajectories
    else:
        raise ValueError('I do not understand what to use as trajectories, or where to find them.')

    if not isinstance(final_frame, int):
        final_frame = trajectories.frame.max()

    print('Final frame: ' + str(final_frame) + '.')

    #print(file_location + trajectory_type + '_trajectories' + label_trajectory)
    if isinstance(final_frame, int):
        trajectories = trajectories[trajectories.frame <= final_frame]
        print('I am including up to frame ' + str(final_frame).zfill(3) + '.')

    label_msd = label_trajectory + '_from_' +  trajectory_type
    if isinstance(starting_snr_range, list):
        label_msd = label_trajectory + '_starting_snr_within' + str(starting_snr_range[0]) + '_' + str(starting_snr_range[1])
    if isinstance(final_frame, int):
        label_msd = label_msd + '_until_lag_' + str(final_frame).zfill(3)
    if isinstance(other_label, str):
        label_msd = label_msd + '_' + other_label

    trajectories.index.names = ['framenumber']  # this is to bypass a current bug in trackpy. the filtered_trajectories have an index named 'frame' and an identical column named 'frame', which confuses trackpy. So here I rename the index.
    drift = trackpy.compute_drift(trajectories)
    corrected_trajectories = trackpy.subtract_drift(trajectories.copy(), drift)

    if isinstance(starting_snr_range, list):
        corrected_trajectories = corrected_trajectories[(corrected_trajectories.max_starting_snr > starting_snr_range[0]) & (corrected_trajectories.max_starting_snr < starting_snr_range[1])]

    if 'imsds' + label_msd in os.listdir(file_location):
        if recalculate_msds:
            answer = 'n'
        else:
            answer = input('Shall I load the existing imsds? Type y if yes, n if you want me to calculate them again.')
        if answer == 'y':
            imsds = pandas.read_pickle(file_location + 'imsds' + label_msd + '.pkl')    # if the imsds have already been calculated, load them
        elif answer == 'n':
            print('I am calculating the imsds again, and I will re-write the previous imsd file.')
            imsds = trackpy.imsd(corrected_trajectories, px_to_micron, fps, max_lagtime = final_frame)
            imsds.to_pickle(file_location + 'imsds' + label_msd + '.pkl')
            file = open(file_location + 'msd_info.txt', 'a')
            file.write('\n' + label_msd[1:] + ':')
            file.write('\n\nParameters used to calculate the individual mean-square displacement,\nusing trajectories with stub length ' + str(stub_length) + ':\n\npx_to_micron: ' + str(px_to_micron) + '\nframes per second: ' + str(fps) + '\n')
                    #the line below is for later, potentially
            file.close()
        else:
            raise ValueError('I do not understand your answer.')
    else:
        print('no imsds saved for these parameters')
        imsds = trackpy.imsd(corrected_trajectories, px_to_micron, fps, max_lagtime = final_frame)
        imsds.to_pickle(file_location + 'imsds' + label_msd + '.pkl')
        file = open(file_location + 'msd_info.txt', 'a')
        file.write('\n' + label_msd[1:] + ':')
        file.write('\n\nParameters used to calculate the individual mean-square displacement,\nusing trajectories with stub length ' + str(stub_length) + ':\n\npx_to_micron: ' + str(px_to_micron) + '\nframes per second: ' + str(fps) + '\n')
        #the line below is for later, potentially
        #if isinstance(selection_rule, list):
        #file.write('Trajectory subset: ' + quantity + ' ' + rule_in_words.replace('_', ' ') + ' ' + str(selection_value) + 'with at least ' + str(min_length) + ' points.\n')
        file.close()
    
    if 'emsd' + label_msd in os.listdir(file_location):
        if recalculate_msds:
            answer = 'n'
        else:
            answer = input('Shall I load the existing emsd? Type y if yes, n if you want me to calculate it again.')
        if answer == 'y':
            emsd = pandas.read_pickle(file_location + 'emsd' + label_msd + '.pkl')
        elif answer == 'n':
            print('I am calculating the emsd again, and I will re-write the previous emsd file.')
            emsd = trackpy.emsd(corrected_trajectories, px_to_micron, fps, detail = True, max_lagtime = final_frame)
            emsd_error_simple = []
            for i in imsds.index:
                value = numpy.std(imsds.loc[i])/numpy.sqrt(len(imsds.loc[i]))  # error due to standard deviation among all particles at that dt
                emsd_error_simple.append(value)
            emsd_error_simple = pandas.Series(emsd_error_simple, index = emsd.index) # from list to Series
            emsd = pandas.DataFrame(data = emsd, index = emsd.index)   # from Series to DataFrame
            emsd['error_simple'] = emsd_error_simple
            emsd['error_finiteness'] = emsd['msd'] / numpy.sqrt(emsd['N'])
            emsd = emsd.set_index('lagt')
            emsd.index.name = 'lagt'
    
            emsd.to_pickle(file_location + 'emsd' + label_msd + '.pkl')
            file = open(file_location + 'msd_info.txt', 'a')
            file.write('\n\nParameters used to calculate the ensemble mean-square displacement,\nusing trajectories with stub length ' + str(stub_length) + ':\n\npx_to_micron: ' + str(px_to_micron) + '\nframes per second: ' + str(fps) + '\n')
            file.close()
        else:
            raise ValueError('I do not understand your answer.')

    else:
        print('no emsd saved  for these parameters')
        emsd = trackpy.emsd(corrected_trajectories, px_to_micron, fps, detail = True, max_lagtime = final_frame)
        emsd_error_simple = []
        for i in imsds.index:
            value = numpy.std(imsds.loc[i])/numpy.sqrt(len(imsds.loc[i]))  # error due to standard deviation among all particles at that dt
            emsd_error_simple.append(value)
        emsd_error_simple = pandas.Series(emsd_error_simple, index = emsd.index) # from list to Series
        emsd = pandas.DataFrame(data = emsd, index = emsd.index)   # from Series to DataFrame
        emsd['error_simple'] = emsd_error_simple
        emsd['error_finiteness'] = emsd['msd'] / numpy.sqrt(emsd['N'])
        emsd = emsd.set_index('lagt')
        emsd.index.name = 'lagt'

        emsd.to_pickle(file_location + 'emsd' + label_msd + '.pkl')
        file = open(file_location + 'msd_info.txt', 'a')
        file.write('\n\nParameters used to calculate the ensemble mean-square displacement,\nusing trajectories with stub length ' + str(stub_length) + ':\n\npx_to_micron: ' + str(px_to_micron) + '\nframes per second: ' + str(fps) + '\n')
        #the line below is for later, potentially
        #if isinstance(selection_rule, list):
        #   file.write('Trajectory subset: ' + quantity + ' ' + rule_in_words.replace('_', ' ') + ' ' + str(selection_value) + 'with at least ' + str(min_length) + ' points.\n')
        file.close()

    return imsds, emsd, label_msd

def linear_fit_emsd(trajectories_location, stub_length, selection_rule = None, px_to_micron = 0.10748, fps = 0.1, max_timelag = None, color = '#00FF00'):
    '''
    Fit a line to the emsd curve, to extract the slope (aka diffusion coefficient) and exponent.
    
    INPUT
    -----
    trajectories_location : string
        The directory that contains the tracked particle trajectories.
    
    stub_length : int
        The stub length of the trajectories, when you filtered them. This need be specified in case you want to examine the effect of different trajectory lengths, for instance; all files for trajectories will be in the same folder.
        
    selection_rule : list of lists of the form ['quantity', 'relation', value]; defaults to None
        Here you specify the selection criteria. For instance, to only include spots with SNR > 10, you would write selection_rule = [['snr', 'greater', 10]]. To add a second rule on size, you would write selection_rule = [['snr', 'greater', 10], ['size','lesser', 100]]. Etc.
        Note the following rules: the first item of the list must be a column of the coordinates DataFrame. The second column must be one of ['greater', 'lesser', 'inbetween', 'equal']. If 'inbetween', then the value entry must be a list of two numbers.

    px_to_micron : float, defaults to 0.10748 μm/pixel
        The pixel-to-micron conversion.
    
    fps : float, defaults to 0.1
        The frames per second during acquisition.
        
    max_timelag : int or float, defaults to None
        The maximum timelag up to which to consider the emsd. Suliana suggests 1/2 of the total movie duration, because this is the maximum timelage for which you have at least two sets of independent measurements.
        
    color : string
        The color of the plot.
        
    '''
    file_location = trajectories_location
    emsd_filename = 'emsd_stub_length' + str(stub_length).zfill(3)
    if isinstance(selection_rule, list):
        emsd_specifier = select_coordinates(None, selection_rule = selection_rule)[1]
        emsd_filename = emsd_filename + '_' + emsd_specifier

    emsd_filename = emsd_filename + '_from_filtered.pkl'

    if any("emsd" in x for x in os.listdir(trajectories_location)):
        emsd = pandas.read_pickle(file_location + emsd_filename)
    else:
        print('no_emsd_saved_in ' + file_location)
    
    if isinstance(max_timelag, float):
        emsd = emsd[emsd.index < max_timelag]
    elif isinstance(max_timelag, int):
        emsd = emsd[emsd.index < max_timelag]

    fig_emsd_fit, ax_emsd_fit = pylab.subplots()
    if isinstance(emsd, pandas.DataFrame):  # If you used detail = True when calculating the emsd, then the emsd is not a Series, but a DataFrame. DataFrames don't have the attribute 'name', which gives a bug when plotting the fit result with trackpy.utils.fit_powerlaw. So, here, just make a Series out of the data in the DataFrame.
        emsd = pandas.Series(emsd['msd'].values, index=emsd.index)
        emsd.name = 'msd'

    linear_fit = trackpy.utils.fit_powerlaw(emsd, plot = True)

    ax_emsd_fit.set_title('linear fit to log-log of ensemble msd\n' + emsd_specifier)
    xlims = ax_emsd_fit.get_xlim()
    ylims = ax_emsd_fit.get_ylim()
    ax_emsd_fit.text(xlims[0]+xlims[0]/10., ylims[1]-ylims[1]/2.5, 'slope: ' + str(round(float(linear_fit.A),4)) + '\nexponent: ' + str(round(float(linear_fit.n),4)), fontsize = 13)
    ax_emsd_fit.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]', xlabel='lag time $t$')

    return linear_fit

def make_msd_plot(title, loglog = False):
    '''
    '''
    if 'i' in title:
        label = 'individual'
    elif 'e' in title:
        label = 'ensemble'

    if not pylab.fignum_exists(title):
        print('making new plot')
        fig, ax = pylab.subplots(num = title)
        ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]', xlabel='lag time $t$')
        ax.set_title(label + ' mean square displacements')
    else:
        fig = pylab.figure(title)
        ax = fig.gca()
        print('getting old plot')

    if loglog:
        ax.set_xscale('log')
        ax.set_yscale('log')

    #pylab.show()

    return fig, ax

def msd_general_diffusion(lagtime, Dcoeff, exponent, offset = 0):
    '''
    The general form of the msd (with a constant offset if necessary).
    
    INPUT
    -----
    lagtime : int or float
        The lagtime at which the mean-square-displacement is evaluated.
        
    Dcoeff : float
        The diffusion coefficient.
        
    exponent : float or int
        The power at which the lagtime is raised.
        
    offset : float or int, defaults to 0
        Option to include a constant offset. This is not physical, but useful when fitting to data since we often observe an offset (if not always).
        
        #dimensions : int, defaults to 2
        The number of dimensions in which the motion takes place.
        
    OUTPUT
    ------
    A float that corresponds to the mean-square-displacement at the given lagtime and diffusion parameters.
    '''
    value = offset + 2 * 2 * Dcoeff * lagtime**exponent
    
    return value

def msd_with_dynamic_error(lagtime, Dcoeff = 2e-3/4., exponent = 0.5, offset = 0, shutter_time = 0.03):
    '''
    ? Do S&D assume continuous exposure throughout the frames?
    
    The msd with dynamic error included (and with an arbitrary constant offset if necessary). This formula is taken from Savin & Doyle, Biophysical Journal 88, 623--638 (2005).
    
    INPUT
    -----
    lagtime : int or float
        The lagtime at which the mean-square-displacement is evaluated.
    
    Dcoeff : float
        The diffusion coefficient.
    
    exponent : float or int
        The power at which the lagtime is raised.
    
    shutter_time : float
        The exposure time of each frame.
        
    offset : float or int, defaults to 0
        Option to include a constant offset. This could physically correspond to a static error, although in our case we can not trivially extent this conclusion to these of the literature (Savin & Doyle, Backlund and Moerner 2015) because our static error varies in space and even more in time. That said, having this offset can be useful when fitting to data since we often observe one (if not always).
    
    OUTPUT
    ------
    A float that corresponds to the mean-square-displacement at the given lagtime and diffusion parameters, including the effect of dynamic error.
    '''
    tau = lagtime / shutter_time
    
    numerator = (tau + 1)**(exponent+2) + (tau - 1)**(exponent+2) - 2*tau**(exponent+2) - 2
    denominator = (1 + exponent) * (2 + exponent)
    
    value = numerator / denominator
    value = value * Dcoeff * 4 *  shutter_time**(exponent)
    
    return value


