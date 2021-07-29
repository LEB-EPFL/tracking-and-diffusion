# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
import pandas
import os
import re
import json

##### NOTE #####
# In this script we often obtain as output dictionaries of DataFrames.
# For information on python dictionaries, see here:
# https://docs.python.org/3/tutorial/datastructures.html
# For information on python DataFrames, see here:
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
# For present purposes, remember that a dictionary is a container of data that can have multiple entries, each entry having a name called a "key". Here, these keys will often be the name of a movie. Later on when we pool data together, these keys will be the names of the experimental conditions.
# You will naturally obtain data in this format using the loading function below, load_all_results(). For more info see the documentation below.

##### SETUP #####

with open('general_info.json') as jf:
    specs = json.load(jf)

basic_directory = specs['basic_directory']
all_starvation_times = specs['all_starvation_times']

##### FUNCTIONS #####

def collect_all_directories(spot_type):  
    '''
    Collect all directories that contain movies that are relevant for your analysis. You will invoke this function later on in the analysis.
    
    This function assumes the following file structure:
    basic_directory/YYMMDD/spot_type/starvation_time/movie_folders
    where:
    - basic_directory is the directory contained all data, specified at the beginning of this script
    - YYMMDD is the date of a dataset (e.x. 210323 for March 23rd, 2021)
    - spot_type is origins, muNS, fixed_origins, or fixed_muNS
    - starvation_time is either 0 or 6 hours for us typically, here formatted as '0h' or '6h'
    - movie_folders are the folders for each movie. Remember that from the tracking step already you should have organized your data such that each movie has a synonymous, dedicated folder.
    
    Here is an example of a full path to a movie for our case:
    basic_directory/201004/muNS/0h/201004_0h_bLR31_210msDelay_004/
    
    INPUT
    -----
    spot_type : str
    A string that describes the type of spot you are interested in. It can be 'origins', 'muNS', 'fixed_origins', or 'fixed_muNS'.

    OUTPUT
    ------
    A dictionary of lists, where each entry of the dictionary corresponds to a day when data was taken, and each list within an entry contains all folders (i.e. all movie names) for that day.
    '''
    
    directories = {}
    
    all_days = specs[spot_type]['all_days']

    for i in all_days:
        directories[i] = []
        for j in all_starvation_times:
            d = basic_directory + i + '/' + spot_type + '/'
            if j in os.listdir(d):
                a = [x + '/' for x in os.listdir(d + j + '/')]
                if '.DS_Store' in a:
                    a.remove('.DS_Store')  # this can happen to mac users: the first listed subdirectory is a created by the operating system and called '.DS_Store'. We don't care about it here.
                a = [x for x in a if 'pre' not in x and i in x and '.nd2' not in x]
                directories[i].append(a)
        directories[i] = sum(directories[i], [])

    return directories

### organizing functions ###

def define_analysis_directories(spot_type):
    '''
    Collect all the directories that contain results of the tracking analysis. These are defined by the tracking parameters you used, and which you should have entered in select_specifications() above. Remember, it assumes that you have used the same parameters for all movies of the same type.
    
    INPUT
    -----
    spot_type : str
    A string that describes the type of spot you are interested in. It can be 'origins', 'muNS', 'fixed_origins', or 'fixed_muNS'.
    
    OUTPUT
    ------
    A dictionary that contains strings describing the location of analysis directories for each movie. This dictionary has two levels of organization. At the first, highest level, each entry of the dictionary corresponds to a day when data was taken. At the second, intermediate level, each day-entry contains an entry for each movie from that day. Finally, for each movie-entry there is a string that describes the path to the analysis folder for that movie.
    '''

    all_days = specs[spot_type]['all_days']
    diameter, minmass, percentile, search_range, memory = specs[spot_type]['diameter'], specs[spot_type]['minmass'], specs[spot_type]['percentile'], specs[spot_type]['search_range'], specs[spot_type]['memory']
    directories = collect_all_directories(spot_type)
    
    analysis_directories = {}

    for d in all_days:
        analysis_directories[d] = {}
        for i in directories[d]:
            analysis_directories[d][i] = 'analysis/diameter' + str(diameter) + '_minmass' + str(minmass) + '_percentile' + str(percentile) + '/search_range' + str(search_range) + '_memory' + str(memory) + '/'

    return analysis_directories

def read(quantity, string, spot_type):
    '''
    Read off information about a movie, from its filename. This function is very useful; we use it a lot in order to group data and to color-code data in figures.
    Note that this function is VERY SPECIFIC to the file conventions we have been using in our own analysis (and has some funny-looking rules to account for the early times when there was no established convention).
    
    INPUT
    -----
    quantity : str
        The quantity you want to read from the string. It can be
        - 'starvation_time' : the time into starvation
        - 'time_between_frames' : the time between two consecutive frames
        - 'strain' : the strain imaged
        - 'condition' : for us, lowN or lowC or nothing if at 0h
        - 'movie' : the movie number
        - 'day' : the day when the movie was taken
        - 'search_range' : the search range used (I don't think I have ever used this).
    
    string : str
        The string from which you wish to read. Typically, this will be the name of a movie.
    
    spot_type : str
    A string that describes the type of spot you are interested in. It can be 'origins', 'muNS', 'fixed_origins', or 'fixed_muNS'.

    OUTPUT
    ------
    A string with the required information.
    '''
    if quantity == 'starvation_time':
        answer = string.split('h')[0].split('_')[::-1][0]
    elif quantity == 'time_between_frames':
        if '1s' in string:
            answer = str(1000)
        elif 'ms' in string:
            answer = string.split('ms')[0]
            answer = answer.split('_')[::-1][0]
        elif spot_type == 'origins':
            if '210' in string:
                answer = '5000'
                print('Origins at 1 frame per 5 seconds, true?')
            else:
                answer = '10000'
                print('Origins at 1 frame per 10 seconds, true?')
        elif spot_type == 'fixed_origins':
            answer = '5000'
        if answer == '0':  # because initially we denoted 30ms as 0ms, as this was the shortest time lag possible with the camera
            answer = '30';
        elif answer == '20':  # because in some movies there was a typo in the filename, we never used 20ms time lags
            answer = '30';
    elif quantity == 'strain':
        answer = 'bL'+ string.split('bL')[1].split('_')[0]
    elif quantity == 'condition':
        if 'low' in string:
            answer = 'low' + string.split('low')[1][0]
        elif '6h' in string:
            answer = 'lowN'
        elif '0h' in string:
            answer = ''
    elif quantity == 'movie':
        answer = string.split('_')[::-1][0]
        answer = answer.strip('/')
    elif quantity == 'search_range':
        answer = string.split('search_range')[1][0]
    elif quantity == 'day':  # NEEDS FIXING BECAUSE IT ASSUMES THE DAY COMES FIRST IN THE FILENAME so I can't use it as is to read the day off of dictionary keys
        answer = string.split('_')[0]

    return answer

def load_all_results(spot_type, file_specifier, days = 'all_days', starvation_times = 'all', avoid = ['1s'], most_recent = True):
    '''
    Load all results of the file type specified.
    
    INPUT
    -----
    spot_type : str
        A string that describes the type of spot you are interested in. It can be 'origins', 'muNS', 'fixed_origins', or 'fixed_muNS'.

    file_specifier : str or list of strings
        If a string, it is a string that describes the name of the file you want to load, as it has been recorded in the analysis folder for each movie. For example, it could be 'filtered_trajectories_all', if the phrase 'filtered_trajectories_all' is only present in one file per analysis folder.
        If a list of strings, this list should contain a group of strings that are all present in the name of the file you want to load, and which together describe this filename. For example, it could be ['filtered', 'trajectories', 'renamed']; then the function will look, in each movie's analysis folder, for all filenames that contain all three strings.
        
    most_recent : boolean
        If there are more than one files with the file specifiers you enter, the function will first try to sort them by date, assuming that the date is appended at the end of the filename - for example 'filtered_trajectories_all_renamed_210712.pkl'. This dating convention is followed by functions in post-processing, so it works with those.
        If this fails, you will get an error indicating that you have not specified the filename accurately enough.
        Remember that, as you move on with the analysis, the script will save results with a standard name that is the same for each movie but saved separately in the analysis folder for each movie. We take advantage of this here to 'fish' these files from all relevant movies, so that we can work with them further.
    
    days : 'all_days' (default), or list of str
        The days of data that you want to consider. If 'all_days', it loads all data from all days.
        
    starvation_times : 'all' (default), or list of str
        The starvation times you want to consider. If 'all', those will be '0h' and '6h'.
        
    avoid : list of str, defaults to ['1s']
        A string that is found in all movies that you do not want to consider here. If you want to consider all movies, enter an empty list.
        
    OUTPUT
    ------
    A dictionary with the loaded results. The keys of this dictionary are the names of the movies where the results are coming from. The results are typically pandas DataFrames.
    '''
    
    loaded_results = {}
    
    directories = collect_all_directories(spot_type)
    analysis_directories = define_analysis_directories(spot_type)
    stub_length = str(specs[spot_type]['stub_length']).zfill(3)
    avoid_typically = specs[spot_type]['typically_avoid']
    diameter = specs[spot_type]['diameter']
    minmass = specs[spot_type]['minmass']
    percentile = specs[spot_type]['percentile']
    search_range = specs[spot_type]['search_range']
    memory = specs[spot_type]['memory']

    if isinstance(avoid, list):
        for i in avoid_typically:
            avoid.append(i)

    if days == 'all_days':
        days = specs[spot_type]['all_days']

    if starvation_times == 'all':
        starvation_times = all_starvation_times

    for d in days:
        for s in starvation_times:
            print(basic_directory + d + '/' + spot_type + '/')
            if s in os.listdir(basic_directory + d + '/' + spot_type + '/'):
                directories_now = [x for x in directories[d] if s in x]
                directories_now = [x for x in directories_now if not any([y in x for y in avoid])]
                for j,i in enumerate(directories_now):
                    print('\n movie: ' + i)
                    movie_location = basic_directory + d + '/' + spot_type + '/' + s + '/' +  i
                    location = movie_location + analysis_directories[d][i]
                    if os.path.isdir(location):
                        analysis_done = True
                    else:
                        analysis_done = False
                        print('You have not analysed this movie with the parameters I found in the general_info.json file.')
                    
                    if analysis_done:
                        if isinstance(file_specifier, str):
                            files_all = [x for x in os.listdir(location) if re.search(file_specifier, x)]
                        elif isinstance(file_specifier, list):
                            files_all = [x for x in os.listdir(location) if all([re.search(y, x) for y in file_specifier])]

                        if ((len(files_all) > 1) and most_recent): # sort particles by date when they are dated and pick the most recent
                            files_all = sorted(files_all, key = lambda x: int(x.split('_')[-1].split('.')[0]))
                            files_all = files_all[::-1]
                            file = files_all[0]
                            flag = True
                        elif ((len(files_all) > 1) and not most_recent):
                            print('I have found these files:')
                            for f in files_all:
                                print(f)
                            raise ValueError('There are more than one files and I do not know how to choose.')
                        elif len(files_all) == 1:
                            file = files_all[0]
                            flag = True
                        else:
                            print('There are no files for ' + i + '.')
                            flag = False

                        if flag:
                            loaded_results[i] = pandas.read_pickle(location + file)
                            print('loaded: ' + file + '\n')
                            
    return loaded_results


