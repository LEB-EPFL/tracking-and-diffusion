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
import os
import re
import time
import load_all_results
import random
import datetime
import json

##### SETUP #####
with open('general_info.json') as jf:
    data = json.load(jf)

px_to_micron = data['px_to_micron']# μm per pixel
basic_directory = data['basic_directory']
all_starvation_times = data['all_starvation_times']

##### FUNCTIONS #####
### Gaussian functions ###

# General 2d Gaussian that can be asymmetric and rotated. In the end I did not use this to fit the intensity profiles of the spots, for simplicity and following Parry et al Cell Press 2014.

def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    xo = float(xo)
    yo = float(yo)
    a = (numpy.cos(theta)**2)/(2*sigma_x**2) + (numpy.sin(theta)**2)/(2*sigma_y**2)
    b = -(numpy.sin(2*theta))/(4*sigma_x**2) + (numpy.sin(2*theta))/(4*sigma_y**2)
    c = (numpy.sin(theta)**2)/(2*sigma_x**2) + (numpy.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*numpy.exp(- (a*((xy[0]-xo)**2) + 2*b*(xy[0]-xo)*(xy[1]-yo) + c*((xy[1]-yo)**2)))
    return g.ravel()

# Simple 2d Gaussian that is symmetric. I used this function to fit the intensity profiles of the found spots, following Parry et al Cell Press 2014. Note, however, one difference in the definition of widths: sigma_parry = numpy.sqrt(2) * sigma_sofia. I account for this throughout the analysis.

def twoD_Gaussian_simple(xy, amplitude, xo, yo, sigma, offset):
    xo = float(xo)
    yo = float(yo)
    a = 1/(2*sigma**2)
    c = 1/(2*sigma**2)
    g = offset + amplitude*numpy.exp(- (a*((xy[0]-xo)**2) + c*((xy[1]-yo)**2)))
    return g.ravel()

### fitting functions ###

def fit_spots(trajectories, moviename, spot_type, particle_area_size, initial_guess, plot = False, simple_Gaussian = True):
    '''
    Fit a 2D Gaussian to spots from a sequence of images for which you have the trajectories (e.x. from tracking).
    Final goal: get a DataFrame of trajectories in. Output a DataFrame of the fit results with particle id, x, y, and fit results.
    Then you can append with other fit results and look for correlations with α, D_app, msd(t=2s), SNR.
    
    INPUT
    -----
    trajectories : pandas DataFrame
        A pandas DataFrame of the trajectories that you want to perform the fits on. Note that it is VERY helpful if these trajectories have already been renamed using rename_particles_in_traj() from process_trajectories.py. Then you can easily merge these fit results with the fit results from the msds.
    
    moviename : string
        The name of the movie that the trajectories correspond to. This is useful for determining the location of the images where you will perform the fit, as well as the location where results will be saved.
    
    spot_type : str
        This can be 'origins', 'muNS', 'fixed_origins', 'fixed_muNS', depending on the object you have tracked. We have only been interested in applying this to muNS.
    
    particle_area_size : int, I used 50
        The size of the patch around the particle on which you will perform the fit.
    
    initial_guess : dictionary of integers, I used {'amplitude' : 100, 'sigma_x' : 3, 'sigma_y' : 3, 'offset' : 100, 'theta' : 0}
        A dictionary with values for the initial guess to use on the fits for all the spots. Each entry in the dictionary corresponds to an input variable of the fit function, in our case Gaussian. To account for the possibility to use an asymmetric Gaussian later in the future, there are enough variables for that function. If you use a simple, i.e. symmetric Gaussian, many of these variables are set to be equal to each other.
        Note that if the amplitude you start with in your initial guess is too small, you risk getting wrong fits at random areas on the image with small intensity fluctuations.

    plot : boolean, defaults to False
        When True, the function will show you contours of the fits over all particles. This can quickly cause the computer to be overwhelmed if you are fitting a whole movie... it is a good idea to do only on a couple of particles at a time.
    
    simple_Gaussian : boolean, defaults to True
        When True, the function you will use to fit will be a simple Gaussian, i.e. a symmetric Gaussian. You can also use the asymmetric one if you like, as it is included in this script.
    
    OUTPUT
    ------
    The output is three items.
    1. A "results" dictionary of pandas DataFrames where each entry corresponds to an entry of the trajectories you use in the input, i.e. most likely a movie. This new dictionary is a copy of the trajectories dictionary, where each DataFrame has a new column for each parameter of the Gaussian function it was fit to.
    2. A "fit_uncertainties" dictionary that contains an entry per entry in the results, i.e. most likely a movie, and in each entry a column per variable of the Gaussian function where are registered the uncertainties in the value of the fit parameters. I decided to save this as a separate dictionary, instead of merging it with the results above, because the results ended up having many columns. In the fit_uncertainties dictionary there is a column for the particle id, so you can keep track of the information across the different DataFrames. This is another reason why it is IMPORTANT to use, in the input, trajectories DataFrames where the particles have already been renamed, as also mentioned in the INPUT above.
    3. The "initial_guess" dictionary - a copy of the initial guess you provided in the input. This is not essential but could be useful.
    The scripts also saves the results and fit_uncertainties in each movie's analysis folder.
    '''
    now = datetime.datetime.now()
    string_now = now.strftime("%Y%m%d")
    if simple_Gaussian:
        string_now = 'simple_Gauss_' + string_now

    notes = open('notes_from_Gaussian_fits_' + string_now + '.txt', 'w+') # Here you will keep track of particles that were not fitted. Note that some particles may be given fit parameters but with an undefined uncertainty (covariance). We will filter those out later, in collect_results.py, using filter_results() (TO BE COMPLETED)
    
    results = trajectories.copy(deep = True) # with deep = True you can edit the new copy without changing the old one, if the item being copied is a pandas DataFrame
    if simple_Gaussian:
        new_columns = ['amplitude', 'xo', 'yo', 'sigma', 'offset']
    else:
        new_columns = ['amplitude', 'xo', 'yo', 'sigma_x', 'sigma_y', 'theta', 'offset']  # these new columns will contain the results from the 2D Gaussian fit to the intensity profile
    new_columns_func = new_columns.copy()
    new_columns_func.append('particle')
    fit_uncertainties = pandas.DataFrame(columns = new_columns_func)
    # this pandas DataFrame will include the values from the goodness of the fits
    
    for i in new_columns:
        results[i] = 0 # initialize the new columns
    for i in new_columns_func:
        fit_uncertainties[i] = 0
    
    day = load_all_results.read('day', moviename, spot_type)
    starvation_time = load_all_results.read('starvation_time', moviename, spot_type) + 'h'
    ana_dirs = load_all_results.define_analysis_directories(spot_type) # all analysis directories
    imagepath = basic_directory + '/' + day + '/' + spot_type + '/' + starvation_time + '/' + moviename + '/' + 'images/'
    
    images = tracking.load_images(imagepath)
    particles = list(set(trajectories.particle))

    n = 0
    for i in results.index:
        #        print('index: ' + str(i) + ' out of ' + str(len(results)))
        n = n + 1
        percent_done = 100 * numpy.float(n) / len(results)
        percent_done = int(numpy.round(percent_done, 0))
        print(str(percent_done) + '% done.')
        
        particle = numpy.int(results.loc[i, 'particle'])
        fit_uncertainties.loc[i, 'particle'] = particle
        f = results.loc[i, 'frame']
        t2 = trajectories[trajectories.particle == particle]
        t3 = t2[t2.frame == f]
        xo = numpy.float(t3.x)
        yo = numpy.float(t3.y)
        
        # Create x and y indices
        x = numpy.linspace(xo - particle_area_size/2., xo + particle_area_size/2.)
        y = numpy.linspace(yo - particle_area_size/2., yo + particle_area_size/2.)
        
        if any([x.min() < 0, y.min()< 0, x.max() > images[0].shape[0], y.max() > images[0].shape[1]]):
            for k in new_columns:
                results.loc[i, k] = numpy.nan  # if a particle is too close to the edge of the image it will not be fit as you cannot capture its full profile well enough
#            print('Particle ' + str(particle) + ' is too close to the edges.')
        else:
            xy = numpy.meshgrid(x, y)
            
        #create data
            xmin_int = int(numpy.round(x.min(), 0))
            xmax_int = int(numpy.round(x.max(), 0))
            ymin_int = int(numpy.round(y.min(), 0))
            ymax_int = int(numpy.round(y.max(), 0))
            
#            print('particle ' + str(particle) + ' in frame ' + str(f))

            data = images[f][ymin_int:ymax_int, xmin_int:xmax_int]  # x, y, are flipped. I am not sure why but it works so move on for now ! (not to self - Ah this is why I can't do it with pooled trajectories as input. Just as well, then I can pool the results like all other quantities with functions I already have.)
            
            initial_guess['xo'] = xo
            initial_guess['yo'] = yo
            if simple_Gaussian:
                initial_guess['sigma'] = initial_guess['sigma_x']
            
            initial_guess_list = []  # prepare the list of values for the initial guess, in the right order
            for m in new_columns:
                if m in initial_guess.keys():
                    initial_guess_list.append(initial_guess[m])
                else:
                    print('I do not have an initial guess for ' + m + '.')
            data = data.ravel()
            
            try:
                if simple_Gaussian:
                    popt, pcov = scipy.optimize.curve_fit(twoD_Gaussian_simple, xy, data, p0=initial_guess_list)
                else:
                    popt, pcov = scipy.optimize.curve_fit(twoD_Gaussian, xy, data, p0=initial_guess_list)
                for j, k in enumerate(popt):
                    c = new_columns[j]
                    results.loc[i, c] = k
                    fit_uncertainties.loc[i, c] = numpy.sqrt(numpy.diag(pcov))[j]
                
                if plot:
                    if simple_Gaussian:
                        data_fitted = twoD_Gaussian_simple(xy, *popt)
                    else:
                        data_fitted = twoD_Gaussian(xy, *popt) # plot twoD_Gaussian data generated above
                    pylab.figure('index_' + str(i) + '_original')
                    pylab.title('index_' + str(i) + '_original')
                    pylab.imshow(data.reshape(particle_area_size, particle_area_size), origin = 'lower')
                    pylab.colorbar()
                    
                    fig, ax = pylab.subplots(1, 1)
                    ax.set_title('index_' + str(i) + '_fitted')
                    ax.imshow(data.reshape(particle_area_size, particle_area_size), extent=(x.min(), x.max(), y.min(), y.max()), origin = 'lower')
                    ax.contour(x, y, data_fitted.reshape(particle_area_size, particle_area_size), 4, colors='w')

                    pylab.show()

            except:
                message = 'particle ' + str(particle) + ' in frame ' + str(f) + ' cannot be fitted.'
                print(message)
                notes.write(message + '\n')
                for k in new_columns:
                    results.loc[i, k] = numpy.nan
                    fit_uncertainties.loc[i, k] = numpy.nan

    target_location = basic_directory + '/' + day + '/' + spot_type + '/' + starvation_time + '/' + moviename + '/' + ana_dirs[day][moviename + '/']
    
    results.to_pickle(target_location + '/traj_with_intensity_Gaussian_fit_results_' + string_now + '.pkl')
    fit_uncertainties.to_pickle(target_location + '/intensity_Gaussian_fit_results_uncertainties_' + string_now + '.pkl')
    notes.close()

    return results, fit_uncertainties, initial_guess

### fit many movies en masse ###

### Note that this can take a couple of hours per movie. Best to set it up overnight, so as to let the computer dedicate its CPU to the tasks.
### Also note that, depending how many cores you have on your computer, you may be able to launch many threads of this in parallel. Each thread should automatically occupy a different core of your CPU without causing processing delays, as it will be processed independently of all other threads.
### To launch multiple threads (e.x. one for all 0h movies and one for all 6h movies of a given day, you can either:
### 1. open many different python terminal windows and run the function multiple times, with a different input that describes a subset of data each time
### 2. Willi had a way with python module... REMINDER TO FILL THIS IN

def fit_spots_all_trajectories(days, spot_type = 'muNS', traj_filename = 'filtered_trajectories_all_renamed', starvation_times = all_starvation_times, avoid = None, simple_Gaussian = True):

    '''
    Fit a 2D Gaussian to spots from images for which you have the coordinates (e.x. from tracking), over many movies. This is essentially a for-loop implementation of the function fit_spots() above.
    
    INPUT
    -----
    days : list of str
        The days that you want to consider in this fit. For example, you may have already fitted the data from two days and just    acquired a new day's worth of data; here you can include that day only.

    spot_type : str, defaults to 'muNS'
        This can be 'origins', 'muNS', 'fixed_origins', 'fixed_muNS', depending on the object you have tracked. We have only been interested in applying these fits to muNS particles.
    
    traj_filename : str
        The filename of the trajectories of the movies on which you want to perform the Gaussian fits (see also function load_all_results() in load_all_results.py).
    
    starvation_times : list of str, defaults to all_starvation_times defined above
        The starvation times you want to consider in this fit. For instance, you might only want to focus on 0h data or on 6h data.
        
    avoid : list of str or None (default)
        A list containing strings that characterize movies that you want to avoid. For example, the dictionary of imsds might contain entries corresponding to movies whose name includes '1s', if, for instance, they were taken at 1 frame per second. You might not be interested in these movies for present purposes, and here you can exclude them from this part of the analysis (and save some time).

    simple_Gaussian : boolean, defaults to True
        With this you choose whether to use the simple Gaussian function above for the fits. If False, you will use the potentially asymmetric Gaussian function from above, that has more variables to describe asymmetry. I have not used this extensively in our analysis.
    
    OUTPUT
    ------
    A dictionary of the fit results and their uncertainties, where each entry corresponds to an entry of the trajectories dictionary, i.e. to a movie.
    The function also saves the results for the values of the fit parameters and their uncertainties in the analysis folder of each movie.
    '''
    t = load_all_results.load_all_results(spot_type, traj_filename, days = days, starvation_times = starvation_times)
    
    f = {} # dictionary of trajectories with fit results
    func = {} # dictionary of uncertainties of fit results
    
    if isinstance(avoid, list):
        keys_of_interest = [x for x in t.keys() if not any([y in x for y in avoid])]
    else:
        keys_of_interest = [x for x in t.keys()]

    print('I will fit the spots in these movies:')
    for i in keys_of_interest:
        print(i)

    for i in keys_of_interest:
        t0 = time.time()
        mn = i.strip('/')
        print('Working on movie ' + str(mn) + '.')
        fs = fit_spots(t[i], mn, spot_type, particle_area_size = 50, initial_guess = {'amplitude' : 100, 'sigma_x' : 3, 'sigma_y' : 3, 'offset' : 100, 'theta' : 0}, plot = False, simple_Gaussian = simple_Gaussian)
        f[i] = fs[0]
        func[i] = fs[1]
        t1 = time.time()
        dt = t1-t0
        dt = numpy.round(dt,0)
        print('This movie took ' + str(dt) + ' sec.')

    return f, func

### TO BE SANITY CHECKED AND COMMENTED IF OF BROADER INTEREST ###

def fit_spots_test(trajectories, imagepath, particle_area_size = 50, initial_guess = {'amplitude' : 100, 'sigma_x' : 3, 'sigma_y' : 3, 'offset' : 100, 'theta' : 0}, plot = True, simple_Gaussian = True):
    '''
    Fit a 2D Gaussian to a spot from an image for which you have the coordinates (e.x. from tracking).
    Final goal: input a DataFrame of trajectories, output a DataFrame of the fit results with particle id, x, y, and fit results.
    Later on, you will combine these with other fit results per-particle, such as the results from fits to the msds.
    
    INPUT
    -----
    
    particle_area_size : int, defaults to 50
    The size of the patch around the particle on which you will perform the fit.
    
    OUTPUT
    ------
    '''
    if simple_Gaussian:
        columns = ['particle', 'amplitude', 'xo', 'yo', 'sigma', 'offset']  # these new columns will contain the results from the 2D Gaussian fit to the intensity profile
    else:
        columns = ['particle', 'amplitude', 'xo', 'yo', 'sigma_x', 'sigma_y', 'theta', 'offset']  # these new columns will contain the results from the 2D Gaussian fit to the intensity profile
    results = pandas.DataFrame(columns = columns)

    for i in columns:
        results[i] = 0 # initialize the new columns
    
    images = tracking.load_images(imagepath)
    # for testing, start with one particle now
    particles = list(set(trajectories.particle))
    particle = particles[220] ## choosing at random
    f = 0  ## choosing frame 0 to start with
    trajectories = trajectories[trajectories.particle == particle]
    trajectories = trajectories[trajectories.frame == f]
    xo = numpy.float(trajectories.x)
    yo = numpy.float(trajectories.y)
    
    # Create x and y indices
    x = numpy.linspace(xo - particle_area_size/2., xo + particle_area_size/2.)
    y = numpy.linspace(yo - particle_area_size/2., yo + particle_area_size/2.)
    xy = numpy.meshgrid(x, y)
    
    #create data - start with frame 0 for now testing
    xmin_int = int(numpy.round(x.min(), 0))
    xmax_int = int(numpy.round(x.max(), 0))
    ymin_int = int(numpy.round(y.min(), 0))
    ymax_int = int(numpy.round(y.max(), 0))
    
    print('testing with particle ' + str(particle) + ' in frame 0')
    print(xmin_int)
    print(xmax_int)
    print(ymin_int)
    print(ymax_int)
    
    data = images[f][ymin_int:ymax_int, xmin_int:xmax_int]  # x, y, are flipped. works so move on for now
    
    initial_guess['xo'] = xo
    initial_guess['yo'] = yo
    
    initial_guess_list = []  # prepare the list of values for the initial guess, in the right order
    for i in columns[1:]:
        if simple_Gaussian:
            initial_guess['sigma'] = initial_guess['sigma_x']
        initial_guess_list.append(initial_guess[i])
    
    data = data.ravel()
    if simple_Gaussian:
        popt, pcov = scipy.optimize.curve_fit(twoD_Gaussian_simple, xy, data, p0=initial_guess_list)
    else:
        popt, pcov = scipy.optimize.curve_fit(twoD_Gaussian, xy, data, p0=initial_guess_list)

    if simple_Gaussian:
        data_fitted = twoD_Gaussian_simple(xy, *popt)
    else:
        data_fitted = twoD_Gaussian(xy, *popt)

    if plot:
        pylab.figure()
        pylab.imshow(data.reshape(particle_area_size, particle_area_size))
        #        pylab.colorbar()

    fig, ax = pylab.subplots(1, 1)
    ax.imshow(data.reshape(particle_area_size, particle_area_size), extent=(x.min(), x.max(), y.min(), y.max()), origin = 'lower')
    ax.contour(x, y, data_fitted.reshape(particle_area_size, particle_area_size), 4, colors='w')
    
    pylab.show()
    
    return popt, pcov, data_fitted



