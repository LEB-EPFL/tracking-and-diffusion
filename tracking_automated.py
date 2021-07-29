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
import plot_tracking_results
import os
import re
import json

# from IPython.display import Audio
# sound_file = './sound/beep.wav'

with open('general_info.json') as jf:
    data = json.load(jf)

px_to_micron = data['px_to_micron'] # μm per pixel
basic_directory = data['basic_directory']

def jupyter_track(movie, acquisition_framerate, dataset_location, diameter, minmass, search_range, memory, stub_length, selection_rule = None, p_to_m = px_to_micron, final_frame = None, use_if_existing_coordinates = True, recalculate_msds = True, max_lag_for_fit = None):
    '''
    Follow the tracking process as outlined in the jupyter notebook up to calculating and fitting msds (the fitting in this script is now obsolete; see process_msds.py for current fitting method). In short: find bright spots in a movie, link them into trajectories, filter those trajectories by their length, and calculate the individual msds and ensemble msd.
    
    INPUT
    -----
    movie : str
        The name of the movie (without file extension). In fact, it is the name of the synonyme folder that contains the movie.
        
    acquisition_framerate : float
        The framerate of acquisition in inverse seconds.
        
    dataset_location : str
        The folder that contains the movie you will track.

    diameter : int, odd
        The diameter of the spots, in pixels (see trackpy documentation for details).
        
    minmass : int
        The minmass, i.e. mininum integrated intensity of the spots you wish to find (see trackpy documentation for details).
        
    search_range : int
        The search range, i.e. largest allowed step of a single particle between frames within (see trackpy documentation for details).
        
    memory : int
        The memory, i.t. the number of frames for which a particle can disappear and resume with the same id when it reappears (see trackpy documentation for details).
        
    stub_length : int
        The stub length, i.e. the length of the shortest trajectory you will keep; very short trajectories are likely noise (see trackpy documentation for details).
    
    selection_rule : list of lists of the form ['quantity', 'relation', value], defaults to None
        Here you specify selection criteria. For instance, to only include spots with SNR > 10, you would write selection_rule = [['snr', 'greater', 10]]. To add a second rule on size, you would write selection_rule = [['snr', 'greater', 10], ['size','lesser', 100]]. Etc.
        Note the following rules: the first item of the list must be a column of the coordinates DataFrame. The second column must be one of ['greater', 'lesser', 'inbetween', 'equal']. If 'inbetween', then the value entry must be a list of two numbers.
        
    p_to_m : float, loaded from the general_info.json file where you have inserted your metadata (for us it is 0.10748 μm)
        The pixel-to-micron conversion in μm.
        
    final_frame : int or None (default)
        The maximum frame considered in the calculation of the msds. When None, it will be the final frame of the movie.
        
    use_if_existing_coordinates : boolean, defaults to True
        If you have already found particle coordinates with these specifications, with this variable you choose whether to load these or to find them anew.
        
    recalculate_msds = boolean, defaults to True
        If you have already calculated msds with these specifications, with this variable you choose whether to load these or to recalculate them.
        
    max_lag_for_fit : float or None (default)
        The max lag time to consider when fitting the diffusion equation to the ensemble msd. If none, this will be half of the movie duration. Note that we no longer use these fits.
        
    OUTPUT
    ------
    Saves coordinates, trajectories, filtered trajectories, individual msds, ensemble msd, and auxiliary figures and text files in a folder called 'analysis' and named according to the tracking parameters you choose.
    Returns the individual msds and ensemble msd DataFrames.
    '''
    
    now = datetime.datetime.now()
    string_now = '_' + now.strftime("%Y%m%d")

    dataset_location = dataset_location + movie + '/'

    directory_now = 'analysis/diameter' + str(d) + '_minmass' + str(mm) + '_percentile65/search_range' + str(sr)+ '_memory' + str(m).zfill(2) + '/'
    
    coordinates, trajectories, filtered_trajectories, filtered_trajectories_filename = tracking.track(dataset_location, diameter, minmass, search_range, memory, stub_length, superimpose_phase=True, selection_rule=selection_rule, use_if_existing_coordinates = use_if_existing_coordinates);
    filtered_trajectories_filename = filtered_trajectories_filename + '_all'

#    Audio(sound_file, autoplay=True)

    filtered_trajectories = tracking.measure_step_size(filtered_trajectories) # add the step sizes in the DataFrme of filtered trajectories

    filtered_trajectories['diagonal_size'] = 0 # we no longer use this

    for p in set(filtered_trajectories.particle):
        ds = trackpy.motion.diagonal_size(filtered_trajectories[filtered_trajectories.particle==p])
        #ds = ds * px_to_micron # stick to pixels so all lengths are in the same units in the dataframe
        filtered_trajectories.loc[filtered_trajectories.particle==p, 'diagonal_size'] = ds
    filtered_trajectories.to_pickle(filtered_trajectories_filename + '_with_dg.pkl')

    # We no longer use ep. ep is a measure of localization uncertainty calculated by trackpy based on the signal & noise, however it is often lower than the empirical uncertainty we measure with fixed cells, and we prefer the empirical one as it is more conservative.
    pylab.close('ep vs previous step size')
    pylab.figure('ep vs previous step size')
    pylab.title('uncertainty vs previous step size')

    pylab.plot(filtered_trajectories.ep * p_to_m, filtered_trajectories.previous_step_size * p_to_m, '.', alpha = 0.1)
    #pylab.plot(numpy.arange(0, filtered_trajectories.ep.max() * px_to_micron, 0.01), numpy.arange(0, filtered_trajectories.ep.max() * px_to_micron, 0.01), color = 'r', label = 'ep = step size')
    remove = filtered_trajectories[filtered_trajectories.ep > filtered_trajectories.previous_step_size]
    pylab.plot(remove.ep * px_to_micron, remove.previous_step_size * px_to_micron, '.', alpha = 0.1)
    #pylab.legend(loc = 1, framealpha = 0)

    problematic = 100 * len(remove) / len(filtered_trajectories)
    problematic = str(round(problematic,1))
    print(str(problematic) + '% of steps are too small compared to the localization uncertainty.')
    pylab.text(0.01, 0.1, str(problematic) + '% below the line', color = 'orange')
    
    f = open(dataset_location + directory_now + 'analysis_comments.txt', 'a')
    f.write(str(problematic) + '% of steps are too small compared to the localization uncertainty.')
    pylab.xlabel('localization uncertainty (μm)')
    pylab.ylabel('previous step size (μm)')
    pylab.savefig(dataset_location + directory_now + 'previous_step_size_vs_loc_uncertainty' + string_now + '.png', bbox_inches = 'tight')

    pylab.close('step sizes')
    pylab.figure('step sizes')

    pylab.hist(filtered_trajectories.previous_step_size * p_to_m, bins = 100, label = 'step size', alpha = 0.5)
    pylab.xlabel(r'step size $(\mu m)$')
    pylab.hist(filtered_trajectories.ep * p_to_m, bins = 100, label = 'ep', alpha = 0.5)

    pylab.axvline(filtered_trajectories.ep.median() * p_to_m, color = 'orange', label = 'median loc. uncertainty: ' + str(round(filtered_trajectories.ep.median() * p_to_m,2)) + ' μm')
    pylab.axvline(sr * p_to_m, color = 'k', label = 'search range')
    pylab.axvline(filtered_trajectories.previous_step_size.median() * p_to_m, color = '#0699F9', label = 'median step size: ' + str(round(filtered_trajectories.previous_step_size.median() * p_to_m, 2)) + ' μm')
    pylab.legend()
    #pylab.ylim(0, 45000)
    pylab.savefig(dataset_location + directory_now + 'step_size_distribution' + string_now + '.png', bbox_inches = 'tight')

    up_to_frame = final_frame
    imsds = {}
    emsd = {}

    color = {}
    c = []

    for i in numpy.linspace(255, 65280, 2):
        i = int(i)
        j = "#%06X" %(i, )
        c.append(j)

    pylab.close('imsds')
    pylab.close('emsd')

    pylab.figure('imsds')
    pylab.figure('emsd')

    imsds['all'], emsd['all'], msd_label =  tracking.calculate_msds(dataset_location + directory_now, stub_length, px_to_micron = p_to_m, fps = acquisition_framerate, trajectory_type = 'filtered', selection_rule=selection_rule, final_frame=up_to_frame, other_label = '_all')

    plot_tracking_results.plot_msds(imsds = imsds['all'], emsd = emsd['all'], loglog = True, color = '#0000FF', alpha_imsd = 0.02, emsd_label = None, interactive_plot = True); # interactive plot is under construction

    pylab.figure('imsds')
    pylab.savefig(dataset_location + directory_now + 'imsds' + string_now + '.png', bbox_inches = 'tight')
    pylab.figure('emsd')
    pylab.savefig(dataset_location + directory_now + 'emsd' + string_now + '.png', bbox_inches = 'tight')

    current_emsd_dataframe = emsd['all']

    # we no longer use these fits; see process_msds.py for more details

    if not isinstance(max_lag_for_fit, float):
        max_lag_for_fit = filtered_trajectories.frame.max() / 2.

    current_emsd_dataframe = current_emsd_dataframe.iloc[:max_lag_for_fit]  # fit up to half the max lag time, as a rule of thumb, to stay within the part of the curve with more datapoints and less noise

    try:
        linear_fit, linear_fit_covariance = scipy.optimize.curve_fit(tracking.msd_general_diffusion, current_emsd_dataframe.index, current_emsd_dataframe.msd)
        linear_fit_s, linear_fit_s_covariance = scipy.optimize.curve_fit(tracking.msd_general_diffusion, current_emsd_dataframe.index, current_emsd_dataframe.msd, sigma=current_emsd_dataframe.error_finiteness, absolute_sigma = True)
        print('Fit results:\nDcoeff = %f +/- %f um^2/s\nexponent = %f +/- %f\noffset = %f +/- %f um^2.' %(linear_fit[0], numpy.sqrt(numpy.diag(linear_fit_covariance))[0], linear_fit[1], numpy.sqrt(numpy.diag(linear_fit_covariance))[1], linear_fit[2], numpy.sqrt(numpy.diag(linear_fit_covariance))[2]))
        
        print('Fit results with uncertainties fitted:\nDcoeff = %f +/- %f um^2/s\nexponent = %f+/- %f\noffset = %f+/- %f.' %(linear_fit_s[0], numpy.sqrt(numpy.diag(linear_fit_s_covariance))[0], linear_fit_s[1], numpy.sqrt(numpy.diag(linear_fit_s_covariance))[1], linear_fit_s[2],   numpy.sqrt(numpy.diag(linear_fit_s_covariance))[2]))

        #pylab.close(i + 'emsd, calculated from ' + current_emsd_key + ', fit to subdiffusive msd with constant offset')

        pylab.figure('emsd, fit to subdiffusive msd with constant offset')

        pylab.title('emsd \nfit to subdiffusive msd, with constant offset')

        pylab.errorbar(current_emsd_dataframe.index, current_emsd_dataframe.msd, yerr = current_emsd_dataframe.error_finiteness, xerr = None, fmt = '.', alpha = 1, color = '#0000FF')

        pylab.plot(current_emsd_dataframe.index, tracking.msd_general_diffusion(current_emsd_dataframe.index, Dcoeff = linear_fit[0], exponent = linear_fit[1], offset=linear_fit[2]), '#0099FF', linewidth = 2, zorder = 10)

        pylab.plot(current_emsd_dataframe.index, tracking.msd_general_diffusion(current_emsd_dataframe.index, Dcoeff = linear_fit_s[0], exponent = linear_fit_s[1], offset=linear_fit_s[2]), 'r', linewidth = 2, zorder = 11)
        
        pylab.xlabel('time lag (s)')
        pylab.ylabel(r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]')
        pylab.ylim(0, None)

        pylab.text(2, 0.05, 'Dcoeff = %f +/- %f um^2/s \nexponent = %f +/- %f \noffset = %f +/- %f um^2' %(linear_fit[0], numpy.sqrt(numpy.diag(linear_fit_covariance))[0], numpy.round(linear_fit[1], 3), numpy.round(numpy.sqrt(numpy.diag(linear_fit_covariance))[1], 3), linear_fit[2],numpy.sqrt(numpy.diag(linear_fit_covariance))[2]), color = '#3EA7F4')

        f.write('\nfit parameters, simple fit: \nDcoeff = %f +/- %f um^2/s \nexponent = %f +/- %f \noffset = %f +/- %f um^2' %(linear_fit[0], numpy.sqrt(numpy.diag(linear_fit_covariance))[0], numpy.round(linear_fit[1], 3), numpy.round(numpy.sqrt(numpy.diag(linear_fit_covariance))[1], 3), linear_fit[2],numpy.sqrt(numpy.diag(linear_fit_covariance))[2]))

        pylab.text(0, 0.0123, 'Dcoeff = %f +/- %f um^2/s \nexponent = %f +/- %f \noffset = %f +/- %f um^2' %(linear_fit_s[0], numpy.sqrt(numpy.diag(linear_fit_s_covariance))[0], numpy.round(linear_fit_s[1], 3), numpy.round(numpy.sqrt(numpy.diag(linear_fit_s_covariance))[1], 3), linear_fit_s[2], numpy.sqrt(numpy.diag(linear_fit_s_covariance))[2]), color = '#FF0000')

        f.write('\nfit parameters, data uncertainty considered in fit: \nDcoeff = %f +/- %f um^2/s \nexponent = %f +/- %f \noffset = %f +/- %f um^2' %(linear_fit_s[0], numpy.sqrt(numpy.diag(linear_fit_s_covariance))[0], numpy.round(linear_fit_s[1], 3), numpy.round(numpy.sqrt(numpy.diag(linear_fit_s_covariance))[1], 3), linear_fit_s[2], numpy.sqrt(numpy.diag(linear_fit_s_covariance))[2]))

        #pylab.axhline((a.ep.median()*px_to_micron)**2, color = 'r', linestyle = '--')

        pylab.savefig(dataset_location + directory_now + 'emsd_with_fits' + string_now + '.png', bbox_inches = 'tight')

    except RuntimeError:
        f.write('\nunable to fit')

    pylab.close('all')
    f.close()

    return imsds['all'], emsd['all']

