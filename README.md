# tracking-and-diffusion
Tracking and post-processing of trajectories to quantify the diffusive motion of fluorescent spots. 
Step-by-step guide for processing the trajectories you have acquired with the tracking pipeline. 

This guide includes details relevant for the present work, which includes on two types of fluorescent spots: chromosome origins and muNS. These two types of spots are processed in slightly different ways.

## general structure

The main steps are:
* post-processing of trajectories
* post-processing of msds
* post-processing of identified bright spots (for muNS only)
* pooling of data from different movies (within a day and across different days)
* binning according to size (for muNS only)

Finally, another chapter of analysis includes
* quantifying cell motion from segmented frames.

The scripts you will need are:
* load_all_results.py: contains basic functions for organizing the data
* process_trajectories.py: contains functions for processing the trajectories
* process_msds.py: contains functions for processing the msds
* process_spots.py: only for muNS, contains functions for fitting Gaussian profiles over the bright spots (used for estimating particle size)
* collect_results.py: contains functions for pooling the data together
* cell_motion.py: contains functions for quantifying cell motion during the course of a segmented movie (used for some movies of chromosome origins).
* tracking.py: this is the main tracking script, you will invoke some of its functions during this process

You can find detailed information about how to apply each function within these scripts. The following instructions walk you through the basic steps, but make sure to refer to the commentary within the scripts to correctly choose parameters for your situation. These are often optional parameters and referred to here as “*params”. 

**naming conventions for the movies**

The scripts read information about the movie from the filenames. The reading is done by a function called read() in load_all_results.py. It has been written for the set of filenames we have been handling, and assumes some conventions. 
In our data we have converged on the conventions followed by these examples:

210323_bLR31_0h_210ms_tc_002 \
210323_bLR31_6hlowN_210ms_tc_6001 \
210323_bLR32_6hlowC_210ms_tc_6029 \

Each bit of information is separated from the others by an underscore ‘_’. 

* The date should be written in the first part of the filename. In the current version, it is important that this be the first part.
* The strain should always begin with ‘bL’.
* The starvation time should always be followed by ‘h’ and written in hours. 
* The possible starvation conditions are ‘lowX’, where for us so far X has been N (nitrogen) or C (carbon). If the starvation time is ‘6h’, the read() function will look for the phrase ‘lowX’ in the filename and identify the starvation condition as lowC or lowN. If the starvation time is ‘0h’, the condition will be an empty string ‘’. 
* The time between frames should be written in milliseconds followed my ‘ms’. That said, in our data this has not always been the case and the read() function knows to recognize these exceptions for our cases. 
* The movie number should always come at the end of the filename.

These conventions arose naturally out of the filenames we have. You can extend or generalize them by editing the function read() in load_all_results.py. 

**starting point: after tracking and before post-processing**

During tracking, the code has generated an ‘analysis’ folder. This folder will likely look like this: #insert image

The tracking script will have made different subfolders for each set of tracking parameters. So if you have tried out a couple of sets of parameters you will have multiple subfolders in analysis/. In the end you should have settled on what you think are optimal parameters, and you will now work with the tracking results from those.

In the directory that corresponds to your final choice of tracking parameters you should have contents like the following (in the example above, these are the contents in ‘search_range4_memory20/‘).

In the post-processing we will use two types of files from tracking: the trajectories and the msds. If you have generated different types of trajectories and msds as you were tuning tracking parameters, you will now need to choose the files that you will process further. Note that you should always use trajectories with ‘filtered’ as a prefix, as those are trajectories where short studs have been removed (see tracking notebook and trackpy tutorial for more details).

**0. preparation**

1. You need to have a dedicated data folder that will be your basic directory. Inside this folder you need to be structured as shown in the following image. # add image

Here each type of data has its own folder where we will put corresponding results. These are the first folders whose names begin with underscore '_'.

Then, the raw data as well as results from analysis on a per-movie basis is contained in folders named after the day the data was taken - here 201004, 201006, etc. (Do not worry about the faint folders.)
In what follows below I will use muNS as an example, when necessary. 

2. You need to write down your experimental parameters in generate_basic_info.py. This script will then generate a .json file that other scripts will read.

For instance, you need to write down the pixel-to-micron conversion as well as the list of all the days for which you have data. If you already have a list and you want to add the data from a new day, you need to update this list. For a full list of experimental parameters, see generate_basic_info.py . 

**1. post-processing of spots** 

1. Fit Gaussian profiles over the movies of interest. These fits can take a long time. It is best to let the computer run them overnight. If you have multiple threads in your computer, use them. I do that by opening multiple python terminals and launching the fits on a subset of movies in each terminal.
To launch the fits on a set of movies, for example movies from 210323, type 

`fit_spots_all_trajectories(spot_type = 'muNS', traj_filename = 'filtered_trajectories_all_renamed', days = ['210323'], starvation_times = all_starvation_times, avoid = None, simple_Gaussian = True)`

script: process_spots.py
saved_output: 
* traj_with_intensity_Gaussian_fit_results_simple_Gauss_YYMMDD.pkl
contains the trajectories you used as input with additional columns, where each new column corresponds to one fit parameter from the Gaussian fit
* intensity_Gaussian_fit_results_uncertainties_simple_Gauss_YYMMDD.pkl
contains the covariance for each fit parameter and each spot (related to the goodness of the fit)
* notes_from_Gaussian_fits_simple_Gauss_YYMMDD.txt
contains a list of the particles for which a Gaussian fit was not found. 

Here ‘YYMMDD’ is the date when you ran the script, and ’simple_Gauss’ will be included in the filename if this is what you have opted for in the fit, as is the default. Another option would be the asymmetric Gaussian, which is included in process_spots.py, but we have not used it and it has not been tested. 
 
2. Calculate the intensity magnitude for each spot:
 
`calculate_intensity_magnitude(‘muNS’, trajectories)`

script: process_trajectories.py. 
saved output: filtered_trajectories_all_with_magnitude_from_simple_Gauss_YYMMDD.pkl

3. Calculate the average starting value of quantities obtained from the fit, such as the magnitude calculated previously. In what follows, we only use the average starting values for the magnitude (to bin particles by size) and for the width (to filter out very large particles), however you will automatically calculate this for other values as well - such as the offset - as they might come in handy later in the future.

`append_starting_fit_values_to_traj(‘muNS’, trajectories)`

script: process_trajectories.py
saved output: filtered_trajectories_all_with_starting_values_from_simple_Gauss_fit_YYMMDD.pkl

4. Classify particles with respect to the width of their intensity distribution. I have used 450 nm as an upper cutoff for the width. 
Parry et al omit particles with width greater than the diffraction limit. This seems like an unnecessarily strict criterion, as I have confirmed after discussing with Suliana; so here we use twice the diffraction limit as a threshold. To perform this classification

`filter_by_width(‘muNS’, trajectories)`

script: process_trajectories.py.
saved output: filtered_trajectories_all_with_starting_values_from_simple_Gauss_fit_classified_by_ave_sigma_450p0nm_YYYYMMDD.pkl

The above filename indicates 450 nm as the threshold for sigma. The filename may vary slightly depending on your choice of optional parameters along the way, which you can read more about in the comments of the script. Again, ‘YYYYMMDD’ shows the day when you ran the last function.

**2. post-processing of trajectories** 

load trajectories with 
`t = load_all_results(‘muNS’, file_specifier, days = [‘new_day’])` 
script: load_all_results.py

Here file_specifier will be a string, or list of strings, that uniquely specifies the filename of the DataFrame of trajectories you have chosen to work with, across all movies. For instance, in the example above you could type 

`t = load_all_results(‘muNS’,[‘filtered’, ‘trajectories’, ‘ep_inbetween_0p0_2p5’, ‘with_dg’], days = [‘new_day’])`

to choose the first pickle file of trajectories shown in the folder. Or, if you have performed Gaussian fits on the spots and you want to use them, you could choose the trajectories DataFrames that have the fit results added to them, mentioned in section 1. With this line you will collect all DataFrames of trajectories that match the strings you specify in file_specifier, across all movies for which you have them. 

append the starting SNR with process_trajectories.py. This is optional as we have not ended up using the SNR. 

`t_snr = append_starting_snr_to_traj(‘muNS’, t)`

script: process_trajectories.py
saved output: filtered_trajectories_all_with_starting_snr.pkl

rename particles, such that their id contains the date and number of the movie in which they belong.
`t_renamed = rename_particles_in_traj(t_snr, ‘muNS’)`

script: process_trajectories.py
saved output: filtered_trajectories_all_renamed.pkl

Note that you will often use command 2.1 above to load dictionaries of trajectories, msds, or other DataFrames so you can handle them. Any time you see that a function requires a dictionary of pandas DataFrames as input line 2.1 shows you how to load these (see also the comments within load_all_results.py). 

If you are working with muNS particles you might also want to estimate their size, by measuring their integrated intensity (see also Parry et al, Cell 2014). This involves fitting every bright spot you have found with a Gaussian and calculating the integrated intensity from the fit results. This is described in section 1 above. 

**3. post-processing of msds**

load the new imsds

`ims = load_all_results( )`

script: load_all_results.py

rename the columns of the msds

`ims_renamed = rename_particles_in_imsds(ims, ‘muNS’)` 

script: process_msds.py
saved output: imsds_all_renamed.pkl

fit msds to get the values of D_app, α. Here you can choose the time window on which to apply the fit; units of time are seconds.

`f = fit_imsds_within_time_range(‘muNS’, ims_renamed, t_start=0.21, t_end=4 * 0.21)` 

script: process_msds.py
saved output: fit_results_alpha_Dapp_individual_tstart0p21_tend0p84sec.pkl

where the filename here denotes the time frame 0.21 - 0.84 seconds. 

**4. combine all results per movie for quantities that characterize each particle**

load the renamed trajectories

`t = load_all_results(‘muNS’, ‘filtered_trajectories_all_renamed.pkl’, most_recent = True)`

script: load_all_results.py 

Note that, if you are working with muNS particles on which you have performed a Gaussian fit to estimate their size, you will want to combine the results from the fits to the msds with the trajectories that contain information from the Gaussian fits on the spots; so, here, instead of ‘filtered_trajectories_all_renamed.pkl’ you will want to use 
‘filtered_trajectories_all_with_starting_values_from_simple_Gauss_fit_classified_by_ave_sigma_YYMMDD.pkl’

load the results from the fits to the imsds

`f = load_all_results(‘muNS’, 'fit_results_alpha_Dapp_individual_tstart0p21_tend0p48sec.pkl’)`

script: load_all_results.py

combine the results, per particle, into a single DataFrame

`c = combine_particle_results(t, f, ‘muNS’, label = ‘tstart0p21_tend0p84sec’)` 

script: collect_results.py
saved output: 'combined_results_per_particle_' + label + '.pkl'

where you have the choice to add a specification to the filename at the function input with ‘label’.

**5. check movie-to-movie and day-to-day variability**

This you can do by plotting histograms of the fit results, for instance, and comparing across movies and days. I can hand over functions that I have written to do that in the near future, but it does not matter how do you do it.
If variability is ok, you can combine data from all movies together into a single DataFrame, as described in section 6 below. 

**6. pool data**

1. If this is the first time you handle this data, or if you are adding a new day’s worth of data, or if you want to start including a new experimental parameter (such as a strain that you had not been considering before), you will first need to define the categories into which movies will be grouped. For more information on this, see the comments in the functions we call below. These are all in collect_results.py. Since the file we generate from now on are relevant for all data of this type, they will be saved in the central folder for the type of spot you are processing (e.x. in folder _muNS/ shown in part 0).


	To define pool categories: 

`define_pool_categories(‘muNS’, *params)`

script: collect_results.py
saved output: 
pool_categories.txt 

This file includes all the different pool categories for all different ways of pooling that we have considered. It is a good idea to open it and check that the categories correspond to what you intended. 

2. Distribute movies to their rightful categories:

`allocate_movies_to_categories(‘muNS’, *params)`

script: collect_results.py
saved output:
* pooled_directories_by_day_YYYYMMDD_HHMMSS.npy
* pooled_directories_by_day_without_timelag_YYYYMMDD_HHMMSS.npy
* pooled_directories_YYYYMMDD_HHMMSS.npy
* pooled_directories_without_timelag_YYYYMMDD_HHMMSS.npy
* pooled_movies_YYYYMMDD_HHMMSS.txt

where YYMMDD_HHMMSS denotes the year, month, day, hour, minute, second if the files’ generation. The .npy files are python dictionaries, where each key denotes a data category and the corresponding entry is a list of all the movies that have been allocated to this category. The text file contains this information for all four dictionaries above, so that you can confirm that pooling has been done correctly. 


3. Copy the newly generated files and rename them such that they no longer include the date and time of their generation. In our example, these files will be saved in the folder ‘_muNS/’. 

I keep all previous files of this type, as it allows me to know how I have grouped movies in the past and can help debug a discrepancy later in the future. However, in order for these files to be found in the following step by the function that performs data pooling, these filenames need to be simple, without the date and time information. Here is what you need to have in the end in that folder:  #image




4. combine the data

`pool(‘muNS’, file_type, ax, *params)`
script: collect_results.py 

Here ‘file_type’ will typically be trajectories or the msds. The script contains instructions, depending on what data it is that you are pooling. There is no saved output here; I preferred to save one final, reference DataFrames myself, later on during steps 7, 8. As these DataFrames grew during the project (for example when we add a new day or starvation condition), I wanted to avoid ending up with many saved DataFrames where it would be cumbersome to keep track of what was included in each case. Instead, once the data was more or less static, I saved a final “master” DataFrame during part 7, 8 below. 

You are now ready to manipulate this collective data as you wish, make histograms, heatmaps, or visualize it in other helpful ways. 

**7. filter particles (optional)**

In this step you can filter out particles that have so far been associated with an unphysical value for some parameter. For instance, you can remove from your analysis particles with D_app < 0 or with a NaN value in any of the entries. For a complete list of the filtering criteria, see the comments inside the filter functions mentioned below. The process goes as follows:


1. filter particles from the DataFrames that contain the combined particle results, those generated from step 4 above. You will use the list of particles from these files to further filter the DataFrames of msds. 

So, first load the files of combined particle results:

`r = load_all_results(‘muNS’, ‘combined_results_per_particle_fit_tstart0p21_tend0p84sec.pkl’)`
script: load_all_results.py

filter them:

`r_filtered = filter_results(‘muNS’, r, *params)`
script: collect_results.py

pool:

`r_filtered_pooled = pool(‘muNS’, r_filtered, 0)` 
script: collect_results.py

and save the result as a dictionary with numpy.save(), see appendix below for syntax. This can serve as your master dictionary of DataFrames for the per-particle results (D_app, α, intensity amplitude, particle size, … ) where all particles have physically meaningful values for all these quantities.

2. now filter the msds, using the indices of the results from step 7.1 to select out particles. To do that:

load the msds 

`ims  = load_all_results(‘muNS’, ‘imsds_all_renamed’, *params)`
script: load_all_results.py

pool them

`ims_pooled = pool(‘muNS’, ims, 1)`
script: collect_results.py

filter them 

`ims_pooled_filtered = filter_msds(‘muNS’, ims_pooled, path-to-file from step 1 above)`

save these pooled and filtered msds in the same location as the per-particle results with numpy.save(). You can use this as a matching “master” dictionary of DataFrames that you will refer to for all your further analysis and plots.  

**8. bin data by particle size (muNS)**

You will typically do this for the data frames that have all results per particle (D_app, α, particle size, etc). This is the data frame you get after step 4.3 above. In our analysis I binned after pooling data over all days. 
To do this, once you have loaded a pooled DataFrame (it could be the master DataFrame you saved above, we will call it df), use 

`binned_df = bin_dataframe(df, ‘particle_size’, below_size_limit_only = True, avoid = ['lowC'], quantiles = False)`
script: collect_results.py. 

This function currently has default bins, based on the distribution of particle sizes for our data. You can plot the distributions of particle sizes and confirm or refine this choice, depending on your data. 

The output, binned_df, is a dictionary of two levels: at a first level, the data is split according to size bin, and then within each bin the data is split according to the condition. You can save this dictionary of DataFrames next to your other “master” DataFrames, if you are satisfied with your choice of bins.

**Appendix: example lines of code**

**1. post-processing of trajectories**

`run load_all_results.py

t = load_all_results(‘muNS’, [‘filtered_trajectories, stub_length020’, ‘ep_inbetween_0p0_2p5’, 'snr_greater_than_1p0.pkl’], days = [‘201004’])

run process_trajectories.py

t_snr = append_starting_snr_to_traj(‘muNS’, t)  

t_renamed = rename_particles_in_traj(t_snr, ‘muNS’)` 

**2. post-processing of spots**

`run process_spots.py

f = fit_spots_all_trajectories(spot_type = 'muNS', traj_filename = 'filtered_trajectories_all_renamed', days = ['201004'], starvation_times = all_starvation_times, avoid = None, simple_Gaussian = True)

t_with_Gauss_info = load_all_results.load_all_results(‘muNS’, 'filtered_trajectories_all_with_starting_values_from_simple_Gauss_fit_classified_by_ave_sigma_450p0nm_20210614', days = [‘201004’])

t = calculate_intensity_magnitude(‘muNS’, t_with_Gauss_info)

t = append_starting_fit_values_to_traj(‘muNS’, t, label = ‘simple’)

t = filter_by_width(‘muNS’, t, width_limit = 0.450, label1 = 'simple', label2 = 'by_ave_sigma')`

**3. post-processing of msds**

`ims = load_all_results.load_all_results(‘muNS’, [‘imsds, stub_length020’, ‘snr_greater_than_1p0’, ‘until_lag_299’, ‘all’], days = [‘201004’])

run process_msds.py 

ims_renamed = rename_particles_in_imsds(ims, ‘muNS’)

f = fit_imsds_within_time_range(‘muNS’, ims_renamed, t_start=0.21, t_end=4 * 0.21)`

**4. combine all results per movie for quantities that characterize each particle**

`t = load_all_results.load_all_results(‘muNS’, ‘filtered_trajectories_all_renamed.pkl’)

f = load_all_results(‘muNS’, ‘fit_results_alpha_Dapp_individual_tstart0p21_tend0p48sec.pkl’)

c = combine_particle_results(t, f, ‘muNS’, label = ‘tstart0p21_tend0p84sec’, transfer_columns = ['starting_snr', 'average_starting_magnitude', 'average_starting_offset', 'below_diffraction_limit'])`

**6. pool data**

`run collect_results.py

d = define_pool_categories(spot_type, time_between_frames = [‘210’, ’30'], strains = ['bLR31', 'bLR32'], starvation_times = all_starvation_times, conditions = [‘lowN'])

a = allocate_movies_to_categories(‘muNS’, movies = ‘all_movies’, time_between_frames = [‘210’, ’30'], strains = ['bLR31', 'bLR32'], starvation_times = all_starvation_times, conditions = [‘lowN’])`

Note: you do not need to run the lines “d = “ and “a = “ above if you are not adding a new category of data to your set. 

to generate a dictionary of pooled DataFrames:

first load all DataFrames of interest, for example for trajectories

`t = load_all_results.load_all_results(‘muNS’, ‘filtered_trajectories_all_renamed.pkl’)`

then pool:

`t_pooled = pool(‘muNS’, t, 0, per_day = False, ignore_timelag = False, days = 'all_days', starvation_times = 'all', load = False, avoid = None)`

**7. filter particles (optional)**

start with the dictionary of all DataFrames that contain the results per particle, for all the movies:
`r = load_all_results.load_all_results('muNS', ‘filtered_trajectories_all_renamed’)

filter the particles in the DataFrames of this dictionary:
r_filtered = filter_results(‘muNS', r, below_size_limit_only=True)`

pool the DataFrames:
`r_filtered_pooled = pool(‘muNS', r_filtered, 0)` 

save the dictionary of pooled DataFrames as a reference, “master” dictionary of DataFrames:
`numpy.save(‘/Volumes/GlennGouldMac/PolyP/papers/ours/data/data_bank/210712_origins_filtered_pooled_results_per_particle.npy', r_filtered_pooled)`  this is my path & filename, here you choose your own. 

now load all the msds:
`ims = load_all_results.load_all_results(‘muNS’, ‘imsds_all_renamed’, *params)`

pool them:
`ims_pooled = pool(‘muNS’, ims, 1) (script: collect_results.py)`

filter them: 
`ims_pooled_filtered = filter_msds(‘muNS’, ims_pooled, path-to-file from step 1 above)`

save: 
`numpy.save(‘/Volumes/GlennGouldMac/PolyP/papers/ours/data/data_bank/210712_origins_filtered_pooled_results_per_particle.npy', ims_pooled_filtered)` again choose your own path and filename here

**8. bin DataFrames by particle size (only relevant for muNS)**

`binned_df = bin_dataframe(df, 'particle_size', below_size_limit_only = True, quantiles = False)` 

where rp is a dictionary of pooled DataFrames for the per-particle results. 
