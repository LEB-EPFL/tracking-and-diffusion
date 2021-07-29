# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

### BEFORE YOU RUN THE SCRIPT ###
# I have been using the hexadecimal color code, which lets you define colors in the format '#RRGGBB' where R, G, B are hex symbols. For more information and colors, see for ex. https://htmlcolorcodes.com/fr/

import matplotlib
#matplotlib.use('Agg')
#matplotlib.use('TkAgg')
import os
import re
import matplotlib.patches
import pylab
import pandas
import numpy
import tracking

colors = ['#000000','#00AAFF','#00AA00','#00FF00','#00FF55','#00FFAA']

def make_msd_plot(name, loglog = False, keyword = None):
    '''
    Creates a plot according to the specifications we like when we plot msds.
    
    INPUT
    -----
    name : str
        The name of the new figure.
        
    loglog : boolean, defaults to False
        Option to make a log-log plot.
        
    keyword : str, defaults to None
        A string you may wish to add to the name of the plot.
        
    OUTPUT
    ------
    The figure and axes instances of the new figure.
    '''
    
    if 'i' in name:
        label = 'individual'
    elif 'e' in name:
        label = 'ensemble'
    else:
        label = ''

    title = label + ' mean square displacements'
    if isinstance(keyword, str):
        title = label + keyword

    if not pylab.fignum_exists(name): #or new_figure:
        fig, ax = pylab.subplots(num = name)
        ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]', xlabel='lag time $t$')
        ax.set_title(title)
    else:
        fig = pylab.figure(name)
        ax = fig.gca()

    if loglog:
        ax.set_xscale('log')
        ax.set_yscale('log')

    #pylab.show()

    return fig, ax

def plot_msds(imsds = None, emsd = None, loglog = False, color = '#000000', symbol = '.', alpha_imsd = 0.075, alpha_emsd = 1.0, linewidth = 2, legend_content = None, legend_location = None, emsd_label = None, interactive_plot = True, new_imsd_figure = True, new_emsd_figure = True, plot_imsds = True, imsd_title = 'imsds', emsd_title = 'emsd', edgecolor = 'None'):
    '''
    Plot the msds given as input.
    
    INPUT
    -----
    imsds : pandas DataFrame, defaults to None
        The dataframe with the imsds you want to plot.
    
    emsd : pandas DataFrame, defaults to None
        The dataframe with the emsd you want to plot. 
        
    loglog : boolean, defaults to False
        Option to make a log-log plot.
        
    color : str
        The color of the plots, here often specified in hexadecimal.
        
    symbol : str
        The symbol that will represent the data points on the plot.
        
    alpha_imsd : float, defaults to 0.075
        The transparency of the imsd curves. It is useful to make them somewhat transparent, because usually there are hundreds of them on the same plot.
        
    alpha_emsd : float, defaults to 1.0
        The transparency of the emsd curve. It is useful to make them somewhat transparent when you want to plot multiple ones - for example, to see the effect on the msd of a parameter change, which may be small, leading to the points overlapping a lot.
    
    linewidth : int
        The thickness of the imsd curves.
        
    legend_content : str, defaults to None
        A word that describes a quantity that all the curves have in common. For instance, this is useful when you plot several groups of imsds together, each from a different bin of SNR. In this case, the identifier could be the SNR bin limits.
  
    emsd_label : str, defaults to None
        The label of this curve for the legend in the plot.
        
    interactive_plot : boolean, defaults to True
        Option to click on the plot of imsds and have the particle id pop out on the terminal.
        
    new_imsd_figure : boolean, defaults to True
        Option to make a new figure for the imsds. When False, if an imsd plot is open already, the new imsds will be plotted on top.
        
    new_emsd_figure : boolean, defaults to True
        Option to make a new figure for the emsd. When False, if an emsd plot is open already, the new emsd will be plotted on top. This is the case where the alpha_emsd will be relevant.
        
    plot_imsds : boolean, defaults to True
        Option to plot or not plot the imsds.
        
    OUTPUT
    ------
    At most two graphs, one for the imsds and one for the emsd if they are not None, as well as the corresponding DataFrames.
    '''
    
    n_particles = len(imsds.columns)
    if isinstance(emsd_label, str):
        emsd_label = emsd_label + ', ' + str(n_particles) + ' particles'
    else:
        emsd_label = str(n_particles) + ' particles'

    if plot_imsds:
        fig_imsd, ax_imsd = make_msd_plot(imsd_title, loglog = loglog)
    
        if isinstance(imsds, pandas.DataFrame):
            #if interactive_plot:
            #   fig_imsd.canvas.mpl_connect('pick_event', onpick)
            for i,j in enumerate(imsds.columns):
                ax_imsd.plot(imsds.index, imsds[imsds.columns[i]], color = color,
                             #picker = 3,
                             alpha = alpha_imsd, label = str(j), linewidth = linewidth)
            l = matplotlib.patches.Patch(color=color, label=legend_content)
        pylab.legend(handles=[l], frameon = False, loc = 2)
    else:
        ax_imsd = None
    
    if isinstance(emsd, pandas.Series):
        emsd = pandas.DataFrame(emsd)

    if isinstance(emsd, pandas.DataFrame):
        fig_emsd, ax_emsd = make_msd_plot(emsd_title, loglog = loglog)

        if 'error_finiteness' not in emsd.columns:
            if 'N' not in emsd.columns:
                raise ValueError('I do not have the number of independent measurements.')
            else:
                emsd['error_finiteness'] = emsd['msd'] / numpy.sqrt(emsd['N'])
        a = ax_emsd.errorbar(emsd.index, emsd.msd, yerr = emsd.error_finiteness, xerr = None, fmt = symbol, mfc = edgecolor, label = emsd_label, alpha = alpha_emsd, color = color)
#l = matplotlib.patches.Patch(color=color, label=legend_content)
#m = pylab.legend([l], [legend_content], loc=2, frameon = False)
        m = pylab.legend(a, [emsd_label],
                         #loc=2,
                         frameon = False)


#pylab.show()

    if not isinstance(emsd, pandas.DataFrame):
        ax_emsd = None
    return ax_imsd, ax_emsd

def load_data(quantity, selection_rules, directory = '.'):
    '''
    Load a set of data that have been calculated according to different selection rules.
    
    INPUT
    -----
    quantity : str
        The quantity you want to load. For instance, 'trajectories' for trajectories, 'imsds' for imsds.
        
    selection_rules : list of lists of the form ['quantity', 'relation', value]
        Here you specify the selection criteria you used to calculate the quantities you now want to visualize. Remember the syntax: for instance, to only include spots with SNR > 10, you would write selection_rule = [['snr', 'greater', 10]]. If you had added a second rule on size, you would write selection_rule = [['snr', 'greater', 10], ['size','lesser', 100]]. Etc.
    Note the following rules: the first item of the list must be a column of the coordinates DataFrame. The second column must be one of ['greater', 'lesser', 'inbetween', 'equal']. If 'inbetween', then the value entry must be a list of two numbers.
        
    directory : string, defaults to '.'
        The directory that contains the files of interest.
        
    OUTPUT
    ------
    A dictionary with the set of data you have selected to load, where the keys correspond to the selection rule for each data set.
    '''

    data = {}

    for i in selection_rules:
        identifier = tracking.translate_selection_rule(i)

        filenames = [x for x in os.listdir(directory) if re.search(quantity, x)]
        filenames = [x for x in filenames if identifier in x]

        identifier = identifier[1:]

        for j in filenames:
            data[identifier] = pandas.read_pickle(directory + j)

    return data

def plot_imsds(selection_rules, directory = '.', colors=['#000000'], alpha = 0.2, linewidth = 2):
    '''
    Plots all groups of imsd curves that have been calculated according to different selection rules. In the current implementation, this function is only able to handle a group of imsds that have been separated according to the same quantity; for instance, according to their SNR. For each SNR bin, the imsds will be shown with a different color and corresponding label in the legend.
        
    INPUT
    -----
    selection_rules : list of lists of the form ['quantity', 'relation', value]
        Here you specify the selection criteria you used to calculate the quantities you now want to visualize. Remember the syntax: for instance, to only include spots with SNR > 10, you would write selection_rule = [['snr', 'greater', 10]]. If you had added a second rule on size, you would write selection_rule = [['snr', 'greater', 10], ['size','lesser', 100]]. Etc.
        Note the following rules: the first item of the list must be a column of the coordinates DataFrame. The second column must be one of ['greater', 'lesser', 'inbetween', 'equal']. If 'inbetween', then the value entry must be a list of two numbers.
        
    directory : string, defaults to '.'
        The directory that contains the files of interest.
    
    colors : list of strings in hex
        The color of the imsd curves.
        
    alpha : number between 0 and 1
        The transparency of the imsd curves.
    
    linewidth : int
        The thickness of the imsd curves.
        
    OUTPUT
    ------
    A dictionary of the groups of imsds that are plotted, with keys according to the selection rule that was used to calculate them.
        
    '''
    
    data = load_data('imsds', selection_rules, directory=directory)
    
    p = []
    
    sorted_keys = list(data.keys())
    sorted_keys.sort(key = lambda x: x.split('-')[1])
    
    for i,j in enumerate(sorted_keys):
        plot_msds(imsds = data[j], emsd = None, loglog = False, color = colors[i], alpha_imsd = alpha, linewidth = linewidth, legend_content = j,  new_imsd_figure = True, new_emsd_figure = True, plot_imsds = True)
        p.append(matplotlib.patches.Patch(color=colors[i], label=j))
    
    pylab.legend(handles = p, loc = 2)

    pylab.show()

    return data

def plot_step_size_distribution(trajectories, px_to_micron, by_particle = False, nbins = 10., alpha = 0.01, new_figure = True, label = None, color = None, figure_name = 'step_size_distribution'):
    '''
    '''
    t = trajectories
    particle_ids = list(set(t.particle))
    
    min_step = t.previous_step_size.min() * px_to_micron
    #print(min_step)
    max_step = t.previous_step_size.max() * px_to_micron
    #print(max_step)
    bin_width = max_step / nbins
    bs = numpy.arange(0, max_step, bin_width)
    #print(bs)
    
    mean_step = t.previous_step_size.mean() * px_to_micron
    step_std = t.previous_step_size.std() * px_to_micron
    
    if new_figure:
        fig = pylab.figure(figure_name)
        ax = fig.add_subplot(111)
    else:
        fig = pylab.figure(figure_name)
        ax = fig.gca()
    
    if by_particle:
        for i,j in enumerate(particle_ids):
            ax.hist(t[t.particle==j].previous_step_size * px_to_micron, bins = bs, alpha = alpha,
                    color = color
                    )
    else:
        ax.hist(t.previous_step_size * px_to_micron, bins = bs, alpha = alpha, label = label, histtype='stepfilled', ec = color, color = color)

    ax.set_xlabel('step size (um)')
    ax.set_ylabel('occurences')

#    print('mean: ' + str(mean_step))
#    print('standard deviation: ' + str(step_std))

    return fig, ax

#### The function below is used for cliking on an imsd and getting out the particle id. It no longer works, TBD. ###
#def onpick(event):
#    thisline = event.artist
#    xdata = thisline.get_xdata()
#    ydata = thisline.get_ydata()
#    ind = event.ind
#    points = tuple(zip(xdata[ind], ydata[ind]))
#    print('onpick points:', points)
#    l = thisline.get_label()
#    #print(2*l)
#    print(l)
#    #text.set_text(l)



