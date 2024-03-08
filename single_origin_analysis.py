# %%
from collections import defaultdict
import json
import glob
import os
import numpy as np
import tifffile
import random
import copy

from microfilm.microplot import microshow
import tracking
import read_and_plot_oufti_output_cleaned as oufti
import load_all_results
import center_profile
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import to_rgb, hsv_to_rgb, to_hex
import colorsys

import re
from collect_results_cleaned import markershapes
from collect_results_cleaned import colors
from collect_results_cleaned import positions
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

SPOT_TYPE = "single_origin"

with open('general_info.json') as jf:
    data = json.load(jf)
data_folder = data['basic_directory']
figure_folder = data['figure_directory']
SETTINGS = data[SPOT_TYPE]


def batch_tracking(folder):
    movies = glob.glob(folder + '*')
    for movie in movies:
        print(os.path.basename(movie))

        movie = movie + "/"
        label1 = ('diameter' + str(SETTINGS['diameter']) + '_minmass' + str(SETTINGS['minmass']) +
                  '_percentile' + str(SETTINGS['percentile']))
        label2 = ('search_range' + str(SETTINGS['search_range']) + '_memory' +
                  str(SETTINGS['memory']).zfill(2))
        os.makedirs(os.path.join(movie, 'analysis', label1, label2), exist_ok=True)
        coordinates = tracking.find_spots_in_movie(movie,
                                                   SETTINGS['diameter'],
                                                   SETTINGS['minmass'],
                                                   SETTINGS['percentile'],
                                                   xlims=None,
                                                   ylims=None,
                                                   superimpose_phase=False,
                                                   extension=".tif",
                                                   use_if_existing_coordinates=False,
                                                   spots_in_phase=True, plot=False)
        coordinates.to_pickle(os.path.join(movie, 'analysis', label1, label2, 'peak_coords.pkl'))


# %% Get the spot positions in the data
folder = ("//lebsrv2.epfl.ch/LEB_SHARED/SHARED/_Scientific projects/polyP_SoM/"
          "paper_material/data/220620/single_origin/0h/")
batch_tracking(folder)
folder = ("//lebsrv2.epfl.ch/LEB_SHARED/SHARED/_Scientific projects/polyP_SoM/"
          "paper_material/data/220815/single_origin/0h/")
batch_tracking(folder)
folder = ("//lebsrv2.epfl.ch/LEB_SHARED/SHARED/_Scientific projects/polyP_SoM/"
          "paper_material/data/220815/single_origin/6h/")
batch_tracking(folder)
folder = ("//lebsrv2.epfl.ch/LEB_SHARED/SHARED/_Scientific projects/polyP_SoM/"
          "paper_material/data/220620/single_origin/6h/")
batch_tracking(folder)
# %%
days = ['220815', '220620']
peaks = load_all_results.load_all_results('single_origin', 'peak_coords', days=days)

# %% Get the cell profiles
seg_data = oufti.load_segmentation_results(days, "single_origin")
cells = center_profile.main(seg_data[0], profile=False)


# %%
for key in seg_data[0].keys():
    seg_data[0][key]['n_peaks'] = np.zeros(len(seg_data[0][key].index))

for key in seg_data[0].keys():
    for idx, peak in peaks[key].iterrows():
        peak_pos = center_profile.geometry.Point(peak['x'], peak['y'])
        for cell in cells[key]:
            if cell.contains(peak_pos):
                seg_data[0][key].loc[cell.cell_id, 'n_peaks'] += 1

# %%
for movie in seg_data[0].keys():
    os.makedirs(os.path.join(data['basic_directory'], '_single_origin', movie), exist_ok=True)
    seg_data[0][movie].to_pickle(os.path.join(data['basic_directory'], '_single_origin', movie,
                                 'segmentations_with_n_points.pkl'))


# %% Check these results visually

def black_to_cyan_cm():
    cdict = {'red':  [[0.0, 0.0, 0.0],
                      [1., 0., 0.]],
            'green': [[0.0,  0.0, 0.0],
                      [1,  1.0, 1.0]],
            'blue':  [[0.0,  0.0, 0.0],
                      [1,  1.0, 1.0]],
            'alpha': [[0.0,  1., 1.],
                      [1.,  1., 1.]]}

    newcmp = LinearSegmentedColormap('testCmap', cdict, N=256)
    return newcmp


def black_to_yellow_cm():
    cdict = {'red':  [[0.0, 0.0, 0.0],
                      [1., 1., 1.]],
            'green': [[0.0,  0.0, 0.0],
                      [1,  1.0, 1.0]],
            'blue':  [[0.0,  0.0, 0.0],
                      [1,  0.0, 0.0]],
            'alpha': [[0.0,  1., 1.],
                      [1.,  1., 1.]]}

    newcmp = LinearSegmentedColormap('testCmap', cdict, N=256)
    return newcmp


def plot_movie_overview(movie, crop=None, frame_size=30, cell_color=None):

    plt.figure()
    starv = '0h' if '0h' in movie else '6h'
    if cell_color is None:
        colors = ['w', 'g', 'r', 'r', 'r', 'r']
    else:
        colors = [cell_color]*6

    for idx, peak in peaks[movie].iterrows():
        plt.scatter(peak['x'], peak['y'], marker='x', color='k')
    for cell in cells[movie]:
        x = [x - 1 for x in cell.exterior.xy[0]]
        y = [y - 1 for y in cell.exterior.xy[1]]
        plt.plot(x, y,
                 color=colors[int(seg_data[0][movie].loc[cell.cell_id, 'n_peaks'])],
                 linewidth=2)
        if cell.nucleoid:
            x = [x - 1 for x in cell.nucleoid.exterior.xy[0]]
            y = [y - 1 for y in cell.nucleoid.exterior.xy[1]]
            plt.plot(x, y, color='#40e5c2', linewidth=3)

    image_folder = os.path.join(data_folder, '220620', 'single_origin', starv, movie, 'images')
    image_path = glob.glob(image_folder + '/*.tif')
    image = tifffile.imread(image_path)
    peaks_folder = os.path.join(data_folder, '220620', 'single_origin', starv, movie,
                                'phase_images')
    peaks_path = glob.glob(peaks_folder + '/*.tif')
    peaks_img = tifffile.imread(peaks_path)
    phase_folder = os.path.join(data_folder, '220620', 'single_origin', starv, movie, 'extra_images')
    phase_path = glob.glob(phase_folder + '/*.tif')
    phase_img = tifffile.imread(phase_path)
    if crop is not None:
        image_vs = image[int(crop[1] - frame_size/2 - 1):int(crop[1] + frame_size/2 + 1),
                         int(crop[0] - frame_size/2 - 1):int(crop[0] + frame_size/2 + 1)]
        limit = [image_vs.min(), image_vs.max()]
        peaks_img_vs = peaks_img[int(crop[1] - frame_size/2 - 1):int(crop[1] + frame_size/2 + 1),
                                 int(crop[0] - frame_size/2 - 1):int(crop[0] + frame_size/2 + 1)]
        peak_limit = [peaks_img_vs.mean(), peaks_img_vs.max()]
        phase_image_vs = phase_img[int(crop[1] - frame_size/2 - 1):int(crop[1] + frame_size/2 + 1),
                                   int(crop[0] - frame_size/2 - 1):int(crop[0] + frame_size/2 + 1)]
        phase_limit = [phase_image_vs.min(), phase_image_vs.max()]
    microshow([image, peaks_img, phase_img],
              cmaps=[black_to_cyan_cm(), black_to_yellow_cm(), 'gray'],
              limits=[limit, peak_limit, phase_limit],
              ax=plt.gca())
    # plt.imshow(image, black_to_cyan_cm(), vmin=image_vs.min(), vmax=image_vs.max())
    # plt.imshow(peaks_img, black_to_yellow_cm(), vmin=peaks_img_vs.mean(), vmax=peaks_img_vs.max())
    plt.gca().axis('equal')


# %%
for movie in cells.keys():
    plot_movie_overview(movie)
    plt.show()

# %% Example frames
key_list = list(cells.keys())
frames_list = {'wt_0h_single': {'movie': key_list[0], 'crop': [874, 674]},
               'wt_0h_dual': {'movie': key_list[0], 'crop': [346, 343]},
               'dpolyp_0h_dual': {'movie': key_list[3], 'crop': [1003, 1142]},
               'dpolyp_0h_single': {'movie': key_list[5], 'crop': [203, 361]},
               'wt_6h_single':  {'movie': key_list[6], 'crop': [687, 576]},
               'wt_6h_dual':  {'movie': key_list[6], 'crop': [452, 581]},
               'dpolyp_6h_single':  {'movie': key_list[9], 'crop': [902, 969]},
               'dpolyp_6h_dual':  {'movie': key_list[9], 'crop': [709, 926]}
               }

frame_size = 30

for key in frames_list.keys():
    data = frames_list[key]
    plot_movie_overview(data['movie'], data['crop'], frame_size, cell_color='#FFA500')
    plt.xlim(data['crop'][0] - frame_size/2, data['crop'][0] + frame_size/2)
    plt.ylim(data['crop'][1] - frame_size/2, data['crop'][1] + frame_size/2)
    plt.title(key)
    plt.savefig(os.path.join(figure_folder, 'single_origin', 'example_frames', key + '.svg'))


# %% Filter the datasets to only look at cells that have one origin
single_origin = {}
dual_origin = {}
for movie in seg_data[0].keys():
    single_origin[movie] = seg_data[0][movie][seg_data[0][movie].n_peaks == 1]
    dual_origin[movie] = seg_data[0][movie][seg_data[0][movie].n_peaks == 2]


# %% Group the data the way we are interested and show
single_ori_data = defaultdict(lambda: defaultdict(list))
dual_ori_data = defaultdict(lambda: defaultdict(list))
for movie in single_origin.keys():
    strain = 'wt' if 'lr379' in movie.lower() else 'dpolyp'
    starv = '6h' if '6h' in movie else '0h'
    group = strain + '_' + starv
    single_ori_data[group]['nuc_area'].extend(single_origin[movie].nucleoid_area)
    single_ori_data[group]['nc_ratio'].extend(single_origin[movie].nc_ratio)
    single_ori_data[group]['cell_area'].extend(single_origin[movie].area)
    single_ori_data[group]['sum_intensity'].extend(single_origin[movie].sum_intensity)
    dual_ori_data[group]['nuc_area'].extend(dual_origin[movie].nucleoid_area)
    dual_ori_data[group]['nc_ratio'].extend(dual_origin[movie].nc_ratio)
    dual_ori_data[group]['cell_area'].extend(dual_origin[movie].area)
    dual_ori_data[group]['sum_intensity'].extend(dual_origin[movie].sum_intensity)


# %%PLot nucleoid Area
plot_data = copy.deepcopy(dual_ori_data)

fig, axs = plt.subplots(2, 2)
fig.suptitle("Nucleoid Area [um**2]")
axs = axs.flatten()
for idx, key in enumerate(plot_data):
    print(key)
    plt.sca(axs[idx])
    data = plot_data[key]['nuc_area']
    data = np.asarray(data)
    plt.hist(data/86.6, bins=np.linspace(0, 3, 20))
    print(len(data[list(~np.isnan(data))]))
    axs[idx].set_title(key)
    axs[idx].set_xlim(0, 3)
    if idx < 2:
        axs[idx].set_xticklabels([])


# %% Plot Area medians
data_1 = copy.deepcopy(single_ori_data)
data_2 = copy.deepcopy(dual_ori_data)
all_data = {'single': data_1, "double": data_2}
from collect_results_cleaned import markershapes, positions, colors

for single_double, this_data in all_data.items():
    for key in this_data.keys():
        data = [x/86.6 for x in this_data[key]['cell_area']]
        n = len(data)
        ms = 12
        strain = 'bLR1' if 'wt' in key.lower() else 'bLR2'
        starvation_time = '0h' if '0h' in key else '6h'
        condition = 'lowN' if starvation_time == '6h' else ''
        my_color = colors[strain][starvation_time + condition] if single_double == 'single' else 'w'
        plt.errorbar(positions[strain][starvation_time+condition],
                        np.nanmedian(data), xerr = 0, yerr = np.nanstd(data) / np.sqrt(n),
                        fmt = markershapes[starvation_time + condition], markersize = ms,
                        markeredgecolor = colors[strain][starvation_time + condition],
                        ecolor = colors[strain][starvation_time + condition],
                        color = my_color,
                        alpha = 1, markeredgewidth = 2, capsize = 10)
    plt.xticks([1, 1.2, 1.4, 1.6], ['WT\n0h', '\n6h lowN', 'ΔpolyP\n0h', '\n6h lowN'])
    plt.ylabel("cell area (μm²)")
    ax = plt.gca()
    plt.title(single_double + " origin")
    # ax.text(1.4, 1.2, "● single origin", color='k')
    # ax.text(1.4, 1.3, "○ dual origin", color='k')

    plt.savefig(figure_folder + "/single_origin/" + "both_cell_area.svg")
    plt.savefig(figure_folder + "/single_origin/" + "both_cell_area.png")

    from collections.abc import MutableMapping

    def flatten(dictionary, parent_key='', separator='_', levels = 2, level=1):
        items = []
        print(level)
        print(parent_key)
        for key, value in dictionary.items():
            new_key = parent_key + separator + key if parent_key else key
            if level == levels:
                items.append((new_key, value))
                continue
            if isinstance(value, MutableMapping):
                items.extend(flatten(value, new_key, separator=separator, levels=levels, level=level+1).items())
            else:
                items.append((new_key, value))
        return dict(items)

    import pandas
    flat_data = flatten(all_data)
    print(flat_data.keys())
    p_values = pandas.DataFrame(columns=list(flat_data.keys()), index=list(flat_data.keys()))
    pandas.set_option('display.float_format', '{:.2E}'.format)
    # all_results = collect_results.pool(spot_type, results, 0, per_day=True, avoid=avoid + ['lowC'])
    from scipy import stats
    for i in flat_data.keys():
        for j in flat_data.keys():
            if i != j:
                d0 = flat_data[i]['cell_area']
                d1 = flat_data[j]['cell_area']
                p_values[i][j] = stats.median_test(d0,d1, nan_policy = 'omit')[1]
                # p_values[i][j] = stats.mannwhitneyu(d0, d1, alternative='two-sided')[1]
    print(p_values)
    p_values.to_csv(figure_folder + "/single_origin/" + "_cell_area_p_values.csv")

# %%
fig, axs = plt.subplots(2, 2)
fig.suptitle("Nucleoid Cell Ratio")
axs = axs.flatten()
for idx, key in enumerate(plot_data):
    print(key)
    plt.sca(axs[idx])
    data = plot_data[key]['nc_ratio']
    plt.hist(data, bins=np.linspace(0.25, 0.8, 20))
    data = np.asarray(data)
    print(len(data[list(~np.isnan(data))]))
    axs[idx].set_title(key)
    axs[idx].set_xlim(0.25, 0.8)
    if idx < 2:
        axs[idx].set_xticklabels([])

# %%
fig, axs = plt.subplots(2, 2)
fig.suptitle("Cell Area [um**2]")
axs = axs.flatten()
for idx, key in enumerate(plot_data):
    print(key)
    plt.sca(axs[idx])
    data = plot_data[key]['cell_area']
    data = np.asarray(data)
    plt.hist(data/86.6, bins=np.linspace(0, 4, 20))
    print(len(data[list(~np.isnan(data))]))
    axs[idx].set_title(key)
    axs[idx].set_xlim(0, 4)
    if idx < 2:
        axs[idx].set_xticklabels([])

# %% Heatmap of Cell size vs nucleoid size
comp = 'starv'
strain = 'wt'
legends = []

# Do this to randomize zorder position of the points
x_data1 = plot_data[strain + '_0h']['cell_area']
x_data2 = plot_data[strain + '_6h']['cell_area']
x_data = np.asarray(x_data1 + x_data2)/86.6

y_data1 = plot_data[strain + '_0h']['nuc_area']
y_data2 = plot_data[strain + '_6h']['nuc_area']
y_data = np.asarray(y_data1 + y_data2)/86.6

legends = [strain + '_0h', strain + '_6h']

my_colors = ['b', 'g']
color1 = [my_colors[0]]*len(x_data1)
color2 = [my_colors[1]]*len(y_data2)
color = color1 + color2

indices = list(range(x_data.shape[0]))
np.random.shuffle(indices)

x_data = x_data[indices]
y_data = y_data[indices]
color = np.asarray(color)[indices]
# for starv in ['_0h', '_6h']:
# legends.append(strain + starv)
# x_data = np.asarray(plot_data[strain + starv]['cell_area'])/86.6
# y_data = np.asarray(plot_data[strain + starv]['nuc_area'])/86.6

# Do a linear fit to these datasets
# fit_x1 = np.asarray(sorted(x_data1))/86.6
# fit_y1 = np.asarray([x for _, x in sorted(zip(x_data1, y_data1))])/86.6
# idx = np.isfinite(fit_x1) & np.isfinite(fit_y1)
# slope1, intercept1 = np.polyfit(fit_x1[idx], fit_y1[idx], 1)
# fit_x2 = np.asarray(sorted(x_data2))/86.6
# fit_y2 = np.asarray([x for _, x in sorted(zip(x_data2, y_data2))])/86.6
# idx = np.isfinite(fit_x2) & np.isfinite(fit_y2)
# slope2, intercept2 = np.polyfit(fit_x2[idx], fit_y2[idx], 1)

# x_fit = np.linspace(0, 10, 100)
# plt.plot(x_fit, slope1*x_fit + intercept1, c=my_colors[0], label=legends[0])
# plt.plot(x_fit, slope2*x_fit + intercept2, c=my_colors[1], label=legends[1])

# Do the main plot
plt.scatter(x_data, y_data, c=color, alpha=0.5, s=15)
plt.xlim(0.7, 4.5)
plt.ylim(0.4, 2.7)
plt.legend()
plt.xlabel('Cell Area [um**2]')
plt.ylabel('Nuceloid Area [um**2]')
plt.show()

plt.savefig(figure_folder + "/single_origin/" + strain + "_single_ori.svg")

# %% WT vs dPolyP at different time points
comp = "strain"
starv = '6h'
legends = []

x_data1 = plot_data['wt_' + starv]['cell_area']
x_data2 = plot_data['dpolyp_' + starv]['cell_area']
x_data = np.asarray(x_data1 + x_data2)/86.6

# y_data1 = plot_data['wt_' + starv]['nuc_area']
# y_data2 = plot_data['dpolyp_' + starv]['nuc_area']
y_data1 = plot_data['wt_' + starv]['nc_ratio']
y_data2 = plot_data['dpolyp_' + starv]['nc_ratio']
# y_data = np.asarray(y_data1 + y_data2)/86.6
y_data = np.asarray(y_data1 + y_data2)

legends = ['wt_' + starv, 'dpolyp_' + starv]
condition = "lowN" if starv == '6h' else ''

my_colors = [colors['bLR1'][starv+condition], colors['bLR2'][starv+condition]]
color1 = [my_colors[0]]*len(x_data1)
color2 = [my_colors[1]]*len(x_data2)
color = color1 + color2

indices = list(range(x_data.shape[0]))
np.random.shuffle(indices)

x_data = x_data[indices]
y_data = y_data[indices]
color = np.asarray(color)[indices]

#%%  Simple plot for this Figure 4C
plot_data = copy.deepcopy(single_ori_data)
my_colors = [colors['bLR1']['0h'], colors['bLR2']['0h']]
x_data, y_data, color, legends = organize_data_comp_starv('0h', plot_data, 'cell_area',
                                                           'nc_ratio', my_colors)
ax = axes_fixed_size(3.5, 2.5)
plt.scatter(x_data/86.6, y_data, c=color, edgecolor='none', alpha=0.5, s=10)
fake_legends({"color": my_colors[0], "edgecolor": 'none', "alpha": 0.5}, "WT\u20090h")
fake_legends({"color": my_colors[1], "edgecolor": 'none', "alpha": 0.5}, "ΔpolyP\u20090h")
write_n_values_starv('0h', plot_data, 'cell_area', 'nc_ratio')
plt.xlim(0.7, 3.5)
plt.ylim(0.2, 1)
plt.legend(loc="upper right")
plt.xlabel('cell area (μm²)')
plt.ylabel('NC ratio')
plt.savefig(figure_folder + "/single_origin/" "nc_ratio_0h.svg")
plt.show()

my_colors = [colors['bLR1']['6hlowN'], colors['bLR2']['6hlowN']]
x_data, y_data, color, legends = organize_data_comp_starv('6h', plot_data, 'cell_area',
                                                           'nc_ratio', my_colors)
ax = axes_fixed_size(3.5, 2.5)
plt.scatter(x_data/86.6, y_data, c=color, edgecolor='none', alpha=0.5, s=10)
fake_legends({"color": my_colors[0], "edgecolor": 'none', "alpha": 0.5}, "WT\u20090h")
fake_legends({"color": my_colors[1], "edgecolor": 'none', "alpha": 0.5}, "ΔpolyP\u20090h")
write_n_values_starv('6h', plot_data, 'cell_area', 'nc_ratio')
plt.xlim(0.7, 3.5)
plt.ylim(0.2, 1)
plt.legend(loc="upper right")
plt.xlabel('cell area (μm²)')
plt.ylabel('NC ratio')
plt.savefig(figure_folder + "/single_origin/" "nc_ratio_6h.svg")
plt.show()


# %%
# region Linear fits
# fit_x1 = np.asarray(sorted(x_data1))/86.6
# fit_y1 = np.asarray([x for _, x in sorted(zip(x_data1, y_data1))])/86.6
# idx = np.isfinite(fit_x1) & np.isfinite(fit_y1)
# slope1, intercept1 = np.polyfit(fit_x1[idx], fit_y1[idx], 1)
# fit_x2 = np.asarray(sorted(x_data2))/86.6
# fit_y2 = np.asarray([x for _, x in sorted(zip(x_data2, y_data2))])/86.6
# idx = np.isfinite(fit_x2) & np.isfinite(fit_y2)
# slope2, intercept2 = np.polyfit(fit_x2[idx], fit_y2[idx], 1)

# x_fit = np.linspace(0, 10, 100)
# plt.plot(x_fit, slope1*x_fit + intercept1, c=my_colors[0], label=legends[0])
# plt.plot(x_fit, slope2*x_fit + intercept2, c=my_colors[1], label=legends[1])
# endregion
plot_data = copy.deepcopy(single_ori_data)
comp='strain'
starv = '6h'
y_data_name = 'nuc_area'
if starv == '0h':
    my_colors = [colors['bLR1']['0h'], colors['bLR2']['0h']]
else:
    my_colors = [colors['bLR1']['6hlowN'], colors['bLR2']['6hlowN']]
x_data, y_data, color, legends = organize_data_comp_starv(starv, plot_data, 'cell_area',
                                                          y_data_name, my_colors)
fig = plt.figure(figsize=(10, 7))
gs = fig.add_gridspec(2, 2, hspace=0, wspace=0, width_ratios=[3, 1], height_ratios=[1, 3])
axs = gs.subplots(sharex='col', sharey='row')
#                         gridspec_kw={'width_ratios': [3, 1],
#                                      'height_ratios': [3, 1]})
axs[0, 1].axis('off')
axs[0, 1].set_facecolor('none')
y_factor = 86.6 if y_data_name == "nuc_area" else 1
axs[1, 0].scatter(x_data/86.6, y_data/y_factor, c=color, edgecolors='none', alpha=0.3, s=15)
axs[1, 0].scatter(-100, -100, color=my_colors[0], label=legends[0])
axs[1, 0].scatter(-100, -100, color=my_colors[1], label=legends[1])

axs[1, 0].legend()
axs[1, 0].set_xlabel('cell area (μm²)')
y_label = 'nucleoid area (μm²)' if y_data_name == "nuc_area" else "NC ratio"
axs[1, 0].set_ylabel(y_label)

for idx, key in enumerate(['wt_' + starv, 'dpolyp_' + starv]):
    density, bins = np.histogram(np.asarray(plot_data[key]['cell_area'])/86.6,
                                 bins=np.linspace(0.7, 3.5, 20))
    density = density / density.sum()
    widths = bins[:-1] - bins[1:]
    axs[0, 0].bar(bins[1:], density, width=widths, color=my_colors[idx])
# Replot the back one in front without fill
density, bins = np.histogram(np.asarray(plot_data['wt_' + starv]['cell_area'])/86.6,
                             bins=np.linspace(0.7, 3.5, 20))
density = density / density.sum()
widths = bins[:-1] - bins[1:]
axs[0, 0].stairs(density, bins - widths[0]/2, edgecolor=my_colors[0], facecolor='none',
                 linewidth=3)

axs[0, 0].set_xlim(0.7, 3.5)
axs[0, 0].set_xticks([1, 1.5, 2, 2.5, 3])
axs[0, 0].tick_params(direction='in')
axs[0, 0].set_ylabel("density")
# axs[0, 0].set_yticks(axs[0, 0].get_yticks()[1:])

y_lim = [0.4, 2.4] if y_data_name == 'nuc_area' else [0.2, 1]
for idx, key in enumerate(['wt_' + starv, 'dpolyp_' + starv]):
    density, bins = np.histogram(np.asarray(plot_data[key][y_data_name])/y_factor,
                                 bins=np.linspace(y_lim[0], y_lim[1], 20))
    # density, bins = np.histogram(np.asarray(plot_data[key]['nc_ratio'])/.866,
    #                              bins=np.linspace(0, 1, 20))
    density = density / density.sum()
    widths = bins[:-1] - bins[1:]
    axs[1, 1].barh(bins[1:], density, height=widths, color=my_colors[idx])
# Replot the back one in front without fill
density, bins = np.histogram(np.asarray(plot_data['wt_' + starv][y_data_name])/y_factor,
                             bins=np.linspace(y_lim[0], y_lim[1], 20))
# density, bins = np.histogram(np.asarray(plot_data[legends[0]]['nc_ratio'])/.866,
#                              bins=np.linspace(0, 1, 20))
density = density / density.sum()
widths = bins[:-1] - bins[1:]
axs[1, 1].stairs(density, bins - widths[0]/2, edgecolor=my_colors[0], facecolor='none',
                 linewidth=3, orientation='horizontal')
y_lim = [0.4, 2.4] if y_data_name == 'nuc_area' else [0.2, 1]
axs[1, 1].set_ylim(y_lim)
axs[1, 1].tick_params(direction='in')
# axs[1, 1].set_ylim(0.2, 1)
axs[1, 1].set_xlabel("density")
write_n_values_starv(starv, plot_data, 'cell_area', y_data_name)
# axs[1, 1].set_xticks(axs[1, 1].get_xticks()[1:])
if comp == 'starv':
    plt.savefig(f"{figure_folder}/single_origin/single_ori{strain}_{y_data_name}.svg")
elif comp == 'strain':
    plt.savefig(f"{figure_folder}/single_origin/single_ori{starv}_{y_data_name}.svg")

plt.show()


#%% Median NC ratio per type

ms = 12
ft = 'NC ratio median values'
plt.figure(ft, figsize=(6, 5))
plt.title(ft)
plot_data = copy.deepcopy(single_ori_data)
for key, value in plot_data.items():
    strain = re.findall(r'[a-z]*', key)[0]
    strain = "bLR1" if strain == "wt" else "bLR2"
    starvation_time = re.findall(r'\dh', key)[0]
    condition = 'lowN' if starvation_time == '6h' else ''
    ratio = [x for x in value['nc_ratio'] if x > 0]
    plt.errorbar(positions[strain][starvation_time + condition],
                 np.nanmedian(value['nc_ratio']), xerr=0, yerr=(np.std(ratio) /
                                                                np.sqrt(len(ratio))),
                 fmt=markershapes[starvation_time + condition], markersize=ms,
                 markeredgecolor='k', ecolor=colors[strain][starvation_time + condition],
                 color=colors[strain][starvation_time + condition], alpha=1,
                 markeredgewidth=2, capsize=10)
    csv_filename = f"{figure_folder}/single_origin/ncratios_{strain}_{starvation_time}.csv"
    np.savetxt(csv_filename, value['nc_ratio'], delimiter=",")
plt.xticks([1, 1.2, 1.4, 1.6], ['WT\n0h', '\n6h lowN', 'ΔpolyP\n0h', '\n6h lowN'])
plt.ylabel("NC ratio")
plt.savefig(figure_folder + "/single_origin/" "medians.svg")
plt.show()

ms = 12
ft = 'Nucleoid area median values'
plt.figure(ft, figsize=(6, 5))
plt.title(ft)
plot_data = copy.deepcopy(single_ori_data)
for key, value in plot_data.items():
    strain = re.findall(r'[a-z]*', key)[0]
    strain = "bLR1" if strain == "wt" else "bLR2"
    starvation_time = re.findall(r'\dh', key)[0]
    condition = 'lowN' if starvation_time == '6h' else ''
    ratio = [x/86.6 for x in value['nuc_area'] if x > 0]
    plt.errorbar(positions[strain][starvation_time + condition],
                 np.nanmedian(ratio), xerr=0, yerr=(np.std(ratio) /
                                                                np.sqrt(len(ratio))),
                 fmt=markershapes[starvation_time + condition], markersize=ms,
                 markeredgecolor='k', ecolor=colors[strain][starvation_time + condition],
                 color=colors[strain][starvation_time + condition], alpha=1,
                 markeredgewidth=2, capsize=10)
    csv_filename = f"{figure_folder}/single_origin/nuc_areas_{strain}_{starvation_time}.csv"
    np.savetxt(csv_filename, value['nuc_area'], delimiter=",")
plt.xticks([1, 1.2, 1.4, 1.6], ['WT\n0h', '\n6h lowN', 'ΔpolyP\n0h', '\n6h lowN'])
plt.ylim(0.65, 1.1)
plt.ylabel("nucleoid area (μm²)")
plt.savefig(figure_folder + "/single_origin/" "medians_nuc_area_11.svg")
plt.show()



# %% single_dual combined

def nuc_vs_cell_area_at_starv(starv, strain, single_ori_data, dual_ori_data, my_colors):
    ax = axes_fixed_size(2, 2)

    plot_data1 = copy.deepcopy(single_ori_data)
    plot_data2 = copy.deepcopy(dual_ori_data)
    x_data, y_data, color, legends = organize_data_comp_nuc(starv, strain, plot_data1, plot_data2,
                                                            'cell_area','nuc_area', my_colors)
    alpha = 0.3
    ax.scatter(x_data/86.6, y_data/86.6, c=color, edgecolors='none', alpha=alpha, s=5)
    ax.scatter(-100, -100, color=my_colors[0], edgecolors='none', alpha=alpha,
               label=legends[0])
    ax.scatter(-100, -100, color=my_colors[1], edgecolors='none', alpha=alpha,
               label=legends[1])
    write_n_values_nuc(starv, strain, plot_data1, plot_data2, 'cell_area','nuc_area')

    ax.set_xlim(0.7, 4.1)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_ylim(0.4, 2.4)
    ax.set_yticks([.5, 1, 1.5, 2])
    ax.legend()
    ax.set_xlabel('cell area (μm²)')
    ax.set_ylabel('nuceloid area (μm²)')

hue_offset = -0.08
my_colors = [colors['bLR1']['0h'], "#888888"]  #  adjust_color(colors['bLR1']['0h'], hue_offset)]
nuc_vs_cell_area_at_starv('0h', 'wt', single_ori_data, dual_ori_data, my_colors)
plt.savefig(figure_folder + "/single_origin/" "single_dual_wt_0h.svg")
my_colors = [colors['bLR1']['6hlowN'], "#888888"]  #  adjust_color(colors['bLR1']['6hlowN'], hue_offset)]
nuc_vs_cell_area_at_starv('6h', 'wt', single_ori_data, dual_ori_data, my_colors)
plt.savefig(figure_folder + "/single_origin/" "single_dual_wt_6h.svg")

my_colors = [colors['bLR2']['0h'], "#888888"]  #  adjust_color(colors['bLR2']['0h'], hue_offset)]
nuc_vs_cell_area_at_starv('0h', 'dpolyp', single_ori_data, dual_ori_data, my_colors)
plt.savefig(figure_folder + "/single_origin/" "single_dual_dpolyp_0h.svg")
my_colors = [colors['bLR2']['6hlowN'], "#888888"]  #  adjust_color(colors['bLR2']['6hlowN'], hue_offset)]
nuc_vs_cell_area_at_starv('6h', 'dpolyp', single_ori_data, dual_ori_data, my_colors)
plt.savefig(figure_folder + "/single_origin/" "single_dual_dpolyp_6h.svg")
plt.show()


#%% ncratio vs cell_ares all
def ncratio_vs_cell_area_at_starv(starv, strain, single_ori_data, dual_ori_data, my_colors):
    ax = axes_fixed_size(2, 2)

    plot_data1 = copy.deepcopy(single_ori_data)
    plot_data2 = copy.deepcopy(dual_ori_data)
    x_data, y_data, color, legends = organize_data_comp_nuc(starv, strain, plot_data1, plot_data2,
                                                            'cell_area', 'nc_ratio', my_colors)
    alpha=0.5
    ax.scatter(x_data/86.6, y_data, c=color, edgecolors='none', alpha=alpha, s=5)
    ax.scatter(-100, -100, color=my_colors[0], edgecolors='none', alpha=alpha,
               label=legends[0])
    ax.scatter(-100, -100, color=my_colors[1], edgecolors='none', alpha=alpha,
               label=legends[1])
    write_n_values_nuc(starv, strain, plot_data1, plot_data2, 'cell_area','nc_ratio')

    ax.set_xlim(0.7, 4.1)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_ylim(0.2, 1)
    ax.set_yticks([.2, .4, .6, .8, 1])
    ax.legend()
    ax.set_xlabel('cell area (μm²)')
    ax.set_ylabel('NC ratio')


my_colors = [colors['bLR1']['0h'], "#888888"]  # adjust_color(colors['bLR1']['0h'], hue_offset)]
ncratio_vs_cell_area_at_starv('0h','wt', single_ori_data, dual_ori_data, my_colors)
plt.savefig(figure_folder + "/single_origin/" "single_dual_ncratio_wt_0h.svg")
my_colors = [colors['bLR1']['6hlowN'], "#888888"]  #  adjust_color(colors['bLR1']['6hlowN'], hue_offset)]
ncratio_vs_cell_area_at_starv('6h', 'wt', single_ori_data, dual_ori_data, my_colors)
plt.savefig(figure_folder + "/single_origin/" "single_dual_ncratio_wt_6h.svg")

my_colors = [colors['bLR2']['0h'], "#888888"]  #  adjust_color(colors['bLR2']['0h'], hue_offset)]
ncratio_vs_cell_area_at_starv('0h','dpolyp', single_ori_data, dual_ori_data, my_colors)
plt.savefig(figure_folder + "/single_origin/" "single_dual_ncratio_dpolyp_0h.svg")
my_colors = [colors['bLR2']['6hlowN'], "#888888"]  #  adjust_color(colors['bLR2']['6hlowN'], hue_offset)]
ncratio_vs_cell_area_at_starv('6h', 'dpolyp', single_ori_data, dual_ori_data, my_colors)
plt.savefig(figure_folder + "/single_origin/" "single_dual_ncratio_dpolyp_6h.svg")

plt.show()

#%% Intensity vs cell area

# Recombine data for sybr_green Fig S7 B & C
all_data =  defaultdict(lambda: defaultdict(list))
for key, value in single_ori_data.items():
    all_data[key] = copy.deepcopy(value)
    for subkey in all_data[key].keys():
        all_data[key][subkey].extend(dual_ori_data[key][subkey])

# all_data = copy.deepcopy(single_ori_data)
y_lim = [0, 4E6]

my_colors = [colors['bLR1']['0h'], colors['bLR1']['6hlowN']]
x_data, y_data, color, legends = organize_data_comp_strain('wt', all_data, 'cell_area',
                                                           'sum_intensity', my_colors)
ax = axes_fixed_size(2, 2)
plt.scatter(x_data/86.6, y_data, c=color, edgecolor='none', alpha=0.5, s=5)
fake_legends({"color": my_colors[0], "edgecolor": 'none', "alpha": 0.5}, "WT\u20090h")
fake_legends({"color": my_colors[1], "edgecolor": 'none', "alpha": 0.5}, "WT\u20096h")
write_n_values('wt', all_data, 'cell_area', 'sum_intensity')
plt.xlim(0.7, 4.1)
plt.ylim(y_lim)
plt.legend()
plt.xlabel('cell area (μm²)')
plt.ylabel('total intensity/cell')
plt.savefig(figure_folder + "/single_origin/" "intensity_wt_sybr.svg")
plt.show()

my_colors = [colors['bLR2']['0h'], colors['bLR2']['6hlowN']]
x_data, y_data, color, legends = organize_data_comp_strain('dpolyp', all_data, 'cell_area',
                                                           'sum_intensity', my_colors)
ax = axes_fixed_size(2, 2)
plt.scatter(x_data/86.6, y_data, c=color, edgecolor='none', alpha=0.5, s=5)
fake_legends({"color": my_colors[0], "edgecolor": 'none', "alpha": 0.5}, "ΔpolyP\u20090h")
fake_legends({"color": my_colors[1], "edgecolor": 'none', "alpha": 0.5}, "ΔpolyP\u20096h")
write_n_values('dpolyp', all_data, 'cell_area', 'sum_intensity',)
plt.xlim(0.7, 4.1)
plt.ylim(y_lim)
plt.legend()
plt.xlabel('cell area (μm²)')
plt.ylabel('total intensity/cell')
plt.savefig(figure_folder + "/single_origin/" "intensity_dpolyp_sybr.svg")
plt.show()

my_colors = [colors['bLR1']['0h'], colors['bLR1']['6hlowN']]
HU_data_plot = copy.deepcopy(HU_data)
y_lim = [0, 2E6]
x_data, y_data, color, legends = organize_data_comp_strain('wt', HU_data_plot, 'area',
                                                           'sum_intensity', my_colors)
ax = axes_fixed_size(2, 2)
plt.scatter(x_data/86.6, y_data, c=color,edgecolor='none', alpha=0.5, s=5)
fake_legends({"color": my_colors[0], "edgecolor": 'none', "alpha": 0.5}, "WT\u20090h")
fake_legends({"color": my_colors[1], "edgecolor": 'none', "alpha": 0.5}, "WT\u20096h")
write_n_values('wt', HU_data_plot, 'area', 'sum_intensity')
plt.xlim(0.7, 4.1)
plt.ylim(y_lim)
plt.legend()
plt.xlabel('cell area (μm²)')
plt.ylabel('total intensity/cell')
plt.savefig(figure_folder + "/single_origin/" "intensity_wt_HU.svg")
plt.show()

my_colors = [colors['bLR2']['0h'], colors['bLR2']['6hlowN']]
y_lim = [0, 2E6]
x_data, y_data, color, legends = organize_data_comp_strain('dpolyP', HU_data_plot, 'area',
                                                           'sum_intensity', my_colors)
ax = axes_fixed_size(2, 2)
plt.scatter(x_data/86.6, y_data, c=color,edgecolor='none', alpha=0.5, s=5)
fake_legends({"color": my_colors[0], "edgecolor": 'none', "alpha": 0.5}, "ΔpolyP\u20090h")
fake_legends({"color": my_colors[1], "edgecolor": 'none', "alpha": 0.5}, "ΔpolyP\u20096h")
write_n_values('dpolyP', HU_data_plot, 'area', 'sum_intensity')
plt.xlim(0.7, 4.1)
plt.ylim(y_lim)
plt.legend()
plt.xlabel('cell area (μm²)')
plt.ylabel('total intensity/cell')
plt.savefig(figure_folder + "/single_origin/" "intensity_dpolyp_HU.svg")
plt.show()

#%% Figure S7 F & G
mpl.rcParams.update({"legend.frameon": False, "legend.loc": "upper left",
                     "legend.handletextpad": 0, "legend.borderpad": 0,
                     "legend.borderaxespad": 0.2, "legend.handlelength": 1.2,
                     "legend.labelspacing": 0.2})
my_colors = [colors['bLR1']['0h'], colors['bLR2']['0h']]
# my_colors = ["#888888", colors['bLR2']['0h']]

HU_data_plot = copy.deepcopy(HU_data)
y_lim = [0.4, 2.4]
x_data, y_data, color, legends = organize_data_comp_starv('0h', HU_data_plot, 'area',
                                                           'nucleoid_area', my_colors)
ax = axes_fixed_size(2, 2)
plt.scatter(x_data/86.6, y_data/86.6, c=color,edgecolor='none', alpha=0.5, s=5)
write_n_values_starv('0h', HU_data_plot, 'area', 'nucleoid_area')
plt.xlim(0.7, 4.1)
plt.ylim(y_lim)
fake_legends({"color": my_colors[0], "edgecolor": 'none', "alpha": 0.5}, "WT\u20090h")
fake_legends({"color": my_colors[1], "edgecolor": 'none', "alpha": 0.5}, "ΔpolyP\u20090h")
plt.legend()
plt.xlabel('cell area (μm²)')
plt.ylabel('nucleoid area (μm²)')
plt.savefig(figure_folder + "/single_origin/" "HU_nuc_0h.svg")
plt.show()


my_colors = [colors['bLR1']['6hlowN'], colors['bLR2']['6hlowN']]
x_data, y_data, color, legends = organize_data_comp_starv('6h', HU_data_plot, 'area',
                                                           'nucleoid_area', my_colors)
ax = axes_fixed_size(2, 2)
plt.scatter(x_data/86.6, y_data/86.6, c=color,edgecolor='none', alpha=0.5, s=5)
write_n_values_starv('6h', HU_data_plot, 'area', 'nucleoid_area')
plt.xlim(0.7, 4.1)
plt.ylim(y_lim)
fake_legends({"color": my_colors[0], "edgecolor": 'none', "alpha": 0.5}, "WT\u20096h")
fake_legends({"color": my_colors[1], "edgecolor": 'none', "alpha": 0.5}, "ΔpolyP\u20096h")
plt.legend()
plt.xlabel('cell area (μm²)')
plt.ylabel('nucleoid area (μm²)')
plt.savefig(figure_folder + "/single_origin/" "HU_nuc_6h.svg")
plt.show()

my_colors = [colors['bLR1']['0h'], colors['bLR2']['0h']]
HU_data_plot = copy.deepcopy(HU_data)
y_lim = [0.2, 1]
x_data, y_data, color, legends = organize_data_comp_starv('0h', HU_data_plot, 'area',
                                                           'nc_ratio', my_colors)
ax = axes_fixed_size(2, 2)
plt.scatter(x_data/86.6, y_data, c=color,edgecolor='none', alpha=0.5, s=5)
write_n_values_starv('0h', HU_data_plot, 'area', 'nc_ratio')
plt.xlim(0.7, 4.1)
plt.ylim(y_lim)
fake_legends({"color": my_colors[0], "edgecolor": 'none', "alpha": 0.5}, "WT\u20090h")
fake_legends({"color": my_colors[1], "edgecolor": 'none', "alpha": 0.5}, "ΔpolyP\u20090h")
plt.legend()
plt.xlabel('cell area (μm²)')
plt.ylabel("NC ratio")
plt.savefig(figure_folder + "/single_origin/" "HU_ncratio_0h.svg")
plt.show()

my_colors = [colors['bLR1']['6hlowN'], colors['bLR2']['6hlowN']]
x_data, y_data, color, legends = organize_data_comp_starv('6h', HU_data_plot, 'area',
                                                           'nc_ratio', my_colors)
ax = axes_fixed_size(2, 2)
plt.scatter(x_data/86.6, y_data, c=color,edgecolor='none', alpha=0.5, s=5)
write_n_values_starv('6h', HU_data_plot, 'area', 'nc_ratio')
plt.xlim(0.7, 4.1)
plt.ylim(y_lim)
fake_legends({"color": my_colors[0], "edgecolor": 'none', "alpha": 0.5}, "WT\u20096h")
fake_legends({"color": my_colors[1], "edgecolor": 'none', "alpha": 0.5}, "ΔpolyP\u20096h")
plt.legend()
plt.xlabel('cell area (μm²)')
plt.ylabel("NC ratio")
plt.savefig(figure_folder + "/single_origin/" "HU_ncratio_6h.svg")
plt.show()


#%% Further functions

def adjust_color(color, hue = -0.2):
    rgb = list(i*255 for i in to_rgb(color))
    hsv = list(colorsys.rgb_to_hsv(*rgb))
    if hsv[0] + hue > 1:
        hsv[0] =  hsv[0] + hue - 1
    elif hsv[0] + hue < 0:
        hsv[0] = 1 + hsv[0] + hue
    else:
        hsv[0] = hsv[0] + hue
    rgb = [i/255 for i in hsv_to_rgb(hsv)]
    return to_hex(rgb)



def organize_data_comp_nuc(starv, strain, plot_data1, plot_data2, x_data_name, y_data_name, my_colors):
    x_data1 = plot_data1[strain + '_' + starv][x_data_name]
    x_data2 = plot_data2[strain + '_' + starv][x_data_name]
    x_data = np.asarray(x_data1 + x_data2)

    y_data1 = plot_data1[strain + '_' + starv][y_data_name]
    y_data2 = plot_data2[strain + '_' + starv][y_data_name]
    y_data = np.asarray(y_data1 + y_data2)

    legends = ['single origin', 'dual origin']

    color1 = [my_colors[0]]*len(x_data1)
    color2 = [my_colors[1]]*len(x_data2)
    color = color1 + color2

    # Randomize order
    indices = list(range(x_data.shape[0]))
    np.random.shuffle(indices)
    x_data = x_data[indices]
    y_data = y_data[indices]
    color = np.asarray(color)[indices]
    return x_data, y_data, color, legends

def organize_data_comp_starv(starv, plot_data, x_data_name, y_data_name, my_colors):
    legends = []

    x_data1 = list(plot_data['wt_' + starv][x_data_name])
    x_data2 = list(plot_data['dpolyp_' + starv][x_data_name])
    x_data = np.asarray(x_data1 + x_data2)

    y_data1 = list(plot_data['wt_' + starv][y_data_name])
    y_data2 = list(plot_data['dpolyp_' + starv][y_data_name])
    y_data = np.asarray(y_data1 + y_data2)

    legends = ['WT\u2009', 'ΔpolyP\u2009']

    color1 = [my_colors[0]]*len(x_data1)
    color2 = [my_colors[1]]*len(x_data2)
    color = color1 + color2

    indices = list(range(x_data.shape[0]))
    np.random.shuffle(indices)

    x_data = x_data[indices]
    y_data = y_data[indices]
    color = np.asarray(color)[indices]
    return x_data, y_data, color, legends

def organize_data_comp_strain(strain, plot_data, x_data_name, y_data_name, my_colors):
    legends = []

    # Do this to randomize zorder position of the points
    x_data1 = list(plot_data[strain + '_0h'][x_data_name])
    x_data2 = list(plot_data[strain + '_6h'][x_data_name])
    x_data = np.asarray(x_data1 + x_data2)

    y_data1 = list(plot_data[strain + '_0h'][y_data_name])
    y_data2 = list(plot_data[strain + '_6h'][y_data_name])
    y_data = np.asarray(y_data1 + y_data2)

    legends = [strain + '_0h', strain + '_6h']

    color1 = [my_colors[0]]*len(x_data1)
    color2 = [my_colors[1]]*len(y_data2)
    color = color1 + color2

    indices = list(range(x_data.shape[0]))
    np.random.shuffle(indices)

    x_data = x_data[indices]
    y_data = y_data[indices]
    color = np.asarray(color)[indices]
    return x_data, y_data, color, legends


def fake_legends(settings, name):
    plt.scatter(-100, -100, label=name, **settings)

def write_n_values(strain, plot_data, x_data_name, y_data_name):
    for i, starv in enumerate(["0h", "6h"]):
        x_data1 = list(plot_data[strain + '_' + starv][x_data_name])
        y_data1 = list(plot_data[strain + '_' + starv][y_data_name])
        print_n = [True for x, y in zip(x_data1, y_data1) if (x > 0 and y > 0)]
        print(f"{strain} {starv}: n = {len(print_n)}")
        ax = plt.gca()
        ax.text(1.1, 1 - 0.1*i, f"{strain} {starv}: n = {len(print_n)}",
                 verticalalignment='top', horizontalalignment='left',
                 transform=ax.transAxes)

def write_n_values_starv(starv, plot_data, x_data_name, y_data_name):
    for i, strain in enumerate(["wt", "dpolyp"]):
        x_data1 = list(plot_data[strain + '_' + starv][x_data_name])
        y_data1 = list(plot_data[strain + '_' + starv][y_data_name])
        print_n = [True for x, y in zip(x_data1, y_data1) if (x > 0 and y > 0)]
        print(f"{strain} {starv}: n = {len(print_n)}")
        ax = plt.gca()
        ax.text(1.1, 1 - 0.1*i, f"{strain} {starv}: n = {len(print_n)}",
                 verticalalignment='top', horizontalalignment='left',
                 transform=ax.transAxes)


def write_n_values_nuc(starv, strain, plot_data1, plot_data2, x_data_name, y_data_name):
    for i, plot_data in enumerate([plot_data1, plot_data2]):
        x_data1 = list(plot_data[strain + '_' + starv][x_data_name])
        y_data1 = list(plot_data[strain + '_' + starv][y_data_name])
        print_n = [True for x, y in zip(x_data1, y_data1) if (x > 0 and y > 0)]
        nuc = 'single' if i == 0 else 'dual'
        print(f"{strain} {starv} {nuc}: n = {len(print_n)}")
        ax = plt.gca()
        ax.text(1.1, 1 - 0.1*i, f"{strain} {starv} {nuc}: n = {len(print_n)}",
                 verticalalignment='top', horizontalalignment='left',
                 transform=ax.transAxes)

from mpl_toolkits.axes_grid1 import Divider, Size


def axes_fixed_size(size_x, size_y):
    fig = plt.figure(figsize=(size_x+1, size_y+1))

    h = [Size.Fixed(.5), Size.Fixed(size_x)]
    v = [Size.Fixed(.5), Size.Fixed(size_y)]

    divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
    # The width and height of the rectangle are ignored.

    ax = fig.add_axes(divider.get_position(),
                      axes_locator=divider.new_locator(nx=1, ny=1))

    return ax



# %% Link to Google Drive Doc
from io import BytesIO

import requests
import pandas as pd


url = "https://docs.google.com/spreadsheets/d/12GUGs8CJnQxjVk--1VxISSNt-7BPlNtbwUUcyuVPGoE/edit#gid=773212145"
url =  url.replace('/edit#gid=', '/export?format=csv&gid=')

r = requests.get(url)
data = r.content
df = pd.read_csv(BytesIO(data), lineterminator='\n')



HU_wt_0h = copy.deepcopy(df[(df["Strain Time"] == "WT 0h") & (df["day\r"] != 201215)])
HU_wt_6h = copy.deepcopy(df[(df["Strain Time"] == "WT 6h") & (df["day\r"] != 201215)])
HU_dpolyp_0h = copy.deepcopy(df[(df["Strain Time"] == "dpolyP 0h") & (df["day\r"] != 201215)])
HU_dpolyp_6h = copy.deepcopy(df[(df["Strain Time"] == "dpolyP 6h") & (df["day\r"] != 201215)])
HU_data = {'wt_0h': HU_wt_0h,
           'dpolyP_0h': copy.deepcopy(HU_dpolyp_0h),
           'dpolyp_0h': HU_dpolyp_0h,
           'wt_6h': HU_wt_6h,
           'dpolyP_6h': HU_dpolyp_6h,
           'dpolyp_6h': copy.deepcopy(HU_dpolyp_6h)}


#%% Plot the HU data into the other means Figure S7H

ft = 'NC ratio median values'
plt.figure(ft, figsize=(6, 5))
plt.title(ft)

plot_data = copy.deepcopy(single_ori_data)
for key, value in plot_data.items():
    print(key)
    strain = re.findall(r'[a-z]*', key)[0]
    strain = "bLR1" if strain == "wt" else "bLR2"
    starvation_time = re.findall(r'\dh', key)[0]
    condition = 'lowN' if starvation_time == '6h' else ''
    print(positions[strain][starvation_time + condition])
    data = copy.deepcopy(value['nc_ratio'])
    data = [x for x in data if x > 0]
    plt.errorbar(positions[strain][starvation_time + condition]-0.03,
                 np.nanmedian(data), xerr=0, yerr=(np.std(data) /
                                                                np.sqrt(len(data))),
                 fmt='o', markersize=ms,
                 markeredgecolor='none', ecolor=colors[strain][starvation_time + condition],
                 color=colors[strain][starvation_time + condition], alpha=1,
                 markeredgewidth=2, capsize=10)


dual_data = dual_ori_data
for key, value in plot_data.items():
    print(key)
    strain = re.findall(r'[a-z]*', key)[0]
    strain = "bLR1" if strain == "wt" else "bLR2"
    starvation_time = re.findall(r'\dh', key)[0]
    condition = 'lowN' if starvation_time == '6h' else ''
    print(positions[strain][starvation_time + condition])
    all_data = copy.deepcopy(value['nc_ratio'])
    all_data.extend(dual_data[key]['nc_ratio'])
    all_data = [x for x in all_data if x > 0]
    plt.errorbar(positions[strain][starvation_time + condition],
                 np.nanmedian(all_data), xerr=0, yerr=(np.std(all_data) /
                                                       np.sqrt(len(all_data))),
                 fmt='o', markersize=ms,
                 markeredgecolor=colors[strain][starvation_time + condition],
                 ecolor=colors[strain][starvation_time + condition],
                 color='none', alpha=1,
                 markeredgewidth=2, capsize=10)

#HU_mcherry data
for key, value in HU_data.items():
    strain = re.findall(r'[a-z]*', key)[0]
    strain = "bLR1" if strain == "wt" else "bLR2"
    starvation_time = re.findall(r'\dh', key)[0]
    condition = 'lowN' if starvation_time == '6h' else ''
    print(key)
    print(np.nanmedian(value['nc_ratio']))
    plt.errorbar(positions[strain][starvation_time + condition]+0.03,
                 np.nanmedian(value['nc_ratio']), xerr=0, yerr=(np.nanstd(value['nc_ratio']) /
                                                                np.sqrt(len(value['nc_ratio']))),
                 fmt='^', markersize=ms,
                 markeredgecolor='none', ecolor=colors[strain][starvation_time + condition],
                 color=colors[strain][starvation_time + condition], alpha=1,
                 markeredgewidth=2, capsize=10)


plt.xticks([1, 1.2, 1.4, 1.6], ['WT\n0h', '\n6h lowN', 'ΔpolyP\n0h', '\n6h lowN'])
plt.ylabel("NC ratio")
plt.savefig(figure_folder + "/single_origin/" "all_medians_incl_HU.svg")
plt.show()


#%% Plot the HU data AREAS into the other means Figure S7H

ft = 'Nucleoid median values'
plt.figure(ft, figsize=(6, 5))
plt.title(ft)

plot_data = copy.deepcopy(single_ori_data)
for key, value in plot_data.items():
    print(key)
    strain = re.findall(r'[a-z]*', key)[0]
    strain = "bLR1" if strain == "wt" else "bLR2"
    starvation_time = re.findall(r'\dh', key)[0]
    condition = 'lowN' if starvation_time == '6h' else ''
    print(positions[strain][starvation_time + condition])
    data = copy.deepcopy(value['nuc_area'])
    data = [x/86.6 for x in data if x > 0]
    plt.errorbar(positions[strain][starvation_time + condition]-0.03,
                 np.nanmedian(data), xerr=0, yerr=(np.std(data) /
                                                                np.sqrt(len(data))),
                 fmt='o', markersize=ms,
                 markeredgecolor='none', ecolor=colors[strain][starvation_time + condition],
                 color=colors[strain][starvation_time + condition], alpha=1,
                 markeredgewidth=2, capsize=10)


dual_data = dual_ori_data
for key, value in plot_data.items():
    print(key)
    strain = re.findall(r'[a-z]*', key)[0]
    strain = "bLR1" if strain == "wt" else "bLR2"
    starvation_time = re.findall(r'\dh', key)[0]
    condition = 'lowN' if starvation_time == '6h' else ''
    all_data = copy.deepcopy(value['nuc_area'])
    all_data.extend(dual_data[key]['nuc_area'])
    all_data = [x/86.6 for x in all_data if x > 0]
    print(np.nanmedian(all_data))
    plt.errorbar(positions[strain][starvation_time + condition],
                 np.nanmedian(all_data), xerr=0, yerr=(np.std(all_data) /
                                                       np.sqrt(len(all_data))),
                 fmt='o', markersize=ms,
                 markeredgecolor=colors[strain][starvation_time + condition],
                 ecolor=colors[strain][starvation_time + condition],
                 color='none', alpha=1,
                 markeredgewidth=2, capsize=10)

#HU_mcherry data
for key, value in HU_data.items():
    strain = re.findall(r'[a-z]*', key)[0]
    strain = "bLR1" if strain == "wt" else "bLR2"
    starvation_time = re.findall(r'\dh', key)[0]
    condition = 'lowN' if starvation_time == '6h' else ''
    print(key)
    data = [x/86.6 for x in value['nucleoid_area']]
    print(np.nanmedian(data))
    plt.errorbar(positions[strain][starvation_time + condition]+0.03,
                 np.nanmedian(data), xerr=0, yerr=(np.nanstd(data) /
                                                                np.sqrt(len(data))),
                 fmt='^', markersize=ms,
                 markeredgecolor='none', ecolor=colors[strain][starvation_time + condition],
                 color=colors[strain][starvation_time + condition], alpha=1,
                 markeredgewidth=2, capsize=10)


plt.xticks([1, 1.2, 1.4, 1.6], ['WT\n0h', '\n6h lowN', 'ΔpolyP\n0h', '\n6h lowN'])
plt.ylabel("nucleoid area (μm²)")
plt.savefig(figure_folder + "/single_origin/" "all_medians_nuc_areas_incl_HU.svg")
plt.show()


# df = pd.read_csv(url, lineterminator='\n', error_bad_lines=False)
# %%
