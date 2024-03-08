# %%
# from collect_results_cleaned import combine_particle_results
import read_and_plot_oufti_output_cleaned as oufti
from load_all_results import load_all_results
import center_profile
import matplotlib.pyplot as plt
from scipy import spatial
import numpy as np
import shapely
import pdb
import csv
import json
import os
import glob
import re
from multiprocessing import Pool
import process_spots
import collect_results
from scipy import stats
from collections import defaultdict
import copy
# %matplotlib qt

# Reset Matplotlib to default style

with open('general_info.json') as jf:
    data = json.load(jf)
figure_folder = data['figure_directory']

"""
Open the .mat file in Oufti
Export as csv
"""
# with open('segmentation_general_info.json') as jf:
#     data = json.load(jf)
# %%
days = ["210323"]
spot_type = "fixed_muNS"
# traj_spec = ['filtered_trajectories_all_renamed']
traj_spec = ['filtered_trajectories_all_with_magnitude']
# imsd_spec = ['imsds_all_renamed']
startvation_times = ["0h", "6h"]

# region Gaussian fits (does not have to be redone)
# %% Fit gaussians to get the magnitudes for these files, takes some time!
with open('general_info.json') as jf:
    data = json.load(jf)
data_folder = data['basic_directory']
traj_filename = 'filtered_trajectories_all_renamed'
folders = [
           {'folder': (data_folder +
                       '210323\\fixed_muNS\\0h\\210323_bLR31_0h_210ms_fixed_tc_011\\'),
            'day': ['210323'],
            'spot_type': 'fixed_muNS',
            'starvation_time': '0h'},
           {'folder': (data_folder +
                       '210323\\fixed_muNS\\0h\\210323_bLR32_0h_210ms_fixed_tc_006\\'),
            'day': ['210323'],
            'spot_type': 'fixed_muNS',
            'starvation_time': '0h'},
           {'folder': (data_folder +
                       '210323\\fixed_muNS\\0h\\210323_bLR31_0h_210ms_fixed_tc_005\\'),
            'day': ['210323'],
            'spot_type': 'fixed_muNS',
            'starvation_time': '0h'},

           {'folder': data_folder + '210323\\muNS\\0h\\210323_bLR31_0h_210ms_tc_016\\',
            'day': ['210323'],
            'spot_type': 'muNS',
            'starvation_time': '0h'},
           {'folder': data_folder + '210323\\muNS\\0h\\210323_bLR32_0h_210ms_tc_003\\',
            'day': ['210323'],
            'spot_type': 'muNS',
            'starvation_time': '0h'},

           {'folder': data_folder + '210323\\muNS\\6h\\210323_bLR31_6hlowN_210ms_tc_6001\\',
            'day': ['210323'],
            'spot_type': 'muNS',
            'starvation_time': '6h'},
           {'folder': data_folder + '210323\\muNS\\6h\\210323_bLR32_6hlowN_210ms_tc_6002\\',
            'day': ['210323'],
            'spot_type': 'muNS',
            'starvation_time': '6h'},
           ]

pool_data = []
analyse_folders = []
all_folders = glob.glob(os.path.join(data_folder, '210323', '**', '*/'), recursive=True)
# %% Actually do all the fits

pattern = re.compile(r".*\\(210323_.*tc_\d{1,4})\\.*")
for folder in folders:
    print("FOLDER:", folder['folder'])
    all_folders.remove(folder['folder'])
    avoid_folds = [pattern.search(fold) for fold in all_folders
                   if pattern.search(fold) is not None]
    avoid_set = list(set([my_pattern.group(1) for my_pattern in avoid_folds]))
    avoid_set.remove(pattern.search(folder['folder']).group(1))
    pool_data.append([folder['day'], folder['spot_type'], 'filtered_trajectories_all_renamed',
                      [folder['starvation_time']],
                      avoid_set])
    all_folders.extend([folder['folder']])

with Pool(5) as p:
    p.starmap(process_spots.fit_spots_all_trajectories, pool_data)
# endregion

# %% Segmentation results and construct cells
seg_data = oufti.load_segmentation_results(days, "localized_metrics")
cells = center_profile.main(seg_data[0], profile=False)
# cells = cells['210323_fixed_muNS']
# with open(cell_file, 'wb') as f:
#      pickle.dump(cells, f)

# %% Trajectories and particle information
trajs_mobile = load_all_results("muNS", traj_spec, days=days, starvation_times=startvation_times)
results_mobile = load_all_results("muNS", ['results_alpha_Dapp'], avoid=['_0ms', '_30ms',
                                                                         '_002', '_016',
                                                                         '90ms'])
# all_mobile = collect_results.combine_particle_results(trajs_mobile, results_mobile, spot_type,
#                                                       transfer_columns=['x', 'y'])
all_mobile = load_all_results("muNS", ['combined_results'], avoid=['_0ms', '_30ms',
                                                                   '_002', '90ms'])


trajs_fixed = load_all_results("fixed_muNS", traj_spec, days=days,
                               starvation_times=startvation_times)
results_fixed = load_all_results("fixed_muNS", ['results_alpha_Dapp'])
all_fixed = collect_results.combine_particle_results(trajs_fixed, results_fixed, spot_type,
                                                     transfer_columns=['x', 'y'])
# trajs = trajs[list(trajs.keys())[0]]  # We only have the first file so far
# trajs = calculate_intensity_magnitude(spot_type, trajs)
# trajs = append_starting_fit_values_to_traj(spot_type, trajs)
particles_fixed = {}
for movie in trajs_fixed.keys():
    particles_fixed[movie] = list(set(trajs_fixed[movie]['particle']))
particles_mobile = {}
for movie in trajs_mobile.keys():
    particles_mobile[movie] = list(set(trajs_mobile[movie]['particle']))


# %% Connect trajectories to cells
for movie in cells.keys():
    try:
        trajs_here = trajs_fixed[movie]
        all_here = all_fixed[movie]
        particles_here = particles_fixed[movie]
    except KeyError:
        trajs_here = trajs_mobile[movie]
        all_here = all_mobile[movie]
        particles_here = particles_mobile[movie]
    for particle in particles_here:
        # Find the corresponding cell
        data = trajs_here[trajs_here['particle'] == particle]
        D_app_data = all_here[all_here.index == particle]
        particle_pos = center_profile.geometry.Point(data.iloc[0]['x'], data.iloc[0]['y'])
        for cell in cells[movie]:
            if cell.contains(particle_pos):
                cell.trajectory = data
                cell.D_app = D_app_data
                # traj_start = cell.centerline.project(particle_pos)
                start_pos = center_profile.geometry.Point(cell.trajectory.iloc[0].x,
                                                          cell.trajectory.iloc[0].y)
                cell.traj_start = shapely.ops.nearest_points(cell.centerline, start_pos)[0]
                mean_pos = center_profile.geometry.Point(np.mean(cell.trajectory.x),
                                                         np.mean(cell.trajectory.y))
                cell.traj_mean = shapely.ops.nearest_points(cell.centerline, mean_pos)[0]
                cell_center = cell.centerline.centroid
                trajectory = center_profile.geometry.LineString(np.array(list(zip(cell.trajectory.x,
                                                                                  cell.trajectory.y
                                                                                  ))))
                cell.traj_center = shapely.ops.nearest_points(trajectory, cell_center)[0]
                cell.all_pos = []
                for x, y in zip(cell.trajectory.x, cell.trajectory.y):
                    pos = center_profile.geometry.Point(x, y)
                    centerline_pos = shapely.ops.nearest_points(cell.centerline, pos)[0]
                    cell.all_pos.append(centerline_pos)
                # cell.traj_start = cell.centerline.interpolate(cell.centerline.interpolate.
                #                   project(particle_pos))
                break

# %% Plot both
for movie in cells.keys():
    pass
movie = '210323_bLR31_6hlowN_210ms_tc_6001/'
print(movie)
plt.figure()
for cell in cells[movie]:
    plt.plot(*cell.exterior.xy)
    try:
        plt.scatter(cell.trajectory.iloc[0].x, cell.trajectory.iloc[0].y)
        plt.plot(*cell.centerline.coords.xy)
        # plt.scatter(cell.traj_start.x, cell.traj_start.y)
    except AttributeError:
        pass
plt.gca().axis('equal')
plt.show()

# %% Get the profile data
profiles = defaultdict(lambda: defaultdict(list))
profile_files = ["WT_0h", "WT_6h_lowN", "DpolyP_0h", "DpolyP_6h_lowN"]
for profile_file in profile_files:
    y_data_norm = []
    with open('data/' + profile_file + '_HUmCherry.csv', newline='') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            # row_list = row[4][1:-1]  # .split(',')
            row_list = list(map(float, row[4][1:-1].split(',')))
            y_data_norm.append(row_list)
    x_profile = list(map(float, row[3][1:-1].split(',')))
    y_profile = np.array(y_data_norm)
    profiles[profile_file]["x_data"] = x_profile
    profiles[profile_file]['profile'] = np.mean(y_profile, axis=0)
profile = profiles["WT_0h"]


# %% Calculate the STD deviation of the half profile point
profiles = defaultdict(lambda: defaultdict(list))
profile_files = ["WT_0h", "WT_6h_lowN", "DpolyP_0h", "DpolyP_6h_lowN"]
for profile_file in profile_files:
    y_data_norm = []
    with open('data/' + profile_file + '_HUmCherry.csv', newline='') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            # row_list = row[4][1:-1]  # .split(',')
            row_list = list(map(float, row[4][1:-1].split(',')))
            y_data_norm.append(row_list)
    x_profile = list(map(float, row[3][1:-1].split(',')))
    y_profile = np.array(y_data_norm)
    half_points = []
    for profile in y_profile:
        x_fine = np.linspace(x_profile[0], x_profile[-1], 100)
        y_fine = np.interp(x_fine, x_profile, profile)
        half_points.append(x_fine[y_fine > y_fine.max()/2][0])
    print(profile_file)
    print(np.round(np.std(half_points)*1000)/1000)

# %% Get distance to cell pole
def distance_along_centerline(centerline, point):
    start = centerline.coords[0]
    end = centerline.coords[-1]
    # traj_start = (cell.trajectory.iloc[0].x, cell.trajectory.iloc[0].y)
    traj_start = (point.x, point.y)
    start_dis = spatial.distance.euclidean(start, traj_start)
    end_dis = spatial.distance.euclidean(end, traj_start)
    full_dis = spatial.distance.euclidean(start, end)
    return min([start_dis, end_dis])/full_dis


# region Mobile vs Fixed
# %%
distances = {"fixed": {"mean": [], "start": [], "center": [], "all": []},
             "mobile": {"mean": [], "start": [], "center": [], "all": []}}
mag = {"fixed": {"mean": [], "start": [], "center": [], "all": []},
       "mobile": {"mean": [], "start": [], "center": [], "all": []}}
D_app = {"fixed": {"mean": [], "start": [], "center": [], "all": []},
         "mobile": {"mean": [], "start": [], "center": [], "all": []}}
snr = {"fixed": {"mean": [], "start": [], "center": [], "all": []},
       "mobile": {"mean": [], "start": [], "center": [], "all": []}}
for movie in cells.keys():
    print(movie)
    if "fixed" in movie:
        key = "fixed"
    else:
        key = "mobile"
    for cell in cells[movie]:
        try:
            for pos_type, traj_point in zip(['mean', 'start', 'center', 'all'],
                                            [cell.traj_mean, cell.traj_start, cell.traj_center,
                                             None]):
                if pos_type != 'all':
                    dis = distance_along_centerline(cell.centerline, traj_point)
                    distances[key][pos_type].append(dis)
                    snr[key][pos_type].append(cell.trajectory.iloc[0].snr)
                    D_app[key][pos_type].append(cell.D_app['D_app'].iloc[0])
                    mag[key][pos_type].append(np.mean(cell.trajectory['magnitude'][:4])**(1./3))
                else:
                    for pos in cell.all_pos:
                        dis = distance_along_centerline(cell.centerline, pos)
                        distances[key][pos_type].append(dis)
                    magn = np.mean(cell.trajectory['magnitude'][:4])**(1./3)
                    mag[key][pos_type].extend(np.ones(len(cell.all_pos)) * magn)
                    D_app_here = cell.D_app['D_app'].iloc[0]
                    D_app[key][pos_type].extend(np.ones(len(cell.all_pos)) * D_app_here)
                    snr_here = cell.trajectory.iloc[0].snr
                    snr[key][pos_type].extend(np.ones(len(cell.all_pos)) * snr_here)
        except AttributeError as e:
            pass
y_data = {'D_app': D_app, "mag": mag, "snr": snr}

# %%
key = "mobile"
pos_type = "mean"
y_value = "D_app"

resolution = 50
if y_value == "D_app":
    min_y, max_y, profile_factor = (0, 0.005, 0.05)
elif y_value == "mag":
    min_y, max_y, profile_factor = (5, 20, 200)
x_bins = np.linspace(0, 0.5, resolution)
y_bins = np.linspace(min_y, max_y, resolution)

X, Y = np.meshgrid(x_bins, y_bins, indexing='ij')
heatmap, xedges, yedges = np.histogram2d(distances[key][pos_type], y_data[y_value][key][pos_type],
                                         bins=[x_bins, y_bins])
print(key, pos_type)
print("N = ", len(distances[key][pos_type]))

plt.pcolormesh(X, Y, heatmap)
plt.xlabel("Normalized Distance to Pole along Centerline")
plt.ylabel(y_value)



# %% Check the profile in magnitudes for the two halfs of the profile
mag_pole = []
mag_center = []
for dis, mag in zip(distances['mobile']['center'], y_data['mag']['mobile']['center']):
    if mag == mag:  # check for nan
        if dis < 0.25:
            mag_pole.append(mag)
        else:
            mag_center.append(mag)

print("Pole median: ", np.nanmedian(mag_pole))
print("Center median: ", np.nanmedian(mag_center))

fig, axs = plt.subplots(2, 1)
axs[0].hist(mag_pole, bins=100)
axs[0].set_title("Pole", fontweight='bold')
axs[0].set_xlim(0, 30)
axs[0].set_xticklabels([])
axs[1].hist(mag_center, bins=100)
axs[1].set_title("Center", fontweight='bold')
axs[1].set_xlabel("Magnitude")
axs[1].set_xlim(0, 30)

_, p_value = stats.ttest_ind(mag_pole, mag_center)
print("P: ", p_value)
# endregion

# region WT 6h vs dpolyp 6h
# %% Do the separation into wt_6h and dpolyp_6h
grouped_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for movie in cells.keys():
    print(movie)
    if 'fixed' in movie.lower():
        continue
    strain = 'wt' if 'blr31' in movie.lower() else 'dpolyp'
    starv = '6h' if '6h' in movie else '0h'
    group = strain + '_' + starv
    print(group)
    for cell in cells[movie]:
        try:
            for pos_type, traj_point in zip(['mean', 'start', 'center', 'all'],
                                            [cell.traj_mean, cell.traj_start, cell.traj_center,
                                                None]):
                if pos_type != 'all':
                    dis = distance_along_centerline(cell.centerline, traj_point)
                    grouped_data[group]['distances'][pos_type].append(dis)
                    grouped_data[group]['snr'][pos_type].append(cell.trajectory.iloc[0].snr)
                    grouped_data[group]['D_app'][pos_type].append(cell.D_app['D_app'].iloc[0])
                    grouped_data[group]['mag'][pos_type].append(
                        np.mean(cell.trajectory['magnitude'][:4])**(1./3))
                else:
                    for pos in cell.all_pos:
                        dis = distance_along_centerline(cell.centerline, pos)
                        grouped_data[group]['distances'][pos_type].append(dis)
                    magn = np.mean(cell.trajectory['magnitude'][:4])**(1./3)
                    grouped_data[group]['mag'][pos_type].extend(np.ones(len(cell.all_pos)) * magn)
                    D_app_here = cell.D_app['D_app'].iloc[0]
                    grouped_data[group]['D_app'][pos_type].extend(np.ones(len(cell.all_pos)) *
                                                                  D_app_here)
                    snr_here = cell.trajectory.iloc[0].snr
                    grouped_data[group]['snr'][pos_type].extend(np.ones(len(cell.all_pos)) *
                                                                snr_here)
        except AttributeError as e:
            pass

# %%
key = "dpolyp_6h"
pos_type = "center"
y_value = "mag"

resolution = 50
if y_value == "D_app":
    min_y, max_y, profile_factor = (0, 0.005, 0.05)
elif y_value == "mag":
    min_y, max_y, profile_factor = (5, 20, 200)
x_bins = np.linspace(0, 0.5, resolution)
y_bins = np.linspace(min_y, max_y, resolution)

X, Y = np.meshgrid(x_bins, y_bins, indexing='ij')
heatmap, xedges, yedges = np.histogram2d(grouped_data[key]['distances'][pos_type],
                                         grouped_data[key][y_value][pos_type],
                                         bins=[x_bins, y_bins])
print(key, pos_type)
print("N = ", len(grouped_data[key]['distances'][pos_type]))

plt.pcolormesh(X, Y, heatmap)
plt.xlabel("Normalized Distance to Pole along Centerline")
plt.ylabel(y_value)

half_profile = int(len(x_profile)/2)+1
plt.plot(x_profile[:half_profile], profile['profile'][:half_profile]*profile_factor + min_y,
         linewidth=3, color='w')

# %% Check the profile in magnitudes for the two halfs of the profile
mag_pole = []
mag_center = []
for dis, mag in zip(grouped_data[key]['distances'][pos_type], grouped_data[key]['mag'][pos_type]):
    if mag == mag:  # check for nan
        if dis < 0.25:
            mag_pole.append(mag)
        else:
            mag_center.append(mag)

print("Pole median: ", np.nanmedian(mag_pole), "N = ", np.sum(~np.isnan(mag_pole)))
print("Center median: ", np.nanmedian(mag_center), "N = ", np.sum(~np.isnan(mag_center)))

fig, axs = plt.subplots(2, 1)
axs[0].hist(mag_pole, bins=100)
axs[0].set_title("Pole", fontweight='bold')
axs[0].set_xlim(0, 30)
axs[0].set_xticklabels([])
axs[1].hist(mag_center, bins=100)
axs[1].set_title("Center", fontweight='bold')
axs[1].set_xlabel("Magnitude")
axs[1].set_xlim(0, 30)

_, p_value = stats.ttest_ind(mag_pole, mag_center)
print("P: ", p_value)
# endregion

# region  ALL 0h vs ALL 6h
# %%
grouped_data_starv = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for movie in cells.keys():
    print(movie)
    if 'fixed' in movie.lower():
        print('skipped')
        continue
    starv = '6h' if '6h' in movie else '0h'
    group = starv
    print(group)
    for cell in cells[movie]:
        try:
            for pos_type, traj_point in zip(['mean', 'start', 'center', 'all'],
                                            [cell.traj_mean, cell.traj_start, cell.traj_center,
                                                None]):
                if pos_type != 'all':
                    dis = distance_along_centerline(cell.centerline, traj_point)
                    grouped_data_starv[group]['distances'][pos_type].append(dis)
                    grouped_data_starv[group]['snr'][pos_type].append(cell.trajectory.iloc[0].snr)
                    grouped_data_starv[group]['D_app'][pos_type].append(cell.D_app['D_app'].iloc[0])
                    grouped_data_starv[group]['mag'][pos_type].append(
                        np.mean(cell.trajectory['magnitude'][:4])**(1./3))
                else:
                    for pos in cell.all_pos:
                        dis = distance_along_centerline(cell.centerline, pos)
                        grouped_data_starv[group]['distances'][pos_type].append(dis)
                    magn = np.mean(cell.trajectory['magnitude'][:4])**(1./3)
                    grouped_data_starv[group]['mag'][pos_type].extend(
                        np.ones(len(cell.all_pos)) * magn)
                    D_app_here = cell.D_app['D_app'].iloc[0]
                    grouped_data_starv[group]['D_app'][pos_type].extend(np.ones(len(cell.all_pos)) *
                                                                        D_app_here)
                    snr_here = cell.trajectory.iloc[0].snr
                    grouped_data_starv[group]['snr'][pos_type].extend(np.ones(len(cell.all_pos)) *
                                                                      snr_here)
        except AttributeError as e:
            pass

# %% PlOT
key = "6h"
pos_type = "center"
y_value = "mag"

resolution = 50
if y_value == "D_app":
    min_y, max_y, profile_factor = (0, 0.005, 0.05)
elif y_value == "mag":
    min_y, max_y, profile_factor = (5, 20, 200)
x_bins = np.linspace(0, 0.5, resolution)
y_bins = np.linspace(min_y, max_y, resolution)

X, Y = np.meshgrid(x_bins, y_bins, indexing='ij')
heatmap, xedges, yedges = np.histogram2d(grouped_data_starv[key]['distances'][pos_type],
                                         grouped_data_starv[key][y_value][pos_type],
                                         bins=[x_bins, y_bins])
print(key, pos_type)
print("N = ", len(grouped_data_starv[key]['distances'][pos_type]))

plt.pcolormesh(X, Y, heatmap)
plt.xlabel("Normalized Distance to Pole along Centerline")
plt.ylabel(y_value)

half_profile = int(len(x_profile)/2) + 1
plt.plot(x_profile[:half_profile], profile['profile'][:half_profile]*profile_factor + min_y, linewidth=3,
         color='w')

# %% Check the profile in magnitudes for the two halfs of the profile
mag_pole = []
mag_center = []
for dis, mag in zip(grouped_data_starv[key]['distances'][pos_type],
                    grouped_data_starv[key]['mag'][pos_type]):
    if mag == mag:  # check for nan
        if dis < 0.25:
            mag_pole.append(mag)
        else:
            mag_center.append(mag)

print("Pole median: ", np.nanmedian(mag_pole), "N = ", np.sum(~np.isnan(mag_pole)))
print("Center median: ", np.nanmedian(mag_center), "N = ", np.sum(~np.isnan(mag_center)))

fig, axs = plt.subplots(2, 1)
axs[0].hist(mag_pole, bins=100)
axs[0].set_title("Pole", fontweight='bold')
axs[0].set_xlim(0, 30)
axs[0].set_xticklabels([])
axs[1].hist(mag_center, bins=100)
axs[1].set_title("Center", fontweight='bold')
axs[1].set_xlabel("Magnitude")
axs[1].set_xlim(0, 30)

_, p_value = stats.ttest_ind(mag_pole, mag_center)
print("P: ", p_value)
# endregion

# region big particles
# %%
bins = np.linspace(7, 12, 4)
grouped_data_bins = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
for movie in cells.keys():
    print(movie)
    if 'fixed' in movie.lower():
        continue
    strain = 'wt' if 'blr31' in movie.lower() else 'dpolyp'
    starv = '6h' if '6h' in movie else '0h'
    group = strain + '_' + starv
    print(group)
    for cell in cells[movie]:
        try:
            particle_size = np.mean(cell.trajectory['magnitude'][:4])**(1./3)
            bin_cat = np.digitize(particle_size, bins)
            bin_cat = 'bin' + str(bin_cat-1)
            for pos_type, traj_point in zip(['mean', 'start', 'center', 'all'],
                                            [cell.traj_mean, cell.traj_start, cell.traj_center,
                                                None]):
                if pos_type != 'all':
                    dis = distance_along_centerline(cell.centerline, traj_point)
                    grouped_data_bins[group]['distances'][pos_type][bin_cat].append(dis)
                    grouped_data_bins[group]['snr'][pos_type][bin_cat].append(
                        cell.trajectory.iloc[0].snr)
                    grouped_data_bins[group]['D_app'][pos_type][bin_cat].append(
                        cell.D_app['D_app'].iloc[0])
                    grouped_data_bins[group]['mag'][pos_type][bin_cat].append(particle_size)
                else:
                    for pos in cell.all_pos:
                        dis = distance_along_centerline(cell.centerline, pos)
                        grouped_data_bins[group]['distances'][pos_type][bin_cat].append(dis)
                    magn = np.mean(cell.trajectory['magnitude'][:4])**(1./3)
                    grouped_data_bins[group]['mag'][pos_type][bin_cat].extend(
                        np.ones(len(cell.all_pos)) * magn)
                    D_app_here = cell.D_app['D_app'].iloc[0]
                    grouped_data_bins[group]['D_app'][pos_type][bin_cat].extend(
                        np.ones(len(cell.all_pos)) * D_app_here)
                    snr_here = cell.trajectory.iloc[0].snr
                    grouped_data_bins[group]['snr'][pos_type][bin_cat].extend(
                        np.ones(len(cell.all_pos)) * snr_here)
        except AttributeError as e:
            pass


# %% [markdown] ## Define Mosaic

# %%
mosaic = [['upper left', '.'],
          ['heat', '.'],
          ['heat', 'bin5'],
          ['heat', '.'],
          ['heat', 'bin3'],
          ['heat', '.'],
          ['heat', 'bin1'],
          ['heat', '.']]
gs_kw = dict(height_ratios=[2.5] + [1]*7, width_ratios=[2, 1], hspace=0.0)
# sp_kw = dict(hspace=0)
fig, axd = plt.subplot_mosaic(mosaic, gridspec_kw=gs_kw, constrained_layout=True, figsize=(12, 9))
# for idx in ['0', '1', '2', '3', '4']:
#     axd[idx].axis('off')
fig.subplots_adjust(hspace=0)
grouped_data_here = copy.deepcopy(grouped_data)
grouped_data_bins_here = copy.deepcopy(grouped_data_bins)

#
key = "dpolyp_0h"
pos_type = "all"
y_value = "mag"
profile_keys = {"dpolyp_0h": "DpolyP_0h", "dpolyp_6h": "DpolyP_6h_lowN", "wt_0h": "WT_0h",
                "wt_6h": "WT_6h_lowN"}
profile_key = profile_keys[key]

resolution = 50
if y_value == "D_app":
    min_y, max_y, profile_factor = (0, 0.005, 0.05)
elif y_value == "mag":
    min_y, max_y, profile_factor = (5, 20.1, 200)
x_bins = np.linspace(0, 0.5, resolution)
y_bins = np.linspace(min_y, max_y, resolution)

X, Y = np.meshgrid(x_bins, y_bins, indexing='ij')
heatmap, xedges, yedges = np.histogram2d(grouped_data_here[key]['distances'][pos_type],
                                         grouped_data_here[key][y_value][pos_type],
                                         bins=[x_bins, y_bins])
print(key, pos_type)
print("N = ", len(grouped_data_bins_here[key]['distances'][pos_type]))

# fig = plt.figure()

# gs = fig.add_gridspec(2, 1, hspace=0, height_ratios=[1, 3])
# axs = gs.subplots(sharex='col')

axs = [axd['upper left'], axd['heat']]
labels = {'mag': 'Magnitude [AU]'}

axs[1].pcolormesh(X, Y, heatmap)
axs[1].set_xlabel("Normalized Distance to Pole along Centerline")
axs[1].set_ylabel(labels[y_value])
bins = np.linspace(7, 14, 7)
# for y_line in bins[1:]:
#     axs[1].axhline(y_line, xmin=0.9, color='w')
# for y, bin_text in zip(bins[1::2], ['Bin 1', 'Bin 3', 'Bin 5']):
#     axs[1].text(0.45, y + (bins[1]-bins[0])/2, bin_text, va='center')


half_profile = int(len(profiles[profile_key]['x_data'])/2)+1
axs[1].plot(profiles[profile_key]['x_data'][:half_profile],
            profiles[profile_key]['profile'][:half_profile]*profile_factor + min_y, linewidth=3,
            color='w')

# Get where the position for where the profile is > 50% intensity/probability
cutoff = 0.5
profile_half = profiles[profile_key]['profile'][:half_profile]
norm_profile_half = np.divide((profile_half - min(profile_half)),
                              max(profile_half - min(profile_half)))
norm_profile_half = np.interp(np.linspace(0, 0.5, 100),
                              profiles[profile_key]['x_data'][:half_profile],
                              norm_profile_half)
x_interp = np.interp(np.linspace(0, 0.5, 100),
                     profiles[profile_key]['x_data'][:half_profile],
                     profiles[profile_key]['x_data'][:half_profile])
first_over = np.argmax(norm_profile_half > cutoff)
pole_cutoff = x_interp[first_over]
axs[1].axvline(pole_cutoff, color="#CC0066")

density, bins = np.histogram(grouped_data_here[key]['distances'][pos_type],
                             bins=np.linspace(-0.01, 0.5, 30))
density = density / density.sum()
widths = bins[:-1] - bins[1:]
axs[0].bar(bins[1:], density, width=widths)
axs[0].axvline(pole_cutoff, color='#CC0066')
axs[0].set_xlim(0, 0.5)
axs[1].set_xlim(0, 0.5)
axs[1].set_ylim(y_bins[0], y_bins[-1])
# axs[0].set_xticks([])
axs[0].set_yticks(axs[0].get_yticks()[1:])
axs[0].set_ylabel('Density')
# Percent values
data = np.asarray(grouped_data_here[key]['distances'][pos_type])
axs[0].text(.8, .8, 'N =' + str(len(data)),
            transform=axs[0].transAxes, fontsize=14)
axs[0].text(pole_cutoff/2, axs[0].get_ylim()[1]/3, str(round(len(data[data < pole_cutoff])/len(data)*100)) + '%',
            fontsize=18, fontweight='bold', ha='center')
axs[0].text(pole_cutoff + (0.5-pole_cutoff)/2, axs[0].get_ylim()[1]/3, str(round(len(data[data >= pole_cutoff])/len(data)*100)) + '%',
            fontsize=18, fontweight='bold', ha='center')
axs[0].set_xticks([])
plt.savefig(os.path.join(figure_folder, 'localized_metrics', key + '.svg'))

# VIOLIN PLOTS


distances = grouped_data_here[key]['distances'][pos_type]
all_data = grouped_data_here[key][y_value][pos_type]

out_data = [x for x, y in zip(all_data, distances) if y <= pole_cutoff and not np.isnan(x)]
in_data = [x for x, y in zip(all_data, distances) if y > pole_cutoff and not np.isnan(x)]

scale_out = len(out_data)/(len(out_data) + len(in_data))
scale_in = len(in_data)/(len(out_data) + len(in_data))

if scale_out > scale_in:
    scale_in = 1/scale_out*scale_in
    scale_out = 1
else:
    scale_out = 1/scale_in*scale_out
    scale_in = 1
print(scale_out, scale_in)
# DISABLE SCALING
# scale_in = 1
# scale_out = 1

fig, ax = plt.subplots(figsize=(5, 5))

v1 = ax.violinplot(out_data, points=100, positions = [0],
                   showmeans=False, showextrema=False, showmedians=False)
for b in v1['bodies']:
    # get the center
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    # modify the paths to not go further right than the center
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
    b.get_paths()[0].vertices[:, 0] = b.get_paths()[0].vertices[:, 0]*scale_out
    b.set_color('r')
    b.set_alpha(.8)

v2 = ax.violinplot(in_data, points=100, positions=[0],
                   showmeans=False, showextrema=False, showmedians=False)

for b in v2['bodies']:
    # get the center
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    # modify the paths to not go further left than the center
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
    b.get_paths()[0].vertices[:, 0] = b.get_paths()[0].vertices[:, 0]*scale_in
    b.set_alpha(.8)
    b.set_color('b')

#statistics
# statistics, p_value = stats.kstest(out_data, in_data)
# ax.text(0.15, 9, f"kstest p-value: \n {p_value}")
# statistics, p_value = stats.mannwhitneyu(out_data, in_data)
# ax.text(0.15, 7, f"mannwithney p-value: \n {p_value}")

ax.legend([v1['bodies'][0],v2['bodies'][0]],['outside', 'inside'])
ax.set_xlim(ax.get_xlim()[0], -ax.get_xlim()[0])
ax.set_ylim(min_y, max_y)
ax.set_ylabel("Magnitude [AU]")
# ax.set_xlabel("Density")
ax.set_xticks([])
# ax.set_xticklabels([abs(float(x.get_text().replace("−", "-"))) for x in list(ax.get_xticklabels())])
print(figure_folder)
plt.savefig(os.path.join(figure_folder, 'localized_metrics', "violin_" +  key + '.svg'))

plt.show()
#%%  Check the profiles in different bins
fig, axs = plt.subplots(8, 1)
bin_keys = sorted(grouped_data_bins_here[key]['distances'][pos_type].keys(), reverse=True)
bin_names = bin_keys
# axs = [axd['bin5'], axd['bin3'], axd['bin1']]
# bin_keys = ['bin5', 'bin3', 'bin1']
# bin_names = ['Bin 5', 'Bin 3', 'Bin 1']
print(len(axs))
for ax, bin_cat, bin_name in zip(axs, bin_keys, bin_names):
    data = grouped_data_bins_here[key]['distances'][pos_type][bin_cat]
    density, bins = np.histogram(data, bins=np.linspace(0, 0.5, 50))
    density = density / density.sum()
    widths = bins[:-1] - bins[1:]

    print(key, pos_type, bin_cat)
    ax.bar(bins[1:], density, width=widths)
    ax.set_xlim(0, 0.5)
    ax.set_xticks(np.linspace(0, 0.5, 6))
    ax.set_xticklabels([])
    data = np.asarray(data)
    ax.text(.7, .8, 'N =' + str(len(data)),
            transform=ax.transAxes, fontsize=10)
    ax.text(pole_cutoff/2, ax.get_ylim()[1]/3, str(round(len(data[data < pole_cutoff])/len(data)*100)) + '%',
            ha='center', fontsize=12, fontweight='bold')
    ax.text(pole_cutoff + (0.5-pole_cutoff)/2, ax.get_ylim()[1]/3, str(round(len(data[data >= pole_cutoff])/len(data)*100)) + '%',
            ha='center', fontsize=12, fontweight='bold')
    ax.axvline(pole_cutoff, color='#CC0066')
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_title(bin_name)

axs[-1].set_xticklabels(np.round(np.linspace(0, 0.5, 6), 1))
axs[-1].set_xlabel("Distance to Pole")
plt.tight_layout()
plt.show()
# _, p_value = stats.ttest_ind(mag_pole, mag_center)
# print("P: ", p_value)
# endregion




# %% Plot nucleoid occupancy vs particle size
grouped_data_bins_here = copy.deepcopy(grouped_data_bins)
colors = {'wt_6h': 'b', 'dpolyp_6h': 'g'}
bin_keys = sorted(grouped_data_bins_here[key]['distances'][pos_type].keys(), reverse=True)
pos_type = 'all'
for strain_starv in colors.keys():
    profile_key = profile_keys[strain_starv]
    half_profile = int(len(profiles[profile_key]['x_data'])/2)+1
    profile_half = profiles[profile_key]['profile'][:half_profile]
    norm_profile_half = np.divide((profile_half - min(profile_half)),
                              max(profile_half - min(profile_half)))
    norm_profile_half = np.interp(np.linspace(0, 0.5, 100),
                              profiles[profile_key]['x_data'][:half_profile],
                              norm_profile_half)
    x_interp = np.interp(np.linspace(0, 0.5, 100),
                     profiles[profile_key]['x_data'][:half_profile],
                     profiles[profile_key]['x_data'][:half_profile])
    first_over = np.argmax(norm_profile_half > cutoff)
    pole_cutoff = x_interp[first_over]
    print(pole_cutoff)
    for idx, bin_cat in enumerate(bin_keys):
        data = grouped_data_bins_here[strain_starv]['distances'][pos_type][bin_cat]
        data = np.asarray(data)
        plt.scatter(idx, len(data[data >= pole_cutoff])/len(data), c=colors[strain_starv])
# %%
