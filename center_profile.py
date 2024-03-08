# https://github.com/ungarj/label_centerlines
from shapely.geometry.linestring import LineString
import read_and_plot_oufti_output_cleaned as oufti
from shapely import geometry
import numpy as np
import re
from label_centerlines import get_centerline
import matplotlib.pyplot as plt
import tifffile
from skimage import measure
import pdb
import geopandas as gpd
from typing import List
from tqdm import tqdm
import os
import json
import pandas as pd
import random


DYE = 'HUmCherry'  # sybrGreen HUmCherry
# days = ['210908'] if DYE == 'sybrGreen' else None
PLOT = True

with open('general_info.json') as jf:
    general_info = json.load(jf)
figure_folder = general_info['figure_directory'] + 'nc_plotting/'

with open('segmentation_general_info.json') as jf:
    segmentation_info = json.load(jf)


def main(results, profile=True):
    cells = {}
    for key in results.keys():
        cells[key] = []
        for idx, celldata in tqdm(results[key].iterrows(),  total=results[key].shape[0]):
            cell = create_cell(celldata)
            if profile:
                cell.calc_full_profile()
            else:
                cell.calc_full_centerline()
            cells[key].append(cell)

    return cells


def load_data(dye=DYE):
    days_here = segmentation_info[dye.lower()]['all_days']
    results = oufti.pool_segmentation_results(days_here)
    return results


class Cell(geometry.Polygon):

    def __init__(self, positions, cell_id=None, image=None, centerline=None, centerline_dis=None,
                 profile=None, dye=None, nucleoid_data=None):
        super().__init__(positions)
        self.image = image
        self.centerline = centerline
        self.centerline_dis = centerline_dis
        self.profile = profile
        self.dye = dye
        self.cell_id = cell_id
        if nucleoid_data is not None:
            self.nucleoid = Cell(nucleoid_data)
        else:
            self.nucleoid = None

    def create_centerline(self, plot=False):
        attributes = {"id": 1, "name": "polygon", "valid": True}
        # line = Centerline(self, interpolation_distance=0.01, **attributes)
        line = get_centerline(self, smooth_sigma=15)
        if plot:
            plt.scatter(list(zip(*list(line.coords)))[0], list(zip(*list(line.coords)))[1])
            bound = self.boundary
            plt.scatter(list(zip(*list(bound.coords)))[0], list(zip(*list(bound.coords)))[1])
        self.centerline = line
        return line

    def enlarge(self):
        old_cell = self
        res = 100
        new_shape = self.buffer(10, join_style=1, mitre_limit=1000, resolution=res)
        new_shape = new_shape.buffer(-2, join_style=1, mitre_limit=1000, resolution=res)
        # self = self.simplify(tolerance=0.7, preserve_topology=False)
        return Cell(new_shape, image=old_cell.image, centerline=old_cell.centerline,
                    centerline_dis=old_cell.centerline_dis)

    def draw(self):
        ax = plt.gca()
        gpd.GeoSeries(self).plot(color='r', alpha=0.5, ax=ax, edgecolor='g', linewidth=5)

    def extend_centerline(self):
        extension = 20
        # Extend by extension/2 px at the start
        for x in range(extension):
            start = self.centerline.coords[0]
            first = self.centerline.coords[1]
            vec = [x - x0 for x, x0 in zip(start, first)]
            vec = vec / np.linalg.norm(vec) / 2
            new_start = [x + x0 for x, x0 in zip(start, vec)]
            self.centerline = LineString([new_start] + list(self.centerline.coords))

        # extend by extension/2 px at the end
        for x in range(extension):
            end = self.centerline.coords[-1]
            last = self.centerline.coords[-2]
            vec = [x - x0 for x, x0 in zip(last, end)]
            vec = vec / np.linalg.norm(vec) / 2
            new_end = [x - x0 for x, x0 in zip(end, vec)]
            self.centerline = LineString(list(self.centerline.coords) + [new_end])

    def constrain_centerline(self):
        self.centerline = self.centerline.intersection(self)

    def calc_centerline_distances(self):
        coords = list(self.centerline.coords)
        distance = [0]
        for idx in range(1, len(coords)):
            point_distance = np.linalg.norm([x - x0 for x, x0 in zip(coords[idx-1], coords[idx])])
            distance = distance + [distance[-1] + point_distance]
        self.centerline_dis = distance

    def redistribute_centerline_points(self, num_vert=None):
        """ Make a uniform distribution of the measurement spots along the original measurement line
        If no number of points (num_vert) is given, space the points by half a pixel."""
        if num_vert is None:
            distance = 0.5
            num_vert = np.max([1, int(round(self.centerline.length / distance))])

        self.centerline = LineString([self.centerline.interpolate(float(n) / num_vert,
                                                                  normalized=True)
                                     for n in range(num_vert + 1)])

    def calc_profile(self):
        profile = []
        image = tifffile.imread(self.image)
        for idx in range(0, len(self.centerline.coords)-1):
            start, *end = measure.profile_line(image.T, self.centerline.coords[idx],
                                               self.centerline.coords[idx+1],
                                               linewidth=3)
            end = end[-1]
            if idx == 0:
                profile.append(start)
            else:
                profile.append(np.mean([old_end, start]))
            # also append the last value if we are at the end
            if idx == len(self.centerline.coords)-2:
                profile.append(end)
            old_end = end
        self.profile = profile

    def calc_full_centerline(self):
        self.create_centerline()
        self.extend_centerline()
        self.constrain_centerline()
        self.redistribute_centerline_points(20)
        self.calc_centerline_distances()

    def calc_full_profile(self):
        self.create_centerline()
        self.extend_centerline()
        self.constrain_centerline()
        self.redistribute_centerline_points(20)
        self.calc_centerline_distances()
        self.calc_profile()


def create_cell(result):
    cell_id = result.name
    x = list(map(np.float64, re.sub(r'[^\d|\.]', ' ', result['cell_xcoords']).split()))
    y = list(map(np.float64, re.sub(r'[^\d|\.]', ' ', result['cell_ycoords']).split()))

    nucleoid_x = list(map(np.float64, re.sub(r'[^\d|\.]', ' ',
                                             result['nucleoid_xcoords']).split()))
    nucleoid_y = list(map(np.float64, re.sub(r'[^\d|\.]', ' ',
                                             result['nucleoid_ycoords']).split()))
    if not nucleoid_y:
        nucleoid_mp = None
    else:
        nucleoid_mp = geometry.MultiPoint(list(zip(nucleoid_x, nucleoid_y)))
    mp = geometry.MultiPoint(list(zip(x, y)))
    conv_hull = mp.convex_hull
    cell = Cell(conv_hull, cell_id=cell_id, nucleoid_data=nucleoid_mp)
    try:
        cell.image = result['image']
    except KeyError:
        cell.image = None
    return cell


def plot_single_cell(cell: Cell):
    image = tifffile.imread(cell.image)
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(image)
    line = cell.centerline
    x_coords, y_coords = list(zip(*list(line.coords)))
    axs[0].scatter(x_coords, y_coords)
    x_coords_cell, y_coords_cell = list(zip(*list(cell.boundary.coords)))
    axs[0].scatter(x_coords_cell, y_coords_cell)

    axs[1].plot(cell.centerline_dis, cell.profile)

    # Zoom in on the cell but allow to go back to full image
    fig.canvas.toolbar.push_current()
    border = 5
    axs[0].set_xlim([np.min(x_coords)-border, np.max(x_coords)+border])
    axs[0].set_ylim([np.min(y_coords)-border, np.max(y_coords)+border])


def plot_profiles(cells: List[Cell], ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    for cell in random.sample(cells, 100):
        # plot_single_cell(cell)
        # plt.show()
        x_data, y_data = prepare_profile(cell)
        ax.plot(x_data, y_data, color='w', alpha=1/50, linewidth=4)
        # print(np.min(y_data))
        # axs.set_ylim(0, 0.025)


def save_cells(cells: List[Cell]):
    results = {}
    for idx, key in enumerate(cells.keys()):
        results[key] = {}
        for idx, cell in enumerate(cells[key]):
            x_data, y_data = prepare_profile(cell)
            celldict = {'x_data': cell.centerline_dis, 'y_data': cell.profile,
                        'x_data_norm': x_data, 'y_data_norm': y_data}
            results[key][idx] = celldict

    for key in results:
        df = pd.DataFrame.from_dict(results[key], orient='index')
        folder = os.path.join(figure_folder, 'profiles')
        os.makedirs(folder, exist_ok=True)
        df.to_csv(os.path.join(folder, key + '_' + DYE + '_210908incl.csv'))
        print(os.path.join(folder, key + '_' + DYE + '_210908incl.csv'))


def prepare_profile(cell: Cell):
    x_data = [position/np.max(cell.centerline_dis) for position in cell.centerline_dis]
    y_data_prel = [position/len(cell.profile) for position in cell.profile]
    y_data_prel = [position - np.min(y_data_prel) for position in y_data_prel]
    y_data = [position/np.sum(y_data_prel) for position in y_data_prel]
    # pdb.set_trace()
    return x_data, y_data


def plot_mean_profiles(cells: List[Cell], key: str):
    profile = np.ndarray([21])
    for cell in cells:
        x_data, y_data = prepare_profile(cell)
        profile = np.vstack((profile, y_data))
        # pdb.set_trace()
    print(profile.shape)
    profile = np.mean(profile, axis=0)
    if y_data[0] > 0.001:
        y_data.reverse()
        plt.plot(x_data, profile, label=key)
    else:
        plt.plot(x_data, profile, label=key)


def plot_mean_profiles_conditions(cells):
    fig, axs = plt.subplots(1, 1)
    for idx, key in enumerate(cells.keys()):
        plot_mean_profiles(cells[key], key)
    plt.legend()
    plt.show()


def plot_profiles_conditions(cells):
    fig, axs = plt.subplots(2, 2, sharey='all')
    axs = axs.flatten()
    for idx, key in enumerate(cells.keys()):
        plot_profiles(cells[key], axs[idx])
        axs[idx].set_title(key, fontsize='x-large')
        axs[idx].set_ylim(0, 0.1)
    plt.show()
