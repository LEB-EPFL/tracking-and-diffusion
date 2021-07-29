# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
import datetime
import json

now = datetime.datetime.now()
string_now = now.strftime("%Y%m%d_%H%M%S")[2:]

json_file = '/Volumes/GlennGouldMac/PolyP/papers/ours/code/210712/general_info.json'
json_file_record = '/Volumes/GlennGouldMac/PolyP/papers/ours/code/210712/record_of_info_files/' + string_now + '_general_info.json'  # It can be useful to keep track of what information you have input at different points in time. The main info file will always be the most recent one; in a separate folder called "record_of_info_files" you can also keep track of all info files you have ever generated.

data = {}

data['spot_types'] = ['origins', 'muNS', 'fixed_origins', 'fixed_muNS']

data['basic_directory'] = '/Volumes/GlennGouldMac/PolyP/data/'
data['px_to_micron'] = 0.10748 # μm per pixel
data['all_starvation_times'] = ['0h', '6h']

data['muNS'] = {}
data['origins'] = {}
data['fixed_origins'] = {}
data['fixed_muNS'] = {}

data['muNS']['Gaussian_fits_initial_guess'] = {'amplitude' : 100, 'sigma_x' : 3, 'sigma_y' : 3, 'offset' : 100, 'theta' : 0}
data['muNS']['Gaussian_fits_other_params'] = {'particle_area_size' : 50, 'simple_Gassian' : True}

data['origins']['all_days'] = ['210227', '210330', '210518']
data['origins']['typical_stub_length'] = '010'
data['origins']['typically_avoid'] = ['bLR3']
data['origins']['diameter'] = 9
data['origins']['minmass'] = 4000
data['origins']['percentile'] = 65
data['origins']['search_range'] = 4
data['origins']['memory'] = 10
data['origins']['stub_length'] = 10
data['origins']['empirical_noise_floor_file'] = data['basic_directory'] + '210227/fixed_origins/6h/210227_bLR2_6h_fixed_20008/analysis/diameter9_minmass4000_percentile65/search_range4_memory10/emsd_stub_length010_from_filtered_until_lag_099__all.pkl'

data['muNS']['all_days'] = ['210323', '201226',
                            '201205',
                            '201009','201006','201004']
data['muNS']['typical_stub_length'] = '020'

data['muNS']['typically_avoid'] = ['bLR33', '201205_bLR31_6hlowN_210ms_tc_2004', '201205_bLR32_6hlowN_210ms_tc_2005']
# I have been focusing on 210ms movies, but for 201205 I am not using lowN data for now because the sample prep was different. We don't yet know whether this affects lowC.

data['muNS']['diameter'] = 7
data['muNS']['minmass'] = 110
data['muNS']['percentile'] = 65
data['muNS']['search_range'] = 4
data['muNS']['memory'] = 20
data['muNS']['stub_length'] = 20
data['muNS']['empirical_noise_floor_file'] = data['basic_directory'] + '210323/fixed_muNS/6h/210323_bLR32_6hlowN_fixed_210ms_tc_6102/analysis/diameter7_minmass110_percentile65/search_range4_memory20/emsd_stub_length020__snr_greater_than_1p0_from_filtered_until_lag_299__all.pkl'

data['muNS_lowC']['all_days'] = ['210714', '210630', '210323', '201226', '201205']
data['muNS_lowC']['typical_stub_length'] = '020' ## does it match the stub length?
data['muNS_lowC']['typically_avoid'] = ['bLR33'] # I have been focusing on 210ms movies, but for 201205 I am not using lowN data for now because the sample prep was different. We don't yet know whether this affects lowC.

data['muNS_lowC']['dense_movies'] = [] # these FOVs are way denser than others, and we have observed that the height of the second peak in D_app increases in dense movies. So, we are removing the densest of those.
        
data['muNS_lowC']['typically_avoid'] = specs['typically_avoid'] + dense_movies
        
data['muNS_lowC']['diameter'] = 7
data['muNS_lowC']['minmass'] = 150
data['muNS_lowC']['percentile'] = 65
data['muNS_lowC']['search_range'] = 4
data['muNS_lowC']['memory'] = 20
data['muNS_lowC']['stub_length'] = 20
data['muNS_lowC']['empirical_noise_floor_file'] = '/Volumes/LEB_SHARED/SHARED/_Scientific projects/polyP_SoM/data/lowC/analysis_WS/210323/muNS_lowC_fixed/0h/210323_bLR32_0h_210ms_fixed_tc_006/analysis/diameter7_minmass130_percentile65/search_range4_memory20/emsd_stub_length020_from_filtered__all.csv'


data['fixed_origins']['all_days'] = ['210227']
data['fixed_origins']['typical_stub_length'] = '010'
data['fixed_origins']['typically_avoid'] = ['bLR33']
data['fixed_origins']['diameter'] = 9
data['fixed_origins']['minmass'] = 4000
data['fixed_origins']['percentile'] = 65
data['fixed_origins']['search_range'] = 4
data['fixed_origins']['memory'] = 10
data['fixed_origins']['stub_length'] = 10

data['fixed_muNS']['all_days'] = ['210323']
data['fixed_muNS']['typical_stub_length'] = '020'
data['fixed_muNS']['typically_avoid'] = ['bLR33']
data['fixed_muNS']['diameter'] = 7
data['fixed_muNS']['minmass'] = 110
data['fixed_muNS']['percentile'] = 65
data['fixed_muNS']['search_range'] = 4
data['fixed_muNS']['memory'] = 20
data['fixed_muNS']['stub_length'] = 20

for i in data['spot_types']:
    data[i]['central_directory'] = '_' + str(i) + '/' + 'diameter' + str(data[i]['diameter']) + '_minmass' + str(data[i]['minmass']) + '_search_range' + str(data[i]['search_range']) + '_memory' + str(data[i]['memory']) + '_stub_length' + str(data[i]['stub_length']) + '/'

with open(json_file, 'w') as jf:
    json.dump(data, jf)

with open(json_file_record, 'w') as jf:
    json.dump(data, jf)

