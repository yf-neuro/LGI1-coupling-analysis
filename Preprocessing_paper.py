import os
import mne


print('Starting now with preprocessing')
base_dir = './derivatives/'

directory_path = "./data/"

# Initialize a set to store unique participant identifiers
unique_participants = set()

# Iterate over the files in the directory
for filename in os.listdir(directory_path):
    # Check if the file is an .edf file
    if filename.endswith('.edf'):
        # Extract the participant identifier (e.g., "pat7" from "pat7_chanfilt.edf")
        participant_id = filename.split('_')[0]
        # Add the participant identifier to the set (automatically handles uniqueness)
        unique_participants.add(participant_id)

# Count the number of unique participants
unique_count = len(unique_participants)

import re

# define a regular expression pattern to match numbers after a dot
pattern = r'pat\s*(\d+)'

# use list comprehension to extract numbers using regular expression
res = [int(re.findall(pattern, string)[0]) for string in unique_participants if re.findall(pattern, string)]

output = base_dir

# Print the result
print(f"Number of unique participants with .edf files: {unique_count}")


def preprocess_eeg_new(id, base_input_path, output,inversion = False):
    """
    :param id: patient id
    :param base_input_path: path to edf file, edf file naming convention has to be pat*_(1).edf
    :return:
    """
    # id must be string
    if isinstance(id, str) != True:
        id = str(id)
        print("Converting ID to string")

    if len(id) == 1:
        sub_id = '0' + id
        patient_id = id
    else:
        sub_id = id
        patient_id = id

    #def PSD_component_analysis(patient_id, sub_id):
    base_output_path = os.path.join(output,f'sub{sub_id}/ses01/eeg/')
    input_edf_path = os.path.join(base_input_path, f'pat{patient_id}_(1).edf')
        # check if folder for PSD analysis exists
    if os.path.isdir(os.path.join(base_output_path, 'preprocessed')):
        print("Directory already exists")
        base_output_path = os.path.join(output,f'sub{sub_id}/ses01/eeg/preprocessed/')
    else:
        os.mkdir(os.path.join(base_output_path, 'preprocessed'))
        base_output_path = os.path.join(output,f'sub{sub_id}/ses01/eeg/preprocessed/')
    output_edf_path = os.path.join(base_output_path, f'pat{patient_id}_preprocessed.edf')

    # Get the standard 10-20 montage
    montage = mne.channels.make_standard_montage('standard_1020')

    # Get a list of standard 10-20 EEG channel names
    standard_1020_channels = montage.ch_names

    raw = mne.io.read_raw_edf(input_edf_path, preload=True)
    if 'Fp1' in raw.ch_names:
        mne.rename_channels(raw.info, {'Fp1': 'EOG1'})
        #raw._orig_units['EOG1'] = raw._orig_units['Fp1']
    if 'Fp2' in raw.ch_names:
        mne.rename_channels(raw.info, {'Fp2': 'EOG2'})

    # Get all the channel names from your data
    all_channel_names = raw.ch_names

    filtered_channels = []
    for ch in all_channel_names:
        if (ch in standard_1020_channels) and ':' not in ch and '-' not in ch:
            filtered_channels.append(ch)

    # Print the filtered channel names
    print("Filtered channel names: ", filtered_channels)

    channels_to_drop = [ch for ch in all_channel_names if ch not in filtered_channels]
    #    Load your raw data

    raw.drop_channels(channels_to_drop)
    raw.filter(l_freq=0.3, h_freq=35)
    raw.set_eeg_reference(ref_channels=['A1','A2'])
    raw.drop_channels(['A1','A2'])

    if inversion == True:
        raw._data = (raw._data[:,:])*-1

    raw.export(output_edf_path, fmt='edf', overwrite=True)
    return

# only uncomment if preprocessing did not happen before
for n in range(14):
    if n == 0:
        continue
    else:
        print(str(n))
        preprocess_eeg_new(n, directory_path, output,inversion = False)


