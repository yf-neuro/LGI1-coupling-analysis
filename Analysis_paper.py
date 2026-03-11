import os
import numpy as np
import matplotlib.pyplot as plt

""" requires the data structure to be
raw data = /derivatives/sub01/ses01/eeg/preprocessed/pat1_preprocessed.edf
spindle data frame = /derivatives/Group/spindles/SP_events.csv
SO data frame = derivatives/Group/slow_waves/SO_events.csv
"""

def epochs_and_events(raw, data, event_time, chan, sfreq, tmax, tmin):
    """
    for spindles and slow waves alike
    :param event_time: timing of events
    :param chan: channel
    :param sfreq: sampling frequency
    :param tmax: upper bound of time interval
    :param tmin: lower bound of time interval
    :return: epochs_data: epoched data; info: mne info object; events: events across channels;
     dict_id_new: ids of channels
    """

    event_time = event_time.to_numpy()
    event_time = event_time * sfreq
    event_time = event_time.astype(int)
    chan = chan.to_numpy()
    chan = chan.astype(int)
    # check whether last and first event are still within data
    for i in range(10):
        if len(data[0, :]) < event_time[len(event_time) - 1] + abs(tmax) * sfreq:
            event_time = event_time[0:len(event_time) - 1]
            chan = chan[0:len(chan) - 1]
        if event_time[0] - abs(tmin) * sfreq < 0:
            event_time = event_time[1:len(event_time)]
            chan = chan[1:len(chan)]
    # define events object
    events = np.empty((len(event_time), 3))
    events[:, 0] = event_time
    events[:, 1] = 0,
    events[:, 2] = chan
    events = events.astype(int)
    dict_id = {"F3": 0, "F4": 1, "C3": 2, "C4": 3, "O1": 4, "O2": 5}
    ch_col = events[:,2]
    ch_names_dict = {k: v in ch_col for k, v in dict_id.items()}
    ch_names = [k for k, v in ch_names_dict.items() if v]
    ch_idx = [dict_id[k] for k, v in ch_names_dict.items() if v]
    # get correct channel index
    ch_idx_new = ch_idx

    dict_id_new = {}
    for key in ch_names:
        for value in ch_idx_new:
            dict_id_new[key] = value
            ch_idx_new.remove(value)
            break

    dict_id = {"F3": 0, "F4": 1, "C3": 2, "C4": 3, "O1": 4, "O2": 5}
    ch_col = events[:,2]
    ch_names_dict = {k: v in ch_col for k, v in dict_id.items()}
    ch_names = [k for k, v in ch_names_dict.items() if v]
    ch_idx = [dict_id[k] for k, v in ch_names_dict.items() if v]

    ch_types = ['eeg'] * len(ch_names)
    # create the mne info object
    info = mne.create_info(ch_names=ch_names, sfreq=raw.info['sfreq'], ch_types=ch_types)
    n_channels = len(ch_names)
    # Compute the number of samples for the time window
    _, n_times = data.shape
    n_times_window = int((tmax - tmin) * sfreq)

    # Preallocate the epochs array (n_epochs, n_channels, n_times_window)
    n_epochs = len(events)
    epochs_data = np.zeros((n_epochs, n_channels, n_times_window))

    # Extract epochs
    for i, event in enumerate(events):
        event_onset_sample = event[0]  # This is the event time in samples

        # Calculate the sample indices for the epoch
        start_sample = int(event_onset_sample + tmin * sfreq)
        stop_sample = int(event_onset_sample + tmax * sfreq)

        # Handle boundaries to ensure indices are within the data range
        if start_sample < 0 or stop_sample > n_times:
            #events = events[0:i-1]
            raise ValueError(f"Event {i} at sample {event_onset_sample} would go out of bounds.")

        # Extract the epoch from the data
        #for u in range(len(ch_idx)):
        epochs_data[i] = data[ch_idx, start_sample:stop_sample]
    # conversion to microVolts
    epochs_data = epochs_data * 1e-6
    return epochs_data, info, events, dict_id_new
def TFR_processing_corrected(base_output_path, folder, patient_id, epochs_data, info, events, dict_id):
    """
    Function to calculate morlet-wavelet transformation across epochs and regions.
    Function splits data into epochs, computes TFR on epochs and plots epochs across channels.
    Averaged across regions: Frontal (All "F" channel), Central (All "C" Channels), and Occipital (All "O" channel).
    :param base_output_path: str; output folder path
    :param folder: str; either "slow_waves" or "spindles"
    :param patient_id: str
    :param epochs_data: nd array, shape (n_epochs, n_channels, n_times)
    :param info: Info Object from Raw
    :param events: nd array, shape (n_events, n_channels)
    :param dict_id: dict of which channel has which index {'Channel': index, ...}
    :return: evoked: Evoked Object
    :return: avgpower : Averaged TFR Object across epochs. Shape (n_channels, n_frequencies, n_times)
    """
    if folder == "spindles":
        tmin_epo = -1
        baseline_epo = [-0.5, 0.5]
        vmin = -10
        vmax = 10
        bl_plot = (-1,-0.6)
        tmin_plot = -0.4
        tmax_plot = 0.4
        sel_dat_min = abs(tmin_epo) + tmin_plot
        sel_dat_max = abs(tmin_epo) + tmax_plot

    elif folder == "slow_waves":
        tmin_epo = -2
        baseline_epo = [-1, 1]
        vmin = -30
        vmax = 30
        bl_plot = (-1.35, -0.85)
        tmin_plot = -0.8
        tmax_plot = 0.8
        sel_dat_min = abs(tmin_epo) + tmin_plot
        sel_dat_max = abs(tmin_epo) + tmax_plot

    # Create an EpochsArray object using the extracted data
    epochs = mne.EpochsArray(epochs_data, info, events, event_id=dict_id, tmin=tmin_epo, baseline=baseline_epo)
    montage = mne.channels.make_standard_montage(
        'standard_1020')  # You can replace 'standard_1020' with your montage type.
    epochs.set_montage(montage)
    epochs.save(os.path.join(base_output_path, folder, f'pat{patient_id}-epo.fif'), overwrite=True)
    print("Succesfully saved epochs data")

    channel_names = info.ch_names
    f_channel = [entry for entry in channel_names if entry.startswith('F')]
    c_channel = [entry for entry in channel_names if entry.startswith('C')]
    o_channel = [entry for entry in channel_names if entry.startswith('O')]

    # check if per channel region there are SP/SO
    if not f_channel:
        print("No frontal spindles found")
    else:
        fig2f = epochs[f_channel].plot_image(picks=f_channel, vmin=vmin, vmax=vmax, title="Epochs of Frontal Channels",
                                              combine='mean')
        for idx, fig in enumerate(fig2f):
            fname = os.path.join(base_output_path, folder, f'pat{patient_id}_frontal_epochs.png').format(idx)
            fig.savefig(fname)

    if not c_channel:
        print("No central spindles found")
    else:
        fig2c = epochs[c_channel].plot_image(picks=c_channel, combine='mean', vmin=vmin, vmax=vmax,
                                              title="Epochs of Central Channels")
        for idx, fig in enumerate(fig2c):
            fname = os.path.join(base_output_path, folder, f'pat{patient_id}_central_epochs.png').format(idx)
            fig.savefig(fname)
    if not o_channel:
        print("No occipital spindles found")
    else:
        fig2o = epochs[o_channel].plot_image(picks=o_channel, combine='mean', vmin=vmin, vmax=vmax,
                                              title="Epochs of Occipital Channels")
        for idx, fig in enumerate(fig2o):
            fname = os.path.join(base_output_path, folder, f'pat{patient_id}_occipital_epochs.png').format(idx)
            fig.savefig(fname)

    freqs = np.arange(0.5, 30.0, 0.5)

    n_cycles = freqs / 2
    # Compute TFR using Morlet wavelets
    power = epochs.compute_tfr(
        method="morlet", freqs=freqs, n_cycles=n_cycles, return_itc=False, average=False
    )
    power.data = np.sqrt(power.data)
    # Average the power data
    avgpower = power.average()
    avgpower.save(os.path.join(base_output_path, folder, f'pat{patient_id}-tfr.hdf5'), overwrite=True)
    if any(e in avgpower.ch_names for e in ['F3', 'F4']):
        power_frontal = epochs[f_channel].compute_tfr(
            method="morlet", freqs=freqs, n_cycles=n_cycles, return_itc=False, average=False
        )

        # Average the power data per region
        avgpower_frontal = power_frontal.average()
        avgpower_frontal.apply_baseline(baseline=bl_plot, mode='logratio')
        power_frontal = np.mean(avgpower_frontal.data[0:1, :, :], axis=0)
        data_frontal = epochs[f_channel].get_data()
        data_frontal = np.mean(data_frontal[:, 0:1, :], axis=1)
        data_frontal = np.mean(data_frontal[:, :], axis=0)

    if any(e in avgpower.ch_names for e in ['C3', 'C4']):
        power_central = epochs[c_channel].compute_tfr(
            method="morlet", freqs=freqs, n_cycles=n_cycles, return_itc=False, average=False
        )
        power_central.data = np.sqrt(power_central.data)
        # Average the power data
        avgpower_central = power_central.average()
        avgpower_central.apply_baseline(baseline=bl_plot, mode='logratio')
        power_central = np.mean(avgpower_central.data[2:3, :, :], axis=0)  # Avg across epochs for central channels
        data_central = epochs[c_channel].get_data()
        data_central = np.mean(data_central[:, 2:3, :], axis=1)
        data_central = np.mean(data_central[:, :], axis=0)

    if any(e in avgpower.ch_names for e in ['O1', 'O2']):
        power_occ = epochs[o_channel].compute_tfr(
            method="morlet", freqs=freqs, n_cycles=n_cycles, return_itc=False, average=False
        )
        power_occ.data = np.sqrt(power_occ.data)
        # Average the power data
        avgpower_occ = power_occ.average()
        avgpower_occ.apply_baseline(baseline=bl_plot, mode='logratio')
        power_occ = np.mean(avgpower_occ.data[4:5, :, :], axis=0)  # Avg across epochs for occipital channels
        data_occ = epochs[o_channel].get_data()
        data_occ = np.mean(data_occ[:, 4:5, :], axis=1)
        data_occ = np.mean(data_occ[:, :], axis=0)

    # Selecting a time range
    sf = epochs.info['sfreq']

    # Apply baseline correction (in dB)
    avgpower.apply_baseline(baseline=bl_plot, mode='logratio')

    # Compute mean power for the frontal and central regions
    def check_channels(ch_names, all_channel):
        # Extract the unique starting letters from all_channel
        starting_letters = {channel[0] for channel in all_channel}

        # Create a set of starting letters present in ch_names
        present_letters = {channel[0] for channel in ch_names}

        # Check if all starting letters are present in present_letters
        return starting_letters.issubset(present_letters)

    # define which channel names the data can have and stack them to save .npy array for later
    all_channel = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2']
    if set(all_channel).issubset(set(avgpower.ch_names)) or check_channels(avgpower.ch_names, all_channel):
        # Stack power data for both regions
        power_chan_occ = np.stack((power_frontal, power_central, power_occ), axis=0)
        np.save(os.path.join(base_output_path, folder, f'pat{patient_id}_pow_front_centr_occ.npy'), power_chan_occ)

        data_chan_occ = np.stack((data_frontal, data_central, data_occ), axis=0)
        np.save(os.path.join(base_output_path, folder, f'pat{patient_id}_data_front_centr_occ.npy'), data_chan_occ)
        print("saving data to ", os.path.join(base_output_path, folder, f'pat{patient_id}_data_front_centr_occ.npy'))
        print("all channels available --------------------------------------------------------------------------------------------")
    else:
        missing = [x for x in all_channel if x not in avgpower.ch_names]
        if 'F3' in missing and 'F4' in missing:#all(e in ['F3', 'F4'] for e in missing):
            print("No frontal channels found -------------------------------------------------------------------------------------")
            power_chan_occ = np.stack((power_central, power_occ), axis=0)
            np.save(os.path.join(base_output_path, folder, f'pat{patient_id}_pow_front_centr_occ.npy'), power_chan_occ)

            data_chan_occ = np.stack((data_central, data_occ), axis=0)
            np.save(os.path.join(base_output_path, folder, f'pat{patient_id}_data_front_centr_occ.npy'), data_chan_occ)
            print("saving data to ", os.path.join(base_output_path, folder, f'pat{patient_id}_data_front_centr_occ.npy'))
        if 'C3' in missing and 'C4' in missing: #all(e in ['C3', 'C4'] for e in missing):
            print(
                "No central channels found -------------------------------------------------------------------------------------")
            power_chan_occ = np.stack((power_frontal, power_occ), axis=0)
            np.save(os.path.join(base_output_path, folder, f'pat{patient_id}_pow_front_centr_occ.npy'), power_chan_occ)

            data_chan_occ = np.stack((data_frontal, data_occ), axis=0)
            np.save(os.path.join(base_output_path, folder, f'pat{patient_id}_data_front_centr_occ.npy'), data_chan_occ)
            print("saving data to ", os.path.join(base_output_path, folder, f'pat{patient_id}_data_front_centr_occ.npy'))
        if 'O1' in missing and 'O2' in missing: #all(e in ['O1', 'O2'] for e in missing):
            print(
                "No occipital channels found -------------------------------------------------------------------------------------")
            power_chan_occ = np.stack((power_frontal, power_central), axis=0)
            np.save(os.path.join(base_output_path, folder, f'pat{patient_id}_pow_front_centr_occ.npy'), power_chan_occ)

            data_chan_occ = np.stack((data_frontal, data_central), axis=0)
            np.save(os.path.join(base_output_path, folder, f'pat{patient_id}_data_front_centr_occ.npy'), data_chan_occ)
            print("saving data to ",os.path.join(base_output_path, folder, f'pat{patient_id}_data_front_centr_occ.npy'))

    #### Plotting
    ### Plotting individual channels
    # Individually scaled per channel
    # Set up subplots: determine the grid size based on number of channels
    num_channels = len(avgpower.data[:, 0, 0])  # number of subplots = number of channels
    rows = int(num_channels ** 0.5)  # Define grid size based on the number of channels
    cols = (num_channels // rows) + (num_channels % rows > 0)  # Adjust for uneven rows

    # Create the figure and subplots
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))  # Adjust figsize as necessary
    axes = axes.flatten()  # Flatten in case of 2D array of subplots

    # Loop over each channel and plot in its corresponding subplot
    for i in range(num_channels):
        select_data = avgpower.data[i, :, round(sel_dat_min * sf):round(sel_dat_max * sf)]
        vlim_min = select_data.min()
        vlim_max = select_data.max()

        # Plot TFR on corresponding subplot axis
        avgpower.plot([i],
                      baseline=None,
                      mode=None,
                      vlim=(vlim_min, vlim_max),
                      axes=axes[i],  # Use the appropriate axis for each subplot
                      title=f'Morlet Wavelets, averaged across epochs\nIndividually scaled per channel',
                      tmin=tmin_plot,
                      tmax=tmax_plot,
                      cmap="turbo",
                      show=False)  # Disable showing for batch plotting
        axes[i].set_title(f'Chan {avgpower.ch_names[i]}')

    # Hide any empty subplots (if cols * rows > num_channels)
    for j in range(num_channels, len(axes)):
        fig.delaxes(axes[j])

    # Save the entire figure with all subplots
    fig.savefig(os.path.join(base_output_path, folder, f'pat{patient_id}_TFR_morlet_indi_scaled.png'))
    plt.show()
    plt.close(fig)

    ## Plotting over all channels, uniformly scaled
    # Set up subplots: determine the grid size based on number of channels
    num_channels = len(avgpower.data[:, 0, 0])  # number of subplots = number of channels
    rows = int(num_channels ** 0.5)  # Define grid size based on the number of channels
    cols = (num_channels // rows) + (num_channels % rows > 0)  # Adjust for uneven rows

    select_data = avgpower.data[:, :, round(sel_dat_min * sf):round(sel_dat_max * sf)]
    vlim_min = select_data.min()
    vlim_max = select_data.max()

    # Create the figure and subplots
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))  # Adjust figsize as necessary
    axes = axes.flatten()  # Flatten in case of 2D array of subplots

    # Loop over each channel and plot in its corresponding subplot
    for i in range(num_channels):
        # Plot TFR on corresponding subplot axis
        avgpower.plot([i],
                      baseline=None,
                      mode=None,
                      vlim=(vlim_min, vlim_max),
                      axes=axes[i],  # Use the appropriate axis for each subplot
                      title=f'Morlet Wavelets, averaged across epochs\nUniformly scaled',
                      tmin=tmin_plot,
                      tmax=tmax_plot,
                      cmap="turbo",
                      show=False)  # Disable showing for batch plotting
        axes[i].set_title(f'Chan {avgpower.ch_names[i]}')

    # Hide any empty subplots (if cols * rows > num_channels)
    for j in range(num_channels, len(axes)):
        fig.delaxes(axes[j])

    # Save the entire figure with all subplots
    fig.savefig(os.path.join(base_output_path, folder, f'pat{patient_id}_TFR_morlet_uni_scaled.png'))
    plt.show()
    plt.close(fig)


    ##### Plot Regions
    # Update frequencies
    frequencies = np.linspace(0.5, 30, power_chan_occ.shape[1])  # Frequency range (0.5 to 30 Hz)

    # Set up subplots: determine the grid size based on number of channels
    num_channels = power_chan_occ.shape[0]  # Number of channels
    rows = int(num_channels ** 1)  # Define grid size based on the number of channels
    cols = (num_channels // rows) + (num_channels % rows > 0)  # Adjust for uneven rows

    # Create the figure and subplots
    fig, axes = plt.subplots(rows, cols, figsize=(15, 28))  # Adjust figsize as necessary
    axes = axes.flatten()  # Flatten in case of 2D array of subplots
    data_chan_occ = data_chan_occ[:,round(sel_dat_min * sf):round(sel_dat_max * sf)]

    # Adjusting data_chan to match the same time range
    data_t = np.linspace(tmin_plot, tmax_plot, data_chan_occ.shape[1])  # Time vector for data_chan (0.2 to 1.8s)
    channel_types = ["Frontal", "Central", "Occipital"]
    # Loop over each channel and plot in its corresponding subplot
    for i in range(num_channels):
        # Select data within the time window (0.2s to 1.8s in your example)
        select_data = power_chan_occ[i, :, round(sel_dat_min * sf):round(sel_dat_max * sf)]
        vlim_min = select_data.min()
        vlim_max = select_data.max()

        # Plot time-frequency data as a heatmap (imshow)
        img = axes[i].imshow(select_data,
                             aspect='auto',
                             extent=[tmin_plot, tmax_plot, frequencies[0], frequencies[-1]],
                             # Consistent time and frequency range
                             origin='lower',
                             cmap='turbo',
                             vmin=vlim_min,
                             vmax=vlim_max)

        # Set title and labels
        axes[i].set_title(f'{channel_types[i]} Channel')
        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel('Frequency (Hz)')

        # Create secondary y-axis for plotting data_chan
        ax2 = axes[i].twinx()  # Create a twin y-axis that shares the same x-axis
        ax2.plot(data_t, data_chan_occ[i,:], color='black',
                 label=f'Averaged {channel_types[i]} Data')  # Plot data_chan over the same time
        ax2.set_ylabel('Amplitude')  # Set the secondary y-axis label
        ax2.legend(loc='upper right')  # Optional: add legend to indicate data plot

        # Add colorbar to the plot
        fig.colorbar(img, ax=axes[i], orientation='vertical')

    # Hide any empty subplots (if cols * rows > num_channels)
    for j in range(num_channels, len(axes)):
        fig.delaxes(axes[j])

    fig.savefig(os.path.join(base_output_path, folder, f'pat{patient_id}_TFR_Regions.png'))

    # evoked data
    evoked = epochs.average()

    if folder == "spindles":
        # tiefpunkt
        down = evoked.data[0, round(0.4 * sf):round(0.6 * sf)].argmin()
        # in idx in array
        idx_down = down + round(0.4 * sf)
        # in sec
        down = abs((idx_down / sf) - 0.5)
        # sattelpunkt
        turn = down / 2
        # half way to top
        turn2 = turn / 2
        evoked.save(os.path.join(base_output_path, folder, f'pat{patient_id}_evoked-ave.fif'), overwrite=True)

        fig4 = evoked.plot_joint(times=[-down, -turn, -turn2, 0, turn2, turn, down],
                                 title="Evoked Response of all Spindles",
                                 picks="all", show=False)
        plt.xlim(-0.4,0.4)
        fig4.savefig(os.path.join(base_output_path, folder, f'pat{patient_id}_evoked_spindles.png'))
        print(f'saved {folder} evoked plot')

    if folder == "slow_waves":
        for k in range(len(evoked.data[:, 0])):
            down_test = evoked.data[k, :].argmin()
            idx_down_test = down_test
            down = None
            if 2.9 * sf > down_test > 1.1 * sf:
                down = down_test
                # in idx in array
                idx_down = down
                # in sec
                down = abs((idx_down / sf) - 1)
            else:
                continue

            up_test = evoked.data[k, :].argmax()
            up = None
            if 1.1 * sf < up_test < 2.9 * sf:
                up = up_test
                idx_up = up
                up = abs((idx_up / sf) - 1)
            else:
                continue
            # if up and down are existant, stop loop
            break

        evoked.save(os.path.join(base_output_path, folder, f'pat{patient_id}_evoked-ave.fif'), overwrite=True)
        if 'up' in locals():
            if up is None:
                # try:
                #     up
                # except NameError or UnboundLocalError:
                #     print("Uppest point of SO not found in range of 0.2 s around 0")
                #     fig4 = evoked.plot_joint(times=[-down_test, 0, up_test],
                #                              title="Evoked Response of all Slow waves",
                #                              picks="all", show=False)
                #     plt.xlim(-0.8,0.8)
                #     fig4.savefig(os.path.join(base_output_path, folder, f'pat{patient_id}_evoked_sw.png'))
                #     print(f'saved {folder} evoked plot')
                print("No UP found. No plotting.")
                print(f'plot pat{patient_id}_evoked_sw.png not created')
            else:
                    fig4 = evoked.plot_joint(times=[-down, 0, up],
                                             title="Evoked Response of all Slow waves",
                                             picks="all", show=False)
                    plt.xlim(-0.8,0.8)
                    fig4.savefig(os.path.join(base_output_path, folder, f'pat{patient_id}_evoked_sw.png'))
                    print(f'saved {folder} evoked plot')

    return evoked, avgpower
def load_pow(sub_arr, base_dir):
    """
    needs folder structure
    ".../sub01/ses01/eeg/slow_waves/pat1_pow_front_centr.npy"
    or
    ".../sub19/ses01/eeg/slow_waves/pat19_data_front_centr.npy"
    data files must be named: patX_data_front_centr.npy or patXX_data_front_centr.npy
    power files must be named: patX_pow_front_centr.npy or patXX_pow_front_centr.npy

    :param sub_arr: holds all subject ids as nums
    :param base_dir: base directory in which sub folders are
    :return: averaged power and data across all subjects
    """
    arrays = []

    # Loop through all subjects
    for subject_id in sub_arr:
        if subject_id == 0:
            continue
        # Format the subject ID with leading zeros (e.g., '01', '19')
        sub_folder = f"sub{subject_id:02d}/ses01/eeg/slow_waves/"

        # File name follows the pattern "patXX_pow_front_centr.npy"
        file_name = f"pat{subject_id}_pow_front_centr_occ.npy"

        # Full path to the numpy file
        file_path = os.path.join(base_dir, sub_folder, file_name)

        # Load the numpy array
        arr = np.load(file_path)

        # Append it to the list of arrays
        arrays.append(arr)

    # Stack all the loaded arrays along a new dimension (subjects)
    stacked_arrays_power = np.stack(arrays, axis=0)

    # Calculate the average across the subject dimension (axis=0)
    average_power = np.mean(stacked_arrays_power, axis=0)

    # Now `average_array` is the 3D array with the same shape as the original arrays
    # You can save it if needed

    #del(stacked_arrays)
    arrays = []

    # Loop through all subjects
    for subject_id in sub_arr:
        if subject_id == 0:
            continue
        # Format the subject ID with leading zeros (e.g., '01', '19')
        sub_folder = f"sub{subject_id:02d}/ses01/eeg/slow_waves/"

        # File name follows the pattern "patXX_pow_front_centr.npy"
        file_name = f"pat{subject_id}_data_front_centr_occ.npy"

        # Full path to the numpy file
        file_path = os.path.join(base_dir, sub_folder, file_name)

        # Load the numpy array
        arr = np.load(file_path)

        # Append it to the list of arrays
        arrays.append(arr)

    # Stack all the loaded arrays along a new dimension (subjects)
    stacked_arrays = np.stack(arrays, axis=0)

    # Calculate the average across the subject dimension (axis=0)
    average_data = np.mean(stacked_arrays, axis=0)

    # Now `average_array` is the 3D array with the same shape as the original arrays
    print("Averaging completed.")
    return average_power, average_data, stacked_arrays_power
def plot_sw(pow, data, channel_types, figsize, sf=256, stop_freq = 23):
    tmin_epo = -2
    tmin_plot = -0.8
    tmax_plot = 0.8
    sel_dat_min = abs(tmin_epo) + tmin_plot
    sel_dat_max = abs(tmin_epo) + tmax_plot

    power_chan_occ = pow
    data_chan = data
    frequencies = np.linspace(0.5, 30, power_chan_occ.shape[1])
    nearest_value = np.abs(frequencies - stop_freq).argmin()

    data_chan = data_chan[:, round(sel_dat_min * sf):round(sel_dat_max * sf)]
    # Set up subplots: determine the grid size based on number of channels
    num_channels = power_chan_occ.shape[0]  # Number of channels
    rows = int(num_channels ** 1)  # Define grid size based on the number of channels
    cols = (num_channels // rows) + (num_channels % rows > 0)  # Adjust for uneven rows

    # Create the figure and subplots
    fig, axes = plt.subplots(rows, cols, figsize=figsize,
                             gridspec_kw={'hspace': 0.35, 'wspace': 0.1})  # Adjust figsize as necessary
    axes = axes.flatten()  # Flatten in case of 2D array of subplots

    # Adjusting data_chan to match the same time range
    data_t = np.linspace(tmin_plot, tmax_plot, data_chan.shape[1])  # Time vector for data_chan (0.2 to 1.8s)
    #channel_types = ["Frontal", "Central"]
    # Loop over each channel and plot in its corresponding subplot
    for i in range(num_channels):
        # Select data within the time window (0.2s to 1.8s in your example)
        select_data = power_chan_occ[i, 0:nearest_value, round(sel_dat_min * sf):round(sel_dat_max * sf)]
        vlim_min = select_data.min()
        vlim_max = select_data.max()

        # Plot time-frequency data as a heatmap (imshow)
        img = axes[i].imshow(select_data,
                             aspect='auto',
                             extent=[tmin_plot, tmax_plot, frequencies[0], frequencies[nearest_value]],
                             # Consistent time and frequency range
                             origin='lower',
                             cmap='turbo',
                             vmin=vlim_min,
                             vmax=vlim_max)

        # Set title and labels
        axes[i].set_title(f'{channel_types[i]} Channels', fontsize=28, pad=25)
        axes[i].set_xlabel('Time (s)', fontsize=23)
        axes[i].set_ylabel('Frequency (Hz)', fontsize=23)
        axes[i].tick_params(axis='x', labelsize=20)
        axes[i].tick_params(axis='y', labelsize=20)
        # Create secondary y-axis for plotting data_chan
        ax2 = axes[i].twinx()  # Create a twin y-axis that shares the same x-axis
        ax2.plot(data_t, data_chan[i], color='black')  # Plot data_chan over the same time
        ax2.set_ylabel('Amplitude', fontsize=23)
        ax2.tick_params(axis='y', labelsize=20)  # Set the secondary y-axis label
        #ax2.legend(loc='upper right')  # Optional: add legend to indicate data plot

        # Add colorbar to the plot
        # Create the colorbar and store it in a variable
        cbar = fig.colorbar(img, ax=axes[i], orientation='vertical', pad=0.1, label='Power in dB')

        # Set the font size for the colorbar label
        cbar.set_label('Power in dB', fontsize=23)  # Adjust 'fontsize' as desired

        # Set the font size for the colorbar tick labels
        cbar.ax.tick_params(labelsize=18)  # Adjust 'labelsize' as desired
    # Hide any empty subplots (if cols * rows > num_channels)
    for j in range(num_channels, len(axes)):
        fig.delaxes(axes[j])
    return fig
def plot_cluster_new(T_obs, clusters, cluster_p_values, data, tmin_plot=-0.8, tmax_plot=0.8):
    tmin_epo = -2
    sel_dat_min = abs(tmin_epo) + tmin_plot
    sel_dat_max = abs(tmin_epo) + tmax_plot
    sf = 128
    min_plt = round(sel_dat_min * sf)
    max_plt = round(sel_dat_max * sf)
    data = data[min_plt:max_plt]
    fig, ax1 = plt.subplots()
    data_t = np.linspace(tmin_plot, tmax_plot, data.shape[0])

    T_obs = T_obs[:, min_plt:max_plt]
    for i in range(len(clusters)):
        clusters[i] = clusters[i][:, min_plt:max_plt]

    # Plot the T values
    im = ax1.imshow(T_obs, aspect='auto', origin='lower',
                    extent=[-0.8, 0.8, 0, 59], cmap='RdBu_r')

    ax1.set_xlabel('Timepoints')
    ax1.set_ylabel('Frequencies (Hz)')

    # Define custom tick locations and labels for the y-axis
    ytick_locs = np.linspace(0, 59, 7)  # Positions in the plot's y-axis range
    ytick_labels = np.linspace(0, 30, 7).astype(int)  # Desired labels from 0 to 30

    # Apply these ticks and labels
    ax1.set_yticks(ytick_locs)
    ax1.set_yticklabels([f"{label:.1f}" for label in ytick_labels])

    ax1.set_autoscaley_on(False)
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('T-value')

    # Mark significant clusters
    for i_c, c in enumerate(clusters):
        if cluster_p_values[i_c] < 0.12:  # Apply threshold on p-value
            x_extent = np.linspace(-0.8, 0.8, c.shape[1])
            ax1.contour(x_extent, np.arange(c.shape[0]), c, colors='k', linewidths=1)

    ax2 = plt.twinx()  # Create a twin y-axis that shares the same x-axis
    ax2.plot(data_t, data, color='black', linewidth=2)  # Plot data_chan over the same time
    ax2.set_ylabel('Amplitude (μV)')  # Set the secondary y-axis label
    ax2.tick_params(labelright=False)
    ax2.set_xlim([-0.8, 0.8])
    return fig
def plot_sw_clusters(pow, data, channel_types, figsize, clusters, cluster_p_values, sf=256, stop_freq=23,
                     tmin_plot=-0.8, tmax_plot=0.8, clusters_plot=True):
    tmin_epo = -2
    sel_dat_min = abs(tmin_epo) + tmin_plot
    sel_dat_max = abs(tmin_epo) + tmax_plot

    power_chan_occ = pow
    data_chan = data
    frequencies = np.linspace(0.5, 30, power_chan_occ.shape[1])
    nearest_value = np.abs(frequencies - stop_freq).argmin()

    data_chan = data_chan[:, round(sel_dat_min * sf):round(sel_dat_max * sf)]
    # Set up subplots: determine the grid size based on number of channels
    num_channels = power_chan_occ.shape[0]  # Number of channels
    rows = int(num_channels ** 1)  # Define grid size based on the number of channels
    cols = (num_channels // rows) + (num_channels % rows > 0)  # Adjust for uneven rows

    # Create the figure and subplots
    fig, axes = plt.subplots(rows, cols, figsize=figsize,
                             gridspec_kw={'hspace': 0.35, 'wspace': 0.1})  # Adjust figsize as necessary
    axes = axes.flatten()  # Flatten in case of 2D array of subplots

    # Adjusting data_chan to match the same time range
    data_t = np.linspace(tmin_plot, tmax_plot, data_chan.shape[1])  # Time vector for data_chan (0.2 to 1.8s)
    # Loop over each channel and plot in its corresponding subplot
    for i in range(num_channels):
        # Select data within the time window (0.2s to 1.8s in your example)
        select_data = power_chan_occ[i, 0:nearest_value, round(sel_dat_min * sf):round(sel_dat_max * sf)]
        vlim_min = select_data.min()
        vlim_max = select_data.max()

        # Plot time-frequency data as a heatmap (imshow)
        img = axes[i].imshow(select_data,
                             aspect='auto',
                             extent=[tmin_plot, tmax_plot, frequencies[0], frequencies[nearest_value]],
                             # Consistent time and frequency range
                             origin='lower',
                             cmap='turbo',
                             vmin=vlim_min,
                             vmax=vlim_max)

        # Set title and labels
        axes[i].set_title(f'{channel_types[i]} Channels', fontsize=28, pad=25)
        axes[i].set_xlabel('Time (s)', fontsize=23)
        axes[i].set_ylabel('Frequency (Hz)', fontsize=23)
        axes[i].tick_params(axis='x', labelsize=20)
        axes[i].tick_params(axis='y', labelsize=20)
        # Create secondary y-axis for plotting data_chan
        ax2 = axes[i].twinx()  # Create a twin y-axis that shares the same x-axis
        ax2.plot(data_t, data_chan[i], color='black', linewidth=2.2)  # Plot data_chan over the same time
        ax2.set_ylabel('Amplitude', fontsize=23)
        ax2.tick_params(axis='y', labelsize=20)  # Set the secondary y-axis label
        # ax2.legend(loc='upper right')  # Optional: add legend to indicate data plot

        # Add colorbar to the plot
        # Create the colorbar and store it in a variable
        cbar = fig.colorbar(img, ax=axes[i], orientation='vertical', pad=0.1, label='Power in dB')

        # Set the font size for the colorbar label
        cbar.set_label('Power in dB', fontsize=23)  # Adjust 'fontsize' as desired

        # Set the font size for the colorbar tick labels
        cbar.ax.tick_params(labelsize=18)  # Adjust 'labelsize' as desired

        if clusters_plot == True:
            # Mark significant clusters
            # select cluster data
            for n in range(len(clusters)):
                if cluster_p_values[n] < 0.12:  # Apply threshold on p-value
                    # cluster_select = clusters[n][0:nearest_value, round(sel_dat_min * sf):round(sel_dat_max * sf)]
                    cluster_select = clusters[n][0:nearest_value, round(sel_dat_min * sf):round(sel_dat_max * sf)]
                    x_extent = np.linspace(tmin_plot, tmax_plot, cluster_select.shape[1])
                    y_extent = np.linspace(frequencies[0], frequencies[nearest_value], cluster_select.shape[0])
                    axes[i].contour(x_extent, y_extent, cluster_select, colors='k', linewidths=2)

    # Hide any empty subplots (if cols * rows > num_channels)
    for j in range(num_channels, len(axes)):
        fig.delaxes(axes[j])

    return fig
def plot_sw_clusters_one(pow, data, channel_types, figsize, clusters, cluster_p_values, sf=256, stop_freq=23,
                         tmin_plot=-0.8, tmax_plot=0.8, clusters_plot=True):
    tmin_epo = -2
    sel_dat_min = abs(tmin_epo) + tmin_plot
    sel_dat_max = abs(tmin_epo) + tmax_plot

    power_chan_occ = pow
    data_chan = data
    frequencies = np.linspace(0.5, 30, power_chan_occ.shape[0])
    nearest_value = np.abs(frequencies - stop_freq).argmin()

    data_chan = data_chan[round(sel_dat_min * sf):round(sel_dat_max * sf)]

    # Create the figure and subplots
    fig, axes = plt.subplots(1, 1, figsize=figsize,
                             gridspec_kw={'hspace': 0.35, 'wspace': 0.1})  # Adjust figsize as necessary

    # Adjusting data_chan to match the same time range
    data_t = np.linspace(tmin_plot, tmax_plot, data_chan.shape[0])  # Time vector for data_chan (0.2 to 1.8s)

    select_data = power_chan_occ[0:nearest_value, round(sel_dat_min * sf):round(sel_dat_max * sf)]
    vlim_min = select_data.min()
    vlim_max = select_data.max()

    # Plot time-frequency data as a heatmap (imshow)
    img = axes.imshow(select_data,
                      aspect='auto',
                      extent=[tmin_plot, tmax_plot, frequencies[0], frequencies[nearest_value]],
                      # Consistent time and frequency range
                      origin='lower',
                      cmap='turbo',
                      vmin=vlim_min,
                      vmax=vlim_max)

    # Set title and labels
    # axes.set_title(f'{channel_types} Channels', fontsize=28, pad=25)
    axes.set_xlabel('Time (s)', fontsize=23)
    axes.set_ylabel('Frequency (Hz)', fontsize=23)
    axes.tick_params(axis='x', labelsize=20)
    axes.tick_params(axis='y', labelsize=20)
    # Create secondary y-axis for plotting data_chan
    ax2 = axes.twinx()  # Create a twin y-axis that shares the same x-axis
    ax2.plot(data_t, data_chan, color='black', linewidth=2.2)  # Plot data_chan over the same time
    ax2.set_ylabel('Amplitude', fontsize=23)
    ax2.tick_params(axis='y', labelsize=20)  # Set the secondary y-axis label
    # ax2.legend(loc='upper right')  # Optional: add legend to indicate data plot

    # Add colorbar to the plot
    # Create the colorbar and store it in a variable
    cbar = fig.colorbar(img, ax=axes, orientation='vertical', pad=0.1, label='Power in dB')

    # Set the font size for the colorbar label
    cbar.set_label('Power in dB', fontsize=23)  # Adjust 'fontsize' as desired

    # Set the font size for the colorbar tick labels
    cbar.ax.tick_params(labelsize=18)  # Adjust 'labelsize' as desired
    # Mark significant clusters
    # select cluster data
    if clusters_plot == True:
        for n in range(len(clusters)):
            if cluster_p_values[n] < 0.12:  # Apply threshold on p-value
                # cluster_select = clusters[n][0:nearest_value, round(sel_dat_min * sf):round(sel_dat_max * sf)]
                cluster_select = clusters[n][0:nearest_value, round(sel_dat_min * sf):round(sel_dat_max * sf)]
                x_extent = np.linspace(tmin_plot, tmax_plot, cluster_select.shape[1])
                y_extent = np.linspace(frequencies[0], frequencies[nearest_value], cluster_select.shape[0])
                axes.contour(x_extent, y_extent, cluster_select, colors='k', linewidths=2)

    return fig


#################################### Single subject analysis ###########################################
# change directory for controls vs. patients
base_dir = '/.'

for i in range(1,12,1):
    if i in [0]:
        print(str(i))
        continue
    patient_id = str(i)
    if len(patient_id) == 2:
        sub_id = patient_id
    else:
        sub_id = "0"+str(i)
    edf_path = os.path.join(base_dir, f'sub{sub_id}/ses01/eeg/preprocessed/pat{patient_id}_preprocessed.edf')
    hypno_dir = os.path.join(base_dir,f'sub{sub_id}/ses01/scoring/')
    eeg_out_path = os.path.join(base_dir,f'sub{sub_id}/ses01/eeg/')
    base_output_path = os.path.join(base_dir,f'sub{sub_id}/ses01/eeg/')
    print(f'Subject ID: {sub_id} starts now')

    print("Getting Data")
    import mne
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw = raw.resample(sfreq=128)

    # read in dataframe
    import pandas as pd

    id_tmp = int(patient_id)
    # replace name for change between controls and patients!
    df_tfr = pd.read_csv("./derivatives/Group/spindles/SP_events.csv")
    # filter by patient ID to feed in only the right ones
    df_tfr = df_tfr[df_tfr['Subject'] == id_tmp]
    df_tfr = df_tfr.drop_duplicates(subset=['Peak'])
    event_time_sp = df_tfr.Peak
    chan_sp = df_tfr.IdxChannel

    # Define parameters
    tmin_sp = -1  # 0.5 seconds before the event
    tmax_sp = 1  # 0.5 seconds after the event
    sfreq_sp = 128

    data = raw.get_data(units="uV")
    print("Doing TFR analysis")

    epochs_data_sp, info_sp, events_sp, dict_id_sp = epochs_and_events(raw, data, event_time_sp, chan_sp,
                                                                                sfreq_sp,
                                                                                tmax_sp,
                                                                                tmin_sp)

    evoked_spi, avgpower_spi = TFR_processing_corrected(base_output_path, folder="spindles", patient_id=patient_id, epochs_data=epochs_data_sp, info=info_sp,events=events_sp, dict_id=dict_id_sp)

    # Slow wave detection
    id_tmp = int(patient_id)
    # replace name for change between controls and patients!
    df_tfr_SO = pd.read_csv("./derivatives/Group/slow_waves/SO_events.csv")
    # filter by patient ID to feed in only the right ones
    df_tfr_SO = df_tfr_SO[df_tfr_SO['Subject'] == id_tmp]
    df_tfr_SO = df_tfr_SO.drop_duplicates(subset=['MidCrossing'])
    event_time_sw = df_tfr_SO.MidCrossing
    chan_sw = df_tfr_SO.IdxChannel
    # Define parameters
    tmin_sw = -2  # 0.5 seconds before the event
    tmax_sw = 2  # 0.5 seconds after the event
    sfreq_sw = 128
    data_sw = raw.get_data(units="uV")
    epochs_data_sw, info_sw, events_sw, dict_id_sw = epochs_and_events(raw, data_sw, event_time_sw, chan_sw, sfreq_sw,
                                                                            tmax_sw,
                                                                            tmin_sw)

    evoked, avgpower = TFR_processing_corrected(base_output_path, folder="slow_waves", patient_id=patient_id, epochs_data=epochs_data_sw,
                        info=info_sw, events=events_sw, dict_id=dict_id_sw)

    del(raw)


############################# Now group analysis ####################################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

base_dir = "./"
d_arr = list(range(1,13))
d_pow, d_data,d_sub = load_pow(d_arr, base_dir)
np.save(os.path.join(base_dir,"Group/slow_waves/left_pow_all.npy"), d_pow)
np.save(os.path.join(base_dir,"Group/slow_waves/left_data_all.npy"), d_data)
np.save(os.path.join(base_dir,"Group/slow_waves/left_all_sub.npy"), d_sub)

contr_dir = "./derivatives/Controls/"

c_arr = list(range(1,15))
c_pow, c_data, c_sub = load_pow(c_arr, contr_dir)
np.save(os.path.join(base_dir,"Group/slow_waves/contr_pow_all.npy"), c_pow)
np.save(os.path.join(base_dir,"Group/slow_waves/contr_data_all.npy"), c_data)
np.save(os.path.join(base_dir,"Group/slow_waves/contr_all_sub.npy"), c_sub)

channel_types = ["Frontal", "Central", "Occipital"]
figsize=(15, 25)

fig2 = plot_sw(pow = d_pow, data=d_data, channel_types=channel_types, figsize=figsize, sf = 128)
fig2.suptitle('LG1 Patients', fontsize=30)
plt.show()
fig2.savefig(os.path.join(base_dir, 'Group/slow_waves/TFR_LG1.png'))

fig99 = plot_sw(pow = c_pow, data=c_data, channel_types=channel_types, figsize=figsize, sf = 128)
fig99.suptitle('Control group', fontsize=30)
plt.show()
fig99.savefig(os.path.join(base_dir, 'Group/slow_waves/TFR_contr.png'))

## cluster based permutation test
from mne.stats import permutation_cluster_test

d_pow, d_data,d_sub = load_pow(d_arr, base_dir)
c_pow, c_data,c_sub = load_pow(c_arr, contr_dir)

tmin_epo = -2
tmin_plot = -0.8
tmax_plot = 0.8
sel_dat_min = abs(tmin_epo) + tmin_plot
sel_dat_max = abs(tmin_epo) + tmax_plot
sf = 128
min_plt = round(sel_dat_min*sf)
max_plt = round(sel_dat_max*sf)

data_cd = np.stack([c_data, d_data], axis=0)
data_cd = np.mean(data_cd, axis=0)
data_cd = data_cd[1,:]

# Power values
data_d = d_sub[:,1,:,:]
data_contr = c_sub[:,1,:,:]

X = []
X = [data_d, data_contr]
# Define the number of permutations
n_permutations = 10000

# Perform cluster-based permutation test
T_obs_rl, clusters_rl, cluster_p_values_rl, _ = permutation_cluster_test(
    X, n_permutations=n_permutations, tail=0, n_jobs=1, out_type='mask'
)
clusters_plot2 = clusters_rl

fig = plot_cluster_new(T_obs_rl, clusters_rl, cluster_p_values_rl, data_cd)
plt.title("T-values and significant clusters")
plt.show()
fig.savefig(os.path.join(base_dir, 'Group/slow_waves/Permut_LG1Contr.png'), dpi=300)


# more plots with cluster overlay
d_arr = list(range(1,13))
d_pow, d_data,d_sub = load_pow(d_arr, base_dir)

contr_dir = "./derivatives/Controls/"
c_arr = list(range(1,15))
c_pow, c_data, c_sub = load_pow(c_arr, contr_dir)

channel_types = ["Frontal", "Central", "Occipital"]
figsize=(15, 25)

fig2 = plot_sw_clusters(pow = d_pow, data=d_data, channel_types=channel_types, figsize=figsize, sf = 128, clusters = clusters_rl, cluster_p_values=cluster_p_values_rl,tmin_plot = -1.2,tmax_plot = 1.2, clusters_plot=False)
fig2.suptitle('LG1 Patients', fontsize=30)
plt.show()
fig2.savefig(os.path.join(base_dir, 'Group/slow_waves/TFR_LG1.png'))

fig99 = plot_sw_clusters(pow = c_pow, data=c_data, channel_types=channel_types, figsize=figsize, sf = 128, clusters = clusters_rl, cluster_p_values=cluster_p_values_rl,tmin_plot = -1.2,tmax_plot = 1.2, clusters_plot=False)
fig99.suptitle('Control group', fontsize=30)
plt.show()
fig99.savefig(os.path.join(base_dir, 'Group/slow_waves/TFR_contr.png'))

#difference plot
diff_pow = d_pow - c_pow
diff_data = d_data - c_data
mean_data = np.mean(np.stack([d_data,c_data]), axis =0)
channel_types = ["Frontal", "Central", "Occipital"]
figsize=(15, 25)
# Perform cluster-based permutation test
T_obs_rl, clusters_rl, cluster_p_values_rl, _ = permutation_cluster_test(
    X, n_permutations=n_permutations, tail=0, n_jobs=1, out_type='mask'
)

fig2 = plot_sw_clusters(pow = diff_pow, data=mean_data, channel_types=channel_types, figsize=figsize, sf = 128, clusters = clusters_rl, cluster_p_values=cluster_p_values_rl,tmin_plot = -1.2,tmax_plot = 1.2)
fig2.suptitle('Difference Patients minus Controls', fontsize=30)
plt.show()
fig2.savefig(os.path.join(base_dir, 'Group/slow_waves/TFR_difference_clusters.png'))

fig2 = plot_sw_clusters(pow = diff_pow, data=mean_data, channel_types=channel_types, figsize=figsize, sf = 128, clusters = clusters_rl, cluster_p_values=cluster_p_values_rl,tmin_plot = -1.2,tmax_plot = 1.2, clusters_plot=False)
fig2.suptitle('Difference Patients minus Controls', fontsize=30)
plt.show()
fig2.savefig(os.path.join(base_dir, 'Group/slow_waves/TFR_difference.png'))


channel_types = ["Frontal", "Central", "Occipital"]
figsize=(15, 25)

fig5 = plot_sw_clusters(pow = d_pow, data=d_data, channel_types=channel_types, figsize=figsize, sf = 128,  clusters=clusters_rl, cluster_p_values=cluster_p_values_rl,tmin_plot = -1.2,tmax_plot = 1.2)
fig5.suptitle('LG1 Patients', fontsize=30)
plt.show()
fig5.savefig(os.path.join(base_dir, 'Group/slow_waves/TFR_LG1_clustered.png'))

fig91 = plot_sw_clusters(pow = c_pow, data=c_data, channel_types=channel_types, figsize=figsize, sf = 128,  clusters=clusters_rl, cluster_p_values=cluster_p_values_rl,tmin_plot = -1.2,tmax_plot = 1.2)
fig91.suptitle('Control group', fontsize=30)
plt.show()
fig91.savefig(os.path.join(base_dir, 'Group/slow_waves/TFR_contr_clustered.png'))

# plot all channels in one
channel_types = ["All"]

avg_c_data = c_data[1,:]
avg_d_data = d_data[1,:]
avg_c_pow = c_pow[1,:,:]
avg_d_pow = d_pow[1,:,:]

figsize =[20,12]
fig97 = plot_sw_clusters_one(pow = avg_d_pow, data=avg_d_data, channel_types=channel_types, figsize=figsize, sf = 128, clusters=clusters_rl, cluster_p_values=cluster_p_values_rl,tmin_plot = -1.2,tmax_plot = 1.2)
fig97.suptitle('Central Channels, Patients', fontsize=30)
plt.show()
fig97.savefig(os.path.join(base_dir, 'Group/slow_waves/TFR_LGI1_centr_clust.png'))

figsize =[20,12]
fig98 = plot_sw_clusters_one(pow = avg_c_pow, data=avg_c_data, channel_types=channel_types, figsize=figsize, sf = 128,  clusters=clusters_rl, cluster_p_values=cluster_p_values_rl,tmin_plot = -1.2,tmax_plot = 1.2)
fig98.suptitle('Central Channels, Controls', fontsize=30)
plt.show()
fig98.savefig(os.path.join(base_dir, 'Group/slow_waves/TFR_contr_centr_clust.png'))

diff_dat_central = mean_data[1,:]
diff_pow_central = diff_pow[1,:,:]

figsize =[20,12]
fig98 = plot_sw_clusters_one(pow = diff_pow_central, data=diff_dat_central, channel_types=channel_types, figsize=figsize, sf = 128,  clusters=clusters_rl, cluster_p_values=cluster_p_values_rl,tmin_plot = -1.2,tmax_plot = 1.2)
fig98.suptitle('Difference Patients minus controls, Central Channels', fontsize=30)
plt.show()
fig98.savefig(os.path.join(base_dir, 'Group/slow_waves/TFR_diff_centr_clust.png'))

figsize =[20,12]
fig97 = plot_sw_clusters_one(pow = avg_d_pow, data=avg_d_data, channel_types=channel_types, figsize=figsize, sf = 128, clusters=clusters_rl, cluster_p_values=cluster_p_values_rl,tmin_plot = -1.2,tmax_plot = 1.2, clusters_plot=False)
fig97.suptitle('Central Channels, Patients', fontsize=30)
plt.show()
fig97.savefig(os.path.join(base_dir, 'Group/slow_waves/TFR_LGI1_centr.png'))

figsize =[20,12]
fig98 = plot_sw_clusters_one(pow = avg_c_pow, data=avg_c_data, channel_types=channel_types, figsize=figsize, sf = 128,  clusters=clusters_rl, cluster_p_values=cluster_p_values_rl,tmin_plot = -1.2,tmax_plot = 1.2, clusters_plot=False)
fig98.suptitle('Central Channels, Controls', fontsize=30)
plt.show()
fig98.savefig(os.path.join(base_dir, 'Group/slow_waves/TFR_contr_centr.png'))

diff_dat_central = mean_data[1,:]
diff_pow_central = diff_pow[1,:,:]

figsize =[20,12]
fig98 = plot_sw_clusters_one(pow = diff_pow_central, data=diff_dat_central, channel_types=channel_types, figsize=figsize, sf = 128,  clusters=clusters_rl, cluster_p_values=cluster_p_values_rl,tmin_plot = -1.2,tmax_plot = 1.2, clusters_plot=False)
fig98.suptitle('Difference Patients minus controls, Central Channels', fontsize=30)
plt.show()
fig98.savefig(os.path.join(base_dir, 'Group/slow_waves/TFR_diff_centr.png'))

