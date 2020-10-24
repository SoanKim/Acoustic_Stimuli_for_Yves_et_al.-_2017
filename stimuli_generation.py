from __future__ import absolute_import
import numpy as np
import json
from datetime import datetime
from scipy.stats import expon
import pickle as pkl
import scipy.io.wavfile as wavf
from matplotlib import colors
from scipy.signal import welch, decimate
from pycochleagram import cochleagram as cgram
from pycochleagram import utils
from pycochleagram.demo import main, make_harmonic_stack

sample_wave_path = '/Users/soankim/Downloads/baechi.wav'
individual_wave_path = '/Users/soankim/Downloads/chords/'
total_wave_path = '/Users/soankim/Downloads/total_wave.wav'
ramped_wave_path = '/Users/soankim/Downloads/ramped_wave.wav'
save_num_of_counts_json_path = '/Users/soankim/Downloads/num_of_cases.json'
save_spectrogram_json_path = '/Users/soankim/Downloads/spectrogram.json'
save_fig_path = "/Users/soankim/Downloads/figures/"
save_dic_path = "/Users/soankim/Downloads/stim_dic.json"
pickle_path = '/Users/soankim/Downloads/stim_img.pkl'

fs = 44100
total_dur_in_sec = 6
dur_in_sec = 0.03
repetition = 10
n_tones = 26
n_bins = 8
f0 = 400
n_trial = 1
ramp_dur = 0.025

marginal_prob = [0.08, 0.08, 0.08, 0.125, 0.125, 0.17, 0.17, 0.17]
np.random.shuffle(marginal_prob)
percentage = np.random.choice([0.3, 0.5, 0.8, 1.1, 1.4])

num_of_counts_json = {'poisson_num_bins': [],
                      'length_to_split': [],
                      'tones_per_bins': [],
                      'local_first_index': [],
                      'local_second_index': [],
                      'non_local_first_index': [],
                      'non_local_second_index': [],
                      'plot_first_index': [],
                      'plot_second_index': [],
                      'chosen_bins': [],
                      'chosen_bin_counts': [],
                      'change_percentage': [],
                      'chosen_f0': [],
                      'chosen_f0_num_idx': [],
                      'chosen_bin_num': [],
                      'exponential_change_time': [],
                      'localized_bins': [],
                      'non_localized_change_size': [],
                      'local_original_prob_by_time': [],
                      'local_changed_prob_by_time': [],
                      'local_changed_prob': [],
                      'changed_time_and_freq_plot_first_index': [],
                      'changed_time_and_freq_plot_second_index': [],
                      'marginal_prob': marginal_prob,
                      'change_detection_label': [],
                      'stim_img': [],
                      'trial': []}

spectrogram_json = {'time_in_mil': [],
                    'freq': [],
                    'bin': []}

def semitone(f0, n_tones):
    semitone_li = []
    for tone_order in range(n_tones-1):
        semitone = f0 * (2 ** (tone_order/12))
        semitone_li.append(semitone)
    point_two_seimitone = 1600 *(2**(2.41960633404/12))
    semitone_li.append(point_two_seimitone)

    return semitone_li

def poisson_num_bins(total_dur_in_sec):
    num_tones_by_poisson = np.random.poisson(lam=4.4, size=int(total_dur_in_sec/dur_in_sec))
    poisson_num_bin_li = list(num_tones_by_poisson)

    return poisson_num_bin_li

def tones_into_bins(semitone_li):
    length_to_split = [3, 3, 3, 3, 4, 4, 3, 3]
    np.random.shuffle(length_to_split)
    freq_cumsum_li = np.cumsum(length_to_split)
    tones_into_bins_li = []
    semitone_num_idx_li = []
    semitone_idx_li = list(range(26))

    for i in range(8):
        if i == 0:
            chop = semitone_li[i:freq_cumsum_li[i]]
            chop_idx = semitone_idx_li[i:freq_cumsum_li[i]]
        else:
            chop = semitone_li[freq_cumsum_li[i - 1]:freq_cumsum_li[i]]
            chop_idx = semitone_idx_li[freq_cumsum_li[i - 1]:freq_cumsum_li[i]]
        tones_into_bins_li.append(chop)
        semitone_num_idx_li.append(chop_idx)

    return length_to_split, tones_into_bins_li, semitone_num_idx_li

def bins_by_marginal_prob(poisson_num_bin_li, marginal_prob):
    chosen_bin_li = []
    for num in poisson_num_bin_li:
        chosen_bins = np.random.choice(list(range(8)), num, p=marginal_prob)
        chosen_bin_li.append(chosen_bins)

    return chosen_bin_li

def bins_by_localized_changed_marginal_prob(percentage, poisson_num_bin_li, marginal_prob):
    localized_change_size = marginal_prob.copy()
    localized_change_idx = np.random.choice([0, 2, 4, 6])
    not_selected_li = list(range(8))

    selected_li = []
    selected_prob = []
    for i, change_time_idx in enumerate(localized_change_size):
        if i == localized_change_idx:
            selected_li.append(i)
            selected_li.append(i+1)
            localized_change_size[i] = marginal_prob[i] + 0.125 * percentage
            localized_change_size[i + 1] = marginal_prob[i + 1] + 0.125 * percentage

    for selected in selected_li:
        not_selected_li.remove(selected)
        selected_prob.append(marginal_prob[selected])
        num_of_counts_json['local_first_index'].append(float(selected))
        num_of_counts_json['local_first_index'].append(float(selected + 1))

    for not_selected in not_selected_li:
        localized_change_size[not_selected] = marginal_prob[not_selected] - 0.125 * percentage/3

    localized_original_prob_li_by_time = []
    localized_changed_prob_li_by_time = []

    for time in range(int(total_dur_in_sec/dur_in_sec)):
        for num in poisson_num_bin_li:
            original_chosen_bins = np.random.choice(list(range(8)), num, p=marginal_prob)
            changed_chosen_bins = np.random.choice(list(range(8)), num, p=localized_change_size)
            for original in original_chosen_bins:
                localized_original_prob_li_by_time.append(original)
            for changed in changed_chosen_bins:
                localized_changed_prob_li_by_time.append(changed)

                num_of_counts_json['local_original_prob_by_time'].append(original.tolist())
                num_of_counts_json['local_changed_prob_by_time'].append(changed.tolist())
    num_of_counts_json['local_changed_prob'].append(percentage.tolist())

    fig, ax = plt.subplots()
    original_u, original_counts = np.unique(localized_original_prob_li_by_time, return_counts=True)
    changed_u, changed_counts = np.unique(localized_changed_prob_li_by_time, return_counts=True)

    width = 0.99
    rects2 = ax.bar(np.arange(8), changed_counts,  width=width, alpha=0.7, label='Increased', color='orange')
    rects1 = ax.bar(np.arange(8), original_counts, width=width, alpha=0.5, label='Decreased', color='grey')

    def autolabel(rects, counts):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height() - 4000
            ax.annotate("{:.2%}".format(height/sum(counts)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', color='k')

    # autolabel(rects1, original_counts)
    # autolabel(rects2, changed_counts)

    plt.xticks(np.arange(8), original_u)
    plt.ylabel('Counts')
    plt.xlabel('Bins')
    plt.ylim(0, 60000)
    plt.legend(loc='best')
    selected_str = ''
    for selected in selected_li:
        selected_str+=str(selected)+', '
        plt.title(str(int(percentage*100))+'% Changed in bin '+ selected_str+ " for "+str(int(total_dur_in_sec/1000))+' s')
    # plt.show()
    # save_fig_name = save_fig_path + datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")+' localized_change.png'
    # fig.savefig(save_fig_name)

    with open(save_num_of_counts_json_path, "w") as fp:
        json.dump(num_of_counts_json, fp)

    return localized_original_prob_li_by_time, localized_changed_prob_li_by_time, localized_change_size, selected_li

### Not used ###
# def bins_by_non_localized_change_size(percentage, poisson_num_bin_li, marginal_prob):
#     non_localized_change_size = marginal_prob.copy()
#     non_local_first_index_li = []
#     non_local_second_index_li = []
#     non_localized_change_idx = np.random.choice(list(range(8)))
#
#     for i, change_time_idx in enumerate(non_localized_change_size):
#         if i == non_localized_change_idx:
#             if i == 0:
#                 sec_idx = 7
#             elif i == 1:
#                 sec_idx = np.random.choice([4, 5, 7])
#             elif i == 2:
#                 sec_idx = np.random.choice([5, 6])
#             elif i == 3:
#                 sec_idx = np.random.choice([0, 6, 7])
#             elif i == 4:
#                 sec_idx = np.random.choice([1, 7])
#             elif i == 5:
#                 sec_idx = np.random.choice([1, 2])
#             elif i == 6:
#                 sec_idx = np.random.choice([0, 2, 3])
#             else:
#                 sec_idx = np.random.choice([1, 3, 4])
#
#             non_localized_change_size[i] = non_localized_change_size[i] + 0.125 * percentage
#             non_localized_change_size[sec_idx] = non_localized_change_size[sec_idx] + 0.125 * percentage
#
#             non_local_first_index_li.append(i)
#             non_local_second_index_li.append(sec_idx)
#
#             # num_of_counts_json['non_local_first_index'].append(float(i))
#             # num_of_counts_json['non_local_second_index'].append(float(sec_idx))
#
#     non_localized_original_prob_li_by_time = []
#     non_localized_changed_prob_li_by_time = []
#
#     normalized_changed_prob = non_localized_change_size / sum(non_localized_change_size)
#     for time in range(int(total_dur_in_sec / dur_in_sec)):
#         for num in poisson_num_bin_li:
#             original_chosen_bins = np.random.choice(list(range(8)), num, p=marginal_prob)
#             changed_chosen_bins = np.random.choice(list(range(8)), num, p=normalized_changed_prob)
#             for original in original_chosen_bins:
#                 non_localized_original_prob_li_by_time.append(original)
#             for changed in changed_chosen_bins:
#                 non_localized_changed_prob_li_by_time.append(changed)
#
#     # print("non_localized")
#     # print("dimension: ", normalized.shape)
#     # print("marginal_prob: ", marginal_prob)
#     # print("non_localized_change_size: ", non_localized_change_size)
#     # print("normalized: ", normalized)
#
#     # num_of_counts_json['non_local_original_prob_by_time'].append(original.tolist())
#     # num_of_counts_json['non_local_changed_prob_by_time'].append(changed.tolist())
#     # num_of_counts_json['non_local_changed_prob'].append(percentage.tolist())
#
#     # fig, ax = plt.subplots()
#     # sns.distplot(non_localized_original_prob_li_by_time, ax=ax, kde=False, label='original', bins=8)
#     # sns.distplot(non_localized_changed_prob_li_by_time, ax=ax, kde=False, label='changed', bins=8)
#     # fig.tight_layout()
#     # plt.ylabel('Counts')
#     # plt.xlabel('Bins')
#     # plt.legend(loc='lower right')
#     # plt.title(('[Non Localized] ' + str(int(percentage * 100)) + '% Changed in ' + str(int(total_dur_in_sec / 1000))) + ' s')
#     # #plt.show()
#     #
#     # save_fig_name = save_fig_path + datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p") + ' non_localized_change.png'
#     # fig.savefig(save_fig_name)
#
#     # with open(save_num_of_counts_json_path, "w") as fp:
#     #     json.dump(num_of_counts_json, fp)
#
#     return non_local_first_index_li, non_local_second_index_li

def bins_by_locally_changed_marginal_prob(localized_change_size, poisson_num_bin_li, marginal_prob):
    originally_chosen_bin_li = []
    for num in poisson_num_bin_li:
        chosen_bins = np.random.choice(list(range(8)), num, p=marginal_prob)

        originally_chosen_bin_li.append(chosen_bins)

    locally_changed_chosen_bin_li = []
    for num in poisson_num_bin_li:
        chosen_bins = np.random.choice(list(range(8)), num, p=localized_change_size)
        locally_changed_chosen_bin_li.append(chosen_bins)
        # num_of_counts_json['chosen_bins'].append(list(chosen_bins.astype(float)))
        # num_of_counts_json['chosen_bin_counts'].append(len(chosen_bins))
    #print('locally_changed_chosen_bin_li: ', locally_changed_chosen_bin_li)

    # with open(save_num_of_counts_json_path, "w") as fp:
    #     json.dump(num_of_counts_json, fp)
    # # print(chosen_bin_li)
    return originally_chosen_bin_li, locally_changed_chosen_bin_li

def original_time_and_freq_plot(length_to_split, originally_chosen_bin_li):
    original_frame = np.zeros((n_tones, int(total_dur_in_sec/dur_in_sec)))

    plot_first_index_li = []
    plot_second_index_li = []

    for time in range(int(total_dur_in_sec/dur_in_sec)):
        for bin in originally_chosen_bin_li[time]:
            ## Added each index by 1 when slicing
            if bin == 0:
                first_idx = bin
                second_idx = list(range(26))[(np.cumsum(length_to_split)-1)[bin]]+1
            else:
                first_idx = list(range(26))[(np.cumsum(length_to_split) - 1)[bin-1]]+1
                second_idx = list(range(26))[(np.cumsum(length_to_split) - 1)[bin]]+1

            plot_first_index_li.append(first_idx)
            plot_second_index_li.append(second_idx)
            original_frame[first_idx+1:second_idx+1, time] = 1

    #         num_of_counts_json['plot_first_index'].append(float(first_idx))
    #         num_of_counts_json['plot_second_index'].append(float(second_idx))
    #
    # fig, ax = plt.subplots()
    # plt.imshow(original_frame)
    # plt.title("Bins by Original Marginal Prob")
    # plt.show()
    #
    # save_fig_name = save_fig_path + datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")+' Original time_and_freq.png'
    # fig.savefig(save_fig_name)
    #
    # with open(save_num_of_counts_json_path, "w") as fp:
    #     json.dump(num_of_counts_json, fp)

    return original_frame, plot_first_index_li, plot_second_index_li

def changed_time_and_freq_plot(length_to_split, locally_changed_chosen_bin_li):
    changed_frame = np.zeros((n_tones, int(total_dur_in_sec/dur_in_sec)))

    changed_plot_first_index_li = []
    changed_plot_second_index_li = []

    for time in range(int(total_dur_in_sec/dur_in_sec)):
        for bin in locally_changed_chosen_bin_li[time]:
            ## Added each index by 1 when slicing
            if bin == 0:
                second_idx = list(range(26))[(np.cumsum(length_to_split)-1)[bin]]+1
            else:
                first_idx = list(range(26))[(np.cumsum(length_to_split) - 1)[bin-1]]+1
                second_idx = list(range(26))[(np.cumsum(length_to_split) - 1)[bin]]+1

                changed_plot_first_index_li.append(first_idx)
                changed_plot_second_index_li.append(second_idx)
                changed_frame[first_idx + 1:second_idx + 1, time] = 1

            # num_of_counts_json['changed_time_and_freq_plot_first_index'].append(float(first_idx))
            # num_of_counts_json['changed_time_and_freq_plot_second_index'].append(float(second_idx))

    #fig =plt.figure()
    #plt.imshow(changed_frame)
    #plt.title("Bins by Locally Changed Marginal Prob, Increased by "+str(percentage))
    # plt.show()
    #save_fig_name = save_fig_path + datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")+' Changed_prob ' +'by'+str(percentage)+'.png'
    #fig.savefig(save_fig_name)

    # with open(save_num_of_counts_json_path, "w") as fp:
    #     json.dump(num_of_counts_json, fp)

    return changed_frame, changed_plot_first_index_li, changed_plot_second_index_li

def change_time_by_exponential_dist(total_dur_in_sec, dur_in_sec):
    rv = expon()
    expon_distribution = np.linspace(0, np.minimum(rv.dist.b, total_dur_in_sec/1000), int(total_dur_in_sec/dur_in_sec))
    #fig, ax = plt.subplots()
    #plt.plot(expon_distribution, rv.pdf(expon_distribution))
    #plt.title("Change Time by Exponential Probability" + " (mean {0:.2f} s)".format(np.mean(expon_distribution)))
    # plt.show()

    #save_fig_name = save_fig_path + datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")+"Change Time by Exponential Probability" + " (mean {0:.2f} s)".format(np.mean(expon_distribution))+'.png'
    #fig.savefig(save_fig_name)

    # num_of_counts_json['exponential_change_time'].append(list(expon_distribution.astype(float)))
    # with open(save_num_of_counts_json_path, "w") as fp:
    #     json.dump(num_of_counts_json, fp)

    return expon_distribution

def concatenate_change(length_to_split, selected_li, n_trial, expon_distribution, locally_changed_chosen_bin_li, percentage):
    normalized_expon = expon.pdf(expon_distribution)/np.sum(expon.pdf(expon_distribution))
    change_detection_label = []
    stim_img = np.zeros((n_tones, int(total_dur_in_sec / dur_in_sec), n_trial))
    for trial in range(n_trial):
        change = np.random.choice(list(range(int(total_dur_in_sec / dur_in_sec))), p=normalized_expon)
        change_time = (change+1)*dur_in_sec # 3180, 1140, 1080...
        change_detection_label.append(int(change_time))

        if change_time > 0:
            for change_time_idx in range(int(change_time / dur_in_sec - 1), int(total_dur_in_sec / dur_in_sec)):
                for changed_bin in locally_changed_chosen_bin_li[change_time_idx]:
                    ## Added each index by 1 when slicing
                    if changed_bin == 0:
                        changed_first_idx = changed_bin
                        changed_second_idx = list(range(26))[(np.cumsum(length_to_split)-1)[changed_bin]]+1
                    else:
                        changed_first_idx = list(range(26))[(np.cumsum(length_to_split) - 1)[changed_bin-1]]+1
                        changed_second_idx = list(range(26))[(np.cumsum(length_to_split) - 1)[changed_bin]]+1

                    stim_img[changed_first_idx :changed_second_idx , change_time_idx, trial] = 1

        for original_time_idx in range(0, int(change_time / dur_in_sec - 1)+1):
            for original_bin in originally_chosen_bin_li[original_time_idx]:
                if original_bin == 0:
                    original_first_idx = original_bin
                    original_second_idx = list(range(26))[(np.cumsum(length_to_split) - 1)[original_bin]] + 1
                else:
                    original_first_idx = list(range(26))[(np.cumsum(length_to_split) - 1)[original_bin - 1]] + 1
                    original_second_idx = list(range(26))[(np.cumsum(length_to_split) - 1)[original_bin]] + 1

                stim_img[original_first_idx :original_second_idx , original_time_idx, trial] = 1

        fig, ax = plt.subplots(figsize=(8, 5))
        bounds = [0, 0.5, 1]
        cmap = colors.ListedColormap(['white', 'black'])
        norm = colors.BoundaryNorm(bounds, cmap.N)
        im = plt.imshow(stim_img, interpolation='nearest', aspect='auto', cmap=cmap, norm=norm)
        cbar = plt.colorbar(im, ticks=range(2), boundaries=bounds, pad=0.02)
        plt.subplots_adjust(left=0.1, right=0.2, top=0.2, bottom=0.1)
        cbar.set_label('Binary Color')
        fig.tight_layout()
        selected_str = ''

        for selected in selected_li:
            selected_str+=str(selected)+', '
        ax.axhspan(np.cumsum(length_to_split)[selected_li[0]-1]-0.45, np.cumsum(length_to_split)[selected_li[1]]-0.45, color='yellow', alpha=0.5)
        title = "Stimuli Increased by " + "{:.0%}".format(percentage) + ' in bin ' + selected_str +' at ' + str(original_time_idx*30/1000) + 's (trial ' + str(trial)+'/'+str(n_trial)+')'
        plt.title(title)

        save_fig_name = save_fig_path+datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")+"_Stimuli_"+str(percentage)+ '_at_time_index_' + str(original_time_idx)+'_(trial_'+ str(trial)+'out of'+str(n_trial)+')'+'.png'
        plt.axvline(x=original_time_idx, color='r', linewidth=2)
        plt.axvspan(0, original_time_idx, color='gray', alpha=0.5)
        plt.yticks(np.arange(26))
        plt.xticks((0, 25, 50, 75, 100, 125, 150, 175, 200), (str(i*dur_in_sec) for i in [0, 25, 50, 75, 100, 125, 150, 175, 200]))
        plt.ylabel('Semitones')
        plt.xlabel('Time (s)')
        fig.tight_layout()  # , cmap=plt.cm.Greens)
        #plt.show()
        #fig.savefig(save_fig_name, dpi=300, bbox_inches='tight')

    num_of_counts_json['stim_img'].append(stim_img.tolist())
    num_of_counts_json['trial'].append(float(trial))
    num_of_counts_json['change_detection_label'].append(change_detection_label)
    num_of_counts_json['change_percentage'].append(percentage)

    with open(save_num_of_counts_json_path, "w") as fp:
        json.dump(num_of_counts_json, fp)

    with open(pickle_path, 'wb') as f:
        pkl.dump(stim_img, f)

    with open(pickle_path, 'rb') as f:
        x = pkl.load(f)

    return stim_img, change_detection_label, percentage

def make_tone_cloud(stim_img, semitone_li, fs):
    mat = np.transpose(stim_img) * semitone_li
    mat_by_tone = np.transpose(mat)
    time = np.arange(0, dur_in_sec, 1/fs)
    total_cosine = []
    for time_point in range(mat_by_tone.shape[1]):
        chord = np.cos(np.pi * 2 * time)
        for tone in mat_by_tone[:, time_point]:
            cosine = np.cos(np.pi * 2 * tone * time)
            chord += cosine
        total_cosine.extend(chord)  # len: 8800
    #print(total_cosine)
    return total_cosine

def ramp(ramp_dur, fs, total_cosine, ramped_wave_path):
    n_steps = np.floor(ramp_dur*fs)
    #print(n_steps)  #1102.0
    time = np.array(np.arange(0, n_steps)).T/fs
    #print(time) #1102
    framp = 1/(2*ramp_dur)
    #print(framp) #20
    ramp = 0.5-0.5*np.cos(2*np.pi*framp*time)
    #print(len(ramp)) # 1102
    total_cosine[0:int(n_steps)] = np.multiply(ramp, total_cosine[0:int(n_steps)])
    total_cosine[-int(n_steps)-1:-1] = np.multiply(np.flipud(ramp), total_cosine[-int(n_steps)-1:-1])

    scaled = np.int16(total_cosine / np.max(np.abs(total_cosine))* 32767)
    wavf.write(ramped_wave_path, int(fs), scaled)
    return total_cosine

#https://github.com/mcdermottLab/pycochleagram/blob/master/pycochleagram/demo.py
if utils.check_if_display_exists():
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import imshow, show
else:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import imshow, show

### Cochleagram Generation from Waveform ###
def demo_human_cochleagram(n=40):
    """Demo to generate the human cochleagrams, displaying various nonlinearity
    and downsampling options. If a signal is not provided, a tone synthesized
    with 40 harmonics and an f0=100 will be used.

    Args:
      signal (array, optional): Signal containing waveform data.
      sr (int, optional): Sampling rate of the input signal.
      n (int, optional): Number of filters to use in the filterbank.

    Returns:
      None
    """
    # get a signal if one isn't provided
    sr, signal = wavf.read(total_wave_path)

    if signal is None:
        signal, signal_params = make_harmonic_stack()
        sr = signal_params['sr']
        n = signal_params['n']
    else:
        assert sr is not None
        assert n is not None

    ### Demo Cochleagram Generation with Predefined Nonlinearities ###
    # no nonlinearity
    coch = demo_human_cochleagram_helper(signal, sr, n, nonlinearity=None)
    # convert to decibel
    coch_log = demo_human_cochleagram_helper(signal, sr, n, nonlinearity='db')
    # 3/10 power compression
    coch_pow = demo_human_cochleagram_helper(signal, sr, n, nonlinearity='power')

    plt.subplot(321)
    plt.title('Signal waveform')
    plt.plot(signal)
    #plt.xticks(np.arange(0, len(signal)), np.linspace(0, total_dur_in_sec, len(signal)))
    plt.ylabel('amplitude')
    plt.xlabel('time')

    plt.subplot(323)
    plt.title('Signal Frequency Content')
    f, Pxx_den = welch(signal.flatten(), sr, nperseg=1024)
    plt.semilogy(f, Pxx_den)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')

    plt.subplot(322)
    plt.title('Cochleagram with no nonlinearity')
    plt.ylabel('filter #')
    plt.xlabel('time')
    utils.cochshow(np.flipud(coch), interact=False)
    plt.gca().invert_yaxis()

    plt.subplot(324)
    plt.title('Cochleagram with nonlinearity: "log"')
    plt.ylabel('filter #')
    plt.xlabel('time')
    utils.cochshow(np.flipud(coch_log), interact=False)
    plt.gca().invert_yaxis()
    print("coch_log:", coch_log.shape) #coch_log: (85, 264600)

    plt.subplot(326)
    plt.title('Cochleagram with nonlinearity: "power"')
    plt.ylabel('filter #')
    plt.xlabel('time')
    utils.cochshow(np.flipud(coch_pow), interact=False)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    ### Demo Cochleagram Generation with Downsampling ###
    plt.figure()
    # no downsampling
    # cochd = demo_human_cochleagram_helper(signal, sr, n, downsample=None)
    # predefined polyphase resampling with upsample factor = 10000, downsample factor = `sr`
    cochd_poly = demo_human_cochleagram_helper(signal, sr, n, downsample=10000)
    # custom downsampling function to use decimate with a downsampling factor of 2
    custom_downsample_fx = lambda x: decimate(x, 2, axis=1, ftype='fir', zero_phase=True)
    cochd_decimate = demo_human_cochleagram_helper(signal, sr, n, downsample=custom_downsample_fx)

    plt.subplot(221)
    plt.title('Signal waveform')
    plt.plot(np.linspace(0, total_dur_in_sec, len(signal)), signal)
    #plt.xticks(np.arange(0, len(signal)), np.linspace(0, total_dur_in_sec, len(signal)))
    plt.ylabel('amplitude')
    plt.xlabel('time')

    plt.subplot(223)
    plt.title('Signal Frequency Content')
    f, Pxx_den = welch(signal.flatten(), sr, nperseg=1024)
    plt.semilogy(f, Pxx_den)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')

    plt.subplot(222)
    plt.title('Cochleagram with 2x default\n(polyphase) downsampling')
    plt.ylabel('filter #')
    plt.xlabel('time')
    #plt.xticks(np.arange(0, len(signal)), np.linspace(0, total_dur_in_sec, len(signal)))
    utils.cochshow(np.flipud(cochd_poly), interact=False)
    plt.gca().invert_yaxis()

    plt.subplot(224)
    plt.title('Cochleagram with 2x custom\n(decimate) downsampling')
    plt.ylabel('filter #')
    plt.xlabel('time')
    utils.cochshow(np.flipud(cochd_decimate), interact=False)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def demo_human_cochleagram_helper(signal, sr, n=40, sample_factor=2, downsample=None, nonlinearity=None):
    """Demo the cochleagram generation.

      signal (array): If a time-domain signal is provided, its
        cochleagram will be generated with some sensible parameters. If this is
        None, a synthesized tone (harmonic stack of the first 40 harmonics) will
        be used.
      sr: (int): If `signal` is not None, this is the sampling rate
        associated with the signal.
      n (int): number of filters to use.
      sample_factor (int): Determines the density (or "overcompleteness") of the
        filterbank. Original MATLAB code supported 1, 2, 4.
      downsample({None, int, callable}, optional): Determines downsampling method to apply.
        If None, no downsampling will be applied. If this is an int, it will be
        interpreted as the upsampling factor in polyphase resampling
        (with `sr` as the downsampling factor). A custom downsampling function can
        be provided as a callable. The callable will be called on the subband
        envelopes.
      nonlinearity({None, 'db', 'power', callable}, optional): Determines
        nonlinearity method to apply. None applies no nonlinearity. 'db' will
        convert output to decibels (truncated at -60). 'power' will apply 3/10
        power compression.

      Returns:
        array:
          **cochleagram**: The cochleagram of the input signal, created with
            largely default parameters.
    """
    sr, signal = wavf.read(total_wave_path)

    human_coch = cgram.human_cochleagram(signal, sr, n=n, sample_factor=sample_factor,
                                         downsample=downsample, nonlinearity=nonlinearity, strict=False)
    img = np.flipud(human_coch)  # the cochleagram is upside down (i.e., in image coordinates)

    return img

if __name__ == "__main__":
    semitone_li = semitone(f0, n_tones)
    length_to_split, tones_into_bins_li, semitone_num_idx_li = tones_into_bins(semitone_li)
    poisson_num_bin_li = poisson_num_bins(total_dur_in_sec)
    chosen_bin_li = bins_by_marginal_prob(poisson_num_bin_li, marginal_prob)
    localized_original_prob_li_by_time, localized_changed_prob_li_by_time, localized_change_size, selected_li = bins_by_localized_changed_marginal_prob(percentage, poisson_num_bin_li, marginal_prob)
    #non_local_first_index_li, non_local_second_index_li = bins_by_non_localized_change_size(percentage, poisson_num_bin_li, marginal_prob)
    originally_chosen_bin_li, locally_changed_chosen_bin_li = bins_by_locally_changed_marginal_prob(localized_change_size, poisson_num_bin_li, marginal_prob)
    original_frame, plot_first_index_li, plot_second_index_li = original_time_and_freq_plot(length_to_split, originally_chosen_bin_li)
    changed_frame, changed_plot_first_index_li, changed_plot_second_index_li = changed_time_and_freq_plot(length_to_split, locally_changed_chosen_bin_li)
    expon_distribution = change_time_by_exponential_dist(total_dur_in_sec, dur_in_sec)
    stim_img, change_detection_label, percentage = concatenate_change(length_to_split, selected_li, n_trial, expon_distribution, locally_changed_chosen_bin_li, percentage)
    total_cosine = make_tone_cloud(stim_img, semitone_li, fs)
    ramped_cosine = ramp(ramp_dur, fs, total_cosine, ramped_wave_path)
    demo_human_cochleagram(n=40)
