import math
import sys
from functools import reduce

import librosa as librosa
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram, find_peaks
import wave


globalVAR = 0


def write_array_to_file(array):
    global globalVAR
    with open('output.txt' + str(globalVAR), 'w') as f:
        globalVAR = globalVAR + 1
        for item in array:
            f.write("%s\n" % item)


def sum_every_256_elements(myList, frame_size):
    sublists = [myList[j:j + frame_size] for j in range(0, len(myList), frame_size)]

    sums = [reduce(lambda a, b: a + b, sublist) for sublist in sublists]

    return sums


def divide_by_n_root(reduced_squares, frame_size_var):
    def sqrt_div_frame_size(x):
        return math.sqrt(x / frame_size_var)

    return list(map(sqrt_div_frame_size, reduced_squares))


def make_array_for_rms(amplitudes, frame_size):

    squared_amplitudes = np.array(amplitudes, dtype=np.float64)#avoid int overflow
    squared_amplitudes = np.square(squared_amplitudes)
    write_array_to_file(amplitudes)
    write_array_to_file(squared_amplitudes)
    # print(len(squared_amplitudes))
    # print(squared_amplitudes)
    reduced_squares = sum_every_256_elements(squared_amplitudes, frame_size)

    write_array_to_file(reduced_squares)
    # print(len(reduced_squares))
    # print(reduced_squares)
    reduced_squares = divide_by_n_root(reduced_squares, frame_size)
    write_array_to_file(reduced_squares)
    # print(reduced_squares)
    return reduced_squares


def zero_crossing_rate1(signal):
    new_array = []
    for j, single_signal in enumerate(signal):
        if j == 0:
            new_array.append(0)
            continue
        if np.sign(signal[j-1]) != np.sign(signal[j]):
            new_array.append(1)
        else:
            new_array.append(0)

    return new_array


def calculate_zcr(signal, frame_size):
    zcr = []
    for i in range(0, len(signal) - frame_size, frame_size):
        frame = signal[i:i+frame_size]
        zcr_count = np.sum(np.abs(np.diff(np.sign(frame))) / 2)
        zcr.append(zcr_count)
    return zcr


def pad_array_with_0(amplitudes):
    padded_array = []
    len_ampli = len(amplitudes)
    ret_len = pow(2, math.ceil(math.log(len_ampli) / math.log(2)))
    # print(len_ampli)
    # print(ret_len)
    for index in range(ret_len):
        if index < len_ampli:
            padded_array.append(amplitudes[index])
        else:
            padded_array.append(0)
    return padded_array


def fft_radix_2_dit(array):

    N = len(array)

    if N == 1:
        return array

    else:
        X_even = fft_radix_2_dit(array[0::2])
        X_odd = fft_radix_2_dit(array[1::2])

        X = np.zeros(N, dtype=complex)

        for m in range(N):
            #find output array for m frequencies, will have to transpose this array later
            m_alias = m % (N // 2)
            X[m] = X_even[m_alias] + np.exp(-2j * np.pi * m / N) * X_odd[m_alias]

    return X


def my_stft(signal, window_size=512, hop_size=256):
    stft_result = []

    num_freq_bins = window_size // 2 + 1

    hamming_window = np.hamming(window_size)

    for i in range(0, len(signal) - window_size + 1, hop_size):
        windowed_signal = signal[i:i + window_size] * hamming_window
        spectrum = fft_radix_2_dit(windowed_signal)

        stft_result.append(spectrum[:num_freq_bins])

    return np.array(stft_result)


def plot_spectrogram(prepared_stft_array, title, y_axis='linear'):

    plt.figure(figsize=(10, 6))
    plt.imshow(prepared_stft_array, aspect='auto', origin='lower', cmap='viridis', interpolation='none')

    plt.xlabel('Time Frame')
    plt.ylabel('Frequency Bin')
    plt.title('Spectrogram')

    if y_axis == 'log':
        plt.yscale('log')
        plt.ylabel('Frequency HZ - Logarithmic Scale')
    else:
        plt.ylabel('Frequency HZ')

    plt.colorbar(label='Decibels')
    plt.title(title)

    plt.show()

def matplotlib_spectrogram(stft_array):
    plt.figure(figsize=(12, 8))

    for i in range(stft_array.shape[1]):
            _, _, Sxx, _ = plt.specgram(stft_array[:, i], NFFT=512, Fs=2, noverlap=256, cmap='viridis', aspect='auto')

    plt.xlabel('Time Frame')
    plt.ylabel('Frequency Bin')
    plt.title('Spectrogram')
    plt.colorbar(label='Intensity (dB)')

    plt.show()

def plot_spectrogram1(Y, sr, hop_length, title, y_axis="linear"):
    plt.figure(figsize=(18, 9))
    times = librosa.times_like(Y, sr=sr, hop_length=hop_length)
    librosa.display.specshow(Y,
                             sr=sr,
                             hop_length=hop_length,
                             x_axis="time",
                             y_axis=y_axis,
                             x_coords=times)

    plt.colorbar(format="%+2.f")
    plt.title(title)
    plt.show()

def estimate_fundamental_frequency(fft_result, sampling_rate):
    peaks, _ = find_peaks(np.abs(fft_result))

    frequencies = np.fft.fftfreq(len(fft_result), 1 / sampling_rate)

    fundamental_frequency_index = peaks[np.argmax(np.abs(fft_result[peaks]))]
    fundamental_frequency = frequencies[fundamental_frequency_index]

    return fundamental_frequency


def plot_magnitude_spectrum(fft, sr, title, f_ratio=0.01):
    X_mag = np.absolute(fft)
    plt.figure(figsize=(18, 5))

    f = np.linspace(0, sr, len(X_mag))
    f_bins = int(len(X_mag) * f_ratio)

    plt.plot(f[:f_bins], X_mag[:f_bins])
    plt.xlabel('Frequency (Hz)')
    plt.title(title)

    fundamental_frequency = estimate_fundamental_frequency(fft, sr)
    plt.axvline(x=fundamental_frequency, color='r', linestyle='--', label='Fundamental Frequency')

    plt.title(title)
    plt.legend()
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    wav = wave.open('my_talking.wav', 'rb')
    samples = wav.readframes(wav.getnframes())

    n_samples = 256
    frame_size = n_samples * wav.getnchannels()
    # print(frames)
    # print(wav.getnchannels())
    amplitudes = np.frombuffer(samples, dtype=np.int16)
    # print(len(amplitudes))

    time_basic = np.arange(0, len(amplitudes)) / (wav.getframerate() * wav.getnchannels())

    sampling_rate = wav.getframerate()

    # print(amplitudes)

    # 1. Анализа у временском домену
    # 1.1. Приказ таласног облика
    plt.figure(figsize=(10, 4))
    plt.plot(time_basic, amplitudes)
    plt.title('Talasni oblik')
    plt.xlabel('Time')
    plt.show()

    # 1.2. Анализа амплитуде
    # RMS график


    rms = make_array_for_rms(amplitudes, frame_size)
    time_rms = np.arange(0, len(rms)) * frame_size / (wav.getframerate()*wav.getnchannels())
    # print("LEN RMS")
    # print(len(rms))
    # print(len(time_rms))

    # MAX & MIN
    max_index_rms = np.argmax(rms)
    min_index_rms = np.argmin(rms)

    max_time_rms = time_rms[max_index_rms]
    min_time_rms = time_rms[min_index_rms]

    plt.figure(figsize=(10, 4))
    plt.plot(time_rms, rms)
    plt.scatter(max_time_rms, rms[max_index_rms], color='red',
                label=f'Max Point RMS ({max_time_rms:.2f} s)', zorder=2)
    plt.scatter(min_time_rms, rms[min_index_rms], color='green',
                label=f'Min Point RMS ({min_time_rms:.2f} s)', zorder=2)
    plt.title('RMS grafik')
    plt.xlabel('Time')
    plt.ylabel('RMS')
    plt.show()

    max_index_amplitude = np.argmax(amplitudes)
    min_index_amplitude = np.argmin(amplitudes)
    closest_to_zero_index = np.argmin(np.abs(amplitudes))

    max_time_amplitude = time_basic[max_index_amplitude]
    min_time_amplitude = time_basic[min_index_amplitude]
    closest_to_zero_amplitude = time_basic[closest_to_zero_index]

    plt.figure(figsize=(10, 6))
    plt.plot(time_basic, amplitudes)
    plt.scatter(max_time_amplitude, amplitudes[max_index_amplitude], color='red', label=f'Max Point ({max_time_amplitude:.2f} s)', zorder=2)
    plt.scatter(min_time_amplitude, amplitudes[min_index_amplitude], color='green', label=f'Min Point ({min_time_amplitude:.2f} s)', zorder=2)
    plt.scatter(closest_to_zero_amplitude, amplitudes[closest_to_zero_index], color='orange',label=f'Min ABS Point - min AMPLITUDE ({closest_to_zero_amplitude:.2f} s)', zorder=2)
    plt.title('Audio Waveform with Max and Min Points')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

    #Calculate and output the avg amplitude to console
    avg_amplitude = sum(abs(amplitudes))/len(amplitudes)
    avg_amplitude1 = np.mean(np.abs(amplitudes))
    print("PROSECNA AMPLITUDA JE:")
    print(avg_amplitude)

    # 1.3. Обележавање почетка и краја звука

    # Проналажење граница звука

    background_noise_level = np.mean(rms[:100])
    # overlap = 128  # 50% overlap
    # background_noise_level = np.mean(amplitudes[:1000])
    threshold_multiplier = 1.7#Threshold
    sound_start = 0
    sound_end = 0
    in_sound = False
    array_of_starts_and_ends = []

    # Apply a simple moving average for smoothing, for every 5 rms values
    smoothed_rms = []
    window_size = 5
    for i in range(len(rms)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(rms), i + window_size // 2 + 1)
        smoothed_value = np.mean(rms[start_idx:end_idx])
        smoothed_rms.append(smoothed_value)

    # Detect speech segments
    for i, rms_val in enumerate(smoothed_rms):
        if rms_val > (background_noise_level * threshold_multiplier) and not in_sound:
            sound_start = i * frame_size/wav.getnchannels() # u pravljenju rms smo vec ukljucili to da idu 2 kanala tkd ovde moramo da delimo sa brojem kanala
            in_sound = True
        elif rms_val <= (background_noise_level * threshold_multiplier) and in_sound:
            sound_end = i * frame_size/wav.getnchannels()
            in_sound = False
            array_of_starts_and_ends.append((sound_start, sound_end))

    write_array_to_file(smoothed_rms)

    plt.figure(figsize=(10, 6))
    plt.plot(time_basic, amplitudes, label='Audio Signal') #mnozimo vreme sa brojem kanala doduse kod je napisan za 2 kanala tkd..
    # for start, end in merged_starts_and_ends:
    #     plt.scatter(time_basic[start], amplitudes[start], color='red', zorder=2)
    #     plt.scatter(time_basic[end], amplitudes[end], color='green', zorder=2)
    for start, end in array_of_starts_and_ends:
        plt.axvline(x=start / wav.getframerate(), color='green', linestyle='--')
        plt.axvline(x=end / wav.getframerate(), color='red', linestyle='--')
    plt.title('Sound Start and Sound End')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

    # 1.4. Стопа преласка преко нуле
    zc_rate = zero_crossing_rate1(amplitudes)
    zc_rate = sum_every_256_elements(zc_rate, frame_size)

    zc_time = np.arange(0, len(zc_rate)) * frame_size / (wav.getframerate() * wav.getnchannels())

    plt.figure(figsize=(10, 4))
    plt.plot(zc_time, zc_rate, color='blue', label='Zero Crossing Rate')
    plt.title('Zero Crossing Rate')
    plt.xlabel('Time')
    plt.ylabel('Rate')
    plt.xticks(np.arange(0, zc_time[-1], 0.25))
    plt.legend()
    plt.show()
    # 2. Анализа у фреквентном домену
    # 2.1. Приказ спектрограма

    padded_amplitudes = pad_array_with_0(amplitudes)
    padded_amplitudes_1 = padded_amplitudes

    write_array_to_file(padded_amplitudes)

    # MY IMPLEMENTATION OF FFT AND STFT

    my_stft_array = my_stft(padded_amplitudes_1)
    my_stft_array = my_stft_array.T  # or result = np.transpose(result)
    my_fft = fft_radix_2_dit(padded_amplitudes_1)
    write_array_to_file(my_fft)
    write_array_to_file(my_stft_array)

    print("ANALIZA 1-duzina mog ffta, 2-duzina mog stft niza, 3-shape mog stft nize, 4-tip podataka u mom stft nizu")
    print(len(my_fft))
    print(len(my_stft_array))
    print(my_stft_array.shape)
    print(type(my_stft_array[0][0]))

    #calculate power

    my_stft_scale = np.abs(my_stft_array) ** 2

    #convert do decibels - dB
    epsilon = 1e-10
    plot_spectrogram(20 * np.log10(np.maximum(my_stft_scale, epsilon)), "Moj spectrogram sa mojim implementacijama")
    plot_spectrogram1(20 * np.log10(np.maximum(my_stft_scale, epsilon)), sampling_rate, 256,
                      "Spectrogram sa casa sa mojim implementacijama", y_axis='log')
    # matplotlib_spectrogram(20 * np.log10(np.maximum(my_stft_scale, epsilon)))

    # NUMPY IMPLEMENTATION OF FFT AND STFT

    np_stft_array = librosa.stft(librosa.load("my_talking.wav")[0], n_fft=512, hop_length=256)
    numpy_fft = np.fft.fft(padded_amplitudes)
    write_array_to_file(numpy_fft)
    write_array_to_file(np_stft_array)

    enemy_stft_scale = np.abs(np_stft_array) ** 2
    plot_spectrogram(librosa.power_to_db(enemy_stft_scale), "Moj spectrogram sa implementacijama biblioteka")

    plot_spectrogram1(librosa.power_to_db(enemy_stft_scale), sampling_rate, 256,
                      "Spectrogram sa casa sa implementacijama biblioteka", y_axis='log')

    # matplotlib.pyplot.specgram(enemy_stft_scale)


    # frequencies, times, Sxx = spectrogram(amplitudes, wav.getframerate())
    # plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx))
    # plt.title('Спектрограм')
    # plt.show()

    # 2.2. Пронаћи основну фреквенцију

    plot_magnitude_spectrum(my_fft, sampling_rate, "My FFT that shows fundamental frequency", 0.01)
    plot_magnitude_spectrum(numpy_fft, sampling_rate, "Numpy FFT that shows fundamental frequency",  0.01)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
