#%%
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import matplotlib.pyplot as plt
import hashlib

# Function to generate a time series instance
def generate_time_series(length):
    return np.random.rand(length)

def perturb_mean(original_ts, start_index, end_index):
    new_signal = original_ts.copy()
    new_signal[start_index:end_index] = np.mean(new_signal[start_index:end_index])
    return new_signal

def perturb_zero(original_ts, start_index, end_index):
    new_signal = original_ts.copy()
    new_signal[start_index:end_index] = 0
    return new_signal

def perturb_total_mean(original_ts, start_index, end_index):
    new_signal = original_ts.copy()
    new_signal[start_index:end_index] = new_signal.mean()
    return new_signal
def perturb_random(original_ts, start_index, end_index):
    new_signal = original_ts.copy()
    # Generate random values to replace the segment
    num_values = end_index - start_index
    random_values = np.random.rand(num_values) * (original_ts.max() - original_ts.min()) + original_ts.min() # Generates random number within the range of the original time series
    new_signal[start_index:end_index] = random_values
    return new_signal

# Function to perturb the time series by randomly masking some parts from the original time series
def perturb_time_series(original_ts, num_masked_segments, max_consecutive_steps, replacement_method):
     perturbed_ts = original_ts.copy()
     
     
     for _ in range(num_masked_segments):
        # Randomly determine the number of consecutive steps
        num_consecutive_steps = np.random.randint(2, max_consecutive_steps + 1)
        
        # Randomly choose the starting position for consecutive masking
        start_index = np.random.randint(0, len(original_ts)-num_consecutive_steps)
        
        # Mask the consecutive steps
        end_index = start_index + num_consecutive_steps
        
        if replacement_method == 'zero':
            perturbed_ts = perturb_zero(original_ts, start_index, end_index)
        elif replacement_method == 'mean':
            perturbed_ts = perturb_mean(original_ts, start_index, end_index)
        elif replacement_method == 'total_mean':
            perturbed_ts = perturb_total_mean(original_ts, start_index, end_index)
        elif replacement_method =='random':
            perturbed_ts = perturb_random(original_ts, start_index, end_index)
            
          
        #perturbed_ts[start_index:start_index + num_consecutive_steps] = 0   
     return perturbed_ts

# Function to compute DTW distance between two time series
def compute_dtw_distance(original_ts, perturbed_ts):
    distance, _ = fastdtw(original_ts, perturbed_ts)
    return distance

def hash_ts(ts):
    return hashlib.md5(ts.astype(np.float32).tobytes()).hexdigest()

# Function to generate multiple perturbations and compute DTW distances
def generate_perturbations(original_time_series, num_perturbations, replacement_method='zero'):
    
    perturbed_list = []
    distances = []
    unique_hashes = set()
    for _ in range(num_perturbations ):
        # Randomly determine the max number of masked segments and max consecutive steps for each perturbation
        # num_masked_segments = np.random.randint(1, 6)
        # max_step = int(0.4 * len(original_time_series))
        # # print(max_step)
        # max_consecutive_steps = np.random.randint(5, max_step)
        # print(f'num_masked_segments_range: {num_masked_segments}')
        # min_consecutive_steps = np.arange(2, 6)
        # print(f'max_consecutive_steps_range : {max_consecutive_steps }')
        num_masked_segments = np.random.randint(1, 6)
        max_consecutive_steps = np.random.randint(5, int(0.4 * len(original_time_series)))
        # Generate a perturbed time series
        perturbed_ts = perturb_time_series(original_time_series, num_masked_segments, max_consecutive_steps, replacement_method)
        # Append perturbed time series to the list
        # ts_hash = hash_ts(perturbed_ts)

        # if ts_hash in unique_hashes:
        #     continue  # skip duplicates
        perturbed_list.append(perturbed_ts)
        # if len(perturbed_list) >= num_perturbations:
        #     break
    # Insert original instance at the beginning of perturbed instances
    perturbed_list.insert(0, original_time_series)

    # Compute distances between the original instance and the perturbed instances
    distances = [compute_dtw_distance(original_time_series, perturbed_time_series) for perturbed_time_series in perturbed_list]

    return perturbed_list, distances



def kernel(distances, kernel_width):
    return np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))


def plot_ts(original_ts, perturbed_ts):
    # Plot the original and perturbed time series
    plt.plot(original_ts, label='Original Time Series', marker='o')
    plt.plot(perturbed_ts, label='Perturbed Time Series', marker='x')
    plt.legend()
    plt.show()

'''
# Number of data points in each time series
length_of_time_series = 100

# Generate an original time series
original_time_series = generate_time_series(length_of_time_series)

# Generate a perturbed time series with randomly masked consecutive time steps
num_masked_segments = 4
min_consecutive_steps = 6
max_consecutive_steps = 10
perturbed_time_series = perturb_time_series(original_time_series, num_masked_segments, max_consecutive_steps)

# Compute DTW distance between the original and perturbed time series
distance, path = fastdtw(original_time_series, perturbed_time_series)

print(f"DTW Distance: {distance}")

# Plot the original and perturbed time series
plt.plot(original_time_series, label='Original Time Series', marker='o')
plt.plot(perturbed_time_series, label='Perturbed Time Series', marker='x')
plt.legend()
plt.show()

'''

# %%
