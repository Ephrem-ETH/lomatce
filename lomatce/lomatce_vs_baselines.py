#%%
import os
import numpy as np
import logging
import platform
import pathlib
import argparse
import shap
from tabulate import tabulate
from tsai.all import *
from fastai.vision.all import *
from captum.attr import IntegratedGradients
from lime_timeseries import LimeTimeSeriesExplainer
from sklearn.metrics import accuracy_score, f1_score
from  lomatce.explainer import LomatceExplainer
import lomatce.utils.test_dataloader as test_loader
from tabulate import tabulate
from joblib import Parallel, delayed, cpu_count
from logging.handlers import QueueHandler, QueueListener
import multiprocessing as mp



# Set multiprocessing start method to 'spawn'
mp.set_start_method('spawn', force=True)
# Define XAI methods as a list 
XAI_METHODS = ['LOMATCE', 'IG', 'LIME', 'SHAP', 'Random']

def create_directories(*directories):
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def configure_logging(log_dir, dataset_name):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(log_dir, dataset_name, f'performance_decrease_{timestamp}.log')
    logging.basicConfig(filename=log_filename, filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    stdout_logfile = os.path.join(log_dir, dataset_name, f'stdout_{timestamp}.log')
    sys.stdout = open(stdout_logfile, 'w')

def parse_arguments():
    parser = argparse.ArgumentParser(description="LOMATCE: LOcal Model Agnostic Time-series Classification Explanation")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--model", required=False, default=FCN, help="Model path")
    parser.add_argument("--class_labels", nargs='+', required=True, help="List of class labels")

    return parser.parse_args()

def compute_integrated_gradients(x, pred, learner):
    integrated_gradients = IntegratedGradients(learner.model.cpu())
    attr, delta = integrated_gradients.attribute(x, target=pred, return_convergence_delta=True)
    return attr.squeeze().cpu().numpy()

def predict_fn(data, learner):
    if len(data.shape) == 2:
        data = data.reshape(data.shape[0], 1, data.shape[1])
    data_dl = test_loader.test_dataloader(learner, data)
    data_probas, _, _ = learner.get_preds(dl=data_dl, with_decoded=True, save_preds=None)
    return data_probas.numpy()
# #For LOMATCE as the method needs both the probabilites and predictions  
# def predict_fn(data, learner):
#     if len(data.shape) == 2:
#         data = data.reshape(data.shape[0], 1, data.shape[1])
#     data_dl = test_dataloader(learner, data)
#     data_probas, _, data_preds = learner.get_preds(dl=data_dl, with_decoded=True, save_preds=None)
#     return data_probas, data_preds

def segment_time_series(X, segment_length, num_features=None):
     # Check if X is 2D (has only two dimensions)
    if len(X.shape) == 2:
        # If the input is multivariate (num_features > 1), reshape it accordingly
        if num_features and num_features > 1:
            # Calculate the sequence length based on the number of features
            sequence_length = X.shape[1] // num_features
            
            # Reshape X to 3D (num_samples, num_features, sequence_length)
            X = X.reshape(X.shape[0], num_features, sequence_length)
        else:
            # For univariate case, reshape X to (num_samples, 1, sequence_length)
            X = X.reshape(X.shape[0], 1, X.shape[1])
    num_samples, num_features, sequence_length = X.shape
    segments = []
    for i in range(sequence_length - segment_length + 1):
        segments.append(X[:, :, i:i+segment_length])
    return np.concatenate(segments, axis=0)

# Flatten each segment to fit SHAP's 2D input requirement
def preprocess_for_shap(X):
    return X.reshape(X.shape[0], -1)


# 1. Define your segmentation
def create_segments(X, num_segments=None):
    """
    If num_segments is None, treat each timestep as one segment.
    """
    num_samples, num_channels, sequence_length = X.shape
    print(f" input shape: {X.shape}")
    if num_segments is None:
        num_segments = sequence_length
    segment_length = sequence_length // num_segments
    return num_segments, segment_length

# 2. Define the predict function for SHAP
def predict_fn_shap(masks, instance, background_data, num_segments, segment_length, learner):
    masked_inputs = []
    
    for mask in masks:
        masked_instance = instance.copy()  # (1, , num_channels, time_steps)
        if masked_instance.ndim == 2:
            masked_instance = masked_instance[None, :, :]
        for seg_idx in range(num_segments):
            start = seg_idx * segment_length
            end = (seg_idx + 1) * segment_length
            
            if mask[seg_idx] == 0:
                # Replace the segment with background
                random_idx = np.random.randint(0, background_data.shape[0])
                background_segment = background_data[random_idx,  :, start:end]
                masked_instance[0,  :, start:end] = background_segment
        
        masked_inputs.append(masked_instance[0])  # remove batch dimension

    masked_inputs = np.stack(masked_inputs)  # (num_masks, sequence_length, channels)
    
    # FastAI expects (batch, channels, sequence_length)
    masked_inputs = np.transpose(masked_inputs, (0, 1, 2))  # swap to (batch, channels, time)
    print(f"masked_input shape: {masked_inputs.shape}")
    # Create dataloader for FastAI model
    data_dl = test_loader.test_dataloader(learner, masked_inputs)
    
    data_probas, _, _ = learner.get_preds(dl=data_dl, with_decoded=True, save_preds=None)
    return data_probas.numpy()

def compute_lime_importances(flat_series, predict_fn, num_slices=None, num_features=25, num_samples=5000, class_names=None):
    if num_slices is None:
        num_slices = len(flat_series) # Len of the timeseries 
    explainer = LimeTimeSeriesExplainer(class_names=class_names)
    exp = explainer.explain_instance(flat_series, predict_fn, num_features=num_features, num_samples=num_samples, num_slices=num_slices, replacement_method='total_mean')
    values_per_slice = math.ceil(len(flat_series) / num_slices)
    # segments_weights = exp.as_list()
    # sorted_segments = sorted(segments_weights, key=lambda x: x[1], reverse=True)
     # Get segment weights and take the absolute values
    segments_weights = exp.as_list()
    absolute_segments_weights = [(feature, abs(weight)) for feature, weight in segments_weights]
    
    # Sort by absolute importance in descending order
    sorted_segments = sorted(absolute_segments_weights, key=lambda x: x[1], reverse=True)
    important_timesteps = []
    for feature, weight in sorted_segments:
        start = feature * values_per_slice
        end = min(start + values_per_slice, len(flat_series))
        important_timesteps.extend(range(start, end))
    return important_timesteps
def compute_shap_values(instance_to_explain, X_train, learner, prediction, num_slices=None, background_proportion=0.3, K=50):
    # if num_slices is None:
    #     num_slices = instance.shape[1]
    # X_train_segments = segment_time_series(X_train, num_slices)
    # X_train_flat = preprocess_for_shap(X_train_segments)
    
    # instance_segments = segment_time_series(instance, num_slices)
    # instance_flat = preprocess_for_shap(instance_segments)
    # # Determine the size of the background dataset dynamically
    # num_background_samples = int(len(X_train_flat) * background_proportion)
    # print(num_background_samples)
    # # If the dataset is small, use all samples as background
    # if num_background_samples < 1:
    #     num_background_samples = len(X_train_flat)
    # if K > len(X_train_flat):
    #     K= len(X_train_flat)
        
        
    # # Option 1: Sample K background samples
    # background = shap.sample(X_train_flat, K )
    
    # # Option 2: Or use K-means to cluster the background into K samples
    # # background = shap.kmeans(X_train_flat, K)
    
    # # background = X_train_flat[:num_background_samples]
    # # Initialize SHAP KernelExplainer
    # explainer = shap.KernelExplainer(predict_fn, background)  # Use a subset for the background dataset

    # # Explain a specific test sample
    # shap_values = explainer.shap_values(instance_flat)
    # shap_values = shap_values[0,:,prediction]  # Extract the SHAP values for the  class or output
    # shap_values_flat = np.array(shap_values).flatten()
    # # Sort the time steps based on their SHAP values (descending order)
    # sorted_indices = np.argsort(-np.abs(shap_values))  # Sort in descending order to get the highest values first
    # # sorted_indices = np.argsort(-shap_values)  # Sort in descending order by absolute value
    # # Return the sorted SHAP values and corresponding time steps
    # sorted_shap_values = shap_values[sorted_indices]
    # important_time_steps = sorted_indices 
    
    # Setup KernelExplainer
    num_segments, segment_length = create_segments(X_train, num_slices)

    # Example instance to explain (1 sample from the test set)
    # instance_to_explain = X_test[0:1]  # (1, sequence_length, num_channels)
    # background_data = X_train[:15]  # 100 background samples for SHAP
    background_data = X_train[np.random.choice(len(X_train), min(100, len(X_train)), replace=False)]


    # Define the KernelExplainer
    predict_fn_shap_ = lambda masks: predict_fn_shap(
        masks,
        instance=instance_to_explain,
        background_data=background_data,
        num_segments=num_segments,
        segment_length=segment_length,
        learner=learner
    )

    # Explain the first test instance using SHAP
    explainer = shap.KernelExplainer(predict_fn_shap_, np.zeros((1, num_segments)))  # Dummy background (zeros)
    shap_values = explainer.shap_values(np.ones((1, num_segments)))  # Explain with full instance (all segments visible)
    shap_value = shap_values[0,:,prediction]  # Extract the SHAP values for the first class or output
    shap_values_flat = np.array(shap_value).flatten()
    # Sort the time steps based on their SHAP values (descending order)
    sorted_indices = np.argsort(-np.abs(shap_values_flat))  # Sort in descending order to get the highest values first
    # sorted_indices = np.argsort(-shap_values)  # Sort in descending order by absolute value
    # Return the sorted SHAP values and corresponding time steps
    sorted_shap_values = shap_values_flat[sorted_indices]
    important_time_steps = sorted_indices 
    
    return important_time_steps, sorted_shap_values, shap_values_flat

def extract_time_steps(events_dict):
    if not isinstance(events_dict, dict):
        raise TypeError(f"Expected events_dict to be a dictionary, but got {type(events_dict).__name__}")
    important_time_steps = []
    for events in events_dict.values():
        for event in events:
            if len(event['event']) == 3:
                start_time, duration = event['event'][:2]
                if duration > 1:
                    important_time_steps.extend(range(start_time, start_time + duration))
            elif len(event['event']) == 2:
                important_time_steps.append(event['event'][0])
    return important_time_steps

def perturb_instance(instance, important_steps, replacement_method):
    instance_2d = instance.reshape(-1, instance.shape[1])
    perturbed_instance = instance_2d.copy()
    if replacement_method == 'zero':
        perturbed_instance[:, important_steps] = 0
    elif replacement_method == 'random':
        perturbed_instance[:, important_steps] = np.random.normal(
            np.mean(perturbed_instance[:, important_steps]),
            np.std(perturbed_instance[:, important_steps]),
            size=(perturbed_instance.shape[0], len(important_steps))
        )
    elif replacement_method == 'swap':
        available_indices = [idx for idx in range(instance_2d.shape[1]) if idx not in important_steps]
        if len(important_steps) > len(available_indices):
            logging.warning(f"Number of important steps ({len(important_steps)}) is greater than available indices for swapping ({len(available_indices)}). Reducing number of important steps.")
            important_steps = important_steps[:len(available_indices)]
        swap_indices = np.random.choice(
            [idx for idx in range(instance_2d.shape[1]) if idx not in important_steps], 
            size=(perturbed_instance.shape[0], len(important_steps)),
            replace=False
        )
        perturbed_instance[:, important_steps] = instance_2d[:, swap_indices]
    elif replacement_method == 'mean':
        perturbed_instance[:, important_steps] = np.mean(instance_2d[:, important_steps])
    else:
        raise ValueError(f"Unknown replacement method: {replacement_method}")
    return perturbed_instance.reshape(instance.shape)

def evaluate_perturbations(learner, original_accuracy, original_f1, origi_labels, perturbed_instances, replacement_method, method_name, performance_dict):
    perturbed_instances_array = np.array(perturbed_instances).reshape(len(perturbed_instances), 1, -1)
    perturb_dl = test_loader.test_dataloader(learner, perturbed_instances_array)
    perturbed_probas, perturbed_labels, perturbed_predictions = learner.get_preds(dl=perturb_dl, with_decoded=True, save_preds=None)
    perturbed_accuracy = accuracy_score(origi_labels, perturbed_predictions)
    perturbed_f1 = f1_score(origi_labels, perturbed_predictions, average='weighted')
    accuracy_decrease = original_accuracy - perturbed_accuracy
    f1_decrease = original_f1 - perturbed_f1
    if replacement_method not in performance_dict:
        performance_dict[replacement_method] = {}
    performance_dict[replacement_method][method_name] = {'accuracy': accuracy_decrease, 'f1_score': f1_decrease}
    
def compute_important_steps(method_name, instance, learner, prediction, class_names, log_dir, num_important_steps=None, background=None):
        lomatce_explainer = LomatceExplainer(base_dir=log_dir)
        if method_name == 'LOMATCE':
            explanation = lomatce_explainer.explain_instance(instance, lambda data: predict_fn(data, learner), num_perturbations=1000, top_n=10, class_names=class_names, replacement_method='zero')[0]
            important_motifs = explanation.important_motifs
            important_steps = list(set(extract_time_steps(important_motifs)))
            
        elif method_name == 'IG':
            instance_tensor = torch.tensor(instance, dtype=torch.float32).unsqueeze(0)
            ig_attributions = compute_integrated_gradients(instance_tensor, prediction, learner)
            important_steps = np.argsort(abs(ig_attributions))[::-1][:num_important_steps]  # Adjust top_n as needed
        elif method_name == 'LIME':
            flat_series = instance.flatten() if len(instance.shape) == 2 else instance
            important_steps = compute_lime_importances(flat_series, lambda data: predict_fn(data, learner), num_slices=None, num_features=25, num_samples=5000, class_names=class_names)
            num_points_to_take = min(len(important_steps), num_important_steps)
            important_steps = important_steps[:num_points_to_take]
        elif method_name == 'SHAP':
            # flat_series = instance.flatten() if len(instance.shape) == 2 else instance
            shap_important_time_steps, sorted_shap_values, _ = compute_shap_values(instance, background, learner, prediction, num_slices=None )
            num_points_to_take = min(len(shap_important_time_steps), num_important_steps)
            important_steps = shap_important_time_steps[:num_points_to_take]
        elif method_name == 'Random':
            if num_important_steps > len(instance[-1]):
                print(f"Requested number of important steps ({num_important_steps}) is greater than the instance length ({len(instance[-1])}). Reducing num_important_steps to {len(instance[-1])}.")
                num_important_steps = len(instance[-1])
            important_steps = np.random.choice(len(instance[-1]), size=num_important_steps, replace=False)
        else:
            raise ValueError(f"Unknown method name: {method_name}")
        return important_steps

def process_instance(instance_idx, instance, export_dir, log_dir, original_predictions,shap_background, class_names, replacement_methods):
    perturbed_instances = {method_name: {replacement_method: [] for replacement_method in replacement_methods} for method_name in XAI_METHODS}
    prediction = original_predictions[instance_idx]
    learner = load_learner_all(path=export_dir, dls_fname='dls', model_fname='model', learner_fname='learner')
    # Compute LOMATCE important steps
    important_steps_lomatce = compute_important_steps('LOMATCE', instance, learner, prediction, class_names, log_dir)
    num_important_steps = len(important_steps_lomatce)
    logging.info(f'Instance {instance_idx}: Length of important_steps_lomatce: {num_important_steps}')
    print(f'Instance {instance_idx}: Length of important_steps_lomatce: {num_important_steps}')
    
    for replacement_method in replacement_methods:
        perturbed_instances['LOMATCE'][replacement_method].append(perturb_instance(instance, important_steps_lomatce, replacement_method))        
    for method_name in XAI_METHODS[1:]:
        important_steps = compute_important_steps(method_name, instance, learner, prediction, class_names, log_dir, num_important_steps, shap_background)
        for replacement_method in replacement_methods:
            perturbed_instances[method_name][replacement_method] = perturb_instance(instance, important_steps, replacement_method)
    return perturbed_instances

def compute_performance_decrease(export_dir, instances, labels, replacement_methods,shap_background, class_names, log_dir, n_jobs=None):
    log_dir_plots = os.path.join(log_dir, 'plots')
    create_directories(log_dir_plots)
    # globxplain_instance = GlobXplain4TSC(base_dir=log_dir)
    learner = load_learner_all(path=export_dir, dls_fname='dls', model_fname='model', learner_fname='learner')
    test_dl = test_loader.test_dataloader(learner, instances, labels)
    test_probas, origi_labels, original_predictions = learner.get_preds(dl=test_dl, with_decoded=True, save_preds=None)
    original_accuracy = accuracy_score(origi_labels, original_predictions)
    original_f1 = f1_score(origi_labels, original_predictions, average='weighted')
    performance_dict = {}


    if n_jobs is None:
        n_jobs = max(1, cpu_count() // 2)  # Use half of the available cores
    else:
        n_jobs = min(n_jobs, cpu_count())
    all_perturbed_instances = Parallel(n_jobs=10, backend='multiprocessing')(delayed(process_instance)(i, instance,export_dir,log_dir, original_predictions, shap_background, class_names, replacement_methods) for i, instance in enumerate(instances))

    for method_name in XAI_METHODS:
        for replacement_method in replacement_methods:
            perturbed_instances = [all_perturbed_instances[i][method_name][replacement_method] for i in range(len(instances))]
            evaluate_perturbations(learner, original_accuracy, original_f1, origi_labels, perturbed_instances, replacement_method, method_name, performance_dict)
    
    for key, value in performance_dict.items():
        logging.info(f'{key}: {value}')
    
    return performance_dict

def save_performance_dict_to_file(performance_dict, filename):
    """
    Save the performance dictionary as a table to a file.

    Parameters:
        performance_dict: Dictionary containing performance decrease information.
        filename: Name of the file to save the table.
    """
    headers = ["XAI Method", "R-Method", "Accuracy Decrease", "F1 Score Decrease"]
    rows = []
    
    for  r_method, xai_methods in performance_dict.items():
        for xai_method, metrics in xai_methods.items():
            rows.append([xai_method, r_method, metrics['accuracy'], metrics['f1_score']])
    
    table = tabulate(rows, headers=headers, tablefmt='grid')
    print(table)
    with open(filename, 'w') as file:
        file.write(table)

if __name__ == "__main__":
    args = parse_arguments()
    dsid = args.dataset
    dataset_name = dsid.lower()
    model = args.model.lower()
    class_labels = args.class_labels

    # Generate day stamp
    day_stamp = datetime.now().strftime("%Y-%m-%d")
    # Configure directories and logging
    current_dir = os.path.dirname(__file__)
    base_dir = os.path.abspath(os.path.join(current_dir, '..'))
    export_dir = os.path.join(base_dir, 'experiments', 'models', dataset_name, model, 'export')
    
    log_dir = os.path.join(base_dir, 'results', 'eval-logs', day_stamp)
    dataset_log_dir = os.path.join(log_dir, dataset_name)
    create_directories(dataset_log_dir)
    configure_logging(log_dir, dataset_name)


    if not os.path.exists(export_dir):
        raise FileNotFoundError(f"Export directory '{export_dir}' does not exist.")

    learn_new = load_learner_all(path=export_dir, dls_fname='dls', model_fname='model', learner_fname='learner')

    testset_dir = os.path.join(base_dir, 'experiments', dataset_name)
    X_train = np.load(os.path.join(testset_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(testset_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(testset_dir, 'y_test.npy'))
    
    
    logging.info(f'Dataset Name: {dataset_name}')
    logging.info(f'N_samples: 1000 , n_top=10, replacement method : zero, num_slice=None and with absolute value')

    train_x = X_train.copy()  # Train data is required for the background data of SHAP

    if len(X_test) > 100:
        test_x = X_test[:100].copy()
        test_y = y_test[:100].copy()
    else:
        test_x = X_test.copy()
        test_y = y_test.copy()
    
    
    replacement_methods = ['zero', 'swap', 'mean']
    # class_names = ['Cylinder', 'Bell', 'Funnel'] #['Myocardial Infarction', 'Normal Heartbeat']

    performance_dict = compute_performance_decrease(export_dir, test_x, test_y, replacement_methods, train_x, class_labels, dataset_log_dir)
    
    # Save the performance log
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = os.path.join(dataset_log_dir, f'performance_decrease_{timestamp}.txt')
    save_performance_dict_to_file(performance_dict, log_file_path)
    
    
# %%
