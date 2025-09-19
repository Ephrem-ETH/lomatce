#%%
import random
import time
import os
import argparse
import logging
import random
import time
import os
import sys
import scipy.stats as stats
from tsai.all import *
from fastai.vision.all import *
from sklearn.model_selection import train_test_split
from fastai.callback.tracker import EarlyStoppingCallback
from fastai.vision.all import *
from torch.utils.data.dataset import ConcatDataset
from tsai.utils import set_seed
from sklearn.preprocessing import MinMaxScaler
from  explainer import LomatceExplainer
import utils.test_dataloader as test_loader
from tabulate import tabulate
from joblib import Parallel, delayed
from logging.handlers import QueueHandler, QueueListener
import multiprocessing



set_seed(1024)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(f"device: {device}")
my_setup()

MODEL_REGISTRY = {
    "fcn": FCN,
    "lstm_fcn": LSTM_FCN
}


def setup_logging(logfile):
    log_queue = multiprocessing.Queue(-1)
    handler = logging.FileHandler(logfile)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = []  # Clear existing handlers
    root.addHandler(QueueHandler(log_queue))

    listener = QueueListener(log_queue, handler)
    listener.start()
    return listener

def parse_arguments():
    parser = argparse.ArgumentParser(description="LOMATCE: LOcal Model-Agnostic Time-series Classification Explanations")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--model", required=True, help="Model name (e.g., fcn, lstm_fcn)")
    parser.add_argument("--num_runs", type=int, default=100, help="Number of Monte Carlo runs")
    parser.add_argument("--class_labels", nargs='+', required=True, help="List of class labels")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("-r", "--replacement_method", default='zero', choices=['zero', 'mean', 'total_mean', 'random'], help="Replacement method: zero, mean, total_mean, or random")
    return parser.parse_args()

def data_preparation(X, y, test_size=0.3, random_state=12):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    #Reserve 10% for validation, validation set is required in fastai
    X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test, random_state=random_state)

    print(f'Shape of X_train :{X_train.shape}')
    print(f'Shape of y_train :{y_train.shape}')
    # X_valid, X_test, y_valid, y_test = train_test_split(X_test,y_test, test_size=0.65, random_state=1024)
    print(f'Shape of X_valid :{X_valid.shape}')
    print(f'Shape of X_test :{X_test.shape}')

    tfms  = [None, [Categorize()]]
    train_ds = TSDatasets(X_train, y_train, tfms=tfms)
    # valid_ds = TSDatasets(X_valid, y_valid, tfms=tfms)
    valid_ds = TSDatasets(X_valid, y_valid, tfms=tfms)


    # combined_ds = ConcatDataset([train_ds, valid_ds])
    # print(combined_ds[0][0].shape)
    tfms = [None, [Categorize()]]
    # dls = get_ts_dls(combined_ds, tfms=tfms, bs=64)
    dls = TSDataLoaders.from_dsets(train_ds, valid_ds, bs=[64, 128], batch_tfms=[TSStandardize()], num_workers=0,  device=device, shuffle=True )
    print(f' Number of classes : {dls.c}')
    return dls, X_test, y_test

def trainer (base_dir, model, dls, epochs=150, learning_rate=1e-3, patience=15):
    kwargs = {}
    metrics=[accuracy]
    path = f'{base_dir}/export'
    os.makedirs(path, exist_ok=True)
    # Create model
    model = create_model(model, dls=dls, **kwargs)
    # Define early stopping criteria
    early_stopping = EarlyStoppingCallback(monitor='valid_loss', min_delta=0.001, patience=patience)

    # Define a call back to save the best model
    save_callback = SaveModelCallback(monitor='accuracy')
    # Train and evaluate model with early stopping
    # set_seed(42)
    cbs = [early_stopping] #ShowGraph(), save_callback
    learner = Learner(dls=dls, model=model, opt_func=Adam, metrics=metrics,cbs=cbs) #
    learner.fit_one_cycle(epochs, learning_rate)
    print("Type of object before save_all:", type(learner))
    learner.save_all(path=path, dls_fname='dls', model_fname='model', learner_fname='learner')
    return learner

def validator(learn, X_test, y_test):
    dls = learn.dls
    valid_dl = dls.valid
    train_dl = dls.train
    # Labelled test data
    test_ds = valid_dl.dataset.add_test(X_test, y_test)# In this case I'll use X and y, but this would be your test data
    test_dl = valid_dl.new(test_ds)
    test_probas, test_targets, test_preds = learn.get_preds(dl=test_dl, with_decoded=True, save_preds=None, save_targs=None)
    # print(f'Test Accuracy: {skm.accuracy_score(test_targets, test_preds):10.6f}')
    valid_probas, valid_targets, valid_preds = learn.get_preds(dl=valid_dl, with_decoded=True)
    train_probas, train_targets, train_preds = learn.get_preds(dl=train_dl, with_decoded=True)
    
    valid_accuracy = (valid_targets == valid_preds).float().mean()
    test_accuracy = (test_targets == test_preds).float().mean()
    train_accuracy = (train_targets == train_preds).float().mean()

    print(f"Validation Accuracy :{valid_accuracy:.2f}")
    print(f"Test Accuracy :{test_accuracy:.2f}")
    print(f"Train Accuracy :{train_accuracy:.2f}")
#     valid_dl.show_batch(sharey=True)
#     test_dl.show_batch(sharey=True)
#     learn.show_results(max_n=6)

    return train_accuracy, valid_accuracy, test_accuracy, test_preds, valid_preds

def predict_fn(data, learner):
    if len(data.shape) == 2:
        data = data.reshape(data.shape[0], 1, data.shape[1])
    data_dl = test_loader.test_dataloader(learner, data)
    data_probas, _, data_preds = learner.get_preds(dl=data_dl, with_decoded=True, save_preds=None)
    return data_probas, data_preds

def run_single_iteration(base_dir, run, X, y, model_name, class_names,replacement_method, num_samples):

    try:
        print(f"Starting iteration {run}")
        
        # Code for a single iteration
        dls, X_test, y_test = data_preparation(X, y, random_state=run)

        if 'cuda' in str(device):
            learn_new = trainer(base_dir, model_name, dls)
        else:
            print('GPU is not available!')

        train_acc, valid_acc, test_acc, test_preds, valid_preds = validator(learn_new, X_test, y_test)
        
        # Randomly pick one index along the first axis (axis=0)
        randomly_picked_index = np.random.choice(X_test.shape[0], size=1, replace=False)

        # Use the index to get the corresponding instance
        randomly_picked_instance = X_test[randomly_picked_index]

        # Squeeze the singleton dimension
        randomly_picked_instance = np.squeeze(randomly_picked_instance, axis=0)

        # The shape of randomly_picked_instance will be (1, timesteps)
        print(randomly_picked_instance.shape)

        # Print the index of the randomly picked instance
        print("Index of the randomly picked instance:", randomly_picked_index)
        explanation = lomatce_explainer.explain_instance(randomly_picked_instance, lambda data: predict_fn(data, learn_new), num_perturbations=num_samples, class_names=class_names, replacement_method=replacement_method)
        
        explanation_summary = explanation.get_explanation_summary()
        local_fidelity = explanation_summary["local_fidelity"]
        local_prediction = explanation_summary["local_prediction"]
       

        print(f"Iteration {run} completed successfully")

        return train_acc, valid_acc, test_acc, local_fidelity, local_prediction
    except Exception as e:
        print(f"Exception occurred in iteration {run}: {e}")
        raise

def monte_carlo_cross_val_parallel(base_dir, model_name, num_runs, class_names, replacement_method='zero', num_samples=1000):
    start_time = time.time()
    
    
    results = Parallel(n_jobs=1, backend='loky')(
        delayed(run_single_iteration)(base_dir, run, X, y, model_name, class_names, replacement_method, num_samples) for run in range(num_runs)
    )

    train_accs, valid_accs, test_accs, prediction_scores,_ = zip(*results)
    # Save the results as a text file with a header
    header = "train_accs, valid_accs, test_accs, fidelity"
    np.savetxt(f'{base_dir}/results.txt', np.array([train_accs, valid_accs, test_accs, prediction_scores]).T, header=header)

    
    # Calculate means and std deviations
    train_mean, train_std = round(np.mean(train_accs), 2), round(np.std(train_accs), 2)
    valid_mean, valid_std = round(np.mean(valid_accs), 2), round(np.std(valid_accs), 2)
    test_mean, test_std = round(np.mean(test_accs), 2), round(np.std(test_accs), 2)
    fidelity_mean, fidelity_std = round(np.mean(prediction_scores), 2), round(np.std(prediction_scores), 2)

    # Calculate the standard errors
    train_se = train_std / np.sqrt(len(train_accs))
    valid_se = valid_std / np.sqrt(len(valid_accs))
    test_se = test_std / np.sqrt(len(test_accs))
    fidelity_se = fidelity_std / np.sqrt(len(prediction_scores))

    # Set the desired confidence level
    confidence_level = 0.95

    # Calculate the Z-score for the confidence level
    z_score = stats.norm.ppf((1 + confidence_level) / 2)

    # Calculate margin of errors
    train_margin_error = z_score * train_se
    valid_margin_error = z_score * valid_se
    test_margin_error = z_score * test_se
    fidelity_margin_error = z_score * fidelity_se

    # Store the results in the dictionary
    result_dict = {
        'v_acc_mean': valid_mean,
        'v_acc_std': valid_std,
        'v_acc_margin_error': valid_margin_error,
        't_acc_mean': test_mean,
        't_acc_std': test_std,
        't_acc_margin_error': test_margin_error,
        'fidelity_mean': fidelity_mean,
        'fidelity_std': fidelity_std,
        'fidelity_margin_error': fidelity_margin_error
    }

    end_time = time.time()
    total_time = round(end_time - start_time, 2)
    print(f"Total time taken: {total_time} seconds")

    return result_dict

if __name__ == "__main__":
    args = parse_arguments()
    dsid = args.dataset
    dataset_name = dsid.lower()
    model_name = args.model.lower()

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not recognized. Available: {list(MODEL_REGISTRY.keys())}")

    model_class = MODEL_REGISTRY[model_name]

    num_runs = args.num_runs
    class_labels = args.class_labels
    r_method = args.replacement_method
    num_samples = args.num_samples

    print(f'{dsid}, {model_class}, {num_runs}')

    cur_time = time.strftime('%Y-%m-%d_%H-%M-%S')
    base_dir = f'results/simulation/{dataset_name}/{model_name}-{r_method}--{cur_time}' if 'cuda' in str(device) else f'results/{dataset_name}--{cur_time}'
    os.makedirs(base_dir, exist_ok=True)

    logfile = f'{base_dir}/logfile.log'
    setup_logging(logfile)

    sys.stdout = open(f'{base_dir}/output.log', 'w')

    logging.info("Starting the script.")

    try:
        X, y, splits = get_UCR_data(dsid, on_disk=True, return_split=False)
        deviation = 0.02
        lomatce_explainer = LomatceExplainer(base_dir=base_dir)
        dls, X_test, y_test = data_preparation(X, y)

        result_dict = monte_carlo_cross_val_parallel(base_dir, model_class, num_runs, class_labels, r_method)

        summary_result = [
            ["Test Accuracy", f"{result_dict['t_acc_mean']:.2f} ± {result_dict['t_acc_margin_error']:.2f}", f"{result_dict['t_acc_std']:.2f}"],
            ["Valid Accuracy", f"{result_dict['v_acc_mean']:.2f} ± {result_dict['v_acc_margin_error']:.2f}", f"{result_dict['v_acc_std']:.2f}"],
            ["Fidelity", f"{result_dict['fidelity_mean']:.2f} ± {result_dict['fidelity_margin_error']:.2f}", f"{result_dict['fidelity_std']:.2f}"],
        ]

        print("Info:")
        print(f"Dataset: {args.dataset}")
        print(f"Model: {args.model}")
        print(f"Num runs: {args.num_runs}")
        print(f"Replacement method: {r_method}")
        print(f"Num of samples: {args.num_samples}")
        table_headers = ["Metric", "Mean (95% CI)", "Std"]
        print(tabulate(summary_result, headers=table_headers, tablefmt="grid"))
        logging.info("Script completed successfully.")
    except Exception as e:
        logging.error(f"An exception occurred: {e}", exc_info=True)

    sys.stdout = sys.__stdout__