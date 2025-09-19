# import all necessary libraries

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from joblib import Parallel, delayed, cpu_count
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import logging
from typing import Dict, List, Optional, Union
from config.settings import LomatceConfig
from scipy.stats import spearmanr
import fastdtw
import torch

# Import local modules based on environment
# Local environment imports
from lomatce.utils.helper_class import HelperClass
import lomatce.perturbation as perturbation
from lomatce.utils.test_dataloader import test_dataloader



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Config:
    MAX_STABLE_ITERATIONS = 4
    DEFAULT_K = 20
    KERNEL_WIDTH_MULTIPLIER = 2
    # ... other configuration values


class LomatceExplainer:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.helper_instance = HelperClass(base_dir=base_dir)

    def preprocessing_data(self, X):
        """
        Preprocesses the input data.

        Args:
            X (np.ndarray): Input data

        Returns:
            pd.DataFrame: Preprocessed data
        """
        logger.info("Processing data...")
        print(f"Shape of data : {X.shape}")

        # Create sample input data
        # data = np.random.rand(90, 24, 51)

        # Reshape input data into a 2D array
        reshaped_data = np.empty((X.shape[0], X.shape[1]), dtype=np.ndarray)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                reshaped_data[i, j] = X[i, j, :]

        # Create a list of column names for the DataFrame
        col_names = [f"ch{i+1}" for i in range(X.shape[1])]

        # Create the DataFrame
        df = pd.DataFrame(reshaped_data, columns=col_names)

        # Print the resulting DataFrame
        # print(df)
        return df

    def extract_inc_dec_events(self, data):
        """
        Extracts increasing and decreasing events from the input data.

        Args:
            data (np.ndarray): Input data

        Returns:
            pd.DataFrame: DataFrame containing increasing and decreasing events
        """
        # Handle 1D input by reshaping to 2D
        if len(data.shape) == 1:
            data = data.reshape(1, 1, -1)
        elif len(data.shape) == 2:
            data = data.reshape(data.shape[0], 1, data.shape[1])

        # Reshape input data into a 2D array
        reshaped_data = np.empty((data.shape[0], data.shape[1]), dtype=np.ndarray)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                reshaped_data[i, j] = data[i, j, :]

        # Create a list of column names for the DataFrame
        col_names = [f"ch{i+1}" for i in range(data.shape[1])]

        # Create the DataFrame
        df = pd.DataFrame(reshaped_data, columns=col_names)

        # Initialize the result DataFrame
        result_df = pd.DataFrame(index=df.index)
        events_dict = {}
        # Extract increasing and decreasing events for each column of each instance
        for col_name in df.columns:
            increasing_events = []
            decreasing_events = []

            for instance_idx in range(len(df)):
                col_values = df.loc[instance_idx, col_name]
                inc_start_time = 0
                dec_start_time = 0
                inc_duration = 0
                dec_duration = 0
                inc_events = []
                dec_events = []
                inc_sum_values = 0
                dec_sum_values = 0
                for i in range(1, len(col_values)):
                    if col_values[i] > col_values[i - 1]:
                        if dec_duration > 0:
                            dec_avg_value = dec_sum_values / dec_duration
                            if dec_duration > 1:
                                dec_events.append(
                                    [dec_start_time, dec_duration, dec_avg_value]
                                )
                            dec_duration = 0
                            dec_sum_values = col_values[i]
                        if inc_duration == 0:
                            inc_start_time = i
                        inc_duration += 1
                        inc_sum_values += col_values[i]
                    elif col_values[i] < col_values[i - 1]:
                        if inc_duration > 0:
                            inc_avg_value = inc_sum_values / inc_duration
                            if inc_duration > 1:
                                inc_events.append(
                                    [inc_start_time, inc_duration, inc_avg_value]
                                )
                            inc_duration = 0
                            inc_sum_values = col_values[i]
                        if dec_duration == 0:
                            dec_start_time = i
                        dec_duration += 1
                        dec_sum_values += col_values[i]
                if inc_duration > 0:
                    inc_avg_value = inc_sum_values / inc_duration
                    inc_events.append([inc_start_time, inc_duration, inc_avg_value])
                if dec_duration > 0:
                    dec_avg_value = dec_sum_values / dec_duration
                    dec_events.append([dec_start_time, dec_duration, dec_avg_value])
                increasing_events.append(inc_events)
                decreasing_events.append(dec_events)
            result_df[f"Increasing_{col_name}"] = increasing_events
            result_df[f"Decreasing_{col_name}"] = decreasing_events

        return result_df

    def extract_local_max_min_events(self, data):
        """
        Extracts local max and min events from the input data.

        Args:
            data (np.ndarray): Input data

        Returns:
            pd.DataFrame: DataFrame containing local max and min events
        """
        # Handle 1D input by reshaping to 2D
        if len(data.shape) == 1:
            data = data.reshape(1, 1, -1)
        elif len(data.shape) == 2:
            data = data.reshape(data.shape[0], 1, data.shape[1])

        # Reshape input data into a 2D array
        reshaped_data = np.empty((data.shape[0], data.shape[1]), dtype=np.ndarray)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                reshaped_data[i, j] = data[i, j, :]

        # Create a list of column names for the DataFrame
        col_names = [f"ch{i+1}" for i in range(data.shape[1])]

        # Create the DataFrame
        df = pd.DataFrame(reshaped_data, columns=col_names)

        # Initialize the result DataFrame
        result_df = pd.DataFrame(index=df.index)
        events_dict = {}
        # Extract local max and local min events for each column of each instance
        for col_name in df.columns:
            local_max_events = []
            local_min_events = []
            for instance_idx in range(len(df)):
                col_values = df.loc[instance_idx, col_name]
                max_events = []
                min_events = []
                for i in range(1, len(col_values) - 1):
                    if (
                        col_values[i] > col_values[i - 1]
                        and col_values[i] > col_values[i + 1]
                    ):
                        max_events.append([i, col_values[i]])
                    elif (
                        col_values[i] < col_values[i - 1]
                        and col_values[i] < col_values[i + 1]
                    ):
                        min_events.append([i, col_values[i]])

                local_max_events.append(max_events)
                local_min_events.append(min_events)

            result_df[f"LocalMax_{col_name}"] = local_max_events
            result_df[f"LocalMin_{col_name}"] = local_min_events

        # Display the resulting DataFrame
        # print(result_df)
        return result_df

    def flatten_nested_events(self, events_list):
        """
        Flattens a nested list of events.

        Args:
            events_list (list): Nested list of events

        Returns:
            list: Flattened list of events
        """
        inner_values = [inner for row in events_list for inner in row]
        inner_values_2d = np.array(inner_values, dtype=object).tolist()
        return inner_values_2d

    def evaluate_kmeans(self, n_clusters, data_transformed):
        """
        Evaluates the K-means clustering algorithm.

        Args:
            n_clusters (int): Number of clusters
            data_transformed (np.ndarray): Transformed data

        Returns:
            tuple: Number of clusters, silhouette score, sum of squared distances, and KMeans model
        """
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=0).fit(data_transformed)
        labels = kmeans.labels_
        silhouette_avg = silhouette_score(data_transformed, labels)
        sse = kmeans.inertia_
        return n_clusters, silhouette_avg, sse, kmeans

    def cluster_events(self, all_events, k=20, col_name="", n_jobs=None):
        """
        Clusters events using the K-means algorithm.

        Args:
            all_events (list): List of events
            k (int): Maximum number of clusters
            col_name (str): Column name
            n_jobs (int): Number of jobs for parallel processing

        Returns:
            tuple: KMeans model and scaler
        """
        # Preprocess the data
        data = np.array(all_events, dtype=np.float32)
        scaler = StandardScaler()
        data_transformed = scaler.fit_transform(data)
        # Convert the scaled data to float64
        data_transformed = data_transformed.astype(np.float64)

        # Use 1/3 of the available cores if less than 21, otherwise use half, at least 1
        # NOTE: This is mainly for local machines. On servers (with more cores), using half is fine.
        # On local machines, using too many cores may cause deadlock if all are occupied.
        if n_jobs is None:
            total_cores = cpu_count()
            if total_cores < 21:
                n_jobs = max(1, total_cores // 4)
            else:
                n_jobs = max(1, total_cores // 2)
            fraction = f"{n_jobs}/{total_cores}"
        else:
            n_jobs = min(n_jobs, cpu_count())
            total_cores = cpu_count()
            fraction = f"{n_jobs}/{total_cores}"
        print(f"Running on {n_jobs} cpu cores ({fraction} of total cores)")

        try:
            # Evaluate K-means clustering in parallel
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(self.evaluate_kmeans)(n_clusters, data_transformed)
                for n_clusters in range(2, k)
            )
        except Exception as e:
            print(f"Parallel execution failed: {e}. Running sequentially.")
            results = [
                self.evaluate_kmeans(n_clusters, data_transformed)
                for n_clusters in range(2, k)
            ]

        # Extract the results
        silhouette_scores = [result[1] for result in results]
        sse = [result[2] for result in results]
        n_clusters_range = [result[0] for result in results]

        # Stability check
        max_stable_iterations = 4
        stable_count = 0
        optimal_k_index = np.argmax(silhouette_scores)
        optimal_k = n_clusters_range[
            optimal_k_index
        ]  # This correctly maps to the number of clusters

        # Perform stability check
        for i in range(max_stable_iterations, len(silhouette_scores)):
            recent_scores = silhouette_scores[i - max_stable_iterations : i]
            if np.all(np.diff(recent_scores) == 0):
                stable_count += 1
            else:
                stable_count = 0

            if stable_count >= max_stable_iterations:
                optimal_k = n_clusters_range[i]
                break

        # Find the optimal number of clusters
        optimal_k_1 = np.argmax(silhouette_scores) + 2
        # Print the optimal number of clusters for the current column
        print(f"Optimal number of clusters for {col_name}: {optimal_k} = {optimal_k_1}")

        # Fit the kmeans with optimal K value
        kmeans = KMeans(n_clusters=optimal_k, random_state=12).fit(data_transformed)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_

        # Plotting (optional)
        #   if col_name:
        #       plt.figure()
        #       plt.plot(range(2, k), silhouette_scores, marker='o')
        #       plt.xlabel('Number of clusters', fontsize=12)
        #       plt.ylabel('Silhouette score', fontsize=12)
        #       plt.title('Silhouette Method', fontsize=14)
        #       plt.grid(True)
        #       # Ensure x-axis has integer steps
        #       plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        #       plt.savefig(f'{self.base_dir}/{col_name}_silhouette_plot.png', dpi=300)
        #       plt.savefig(f'{self.base_dir}/{col_name}_silhouette_plot.pdf', format='pdf', dpi=300, bbox_inches='tight')
        #       plt.show()

        #       plt.figure()
        #       plt.plot(range(2, k), sse)
        #       plt.xlabel('Number of clusters', fontsize=12)
        #       plt.ylabel('Sum of squared distances', fontsize=12)
        #       plt.title('Elbow Method', fontsize=12)
        #       plt.grid(True)
        #       # Ensure x-axis has integer steps
        #       plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        #       plt.savefig(f'{self.base_dir}/{col_name}_elbow_plot.png', dpi=300)
        #       plt.savefig(f'{self.base_dir}/{col_name}_elbow_plot.pdf', format='pdf', dpi=300, bbox_inches='tight')
        #       plt.show()

        return kmeans, scaler

    def event_attribution(self, kmeans, parametrized_events, scaler):
        """
        Performs event attribution by mapping the extracted events to the clusters.

        Args:
            kmeans (KMeans): KMeans model
            parametrized_events (list): List of parametrized events
            scaler (StandardScaler): Scaler object

        Returns:
            dict: Dictionary containing the count of 'yes' for each cluster
        """
        # Initialize the number of clusters
        n_clusters = len(kmeans.cluster_centers_)

        rows = {}

        for i, row in enumerate(parametrized_events):
            # Initialize clusters with empty lists
            clusters = {j: [] for j in range(n_clusters)}

            for event in row:
                # Scale the event and predict its cluster
                event_scaled = scaler.transform([event])
                label = kmeans.predict(event_scaled)[0]

                # Append 'yes' to the appropriate cluster and 'no' to others
                for cluster_label in clusters:
                    if cluster_label == label:
                        clusters[cluster_label].append("yes")
                    else:
                        clusters[cluster_label].append("no")

            rows[i] = clusters

        # Calculate the count of 'yes' for each cluster
        cluster_event_counts = {}
        for key, values in rows.items():
            yes_prob = [Counter(value).get("yes", 0) for value in values.values()]
            cluster_event_counts[key] = yes_prob

        return cluster_event_counts

    def merge_event_df(self, df_inc_dec, df_max_min):
        """
        Merges the increasing/decreasing and local max/min events DataFrames.

        Args:
            df_inc_dec (pd.DataFrame): DataFrame containing increasing/decreasing events
            df_max_min (pd.DataFrame): DataFrame containing local max/min events

        Returns:
            pd.DataFrame: Merged DataFrame
        """
        # Merge the DataFrames by index
        merged_df = df_inc_dec.merge(df_max_min, left_index=True, right_index=True)
        merged_df.head()
        # merged_df = df_max_min.copy()
        return merged_df

    def prepare_data4DT(
        self,
        merged_df,
        k=20,
        kmeans_dict=None,
        scaler_dict=None,
        for_eval=False,
        n_jobs=None,
    ):
        """
        Prepares the data for linear surrogate model.

        Args:
            merged_df (pd.DataFrame): Merged DataFrame containing events
            k (int): Maximum number of clusters
            kmeans_dict (dict): Dictionary containing KMeans models
            scaler_dict (dict): Dictionary containing scaler objects
            for_eval (bool): Flag indicating whether to prepare data for evaluation
            n_jobs (int): Number of jobs for parallel processing

        Returns:
            tuple: Appended DataFrame, master dictionary, cluster centroids, KMeans dictionary, and scaler dictionary
        """
        #   helper_instance = HelperClass(base_dir="results")

        appended_df = pd.DataFrame()
        count = 0
        master_dict = {}
        cluster_centroids = {}
        if kmeans_dict is None:
            kmeans_dict = {}
            scaler_dict = {}
        for col_name in merged_df.columns:
            parametrized_events = merged_df[col_name]
            # print(type(col_name))
            flatten_data = self.flatten_nested_events(parametrized_events)
            # Extract the part of the column name before the second underscore
            col_prefix = "_".join(col_name.split("_")[:2])
            if for_eval:
                # print(kmeans_dict[col_name])
                # print(kmeans_dict[f'{col_name}'])
                kmeans = kmeans_dict[col_name]
                scaler = scaler_dict[col_name]

                attributed_data = self.event_attribution(
                    kmeans, parametrized_events, scaler
                )
            else:
                kmeans, scaler = self.cluster_events(flatten_data, k, col_name=col_name)
                kmeans_dict[col_name] = kmeans
                scaler_dict[col_name] = scaler
                attributed_data = self.event_attribution(
                    kmeans, parametrized_events, scaler
                )
            # Determine the maximum length of the values
            max_length = max(len(values) for values in attributed_data.values())

            # Generate dynamic column names
            column_names = [f"{col_name}_c{i+1}" for i in range(max_length)]

            # Convert the dictionary to DataFrame with dynamic column names
            df = pd.DataFrame(attributed_data.values(), columns=column_names)
            appended_df = pd.concat([appended_df, df], axis=1)
            post_processed_col_name, cluster_centroid = (
                self.helper_instance.post_processed(kmeans, scaler, col_name)
            )
            master_dict.update(post_processed_col_name)
            cluster_centroids.update(cluster_centroid)
            # if "increasing" in col_name.lower() or "decreasing" in col_name.lower():
            #     self.helper_instance.plot_3D(
            #         kmeans=kmeans,
            #         flatten_events=flatten_data,
            #         scaler=scaler,
            #         col_name=col_name,
            #     )
            # elif "localmax" in col_name.lower() or "localmin" in col_name.lower():
            #     self.helper_instance.plot_2D(
            #         kmeans=kmeans,
            #         flatten_events=flatten_data,
            #         scaler=scaler,
            #         col_name=col_name,
            #     )
        # Save the dictionaries containing all models
        joblib.dump(kmeans_dict, f"{self.base_dir}/kmeans_models_dict.pkl")
        joblib.dump(scaler_dict, f"{self.base_dir}/scaler_models_dict.pkl")

        return appended_df, master_dict, cluster_centroids, kmeans_dict, scaler_dict

    def combine_data(self, X_test, kmeans_dict=None, scaler_dict=None, for_eval=False):
        """
        Combines the data from different sources.

        Args:
            X_test (np.ndarray): Test data
            kmeans_dict (dict): Dictionary containing KMeans models
            scaler_dict (dict): Dictionary containing scaler objects
            for_eval (bool): Flag indicating whether to prepare data for evaluation

        Returns:
            tuple: Full data, KMeans dictionary, scaler dictionary, cluster centroids, and master dictionary
        """
        df = self.preprocessing_data(X_test)
        df_inc_dec = self.extract_inc_dec_events(X_test)
        df_max_min = self.extract_local_max_min_events(X_test)
        merged_df = self.merge_event_df(df_inc_dec=df_inc_dec, df_max_min=df_max_min)
        if for_eval:
            # print(kmeans_dict)
            kmeans_dict = kmeans_dict
            scaler_dict = scaler_dict
            appended_df, master_dict, cluster_centroids, kmeans_dict, scaler_dict = (
                self.prepare_data4DT(
                    merged_df,
                    kmeans_dict=kmeans_dict,
                    scaler_dict=scaler_dict,
                    for_eval=True,
                )
            )
            # print(kmeans_dict)
        else:
            appended_df, master_dict, cluster_centroids, kmeans_dict, scaler_dict = (
                self.prepare_data4DT(merged_df)
            )
        # master_dict = self.helper_instance.update_master_dict(master_dict)
        full_data = appended_df.copy()

        return full_data, kmeans_dict, scaler_dict, cluster_centroids, master_dict

    def apply_lr(
        self, processed_data, target, weights, class_names, master_dict, model_regressor
    ):
        try:
            logger.info("Applying linear regression")

            # Initialize model based on type
            if model_regressor is None:
                model_regressor = Ridge()
            elif model_regressor == "LinearRegression":
                model_regressor = LinearRegression()
            elif model_regressor == "Lasso":
                model_regressor = Lasso()
            else:
                logger.warning(
                    f"Unknown model type: {model_regressor}. Using Ridge regression."
                )
                model_regressor = Ridge()

            # Standardize features
            scaler = StandardScaler()
            processed_data_scaled = scaler.fit_transform(processed_data)

            # Fit model
            model = model_regressor
            model.fit(processed_data_scaled, target, sample_weight=weights)

            # Calculate predictions and scores
            prediction_score = model.score(
                processed_data_scaled, target, sample_weight=weights
            )

            # Get prediction for the first instance using the linear model
            first_instance_scaled = scaler.transform(
                processed_data.iloc[0].values.reshape(1, -1)
            )
            local_pred = model.predict(first_instance_scaled)[0]

            # Get feature importance
            feature_importance = model.coef_ / scaler.scale_
            feature_names = [master_dict.get(col) for col in processed_data.columns]

            # Filter and sort important features
            threshold = LomatceConfig.FEATURE_IMPORTANCE_THRESHOLD
            filtered_importance = [
                (name, coef)
                for name, coef in zip(feature_names, feature_importance)
                if abs(coef) > threshold
            ]

            if not filtered_importance:
                logger.warning(
                    "No features passed importance threshold. Using top 10 by absolute value."
                )
                filtered_importance = sorted(
                    zip(feature_names, feature_importance),
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )[: LomatceConfig.TOP_N_FEATURES]

            # Sort for visualization
            filtered_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            filtered_feature_names, filtered_scores = zip(*filtered_importance)

            # Plot feature importance
            #   self._plot_feature_importance(filtered_feature_names, filtered_scores)

            # Log results
            logger.info(f"R-Score (Local Fidelity): {np.round(prediction_score, 2)}")
            logger.info(f"Local Prediction (Linear Model): {np.round(local_pred, 2)}")
            logger.info(f"Model Type: {type(model).__name__}")

            return model, feature_importance, prediction_score, local_pred
        except ValueError as e:
            logger.error("Invalid data in linear regression analysis: %s", str(e))
            raise
        except Exception as e:
            logger.error("Unexpected error in linear regression analysis: %s", str(e))
            raise

    def _plot_feature_importance(self, feature_names, scores):
        """
        Helper method to plot feature importance.

        Args:
            feature_names (tuple): Names of features
            scores (tuple): Importance scores
        """
        plt.figure(figsize=LomatceConfig.PLOT_FIGSIZE, dpi=LomatceConfig.PLOT_DPI)
        plt.barh(feature_names, scores, color="skyblue")
        plt.xlabel("Feature Importance", fontsize=12)
        plt.ylabel("Features", fontsize=12)
        plt.title("Linear Regression Feature Importance", fontsize=14)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.gca().invert_yaxis()
        plt.grid(axis="x", linestyle="--", alpha=0.7)

        # Save plots
        plt.savefig(
            f"{self.base_dir}/lr_fi_plot.png",
            dpi=LomatceConfig.PLOT_DPI,
            bbox_inches="tight",
        )
        plt.savefig(
            f"{self.base_dir}/lr_fi_plot.pdf",
            format="pdf",
            dpi=LomatceConfig.PLOT_DPI,
            bbox_inches="tight",
        )
        plt.show()

    def _process_events(self, perturbed_instances, n_clusters):
        """
        Processes events from perturbed instances.

        Args:
            perturbed_instances (np.ndarray): Perturbed instances
            n_clusters (int): Number of clusters

        Returns:
            tuple: Processed data and related dictionaries
        """
        logger.info(f"Processing events for {len(perturbed_instances)} instances")

        # Extract events for all instances including original
        df_inc_dec = self.extract_inc_dec_events(perturbed_instances)
        df_max_min = self.extract_local_max_min_events(perturbed_instances)
        merged_df = self.merge_event_df(df_inc_dec=df_inc_dec, df_max_min=df_max_min)

        # Process the data
        final_data, master_dict, cluster_centroids, kmeans_dict, scaler_dict = (
            self.prepare_data4DT(merged_df, k=n_clusters)
        )

        # Verify the shape includes original instance
        logger.info(f"Final data shape: {final_data.shape} (instances x features)")
        logger.info(f"Number of instances: {final_data.shape[0]}")
        logger.info(f"Number of features (clusters): {final_data.shape[1]}")

        if final_data.shape[0] != len(perturbed_instances):
            logger.warning(
                f"Shape mismatch: final_data has {final_data.shape[0]} rows but perturbed_instances has {len(perturbed_instances)} instances"
            )
            # Ensure consistent shapes by truncating or padding
        #   if final_data.shape[0] > len(perturbed_instances):
        #       final_data = final_data.iloc[:len(perturbed_instances)]
        #   else:
        #       # Pad with zeros if needed
        #       padding = pd.DataFrame(0, index=range(len(perturbed_instances) - final_data.shape[0]), columns=final_data.columns)
        #       final_data = pd.concat([final_data, padding])

        return final_data, master_dict, cluster_centroids, kmeans_dict, scaler_dict

    def _calculate_important_features(
        self,
        final_data,
        instances_probs,
        weights,
        class_names,
        master_dict,
        model_regressor,
        top_n,
    ):
        """
        Calculates important features using linear regression.

        Args:
            final_data (pd.DataFrame): Final processed data
            instances_probs (np.ndarray): Instance probabilities
            weights (np.ndarray): Weights for instances
            class_names (list): List of class names
            master_dict (dict): Master dictionary
            model_regressor (str): Regressor model
            top_n (int): Number of top features to return

        Returns:
            tuple: Important features, prediction score, and local prediction
        """
        _, important_features_lr, prediction_score, local_pred = self.apply_lr(
            processed_data=final_data,
            target=instances_probs,
            weights=weights,
            class_names=class_names,
            master_dict=master_dict,
            model_regressor=model_regressor,
        )

        # Calculate top_n dynamically
        if isinstance(top_n, (int, float)) and top_n <= 100:
            top_n = max(1, round((top_n / 100) * len(final_data.columns)))
        else:
            top_n = min(top_n, len(final_data.columns))

        if top_n > len(final_data.columns):
            logger.warning(
                f"top_n exceeds available features. Adjusting to {len(final_data.columns)}"
            )

        # Filter and sort features
        non_zero_features = [
            (index, importance)
            for index, importance in enumerate(important_features_lr)
            if importance > 0
        ]
        sorted_features = sorted(non_zero_features, key=lambda x: x[1], reverse=True)
        selected_features_indices = [index for index, _ in sorted_features[:top_n]]

        return (
            {
                final_data.columns[index]: importance
                for index, importance in enumerate(important_features_lr)
                if index in selected_features_indices
            },
            prediction_score,
            local_pred,
        )

    def explain_instance(
        self,
        origi_instance,
        classifier_fn,
        num_perturbations=1000,
        n_clusters=20,
        kernel_scale=None,
        top_n=10,
        class_names=[0, 1],
        model_regressor=None,
        replacement_method="zero",
        n_jobs=None,
        evaluate_quality=False,
    ):
        """
        Performs time series local explanation using LOMATCE.

        This method:
        1. Generates perturbations around the instance
        2. Gets predictions from the black box model
        3. Fits a local linear model to explain the predictions
        4. Identifies important regions in the time series
        5. Evaluates explanation quality (optional)

        Args:
            origi_instance (np.ndarray): Original time series instance
            classifier_fn (function): Black box model prediction function
            num_perturbations (int): Number of perturbations to generate
            n_clusters (int): Number of clusters for event detection
            kernel_width (float): Width of the kernel for weighting perturbations
            top_n (int): Number of top features to return
            class_names (list): Names of the classes
            model_regressor (str): Type of regressor to use ('LinearRegression', 'Ridge', 'Lasso')
            replacement_method (str): Method for generating perturbations
            n_jobs (int): Number of parallel jobs
            evaluate_quality (bool): Whether to evaluate explanation quality

        Returns:
            TimeSeriesExplanation: Object containing the explanation results
        """
        try:
            logger.info("Starting time series local explanation")

            # Store classifier_fn as instance attribute
            self.classifier_fn = classifier_fn

            # Ensure instance is 1D
            #   origi_instance = origi_instance.reshape(-1)  # Flatten to 1D
            #   logger.info(f"Original instance shape: {origi_instance.shape}")
            # Generate and process perturbations
            perturbed_data, distances = perturbation.generate_perturbations(
                origi_instance[0], num_perturbations, replacement_method
            )

            perturbed_instances = np.array(perturbed_data).reshape(
                (-1, 1, origi_instance.shape[1])
            )
            logger.info(
                f"Perturbed instances shap for classifier: {perturbed_instances.shape}"
            )
            # Process events
            final_data, master_dict, cluster_centroids, kmeans_dict, scaler_dict = (
                self._process_events(perturbed_instances, n_clusters)
            )

            # Calculate kernel width and weights
            # Kernel width scales with number of sqrt(clusters)
            n_clusters = len(final_data.columns)
            if kernel_scale is None:
                kernel_width = 2 * np.sqrt(n_clusters)
            else:
                kernel_width = kernel_scale * np.sqrt(n_clusters)
            weights = perturbation.kernel(np.array(distances), kernel_width)

            # Get predicitons from classifier
            #   perturbed_instances = np.array(perturbed_data).reshape((-1, 1, origi_instance.shape[1]))
            #   logger.info(f"Perturbed instances shap for classifier: {perturbed_instances.shape}")

            # Get probabilities for all instances at once
            perturb_probas, perturb_preds = classifier_fn(perturbed_instances)

            # Convert to PyTorch tensor if needed
            if isinstance(perturb_probas, np.ndarray):
                perturb_probas = torch.from_numpy(perturb_probas)

            # Get max probabilities for each instance
            instances_probs, _ = torch.max(perturb_probas, dim=1)

            # Calculate important features
            selected_features, prediction_score, local_pred = (
                self._calculate_important_features(
                    final_data,
                    instances_probs,
                    weights,
                    class_names,
                    master_dict,
                    model_regressor,
                    top_n,
                )
            )

            # Get original instance events and important motifs
            origi_instance_events = self.merge_event_df(
                self.extract_inc_dec_events(origi_instance),
                self.extract_local_max_min_events(origi_instance),
            ).iloc[0]

            important_motifs, important_motifs_with_cluster  = self.helper_instance.events_in_topK_clusters(
                selected_features,
                origi_instance_events,
                kmeans_dict=kmeans_dict,
                scaler_dict=scaler_dict,
            )

            # Create explanation object
            explanation = LomatceExplanation(
                important_features=selected_features,
                prediction_score=prediction_score,
                local_prediction=local_pred,
                original_prediction=perturb_preds[0].item(),
                important_motifs=important_motifs,
                important_motifs_with_cluster=important_motifs_with_cluster,
                feature_names=list(final_data.columns),
                class_names=class_names,
                raw_probas=perturb_probas,
                cluster_centroids=cluster_centroids,
                helper=self.helper_instance,
            )

            # Evaluate explanation quality if requested
            if evaluate_quality:
                quality_metrics = self._evaluate_explanation_quality(
                    explanation, perturbed_instances, perturb_probas, weights
                )
                logger.info("Explanation quality metrics:")
                for metric, value in quality_metrics.items():
                    logger.info(f"{metric}: {value:.3f}")

            # Plot explanations
            #   self.plot_explanations(
            #       origi_instance, perturb_preds[0].item(), selected_features, important_motifs,
            #       cluster_centroids, class_names, perturb_probas origi_instance_pred = perturb_preds[0].item()
            #   )
            self.helper_instance.plot_events_as_line_on_timeseries_1(
                origi_instance,
                perturb_preds[0].item(),
                selected_features,
                cluster_centroids,
                class_names,
            )
            self.helper_instance.plot_events_on_time_series(
                origi_instance, perturb_preds[0].item(), important_motifs, class_names
            )
            print(f"Predection probability : {perturb_probas[0]}")
            probabilities = perturb_probas[0].squeeze().tolist()

            # Print colored filled boxes for each class
            for i, prob in enumerate(probabilities):
                self.helper_instance.print_colored_filled_boxes(class_names[i], i, prob)

            return explanation

        except Exception as e:
            logger.error(f"Error in time series local explanation: {str(e)}")
            raise


class LomatceExplanation:
    """
    A class to store and manage time series explanations.

    Attributes:
        important_features (dict): Dictionary of important features and their importance scores
        prediction_score (float): Local fidelity score (R² score of the linear model)
        local_prediction (float): Prediction of the local linear model
        original_prediction (float): Prediction of the black box model
        important_motifs (dict): Dictionary of important motifs and their locations
        feature_names (list): Names of the features
        class_names (list): Names of the classes
    """

    def __init__(
        self,
        important_features,
        prediction_score,
        local_prediction,
        original_prediction,
        important_motifs,
        important_motifs_with_cluster,
        feature_names,
        class_names,
        raw_probas=None,
        cluster_centroids=None,
        helper=None,
    ):
        self.important_features = important_features
        self.prediction_score = prediction_score
        self.local_prediction = local_prediction
        self.original_prediction = int(original_prediction)
        self.important_motifs = important_motifs
        self.important_motifs_with_cluster = important_motifs_with_cluster
        self.feature_names = feature_names
        self.class_names = class_names
        
        # Optional internal storage for visualization/evaluation
        self.raw_probas = raw_probas
        self.cluster_centroids = cluster_centroids
        self.helper = helper

    def get_top_features(self, n=10):
        """Returns the top n most important features."""
        return dict(
            sorted(
                self.important_features.items(), key=lambda x: abs(x[1]), reverse=True
            )[:n]
        )

    def get_explanation_summary(self):
        """
        Generates a concise summary of the explanation results.

        Returns:
            dict: A dictionary containing:
                - 'local_fidelity' (float): R² score indicating the faithfulness of the local surrogate model.
                - 'local_prediction' (float): Prediction from the interpretable (local) model.
                - 'original_prediction' (float): Prediction from the black-box model.
                - 'top_features' (dict): Top important features (clusters) and their importance scores.
        """

        return {
            "local_fidelity": self.prediction_score,
            "local_prediction": self.local_prediction,
            "original_prediction": self.original_prediction,
            "top_features": self.get_top_features(),
        }
    
    
    def visualise(self, origi_instance, show_probas=True, save_path=None):
        """
        Visualises the explanation of a time series instance using LOMATCE.

        This method produces two plots:
        1. Cluster centroids overlaid on the original time series.
        2. Important events from top-k clusters overlaid on the original time series.

        It also prints a summary including:
        - The local model's prediction
        - The original (black-box) model's prediction
        - The local fidelity score (R²)
        - (Optional) Class probabilities from the black-box model

        Args:
            origi_instance (np.ndarray): The original time series instance (2D).
            show_probas (bool, optional): Whether to display class probabilities. Defaults to True.
            save_path (str, optional): If provided, saves the final visualisation to the specified path.

        Returns:
            None
        """

        # --- Plot 1: Cluster centroids ---
        print("\n📊 Visual 1: Cluster centroids overlaid on the original time series.")
        self.helper.plot_events_as_line_on_timeseries_1(
            origi_instance,
            self.original_prediction,
            self.important_features,
            self.cluster_centroids,
            self.class_names,
        )

        # --- Plot 2: Important events ---
        print("📌 Visual 2: Important events belong to the top-k clusters overlaid on the original time series.")
        self.helper.plot_events_on_time_series(
            origi_instance,
            self.original_prediction,
            self.important_motifs,
            self.class_names,
        )

        # --- Explanation Summary ---
        class_index = int(self.original_prediction)
        print("🧠 Explanation Summary:")
        print(f"• Local prediction (interpretable model): {self.local_prediction:.4f}")
        print(f"• Black-box model prediction: {class_index} or '{self.class_names[class_index]}'")
        print(f"• Local fidelity score (R²): {self.prediction_score:.4f}")

        # --- Optional: Class probabilities ---
        if show_probas and self.raw_probas is not None:
            print("\n Class Probabilities (from black-box model):")
            probabilities = self.raw_probas[0].squeeze().tolist()
            for i, prob in enumerate(probabilities):
                self.helper.print_colored_filled_boxes(self.class_names[i], i, prob)

        # Optionally save visualisation as an image, if applicable
        if save_path:
            plt.savefig(save_path)
            print(f"\n Visualisation saved to: {save_path}")

