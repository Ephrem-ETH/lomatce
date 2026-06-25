import numpy as np
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.cluster.hierarchy import linkage, fcluster
from scipy import stats
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec 
from matplotlib.colors import LinearSegmentedColormap
import textwrap


class SPLOMATCE:
    def __init__(self, dataset_name, class_labels, lomatce_explainer, predict_fn, output_dir=None):
        self.dataset_name = dataset_name
        # self.model_name = model_name
        self.class_labels = class_labels
        self.explainer = lomatce_explainer
        self.predict_fn = predict_fn
        self.output_dir = output_dir

        # Store results
        self.explanations = []
        self.merge_map = None
        self.global_clusters = None
        self.global_importance_raw = None
        self.global_importance = None
        self.selected_indices = None
        self.selected_events = None
        self.summaries = None
        self.global_faithfulness = None
        self.class_results = {}

    # ---------------------
    # Main entrypoint
    # ---------------------
    def run(self, X, y, B=5, class_wise=True, n_per_class=None, top_n=15, merge_percentile=75, merge_method='average'):
        if class_wise:
            classes = np.unique(y)
            label_map = dict(zip(classes, self.class_labels))
            for cls in classes:
                class_name = label_map.get(cls, str(cls))  # fallback to numeric if not mapped
                print(f"\n=== Running SP-LOMATCE for class {class_name} ===")
                cls_indices = np.where(y == cls)[0]
                X_cls, y_cls = X[cls_indices], y[cls_indices]
                self._execute_pipeline(X_cls, y_cls, B, n_per_class, top_n, class_name=class_name, merge_percentile=merge_percentile, merge_method=merge_method)
        
        else:
            self._execute_pipeline(X, y, B, n_per_class, top_n, class_name=None, merge_percentile=merge_percentile, merge_method=merge_method)
        #     class_results.append({
        #     "class_name": self.dataset_name,
        #     "global_clusters": self.global_clusters,
        #     "global_importance": self.global_importance,
        #     "summaries": self.summaries,
        #     "global_faithfulness": self.global_faithfulness,
        #     "dataset_name": self.dataset_name
        # })
         # --- After all classes processed, visualize everything together ---
        print("\n=== 📊 Final Visualization for All Classes ===\n")
        for class_name, results in self.class_results.items():
            self._plot_global_summaries_(
                results["global_clusters"],
                results["global_importance"],
                results["summaries"],
                dataset_name=self.dataset_name,
                class_name=class_name,
                global_faithfulness=results["faithfulness"]
            )
            
        return self.class_results

    # ---------------------
    # Pipeline for one group
    # ---------------------
    def _execute_pipeline(self, X, y, B, n_per_class, top_n, class_name=None, merge_percentile=75, merge_method='average'):
        self.explanations = self._generate_explanations(X, y, n_per_class, top_n)
        self.merge_map, _ = self._merge_clusters(self.explanations, top_n=top_n, merge_percentile=merge_percentile, method=merge_method)
        instance_matrix, self.global_clusters = self._build_instance_cluster_matrix(self.explanations, self.merge_map)
        self.global_importance_raw = self._compute_global_cluster_importance(instance_matrix)
        self.selected_indices = self._submodular_pick(instance_matrix, self.global_importance_raw, B)
        self.selected_events = self._get_selected_events(self.explanations, self.merge_map, self.selected_indices)
        self.summaries = self._summarize_global_events(self.selected_events)
        
        # Normalize for visualization
        total = self.global_importance_raw.sum()
        if total > 0:
            self.global_importance = self.global_importance_raw / total
        else:
            self.global_importance = self.global_importance_raw

        # if class_name:
        #     self._plot_global_summaries(self.global_clusters, self.global_importance, self.summaries, dataset_name= self.dataset_name, class_name=class_name)
        # else:
        #     self._plot_global_summaries(self.global_clusters, self.global_importance, self.summaries, dataset_name=self.dataset_name)
            
        # ---- Compute Global Faithfulness ----
        local_fidelities = [self.explanations[i].prediction_score for i in self.selected_indices]
        self.global_faithfulness = np.mean(local_fidelities) if local_fidelities else 0.0
        print(f"\n📈 Global Faithfulness (Avg Local Fidelity): {self.global_faithfulness:.4f}")
        
         # --- Store all results for later visualization ---
        self.class_results[class_name or 'All Data'] = {
            "global_clusters": self.global_clusters,
            "global_importance": self.global_importance,
            "global_importance_raw": self.global_importance_raw,
            "summaries": self.summaries,
            "faithfulness": self.global_faithfulness,
        }

    # ---------------------
    # Step 1: Explanations
    # ---------------------
    def _generate_explanations(self, X, y, n_per_class, top_n):
        selected_indices = []
        classes = np.unique(y)

        if n_per_class is not None:  # balanced sampling
            for cls in classes:
                cls_indices = np.where(y == cls)[0]
                chosen = np.random.choice(cls_indices, min(n_per_class, len(cls_indices)), replace=False)
                selected_indices.extend(chosen)
        else:  # global random
            n_global = min(len(X), 10)
            selected_indices = np.random.choice(np.arange(len(X)), n_global, replace=False)

        explanations = []
        for idx in selected_indices:
            instance = X[idx]
            exp = self.explainer.explain_instance(
                instance,
                lambda data: self.predict_fn(data),
                num_perturbations=1000,
                class_names=self.class_labels,
                replacement_method='zero',
                top_n=top_n,
            )
            explanations.append(exp)
        return explanations

    # ---------------------
    # Step 2: Merge clusters
    # ---------------------
    def _merge_clusters(self, explanations, merge_percentile=75, method='average', top_n=None):
        merged_cluster_map = {}
        global_clusters_by_type = {}
        centroids_by_type = defaultdict(list)
        local_keys_by_type = defaultdict(list)

        for inst_idx, exp in enumerate(explanations):
            if not exp.cluster_centroids:
                continue
            if top_n is None:
                top_n = len(exp.important_features)
            sorted_top_clusters = sorted(
                exp.cluster_centroids.items(),
                key=lambda kv: exp.important_features.get(kv[0], 0),
                reverse=True
            )[:top_n]
            for local_name, centroid in sorted_top_clusters:
                inst_local_name = f"inst{inst_idx}_{local_name}"
                event_type = local_name.split('_')[0]
                centroids_by_type[event_type].append(centroid)
                local_keys_by_type[event_type].append(inst_local_name)

        for event_type, centroids in centroids_by_type.items():
            if len(centroids) == 0:
                continue
             # FIX: cannot run linkage with 1 centroid
            if len(centroids) == 1:
                only_key = local_keys_by_type[event_type][0]
                global_name = f"Global_{event_type}_0"
                merged_cluster_map[only_key] = global_name
                global_clusters_by_type[event_type] = [global_name]
                continue
            centroids_array = np.vstack(centroids)
            Z = linkage(centroids_array, method=method)
            threshold = np.percentile(Z[:, 2], merge_percentile)
            labels = fcluster(Z, t=threshold, criterion='distance')

            for local_key, label in zip(local_keys_by_type[event_type], labels):
                global_name = f"Global_{event_type}_{label-1}"
                merged_cluster_map[local_key] = global_name

            global_clusters_by_type[event_type] = sorted(set(
                [merged_cluster_map[k] for k in local_keys_by_type[event_type]]
            ))
        return merged_cluster_map, global_clusters_by_type

    # ---------------------
    # Step 3: Build instance-cluster matrix
    # ---------------------
    def _build_instance_cluster_matrix(self, explanations, merged_cluster_map):
        global_clusters = sorted(set(merged_cluster_map.values()))
        n_instances = len(explanations)
        n_clusters = len(global_clusters)
        matrix = np.zeros((n_instances, n_clusters))

        for i, exp in enumerate(explanations):
            try:
                top_features = exp.get_top_features(n=None)
            except AttributeError:
                top_features = exp.important_features

            for local_name, imp in top_features.items():
                inst_local_name = f"inst{i}_{local_name}"
                if inst_local_name in merged_cluster_map:
                    global_name = merged_cluster_map[inst_local_name]
                    j = global_clusters.index(global_name)
                    matrix[i, j] = imp
        return matrix, global_clusters

    # ---------------------
    # Step 4: Global importance
    # ---------------------
    def _compute_global_cluster_importance(self, W):
        return np.sqrt(np.sum(np.abs(W), axis=0))

    # ---------------------
    # Step 5: Submodular pick
    # ---------------------
    def _submodular_pick(self, X, I, B):
        n_instances, n_clusters = X.shape
        selected = []
        covered = np.zeros(n_clusters)

        while len(selected) < B:
            best_gain, best_instance = -1, None
            for i in range(n_instances):
                if i in selected:
                    continue
                incremental_gain = np.sum(I * ((X[i] > 0) & (covered == 0)))
                if incremental_gain > best_gain:
                    best_gain, best_instance = incremental_gain, i
            if best_instance is None:
                break
            selected.append(best_instance)
            covered = np.maximum(covered, (X[best_instance] > 0) * I)
        return selected

    # ---------------------
    # Step 6: Collect events
    # ---------------------
    def _get_selected_events(self, explanations, merged_cluster_map, selected_indices):
        global_to_local = defaultdict(list)
        for local, global_c in merged_cluster_map.items():
            global_to_local[global_c].append(local)
        aggregated_events = defaultdict(list)

        for idx in selected_indices:
            exp = explanations[idx]
            instance_local_clusters = {
                global_c: [lc for lc in local_list if lc.startswith(f"inst{idx}_")]
                for global_c, local_list in global_to_local.items()
            }
            for global_c, local_clusters in instance_local_clusters.items():
                for event_family, events in (exp.important_motifs_with_cluster or {}).items():
                    for e in events:
                        cluster_label = e.get("cluster")
                        matching_local_clusters = [
                            lc for lc in local_clusters
                            if lc.endswith(f"_c{cluster_label}") and lc.split("_", 1)[1].startswith(event_family)
                        ]
                        if matching_local_clusters:
                            event_copy = e.copy()
                            event_copy["instance_idx"] = idx
                            aggregated_events[global_c].append(event_copy)
        return dict(aggregated_events)

    # ---------------------
    # Step 7: Summarize events
    # ---------------------
    def _summarize_global_events(self, global_events_dict, ci=False):
        summary_dict = {}
        for global_name, events in global_events_dict.items():
            if not events:
                continue
            all_events = np.array([e['event'] for e in events])
            cluster_type = global_name.split("_")[1].lower()

            if cluster_type in ['increasing', 'decreasing']:
                start_times = all_events[:, 0]
                durations = all_events[:, 1]
                start_mean, dur_mean = int(np.mean(start_times)), int(np.mean(durations))
                if len(start_times) > 1:
                    if ci:
                        start_ci, dur_ci = int(stats.sem(start_times) * 1.96), int(stats.sem(durations) * 1.96)
                    else:
                        start_ci, dur_ci = int(np.std(start_times)), int(np.std(durations))
                else:
                    start_ci, dur_ci = 0, 0
                summary = f"{cluster_type.capitalize()} from {start_mean} ± {start_ci} to duration {start_mean + dur_mean} ± {dur_ci}"

            elif cluster_type in ['localmax', 'localmin']:
                times, values = all_events[:, 0].astype(int), all_events[:, 1]
                time_mean, value_mean = int(np.mean(times)), np.mean(values)
                if len(times) > 1:
                    if ci:
                        time_ci, value_ci = int(stats.sem(times) * 1.96), stats.sem(values) * 1.96
                    else:
                        time_ci, value_ci = int(np.std(times)), np.std(values)
                else:
                    time_ci, value_ci = 0, 0
                summary = f"{cluster_type.capitalize()} at {time_mean} ± {time_ci} with value {value_mean:.3f} ± {value_ci:.3f}"

            summary_dict[global_name] = summary
        return summary_dict

    # ---------------------
    # Step 8: Plot
    # ---------------------
    def _plot_global_summaries(self, global_clusters, global_importance, summaries, dataset_name, model_name=None, class_name=None):
        cluster_info = [
            (cluster, imp, summaries[cluster])
            for cluster, imp in zip(global_clusters, global_importance)
            if cluster in summaries
        ]
        cluster_info = sorted(cluster_info, key=lambda x: x[1], reverse=True)

        labels = [f"{summary}" for cluster, _, summary in cluster_info]
        importances = [imp for _, imp, _ in cluster_info]
        clusters = [c for c, _, _ in cluster_info]

        def get_color(cluster_name):
            if "Increasing" in cluster_name:
                return "steelblue"
            elif "Decreasing" in cluster_name:
                return "indianred"
            elif "LocalMax" in cluster_name:
                return "seagreen"
            elif "LocalMin" in cluster_name:
                return "purple"
            return "gray"

        colors = [get_color(c) for c in clusters]

        fig, ax = plt.subplots(figsize=(11, len(labels) * 0.55))
        y_pos = np.arange(len(labels)-1, -1, -1)
        bars = ax.barh(y_pos, importances, align='center', color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Global Importance')
        ax.set_ylim(-0.7, len(bars)-0.3)

        for bar, value in zip(bars, importances):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f"{value:.3f}", va='center', fontsize=8)

        title = f"Global Importance of Events on {dataset_name} (Class {class_name})" if class_name else f"Global Importance of Events on {dataset_name}"
        if model_name:
            title = f"Global Importance of Events Learned by {model_name} on {dataset_name} (Class {class_name})" if class_name else f"Global Importance of Events Learned by {model_name} on {dataset_name}"
        ax.set_title(title)

        legend_elements = [
            Patch(facecolor="steelblue", label="Increasing"),
            Patch(facecolor="indianred", label="Decreasing"),
            Patch(facecolor="seagreen", label="LocalMax"),
            Patch(facecolor="purple", label="LocalMin")
        ]
        ax.legend(handles=legend_elements, title="Event Type", loc="lower right")
        plt.tight_layout()  

        # --- Save figure if path provided ---
        plot_filename = f"{dataset_name}_{class_name}.png"

        # Full save path
        save_path = os.path.join(self.output_dir, plot_filename)
        if self.output_dir is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)  # close to avoid memory leak / displaying
        else:
            plt.show()
        
    

   
    def _plot_global_summaries_(
        self,
        global_clusters,
        global_importance,
        summaries,
        dataset_name,
        class_name=None,
        model_name=None,
        global_faithfulness=None
    ):
        # -----------------------------
        # Prepare data
        # -----------------------------
        cluster_info = [
            (cluster, imp, summaries[cluster])
            for cluster, imp in zip(global_clusters, global_importance)
            if cluster in summaries
        ]
        cluster_info = sorted(cluster_info, key=lambda x: x[1], reverse=True)

        # Wrap labels to 50 characters to prevent them from cutting off
        labels = [textwrap.fill(summary, width=50) for _, _, summary in cluster_info]
        importances = np.array([imp for _, imp, _ in cluster_info])
        clusters = [c for c, _, _ in cluster_info]

        # -----------------------------
        # Color mapping
        # -----------------------------
        def get_color(cluster_name):
            if "Increasing" in cluster_name:
                return "#4C72B0"   # muted blue
            elif "Decreasing" in cluster_name:
                return "#DD8452"   # muted red
            elif "LocalMax" in cluster_name:
                return "#55A868"   # muted green
            elif "LocalMin" in cluster_name:
                return "#8172B3"   # muted purple
            return "gray"

        colors = [get_color(c) for c in clusters]

        # -----------------------------
        # Figure setup
        # -----------------------------
        # Increased height per bar (0.8) to account for multi-line wrapped text
        fig_height = max(5, len(labels) * 0.75)
        fig, ax = plt.subplots(figsize=(14, fig_height))

        y_pos = np.arange(len(labels))[::-1]

        bars = ax.barh(
            y_pos,
            importances,
            height=0.7, # Slightly thinner bars look more professional
            color=colors,
            edgecolor="black",
            linewidth=0.6
        )

        # -----------------------------
        # Axis Formatting
        # -----------------------------
        ax.set_yticks(y_pos)
        # ax.set_yticklabels(labels, fontsize=10, va='center')
        # ax.set_xlabel("Global Importance", fontsize=12, fontweight='bold')
        ax.set_yticklabels(labels, fontsize=14, va='center')
        ax.set_xlabel("Global Importance", fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', labelsize=13)


        # FIX: Remove gaps at the top and bottom
        ax.set_ylim(min(y_pos) - 0.6, max(y_pos) + 0.6)

        # QUALITY: Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Dynamic x-limits
        max_val = importances.max()
        ax.set_xlim(0, max_val * 1.18)

        # -----------------------------
        # Value annotations
        # -----------------------------
        for bar, value in zip(bars, importances):
            x = bar.get_width()
            y = bar.get_y() + bar.get_height() / 2

            if x > 0.12 * max_val:
                ax.text(
                    x - 0.01 * max_val, y,
                    f"{value:.3f}",
                    va="center", ha="right",
                    fontsize=11, color="white", fontweight='bold'
                )
            else:
                ax.text(
                    x + 0.01 * max_val, y,
                    f"{value:.3f}",
                    va="center", ha="left",
                    fontsize=11, color="black"
                )

        # -----------------------------
        # Title
        # -----------------------------
        title_parts = [] 
        if model_name:
            title_parts.append(model_name)
        title_parts.append(dataset_name)
        if class_name:
            title_parts.append(f"Class: {class_name}")

        ax.set_title(
            " – ".join(title_parts),
            fontsize=17,
            pad=18,
            fontweight='bold'
        )

        # -----------------------------
        # Legend
        # -----------------------------
        legend_elements = [
            Patch(facecolor="#4C72B0", label="Increasing"),
            Patch(facecolor="#DD8452", label="Decreasing"),
            Patch(facecolor="#55A868", label="Local Maximum"),
            Patch(facecolor="#8172B3", label="Local Minimum"),
        ]
        ax.legend(
            handles=legend_elements,
            title="Event Type",
            fontsize=12,
            title_fontsize=13,
            loc="lower right",
            frameon=True
        )

        # -----------------------------
        # Global faithfulness annotation
        # -----------------------------
        # if global_faithfulness is not None:
        #     # Move the box slightly lower to avoid overlapping with axis labels
        #     ax.text(
        #         0.10, -0.10,
        #         f"Global Faithfulness (Avg Local Fidelity): {global_faithfulness:.3f}",
        #         ha="center",
        #         va="center",
        #         fontsize=11,
        #         fontweight="bold",
        #         transform=ax.transAxes,
        #         bbox=dict(
        #             boxstyle="round,pad=0.5",
        #             facecolor="#F8F9F9",
        #             edgecolor="#D5D8DC"
        #         )
        #     )
        # -----------------------------
        # 7. Global Faithfulness (Safely at the Bottom Left)
        # -----------------------------
        if global_faithfulness is not None:
                # xy=(0,0) is bottom-left of the chart area
                # xytext=(0, -50) means "exactly 50 points down from the chart"
                ax.annotate(
                    f"Global Faithfulness (Avg Local Fidelity): {global_faithfulness:.3f}",
                    xy=(0, 0), 
                    xycoords='axes fraction',
                    xytext=(0, -50), 
                    textcoords='offset points',
                    ha="left", va="top",
                    fontsize=12, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8F9F9", edgecolor="#D5D8DC")
                )

        # -----------------------------
        # 8. The "Safety Gutter" Fix
        # -----------------------------
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

        # -----------------------------
        # Final polish
        # -----------------------------
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

        # tight_layout handles the label clipping, bbox_inches="tight" handles the save
        fig.subplots_adjust(left=0.42)  # key for long y labels
        plt.tight_layout()
        # plt.tight_layout(rect=[0, 0.05, 1, 1])  # leaves bottom margin for annotation


        # -----------------------------
        # 8. Save both PDF and PNG
        # -----------------------------
        # ax.grid(axis="x", linestyle="--", alpha=0.3, linewidth=0.5)
        # ax.set_axisbelow(True)

        base_name = f"{dataset_name}_{class_name}" if class_name else f"{dataset_name}"
        
        if self.output_dir is not None:
            # Save Vector (Best for PDF LaTeX)
            fig.savefig(os.path.join(self.output_dir, f"{base_name}.pdf"), bbox_inches="tight")
            # Save Raster (Best for presentations/web)
            fig.savefig(os.path.join(self.output_dir, f"{base_name}.png"), dpi=600, bbox_inches="tight")
            plt.show()
            plt.close(fig)
        else:
            
            plt.show()


    # def _plot_global_summaries_(
    #     self,
    #     global_clusters,
    #     global_importance,
    #     summaries,
    #     dataset_name,
    #     class_name=None,
    #     model_name=None,
    #     global_faithfulness=None
    # ):
    #     # -----------------------------
    #     # 1. Dimensions for Springer Single-Column
    #     # -----------------------------
    #     # Standard Springer LNCS/Nature width is ~4.8 inches (12.2cm)
    #     fig_width = 4.8 
    #     height_per_row = 0.45 
    #     # Minimum height of 3.5 to accommodate title, legend, and bottom box
    #     fig_height = max(3.5, len(global_clusters) * height_per_row + 1.2)

    #     fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)

    #     # -----------------------------
    #     # 2. Data Preparation & Wrapping
    #     # -----------------------------
    #     cluster_info = [
    #         (cluster, imp, summaries[cluster])
    #         for cluster, imp in zip(global_clusters, global_importance)
    #         if cluster in summaries
    #     ]
    #     cluster_info = sorted(cluster_info, key=lambda x: x[1], reverse=True)

    #     # Wrap labels to 35 chars for narrow columns
    #     labels = [textwrap.fill(summary, width=35) for _, _, summary in cluster_info]
    #     importances = np.array([imp for _, imp, _ in cluster_info])
    #     clusters = [c for c, _, _ in cluster_info]
    #     y_pos = np.arange(len(labels))[::-1]

    #     # -----------------------------
    #     # 3. Plotting
    #     # -----------------------------
    #     def get_color(cluster_name):
    #         colors = {"Increasing": "#4C72B0", "Decreasing": "#DD8452", 
    #                 "LocalMax": "#55A868", "LocalMin": "#8172B3"}
    #         for key, val in colors.items():
    #             if key in cluster_name: return val
    #         return "#7F7F7F"

    #     bars = ax.barh(y_pos, importances, height=0.7, 
    #                     color=[get_color(c) for c in clusters], 
    #                     edgecolor="black", linewidth=0.5)

    #     # -----------------------------
    #     # 4. Axes & Spines (The Fix for Gaps)
    #     # -----------------------------
    #     ax.set_yticks(y_pos)
    #     ax.set_yticklabels(labels, fontsize=8) 
    #     ax.set_xlabel("Global Importance", fontsize=9, fontweight='bold')
        
    #     # REMOVE TOP/BOTTOM GAPS
    #     ax.set_ylim(min(y_pos) - 0.6, max(y_pos) + 0.6)
        
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)

    #     max_val = importances.max() if len(importances) > 0 else 1
    #     ax.set_xlim(0, max_val * 1.25) 

    #     # -----------------------------
    #     # 5. Value Annotations
    #     # -----------------------------
    #     for bar, value in zip(bars, importances):
    #         x = bar.get_width()
    #         y = bar.get_y() + bar.get_height() / 2
    #         offset = max_val * 0.015
    #         if x > 0.20 * max_val:
    #             ax.text(x - offset, y, f"{value:.3f}", 
    #                     va="center", ha="right", fontsize=7, color="white", fontweight='bold')
    #         else:
    #             ax.text(x + offset, y, f"{value:.3f}", 
    #                     va="center", ha="left", fontsize=7, color="black")

    #     # -----------------------------
    #     # 6. Title & Legend
    #     # -----------------------------
    #     title_parts = [model_name, dataset_name, f"Class: {class_name}" if class_name else None]
    #     ax.set_title(" – ".join(filter(None, title_parts)), fontsize=10, pad=12, fontweight='bold')

    #     legend_elements = [Patch(facecolor="#4C72B0", label="Incr."),
    #                     Patch(facecolor="#DD8452", label="Decr."),
    #                     Patch(facecolor="#55A868", label="L. Max"),
    #                     Patch(facecolor="#8172B3", label="L. Min")]
    #     ax.legend(handles=legend_elements, loc="lower right", frameon=True, fontsize=7, ncol=2)

    #     # -----------------------------
    #     # 7. Faithfulness (Fixed Offset - Left Aligned)
    #     # -----------------------------
    #     if global_faithfulness is not None:
    #         ax.annotate(
    #             f"Global Faithfulness: {global_faithfulness:.3f}",
    #             xy=(0, 0), xycoords='axes fraction',
    #             xytext=(0, -45), textcoords='offset points',
    #             ha="left", va="top", fontsize=8, fontweight="bold",
    #             bbox=dict(boxstyle="round,pad=0.3", facecolor="#F8F9F9", edgecolor="#D5D8DC")
    #         )

    #     # -----------------------------
    #     # 8. Save both PDF and PNG
    #     # -----------------------------
    #     ax.grid(axis="x", linestyle="--", alpha=0.3, linewidth=0.5)
    #     ax.set_axisbelow(True)

    #     base_name = f"{dataset_name}_{class_name}" if class_name else f"{dataset_name}"
        
    #     if self.output_dir is not None:
    #         # Save Vector (Best for PDF LaTeX)
    #         fig.savefig(os.path.join(self.output_dir, f"{base_name}.pdf"), bbox_inches="tight")
    #         # Save Raster (Best for presentations/web)
    #         fig.savefig(os.path.join(self.output_dir, f"{base_name}.png"), dpi=600, bbox_inches="tight")
    #         plt.close(fig)
    #     else:
    #         plt.show()

    
        
    @staticmethod
    def to_serializable(obj):
        """Convert any object to a JSON-serializable form (recursive)."""
        import numpy as np

        # primitives
        if obj is None or isinstance(obj, (str, bool, int, float)):
            return obj

        # numpy arrays → lists
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        # numpy scalars → python scalars
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()

        # lists / tuples → recurse
        if isinstance(obj, (list, tuple, set)):
            return [SPLOMATCE.to_serializable(x) for x in obj]

        # dictionaries → recurse
        if isinstance(obj, dict):
            return {str(k): SPLOMATCE.to_serializable(v) for k, v in obj.items()}

        # objects with attributes → serialize their __dict__
        if hasattr(obj, "__dict__"):
            return SPLOMATCE.to_serializable(obj.__dict__)

        # fallback: convert to string
        return str(obj)



