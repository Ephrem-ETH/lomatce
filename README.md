# LOMATCE: LOcal Model-Agnostic Time-series Classification Explanations

We propose LOcal Model-Agnostic Time-series Classification Explanations (LOMATCE, pronounced "lom-te-see"), a method akin to LIME, to explain deep-learning-based time series classifiers. LOMATCE uses parametrised event primitives (PEPs) to extract and parameterise events like increasing trends, decreasing trends, local maxima, and local minima from time series data. These PEPs capture temporal patterns, and facilitating the training of simple, interpretable models like linear regression. Discussing time series in terms of these familiar events makes the explanations more intuitive and comprehensible. This repository also includes SP-LOMATCE (L2GTX), which extends LOMATCE to generate global explanations by aggregating local explanations using submodular selection.

## File description

- **lomatce**: This directory contains code files.

  - **examples**: Includes notebook examples for each dataset.

    <!-- - output.log:This file contains the fidelity scores for each dataset and perturbation strategy. -->

    <!-- * Example: Evaluation result for FCN model architecture trained on the ECG dataset and zero perturbation strategy can be found at `examples\results\simulation\ecg200\fcn-zero--2024-04-22_22-04-47\output.log.` -->

  - **utils**: Contains utility files:

    - `helper_class.py`: Functions for clustering, explanation plots, etc.
    - `test_dataloader.py`: Dataloader for the test set.

  - `explainer.py`: Core method implementation, from PEP extraction to applying interpretable models like linear regression to mimic deep learning inference.
  - `perturbation.py`: Applies various perturbation strategies and generates neighboring samples.
  - `lomatce_simulation.py`: Runs the FCN model multiple times with random train-test splits to ensure robustness of results.
  - `lomatce_vs_baseline.py`: Code to compare LOMATCE against LIME, SHAP, Integrated Gradients (IG), and a random baseline.
  `sp_lomatce.py`: Implements **SP-LOMATCE (L2GTX)**, which aggregates LOMATCE local explanations into global explanations.

## Method Design

<!-- <img src="design\lomatce_design.png" alt="Method Design Diagram" width="100%" /> -->
### LOMATCE Method Design

![center w:13in](./design/lomatce_final.png)

**Fig 1:** Overview of the LOMATCE framework for generating local explanations.*

### SP-LOMATCE (L2GTX) Method Design
![center w:13in](./design/L2GX_new.png)

*Fig 2. Overview of the SP-LOMATCE (L2GTX) framework for generating global explanations from LOMATCE local explanations.*


## How to Use LOMATCE

### 1. Install LOMATCE

```bash
git clone https://github.com/yourusername/lomatce.git
cd lomatce
pip install -r requirements.txt
```

### 2. Explain Predictions with LOMATCE

#### Import LOMATCE:

```python
from lomatce.explainer import LomatceExplainer
```

#### Instantiate the Explainer:

```python

lomatce_explainer = LomatceExplainer(basic_dir='path/to/data_directory')
```

#### Explain an Instance:

```python
explanation = lomatce_explainer.explain_instance(
    origi_instance=your_ts_instance,
    classifier_fn=your_model_predict_function,
    num_perturbations=1000,  # Number of perturbations
    n_clusters=20,           # Number of event clusters
    top_n=15,                # Top features to show
    class_names=["Class1", "Class2"]
)
```

#### Visualise the Explanation:

```python
explanation.visualise(your_ts_instance, show_probas=True)
```

### 3. Get Explanation Summary

```Python
summary = explanation.get_explanation_summary()
print(summary)
```

You will get key info like local model prediction, original (black-box model) prediciton and local fidelity score.

#### Sample Explanation Output

<!-- ##### Table **1**: Explanation faithfulness, with 95% confidence interval, across various perturbation methods.

|   Dataset    |      Zero       |      Mean       |   Total_mean    |     Random      |
| :----------: | :-------------: | :-------------: | :-------------: | :-------------: |
|   **ECG**    | $0.82 \pm 0.02$ | $0.70 \pm 0.02$ | $0.81 \pm 0.01$ | $0.54 \pm 0.10$ |
| **GunPoint** | $0.72 \pm 0.02$ | $0.52 \pm 0.03$ | $0.75 \pm 0.08$ | $0.75 \pm 0.08$ | -->

Here's an example of LOMATCE highlighting important regions of a time series:

![center w:13in](./design/lomatce_explanation_example.png)
**Fig 3:** Explanation highlights segment significance, relevance scores, and event types (e.g., increasing, decreasing, maxima, minima).

You can explore the example noteboks for each dataset in the [`examples/`] folder.

### 4. How to Use SP-LOMATCE (L2GTX)

#### Import SP-LOMATCE
```python
from lomatce.sp_lomatce import SPLOMATCE
```

#### Instantiate the Explainer

```python
splomatce_explainer = SPLOMATCE(
    class_labels=class_labels,
    dataset_name="ECG200",
    lomatce_explainer=lomatce_explainer,
    predict_fn=predict_fn,
    output_dir=base_dir
)
```
> **Note:** `predict_fn` should return the class probability distribution and predicted class labels. If your model does not directly provide probabilities, use a wrapper function.

#### Generate Global Explanations

```python
result_p95 = splomatce_explainer.run(
    X_merged,
    y_merged,
    n_per_class=20,
    B=20,
    merge_percentile=95
)
```
#### Example Global Explanation

![SP-LOMATCE Global Explanation](./design/sp_lomatce_ecg200_normal_heartbeat.png)

**Fig 4.** Example of a global explanation generated by SP-LOMATCE (L2GTX)
### 5. To Reproduce the Experiment

##### To evaluate the model perfromance and local fidelity over multiple random train-val-test splits:

```python

python lomatce_simulation.py --model [model-name] --dataset [dataset-name] --num_runs [100] --class_labels [list-of-classes] --replacement_method random --num_samples 1000

```

##### Example

```
python lomatce_simulation.py -- model FCN --dataset Coffee --num_runs 100 --class_labels Arabica Robusta --replacement_method random --num_samples 1000
```

<!-- ##### Compare LOMATCE with Other XAI Methods -->

##### To compare LOMATCE against LIME, SHAP, IG and Random baseline.

```python
python lomatce_vs_baseline.py --dataset [dataset-name] --model [model-checkpoint] --class_labels [list-of-classes]
```

<!-- ## Usage

To run the the simulation of the experiment, use the following command:

- For FCN model

```
python fcn_simulation --dataset [dataset-name] --num_runs [100 ]  --class_labels [list of the class names]
``` -->

<!-- ## Requirments

- tsai (State-of-the-art Deep Learning library for Time Series and Sequences)
- python >= 3.8
- pytorch -->

## Citation

If you use **LOMATCE** in your research, please cite:

```bibtex
@ARTICLE{11216415,
  author={Mekonnen, Ephrem Tibebe and Longo, Luca and Dondio, Pierpaolo},
  journal={IEEE Access}, 
  title={LOMATCE: LOcal Model-Agnostic Time Series Classification Explanations}, 
  year={2025},
  volume={13},
  number={},
  pages={185218-185232},
  doi={10.1109/ACCESS.2025.3625442}}

```

If you use **SP-LOMATCE (L2GTX)** in your research, please cite:

```bibtex
@article{mekonnen2026l2gtx,
  title={L2GTX: From Local to Global Time Series Explanations},
  author={Mekonnen, Ephrem Tibebe and Longo, Luca and Rizzo, Lucas and Dondio, Pierpaolo},
  journal={arXiv preprint arXiv:2603.13065},
  year={2026}
}
```
