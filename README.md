# Microbiota-Based IBD Classifier

This project processes microbial taxonomic profiles to classify Inflammatory Bowel Disease (IBD) subtypes, specifically **Ulcerative Colitis (UC)** and **Crohn’s Disease (CD)**, using machine learning techniques.

## Dataset

- **Microbiota Profiles**: `taxonomic_profiles.tsv`
- **Metadata**: `hmp2_metadata_2018-08-20.csv`
- Source: [Human Microbiome Project (HMP2)](https://ibdmdb.org/)

## Pipeline Overview

1. **Data Loading**
   - Reads taxonomic profiles and metadata.
   - Transposes taxonomic data if needed and merges both datasets on sample ID.

2. **Filtering & Preprocessing**
   - Retains only features (OTUs) with non-zero variance.
   - Keeps only samples from the most common biopsy location.
   - Filters for samples diagnosed with `UC` or `CD`.
   - One-hot encodes biopsy location.
   - Scales features using `StandardScaler`.

3. **Feature Selection**
   - Performs **Kruskal-Wallis test** on each bacterial feature.
   - Selects top 10 bacteria with p < 0.05 and largest fold change.

4. **Dimensionality Reduction**
   - Applies **PCA** to visualize the data in 2D.

5. **Model Training & Evaluation**
   - Models used:
     - `SVC` (with pipeline and grid search)
     - `RandomForest`
     - `XGBoost`
     - `CatBoost`
   - Evaluates both **VotingClassifier** and **StackingClassifier**.
   - Hyperparameter tuning with `GridSearchCV`.
   - Optimizes for **recall**.
   - Saves confusion matrix plot and trained models (`.joblib` files).

## Output

- `confusion_matrix_voting_holdout.png`: Confusion matrix image.
- `voting_weights_comparison.csv`: Recall and AUC scores for different weight combinations.
- `best_voting_model_holdout.joblib`: Trained ensemble model.
- `stacking_model_holdout.joblib`: Trained stacking model.

## How to Run

You can run this script inside a Kaggle notebook or locally if you have the data and required packages. Just make sure you upload both input files and run all cells in order.

## Dependencies

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`
- `xgboost`
- `catboost`
- `joblib`, `scipy`

Install via:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost joblib

