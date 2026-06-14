# Amex Offer Redemption Prediction

This project aims to predict the likelihood of a user redeeming an offer using advanced feature engineering and machine learning techniques. The workflow leverages LightGBM and extensive data processing to maximize predictive performance.

## Project Structure

```
├── train.ipynb                # Main notebook for data processing and model training
├── prediction.ipynb           # Notebook for generating predictions on test data
├── train_data.parquet         # Training data
├── test_data.parquet          # Test data
├── offer_metadata.parquet     # Offer metadata
├── add_trans.parquet          # Additional transaction data
├── add_event.parquet          # Additional event data
├── feature_cols.pkl           # Saved feature columns used by the model
├── lgbm_model.pkl             # Trained LightGBM model
├── top_spend_encoder.pkl      # Label encoder for top spend features
├── f_series_label_encoders.pkl# Label encoders for categorical f-series features
├── offer_agg.parquet          # Offer-level aggregated features
├── global_max_time.pkl        # Global max time for recency features
├── enhanced_lgbm_model.pkl    # (Optional) Enhanced model
├── enhanced_feature_cols.pkl  # (Optional) Enhanced feature columns
├── enhanced_categorical_features.pkl # (Optional) Enhanced categorical features
├── feature_importance_analysis.csv   # Feature importance analysis
├── r2_submission_file_ThakGyaHu.csv  # Example submission file
├── 685404e30cfdb_submission_template.csv # Submission template
├── data_dictionary.csv        # Data dictionary
├── catboost_info/             # CatBoost training logs and outputs
```

## Main Features

- **Feature Engineering:**
  - Smoothed offer-level CTR
  - Time-based features (day of week, hour)
  - User-offer interaction and recency features
  - Spending pattern features (short-term, long-term, ratios, shifts)
  - Label encoding for categorical features
- **Model Training:**
  - LightGBM classifier with time-based and K-Fold cross-validation
  - Hyperparameter tuning and early stopping
  - MAP@7 evaluation metric
- **Artifacts:**
  - Trained model and encoders saved for reproducibility
  - Feature columns and aggregated features stored for future use

## Getting Started

1. **Install Dependencies**
   - Python 3.8+
   - Required packages: `pandas`, `numpy`, `lightgbm`, `scikit-learn`
   - Install with:
     ```bash
     pip install pandas numpy lightgbm scikit-learn
     ```

2. **Run Training**
   - Open `train.ipynb` and execute all cells to process data and train the model.
   - Artifacts will be saved automatically.

3. **Generate Predictions**
   - Use `prediction.ipynb` to load the trained model and generate predictions on test data.

## Data Files

- `train_data.parquet`: Main training dataset
- `test_data.parquet`: Test dataset for prediction
- `offer_metadata.parquet`: Metadata for offers
- `add_trans.parquet`, `add_event.parquet`: Additional transaction and event data

## Outputs

- `lgbm_model.pkl`: Trained LightGBM model
- `feature_cols.pkl`: List of features used for training
- `r2_submission_file_ThakGyaHu.csv`: Example submission file

## Notebooks

- `train.ipynb`: Data loading, feature engineering, model training, evaluation
- `prediction.ipynb`: Model inference and submission file generation
