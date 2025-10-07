# AIREADI_Multimodal_Diabetes_Risk_Prediction

## Overview
This repository contains code for training models for AI-based multimodal diabetes progression prediction using clinical features, blood test labs, continuous glucose monitoring, and self-reported questionnaire data. 

---

## Universal Deep Learning Model
Run **`train_universal_deep_learning_model.ipynb`** to train a universal model on the top 16 features for different study group splits, and then test across all different feature sets. 

---

## Random Forest
Run **`RandomForest.ipynb`** to train and test a random forest for each choice of features. 

---

## Data Extraction from AI READI
To extract and preprocess data from the AI READI dataset:

1. Run **`final_clinicaldata.ipynb`** to generate  
   `all_features_all_patients_binned.csv` — creates a dataframe with ~270 features containing measurements, observations, and conditions in the AI-READI dataset with data processing done to bin the features.

2. Run **`create_df_cgm_data.ipynb`** to generate  
   `dataframe_glucose_feats.csv` — contains extracted CGM metrics for all patients in the AI-READI dataset.

3. Run **`create_dataset.ipynb`** to generate  
   `dataset_progression_analysis_0_1.csv`, `dataset_progression_analysis_1_2.csv`, and `dataset_progression_analysis_2_3.csv` — contains the top 16 correlated features for each cohort of patients.

> **Note:** You must have the **AI READI dataset** downloaded locally to execute these scripts.
