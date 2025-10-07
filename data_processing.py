import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def handle_sentinels(final_df):
    # Sentinel values to replace
    sentinels = [55, 555, 77, 777, 88, 99, 888, 999]
    
    # Select only numeric columns
    numeric_cols = final_df.select_dtypes(include=np.number).columns
    
    for col in numeric_cols:
        # Count how many sentinel values exist
        n_replaced = final_df[col].isin(sentinels).sum()
        if n_replaced == 0:
            continue  # skip if no sentinels in this column
    
        # Compute median excluding sentinel values
        median_val = final_df.loc[~final_df[col].isin(sentinels), col].median()
    
        # Replace sentinel values with the column median
        final_df[col] = final_df[col].replace(sentinels, median_val)
    
        # Print summary
       # print(f"{col}: replaced {n_replaced} sentinel value(s) with median = {median_val:.3f}")
    return final_df

def drop_unbalanced_columns(final_df, thresh):
    drop_columns = []
    for col_name in final_df.select_dtypes(include=[np.number]).columns:
        #print(col_name)
        x = final_df[col_name] 
        num_non_zero = x.sum()
        if num_non_zero < thresh:
            print(col_name)
            drop_columns.append(col_name)
    
    final_df.drop(columns=drop_columns, inplace=True)
    return final_df
        

def bin_frame(final_df, skip_columns, num_bins =16, alpha=3):
    numeric_cols = final_df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        series = final_df[col].dropna()
        #print(f"\nColumn: {col}")
        #print(f"Raw Min: {series.min():.3f}, Raw Max: {series.max():.3f}")
        if col in skip_columns:
            #print(f"\nSkipping column: {col}")
            continue
        # Skip binary columns
        if set(series.unique()).issubset({0, 1}):
          #  print("Binary column — skipping binning.")
            continue
    
        # Compute mean and std
        mean, std = series.mean(), series.std()
    
        # Clip outliers to ±2*std and bin directly
        final_df[col] = pd.cut(
            final_df[col].clip(lower=mean - alpha*std, upper=mean + alpha*std),
            bins=num_bins,
            labels=False
        )
    
        # Print summary
        #print(f"Binned range: {mean - alpha*std:.3f} to {mean + alpha*std:.3f}")
        #print(f"Binned sample (first 10): {final_df[col].head(10).tolist()}")
    return final_df
    
def clean_and_clip_df(df, min_val, max_val):
    df = df.clip(lower=min_val, upper=max_val) 
    return df
    
def bin_features(df, col_list, num_bins):     
    for col in col_list:
        if col in df.columns:
            df[col] = clean_and_clip_df(df[col],min_val = df[col].mean() - 3*df[col].std(),max_val=df[col].mean() + 3*df[col].std()) 
            df[col] = (
                    pd.cut(df[col], bins=num_bins, labels=False)
                )
    return df

def extract_top_features(df, condition_col, num_top_features):
    
    
    measurement_cols = df.columns.drop(condition_col)
    
    # --- Ensure numeric ---
    df_numeric = df.apply(pd.to_numeric, errors='coerce')
    
    # --- Compute average absolute correlation ---
    avg_correlations = {
        col: abs(df_numeric[condition_col].corr(df_numeric[col])) 
        for col in measurement_cols
    }
    
    # --- Convert to DataFrame and get top features ---
    avg_corr_df = (
        pd.DataFrame(list(avg_correlations.items()), columns=['Measurement', 'AvgAbsCorrelation'])
        .sort_values(by='AvgAbsCorrelation', ascending=False)
    )
    
    top_features = avg_corr_df.head(num_top_features+1)
    top_features = top_features[top_features['Measurement'] != 'participant_id']
    top_features = top_features[:num_top_features]
    return top_features


def extract_df_with_features(df, study_groups, num_top_features):
    df_temp = df[df['study_group_id'].isin(study_groups)]
    condition_col = 'study_group_id'
    top_features = extract_top_features(df_temp, condition_col, num_top_features)
    print(top_features)
    print(list(top_features["Measurement"]))
    
    # Convert to a Python list
    key_feats = top_features["Measurement"].tolist()
    key_feats.append("participant_id")
    key_feats.append("study_group_id")
    key_feats.append("recommended_split")
    # Now select from df
    df2 = df[key_feats]
    df2 = df2[df2['study_group_id'].isin(study_groups)]
    df2['healthy'] = df2['study_group_id']-study_groups[0]
    
    df2.drop(columns=['study_group_id'])
    
    return df2