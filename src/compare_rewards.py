import pandas as pd
import numpy as np
import os

def main():
    # File paths
    llm_csv_path = 'doorkey_completed_llm.csv'
    multiverse_csv_path = 'doorkey_sampled_multiverse_qlearning.csv'
    
    if not os.path.exists(llm_csv_path) or not os.path.exists(multiverse_csv_path):
        print("Assicurati di eseguire lo script dalla cartella in cui si trovano i file CSV.")
        return

    # Load CSV files
    print(f"Caricamento {llm_csv_path}...")
    df_llm = pd.read_csv(llm_csv_path)
    
    print(f"Caricamento {multiverse_csv_path}...")
    df_multi = pd.read_csv(multiverse_csv_path)
    
    # Define observation and action columns (common columns to group by)
    merge_cols = [
        'obs_agent_x', 'obs_agent_y', 'obs_agent_dir', 
        'obs_key_pos', 'obs_door_open', 'obs_stage', 
        'action_taken'
    ]
    
    # Select only a single seed from the multiverse dataset
    target_seed = df_multi['seed_idx'].unique()[0]
    print(f"\nFiltraggio del multiverse dataset usando solo seed_idx = {target_seed}...")
    df_multi_single_seed = df_multi[df_multi['seed_idx'] == target_seed].copy()
    
    df_multi_single_seed.rename(columns={'reward_obtained': 'reward_multiverse'}, inplace=True)
    df_multi_agg = df_multi_single_seed.drop(columns=['seed_idx'], errors='ignore')
    
    # Rename LLM reward column for clarity
    df_llm_renamed = df_llm.rename(columns={'reward_obtained': 'reward_llm'})
    
    # Merge the datasets
    print("Unione dei dataset in base alle osservazioni e azioni...")
    merged_df = pd.merge(df_llm_renamed, df_multi_agg, on=merge_cols, how='inner')
    
    n_llm = len(df_llm)
    n_multi = len(df_multi_agg)
    n_merged = len(merged_df)
    
    print(f"Righe in LLM CSV: {n_llm}")
    print(f"Righe in Multiverse CSV (per il seed {target_seed}): {n_multi}")
    print(f"Righe combinate (matching): {n_merged}")
    
    if n_merged == 0:
        print("\nNessuna corrispondenza trovata tra i due file per lo stesso set di osservazioni/azioni.")
        return
        
    # Calculate differences
    merged_df['diff'] = merged_df['reward_llm'] - merged_df['reward_multiverse']
    merged_df['abs_diff'] = merged_df['diff'].abs()
    
    # Metrics
    mae = merged_df['abs_diff'].mean()
    mse = (merged_df['diff'] ** 2).mean()
    rmse = np.sqrt(mse)
    
    # Correlation
    # Compute correlation only if variance > 0
    if merged_df['reward_llm'].std() > 0 and merged_df['reward_multiverse'].std() > 0:
        correlation = merged_df['reward_llm'].corr(merged_df['reward_multiverse'])
    else:
        correlation = float('nan')
    
    print("\n" + "="*40)
    print("RISULTATI DI SIMILARITÀ DEI REWARD")
    print("="*40)
    print(f"Mean Absolute Error (MAE) : {mae:.6f}")
    print(f"Mean Squared Error (MSE)  : {mse:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    if not np.isnan(correlation):
        print(f"Correlazione di Pearson   : {correlation:.6f}")
    else:
        print("Correlazione di Pearson   : N/A (varianza zero in uno dei dataset)")
        
    # Show sensible samples (where rewards are not just the baseline -0.001)
    print("\n" + "="*40)
    print("ESEMPI DI REWARD 'NON BANALI' (Diversi dal classico -0.001)")
    print("="*40)
    
    baseline = -0.001
    meaningful_rows = merged_df[(abs(merged_df['reward_llm'] - baseline) > 0.001) | 
                                (abs(merged_df['reward_multiverse'] - baseline) > 0.001)]
    
    if len(meaningful_rows) > 0:
        # Prendi alcuni dei casi con la differenza più alta
        print("\n--> Top 5 differenze più alte tra reward 'non banali':")
        top_diffs = meaningful_rows.sort_values(by='abs_diff', ascending=False).head(5)
        print(top_diffs[merge_cols + ['reward_llm', 'reward_multiverse', 'abs_diff']].to_string(index=False))
        
        # Prendi un sample casuale tra quelli non banali
        sample_size = min(5, len(meaningful_rows))
        print(f"\n--> {sample_size} esempi casuali di reward 'non banali':")
        sensible_sample = meaningful_rows.sample(sample_size, random_state=42)
        print(sensible_sample[merge_cols + ['reward_llm', 'reward_multiverse', 'abs_diff']].to_string(index=False))
    else:
        print("Tutti i reward in entrambi i dataset sono prossimi a -0.001!")

if __name__ == "__main__":
    main()
