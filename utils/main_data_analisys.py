#%%
"""
Main data analysis script for loading and analyzing debate summaries.
"""

import pandas as pd
import os
import sys
import argparse
import yaml
from pathlib import Path

# Get the project root directory (parent of 'code' folder)
PROJECT_ROOT = Path(__file__).parent.parent

# Default path to the Excel file (can be overridden via command line argument)
DEFAULT_EXCEL_FILE_PATH = PROJECT_ROOT / "data" / "results" / "results" / "zipped_results" / "20251102_171219_4omini_4omini" / "all_debates_summary.xlsx"


def load_debates_summary(excel_file_path):
    """
    Load the all_debates_summary.xlsx file.
    
    Args:
        excel_file_path: Path to the Excel file to load
    
    Returns:
        pd.DataFrame: The loaded Excel data as a pandas DataFrame
    """
    if not os.path.exists(excel_file_path):
        raise FileNotFoundError(f"Excel file not found at: {excel_file_path}")
    
    print(f"Loading Excel file from: {excel_file_path}")
    df = pd.read_excel(excel_file_path)
    print(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
    print(f"\nColumn names: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    return df


def parse_llm_names_from_settings(settings_path):
    """
    Parse LLM model names from a settings YAML file.
    
    Args:
        settings_path: Path to the settings YAML file
        
    Returns:
        Tuple of (persuader_model_name, debater_model_name) or (None, None) if not found
    """
    try:
        with open(settings_path, 'r', encoding='utf-8') as f:
            settings = yaml.safe_load(f)
        
        if not settings or 'agent_configurations' not in settings:
            print(f"Warning: Could not find 'agent_configurations' in settings file: {settings_path}")
            return None, None
        
        agent_configs = settings['agent_configurations']
        
        # Try to find the first configuration (usually Default_No_Helper)
        # The model names should be consistent across all helper types
        config_name = None
        for name in ['Default_No_Helper', 'Default_Fallacy_Helper', 'Default_Logical_Helper']:
            if name in agent_configs:
                config_name = name
                break
        
        # If none of the standard names found, use the first one
        if config_name is None and agent_configs:
            config_name = list(agent_configs.keys())[0]
        
        if config_name is None:
            print(f"Warning: No agent configurations found in settings file: {settings_path}")
            return None, None
        
        config = agent_configs[config_name]
        
        # Extract persuader and debater model names
        persuader_model = None
        debater_model = None
        
        if 'persuader' in config and 'model_name' in config['persuader']:
            persuader_model = config['persuader']['model_name']
        
        if 'debater' in config and 'model_name' in config['debater']:
            debater_model = config['debater']['model_name']
        
        if persuader_model and debater_model:
            print(f"Parsed LLM names from settings:")
            print(f"  Persuader (LLM1): {persuader_model}")
            print(f"  Debater (LLM2): {debater_model}")
            return persuader_model, debater_model
        else:
            print(f"Warning: Could not extract model names from configuration '{config_name}'")
            return None, None
            
    except FileNotFoundError:
        print(f"Warning: Settings file not found: {settings_path}")
        return None, None
    except yaml.YAMLError as e:
        print(f"Warning: Error parsing YAML file {settings_path}: {e}")
        return None, None
    except Exception as e:
        print(f"Warning: Error reading settings file {settings_path}: {e}")
        return None, None


def normalize_helper_type_name(helper_type):
    """
    Normalize helper type name to standard format.
    
    Args:
        helper_type: Original helper type name
        
    Returns:
        Normalized helper type name
    """
    helper_type_mapping = {
        'No_Helper': 'No Helper',
        'Default_No_Helper': 'No Helper',
        'Fallacy_Helper': 'Fallacy Helper',
        'Logical_Helper': 'Logical Helper'
    }
    return helper_type_mapping.get(helper_type, helper_type)


def get_helper_type_order():
    """
    Returns the standard order for helper types in visualizations.
    
    Returns:
        List of helper type names in order: No Helper, Logical Helper, Fallacy Helper
    """
    return ['No Helper', 'Logical Helper', 'Fallacy Helper']


def get_figures_subfolder(excel_file_path):
    """
    Generate a subfolder name based on the Excel file path.
    Creates a unique subfolder name based on the relative path from the data folder.
    
    Args:
        excel_file_path: Path to the Excel file
    
    Returns:
        Path: Path to the figures subfolder
    """
    excel_path = Path(excel_file_path)
    
    # Get relative path from data folder
    try:
        relative_path = excel_path.relative_to(PROJECT_ROOT / "data")
        # Remove the filename and get the directory path
        folder_parts = relative_path.parent.parts
        # Create folder name from path parts, joining with underscores
        if folder_parts:
            folder_name = "_".join(folder_parts)
        else:
            folder_name = excel_path.parent.name
    except ValueError:
        # If not relative to data folder, use parent directory name
        folder_name = excel_path.parent.name
    
    # Create figures directory with subfolder
    figures_dir = PROJECT_ROOT / "figures" / folder_name
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    return figures_dir

def run_analysis_for_file(excel_file_path):
    """
    Run the complete analysis for a single Excel file.
    
    Args:
        excel_file_path: Path to the Excel file to analyze
    """
    print("\n" + "="*80)
    print(f"Processing: {excel_file_path}")
    print("="*80)
    
    # Load the data
    df = load_debates_summary(excel_file_path)
    
    # Get the appropriate figures subfolder
    FIGURES_DIR = get_figures_subfolder(excel_file_path)
    print(f"\nFigures will be saved to: {FIGURES_DIR}")
    
    # Display basic information about the dataset
    print("\n" + "="*50)
    print("Dataset Info:")
    print("="*50)
    print(df.info())
    print("\n" + "="*50)
    print("Dataset Statistics:")
    print("="*50)
    print(df.describe())

    # %%
    import seaborn as sns
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    import ast

    # Ensure figures directory exists
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Figures directory: {FIGURES_DIR}")

    # Check if required columns exist
    has_finish_reason = 'finish_reason' in df.columns
    has_conviction_rates = 'conviction_rates_vector' in df.columns
    has_feedback_tags = 'feedback_tags_vector' in df.columns
    
    # Filter for debates where debater was convinced and other cases (if column exists)
    if has_finish_reason:
        convinced_df = df[df['finish_reason'] == 'Debater convinced']
        not_convinced_df = df[df['finish_reason'] != 'Debater convinced']
        print(f"Number of convinced debates: {len(convinced_df)}")
        print(f"Number of other debates: {len(not_convinced_df)}")
    else:
        print("Warning: 'finish_reason' column not found. Skipping finish_reason-based analyses.")
        convinced_df = pd.DataFrame()
        not_convinced_df = pd.DataFrame()
    
    # Create a figure with two subplots side by side (only if finish_reason exists)
    if has_finish_reason:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left subplot: Debater Convinced
        if len(convinced_df) > 0:
            sns.histplot(data=convinced_df, x='rounds', hue='helper_type', multiple='dodge', bins=20, ax=axes[0])
            axes[0].set_title('Debater Convinced', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Number of Rounds')
            axes[0].set_ylabel('Count')
        else:
            axes[0].text(0.5, 0.5, 'No convinced debates found', 
                        ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title('Debater Convinced', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Number of Rounds')
            axes[0].set_ylabel('Count')
        
        # Right subplot: Other Cases
        if len(not_convinced_df) > 0:
            sns.histplot(data=not_convinced_df, x='rounds', hue='helper_type', multiple='dodge', bins=20, ax=axes[1])
            axes[1].set_title('Other Cases', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Number of Rounds')
            axes[1].set_ylabel('Count')
        else:
            axes[1].text(0.5, 0.5, 'No other debates found', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Other Cases', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Number of Rounds')
            axes[1].set_ylabel('Count')
        
        plt.suptitle('Distribution of Rounds by Finish Reason', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save the figure
        output_path = FIGURES_DIR / "rounds_distribution_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
        plt.close()  # Close the figure to free memory
    else:
        print("Skipping rounds_distribution_comparison.png (finish_reason column missing)")

    # %% Basic Analysis Plots
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    
    # 1. Distribution of Finish Reasons (if column exists)
    if has_finish_reason:
        fig, ax = plt.subplots(figsize=(12, 6))
        finish_reason_counts = df['finish_reason'].value_counts()
        # Group error messages together
        error_mask = finish_reason_counts.index.str.contains('ERROR', na=False)
        error_count = finish_reason_counts[error_mask].sum()
        non_error_counts = finish_reason_counts[~error_mask]
        if error_count > 0:
            plot_counts = pd.concat([non_error_counts, pd.Series({'ERROR': error_count})])
        else:
            plot_counts = non_error_counts
        
        bars = ax.bar(range(len(plot_counts)), plot_counts.values, color='steelblue', 
                      yerr=np.sqrt(plot_counts.values), capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})
        ax.set_xticks(range(len(plot_counts)))
        ax.set_xticklabels(plot_counts.index, rotation=45, ha='right')
        ax.set_title('Distribution of Finish Reasons', fontsize=14, fontweight='bold')
        ax.set_xlabel('Finish Reason', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "finish_reasons_distribution.png", dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {FIGURES_DIR / 'finish_reasons_distribution.png'}")
        plt.close()
    else:
        print("Skipping finish_reasons_distribution.png (finish_reason column missing)")
    
    # 2. Distribution of Helper Types
    fig, ax = plt.subplots(figsize=(8, 6))
    helper_counts = df['helper_type'].value_counts()
    bars = ax.bar(range(len(helper_counts)), helper_counts.values, 
                  color=['#1f77b4', '#ff7f0e', '#2ca02c'],
                  yerr=np.sqrt(helper_counts.values), capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})
    ax.set_xticks(range(len(helper_counts)))
    ax.set_xticklabels(helper_counts.index, rotation=0)
    ax.set_title('Distribution of Helper Types', fontsize=14, fontweight='bold')
    ax.set_xlabel('Helper Type', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.tick_params(axis='x', rotation=0, labelsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "helper_types_distribution.png", dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {FIGURES_DIR / 'helper_types_distribution.png'}")
    plt.close()
    
    # 3. Result Distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    result_counts = df['result'].value_counts().sort_index()
    bars = ax.bar(range(len(result_counts)), result_counts.values, color='coral',
                  yerr=np.sqrt(result_counts.values), capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})
    ax.set_xticks(range(len(result_counts)))
    ax.set_xticklabels(result_counts.index, rotation=0)
    ax.set_title('Distribution of Results', fontsize=14, fontweight='bold')
    ax.set_xlabel('Result', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.tick_params(axis='x', rotation=0, labelsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "result_distribution.png", dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {FIGURES_DIR / 'result_distribution.png'}")
    plt.close()
    
    # 4. Success Rate by Helper Type (excluding 1-round successes for more valid statistics)
    fig, ax = plt.subplots(figsize=(8, 6))
    success_by_helper = df.groupby('helper_type').apply(
        lambda x: (
            # Count convinced debates with 2+ rounds
            len(x[(x['result'] > 0) & (x['rounds'] > 1)]) / 
            # Divide by conclusive debates (convinced 2+ rounds + not convinced)
            len(x[((x['result'] > 0) & (x['rounds'] > 1)) | (x['result'] == 0)])
            * 100
            if len(x[((x['result'] > 0) & (x['rounds'] > 1)) | (x['result'] == 0)]) > 0 
            else 0
        )
    )
    
    # Normalize helper type names and order them
    helper_type_order = get_helper_type_order()
    normalized_success = {}
    for helper_type, rate in success_by_helper.items():
        normalized_name = normalize_helper_type_name(helper_type)
        normalized_success[normalized_name] = rate
    
    # Order according to desired order
    ordered_helpers = []
    ordered_rates = []
    for desired_helper in helper_type_order:
        if desired_helper in normalized_success:
            ordered_helpers.append(desired_helper)
            ordered_rates.append(normalized_success[desired_helper])
    
    # Add any remaining helper types not in the standard order
    for helper_type, rate in normalized_success.items():
        if helper_type not in ordered_helpers:
            ordered_helpers.append(helper_type)
            ordered_rates.append(rate)
    
    # Calculate standard error for proportions in the same order
    success_errors = []
    for helper in ordered_helpers:
        # Find original helper type names that map to this normalized name
        matching_helpers = [ht for ht in df['helper_type'].unique() 
                           if normalize_helper_type_name(ht) == helper]
        group = df[df['helper_type'].isin(matching_helpers)]
        # Exclude 1-round successes and errors
        convinced_2plus = len(group[(group['result'] > 0) & (group['rounds'] > 1)])
        not_convinced = len(group[group['result'] == 0])
        conclusive_2plus = convinced_2plus + not_convinced
        
        if conclusive_2plus > 0:
            p = convinced_2plus / conclusive_2plus
            se = np.sqrt(p * (1 - p) / conclusive_2plus) * 100
        else:
            se = 0
        success_errors.append(se)
    
    bars = ax.bar(range(len(ordered_helpers)), ordered_rates, 
                  color=['#1f77b4', '#ff7f0e', '#2ca02c'],
                  yerr=success_errors, capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})
    ax.set_xticks(range(len(ordered_helpers)))
    ax.set_xticklabels(ordered_helpers, rotation=0)
    ax.set_title('Success Rate by Helper Type', fontsize=14, fontweight='bold')
    ax.set_xlabel('Helper Type', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.tick_params(axis='x', rotation=0, labelsize=10)
    ax.set_ylim([0, 100])
    for i, (v, err) in enumerate(zip(ordered_rates, success_errors)):
        ax.text(i, v + err + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "success_rate_by_helper.png", dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {FIGURES_DIR / 'success_rate_by_helper.png'}")
    plt.close()
    
    # 5. Average Rounds by Helper Type
    fig, ax = plt.subplots(figsize=(8, 6))
    rounds_stats = df.groupby('helper_type')['rounds'].agg(['mean', 'sem']).sort_values('mean', ascending=False)
    avg_rounds_by_helper = rounds_stats['mean']
    rounds_errors = rounds_stats['sem']
    
    bars = ax.bar(range(len(avg_rounds_by_helper)), avg_rounds_by_helper.values,
                  color=['#1f77b4', '#ff7f0e', '#2ca02c'],
                  yerr=rounds_errors.values, capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})
    ax.set_xticks(range(len(avg_rounds_by_helper)))
    ax.set_xticklabels(avg_rounds_by_helper.index, rotation=0)
    ax.set_title('Average Number of Rounds by Helper Type', fontsize=14, fontweight='bold')
    ax.set_xlabel('Helper Type', fontsize=12)
    ax.set_ylabel('Average Rounds', fontsize=12)
    ax.tick_params(axis='x', rotation=0, labelsize=10)
    for i, (v, err) in enumerate(zip(avg_rounds_by_helper.values, rounds_errors.values)):
        ax.text(i, v + err + 0.1, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "avg_rounds_by_helper.png", dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {FIGURES_DIR / 'avg_rounds_by_helper.png'}")
    plt.close()
    
    # 6. Rounds Distribution by Helper Type (Box Plot)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='helper_type', y='rounds', hue='helper_type', ax=ax, palette=['#1f77b4', '#ff7f0e', '#2ca02c'], legend=False)
    ax.set_title('Distribution of Rounds by Helper Type', fontsize=14, fontweight='bold')
    ax.set_xlabel('Helper Type', fontsize=12)
    ax.set_ylabel('Number of Rounds', fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rounds_boxplot_by_helper.png", dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {FIGURES_DIR / 'rounds_boxplot_by_helper.png'}")
    plt.close()
    
    # 7. Result Distribution by Helper Type (Box Plot)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='helper_type', y='result', hue='helper_type', ax=ax, palette=['#1f77b4', '#ff7f0e', '#2ca02c'], legend=False)
    ax.set_title('Distribution of Results by Helper Type', fontsize=14, fontweight='bold')
    ax.set_xlabel('Helper Type', fontsize=12)
    ax.set_ylabel('Result', fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "result_boxplot_by_helper.png", dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {FIGURES_DIR / 'result_boxplot_by_helper.png'}")
    plt.close()
    
    # 8. Finish Reason Distribution by Helper Type (Stacked Bar Chart)
    fig, ax = plt.subplots(figsize=(14, 6))
    # Simplify finish reasons for better visualization
    df_plot = df.copy()
    df_plot['finish_reason_simple'] = df_plot['finish_reason'].apply(
        lambda x: 'ERROR' if 'ERROR' in str(x) else x
    )
    # Get top finish reasons
    top_reasons = df_plot['finish_reason_simple'].value_counts().head(5).index.tolist()
    df_plot['finish_reason_simple'] = df_plot['finish_reason_simple'].apply(
        lambda x: x if x in top_reasons else 'Other'
    )
    
    finish_by_helper = pd.crosstab(df_plot['helper_type'], df_plot['finish_reason_simple'])
    finish_by_helper.plot(kind='bar', stacked=True, ax=ax, colormap='Set3')
    ax.set_title('Finish Reason Distribution by Helper Type', fontsize=14, fontweight='bold')
    ax.set_xlabel('Helper Type', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.tick_params(axis='x', rotation=0, labelsize=10)
    ax.legend(title='Finish Reason', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "finish_reason_by_helper_stacked.png", dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {FIGURES_DIR / 'finish_reason_by_helper_stacked.png'}")
    plt.close()
    
    # 9. Summary Statistics Table Visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    summary_stats = df.groupby('helper_type').agg({
        'rounds': ['mean', 'std', 'min', 'max'],
        'result': ['mean', 'std', 'min', 'max']
    }).round(2)
    
    summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]
    summary_stats = summary_stats.reset_index()
    
    table = ax.table(cellText=summary_stats.values,
                     colLabels=summary_stats.columns,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(summary_stats.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Summary Statistics by Helper Type', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(FIGURES_DIR / "summary_statistics_table.png", dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {FIGURES_DIR / 'summary_statistics_table.png'}")
    plt.close()
    
    # 10. Correlation Heatmap (if numeric columns exist)
    fig, ax = plt.subplots(figsize=(8, 6))
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax, square=True)
        ax.set_title('Correlation Heatmap of Numeric Variables', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {FIGURES_DIR / 'correlation_heatmap.png'}")
    plt.close()
    
    # 11. Boxplot of Last Conviction Rate by Finish Reason (if columns exist)
    if has_finish_reason and has_conviction_rates:
        # Parse conviction_rates_vector and extract last value
        def extract_last_conviction_rate(conviction_str):
            """Extract the last value from conviction_rates_vector string."""
            try:
                if pd.isna(conviction_str):
                    return np.nan
                conviction_list = ast.literal_eval(str(conviction_str))
                if isinstance(conviction_list, list) and len(conviction_list) > 0:
                    return float(conviction_list[-1])
                return np.nan
            except (ValueError, SyntaxError, TypeError):
                return np.nan
        
        df['last_conviction_rate'] = df['conviction_rates_vector'].apply(extract_last_conviction_rate)
        
        # Filter for the two specific finish reasons
        finish_reasons_to_plot = ['Debater convinced', 'Max rounds reached']
        df_conviction = df[df['finish_reason'].isin(finish_reasons_to_plot)].copy()
        df_conviction = df_conviction.dropna(subset=['last_conviction_rate'])
        
        if len(df_conviction) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=df_conviction, x='finish_reason', y='last_conviction_rate', 
                       hue='finish_reason', ax=ax, palette=['#2ecc71', '#e74c3c'], 
                       width=0.6, legend=False)
            ax.set_title('Last Conviction Rate by Finish Reason', fontsize=14, fontweight='bold')
            ax.set_xlabel('Finish Reason', fontsize=12)
            ax.set_ylabel('Last Conviction Rate', fontsize=12)
            ax.tick_params(axis='x', rotation=45, labelsize=10)
            
            # Add sample size annotations
            for i, reason in enumerate(finish_reasons_to_plot):
                n = len(df_conviction[df_conviction['finish_reason'] == reason])
                if n > 0:
                    ax.text(i, ax.get_ylim()[1] * 0.95, f'n={n}', ha='center', va='top', 
                           fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "last_conviction_rate_boxplot.png", dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {FIGURES_DIR / 'last_conviction_rate_boxplot.png'}")
            print(f"Number of valid conviction rate values: {len(df_conviction)}")
            plt.close()
        else:
            print("No valid data found for conviction rate boxplot")
            plt.close()
    else:
        print("Skipping last_conviction_rate_boxplot.png (required columns missing)")
    
    # 12. Ratio of "Debater convinced" vs "Max rounds reached" by Helper Type (if finish_reason exists)
    if has_finish_reason:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Filter for the two finish reasons
        finish_reasons_filter = ['Debater convinced', 'Max rounds reached']
        df_ratio = df[df['finish_reason'].isin(finish_reasons_filter)].copy()
        
        # Calculate ratios for each helper type
        ratio_data = []
        helper_types = df_ratio['helper_type'].unique()
        
        for helper in helper_types:
            helper_data = df_ratio[df_ratio['helper_type'] == helper]
            convinced_count = len(helper_data[helper_data['finish_reason'] == 'Debater convinced'])
            max_rounds_count = len(helper_data[helper_data['finish_reason'] == 'Max rounds reached'])
            
            if max_rounds_count > 0:
                ratio = convinced_count / max_rounds_count
                # Calculate error using propagation of errors for ratio
                ratio_error = ratio * np.sqrt((1/convinced_count if convinced_count > 0 else 0) + 
                                              (1/max_rounds_count if max_rounds_count > 0 else 0))
            else:
                ratio = np.inf if convinced_count > 0 else 0
                ratio_error = 0
            
            ratio_data.append({
                'helper_type': helper,
                'ratio': ratio,
                'ratio_error': ratio_error,
                'convinced': convinced_count,
                'max_rounds': max_rounds_count
            })
        
        ratio_df = pd.DataFrame(ratio_data)
        ratio_df = ratio_df.sort_values('ratio', ascending=False)
        
        # Handle infinite ratios (set to max ratio + 1 for visualization)
        max_finite_ratio = ratio_df[ratio_df['ratio'] != np.inf]['ratio'].max() if len(ratio_df[ratio_df['ratio'] != np.inf]) > 0 else 1
        ratio_df_plot = ratio_df.copy()
        ratio_df_plot.loc[ratio_df_plot['ratio'] == np.inf, 'ratio'] = max_finite_ratio + 1
        
        bars = ax.bar(range(len(ratio_df_plot)), ratio_df_plot['ratio'].values,
                      color=['#1f77b4', '#ff7f0e', '#2ca02c'],
                      yerr=ratio_df_plot['ratio_error'].values, capsize=5, 
                      error_kw={'elinewidth': 2, 'capthick': 2})
        ax.set_xticks(range(len(ratio_df_plot)))
        ax.set_xticklabels(ratio_df_plot['helper_type'], rotation=0)
        ax.set_title('Ratio: "Debater convinced" / "Max rounds reached" by Helper Type', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Helper Type', fontsize=12)
        ax.set_ylabel('Ratio (Convinced / Max Rounds)', fontsize=12)
        ax.axhline(y=1, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Ratio = 1')
        
        # Add annotations with counts
        for i, row in enumerate(ratio_df_plot.itertuples()):
            label = f'{row.convinced}/{row.max_rounds}'
            if ratio_df.iloc[i]['ratio'] == np.inf:
                label += ' (∞)'
            ax.text(i, row.ratio + row.ratio_error + 0.1, label, ha='center', va='bottom', 
                   fontsize=9, fontweight='bold')
        
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "convinced_maxrounds_ratio_by_helper.png", dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {FIGURES_DIR / 'convinced_maxrounds_ratio_by_helper.png'}")
        plt.close()
    else:
        print("Skipping convinced_maxrounds_ratio_by_helper.png (finish_reason column missing)")
    
    # 13. Analysis of feedback_tags_vector vs conviction_rates_vector differences (if columns exist)
    if has_conviction_rates and has_feedback_tags:
        def parse_vector(vector_str):
            """Parse vector string, handling null values."""
            try:
                if pd.isna(vector_str):
                    return []
                # Replace null with None for JSON parsing
                vector_str_clean = str(vector_str).replace('null', 'None')
                parsed = ast.literal_eval(vector_str_clean)
                if isinstance(parsed, list):
                    # Convert None to np.nan for numeric operations
                    return [np.nan if x is None else x for x in parsed]
                return []
            except (ValueError, SyntaxError, TypeError):
                return []
        
        # Parse vectors
        df['parsed_feedback_tags'] = df['feedback_tags_vector'].apply(parse_vector)
        df['parsed_conviction_rates'] = df['conviction_rates_vector'].apply(parse_vector)
        
        def normalize_tag_case(tag):
            """Normalize tag to title case for case-insensitive matching."""
            if pd.isna(tag):
                return tag
            return str(tag).title()
        
        # Create exploded dataframe
        exploded_data = []
        for idx, row in df.iterrows():
            tags = row['parsed_feedback_tags']
            conv_rates = row['parsed_conviction_rates']
            
            # Ensure both lists have the same length
            min_len = min(len(tags), len(conv_rates))
            if min_len > 0:
                for i in range(min_len):
                    tag = tags[i]
                    conv_rate = conv_rates[i]
                    
                    # Normalize tag to title case for case-insensitive matching
                    tag_normalized = normalize_tag_case(tag)
                    
                    # Calculate difference from previous conviction rate
                    if i > 0 and not pd.isna(conv_rates[i-1]) and not pd.isna(conv_rate):
                        conv_diff = conv_rate - conv_rates[i-1]
                    else:
                        conv_diff = np.nan
                    
                    exploded_data.append({
                        'helper_type': row['helper_type'],
                        'finish_reason': row['finish_reason'],
                        'tag': tag_normalized,
                        'conviction_rate': conv_rate,
                        'conviction_diff': conv_diff,
                        'position': i
                    })
        
        df_exploded = pd.DataFrame(exploded_data)
        
        # Filter out null tags and create visualization
        df_exploded_clean = df_exploded[~pd.isna(df_exploded['tag'])].copy()
        
        if len(df_exploded_clean) > 0:
            # Get unique tags with count >= 5
            tag_counts = df_exploded_clean['tag'].value_counts()
            unique_tags = tag_counts[tag_counts >= 5].index.tolist()
            df_exploded_plot = df_exploded_clean[df_exploded_clean['tag'].isin(unique_tags)].copy()
            
            # Boxplot of conviction rate differences by tag
            fig, ax = plt.subplots(figsize=(14, 8))
            df_exploded_plot_diff = df_exploded_plot[~pd.isna(df_exploded_plot['conviction_diff'])]
            
            if len(df_exploded_plot_diff) > 0:
                # Filter tags to only those with n>=5 in the diff data
                tag_diff_counts = df_exploded_plot_diff['tag'].value_counts()
                tags_with_sufficient_data = tag_diff_counts[tag_diff_counts >= 5].index.tolist()
                
                if len(tags_with_sufficient_data) > 0:
                    sns.boxplot(data=df_exploded_plot_diff[df_exploded_plot_diff['tag'].isin(tags_with_sufficient_data)], 
                               x='tag', y='conviction_diff', ax=ax,
                               palette='Set2', order=tags_with_sufficient_data)
                    ax.set_title('Conviction Rate Change by Feedback Tag (Boxplot)', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Feedback Tag', fontsize=12)
                    ax.set_ylabel('Change in Conviction Rate', fontsize=12)
                    ax.tick_params(axis='x', rotation=45, labelsize=9)
                    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
                    plt.tight_layout()
                    plt.savefig(FIGURES_DIR / "conviction_diff_by_tag.png", dpi=300, bbox_inches='tight')
                    print(f"Figure saved to: {FIGURES_DIR / 'conviction_diff_by_tag.png'}")
                    print(f"Number of tag entries analyzed: {len(df_exploded_plot_diff[df_exploded_plot_diff['tag'].isin(tags_with_sufficient_data)])}")
                    plt.close()
                else:
                    print("No tags with n>=5 found for conviction_diff_by_tag boxplot")
                    plt.close()
                
                # Barplot of mean conviction rate differences by tag
                fig, ax = plt.subplots(figsize=(14, 8))
                tag_diff_stats = df_exploded_plot_diff.groupby('tag')['conviction_diff'].agg(['mean', 'sem', 'count']).sort_values('mean', ascending=False)
                # Filter to only tags with count >= 5
                tag_diff_stats = tag_diff_stats[tag_diff_stats['count'] >= 5]
                
                if len(tag_diff_stats) > 0:
                
                    # Color bars based on positive/negative values
                    colors = ['#2ecc71' if x >= 0 else '#e74c3c' for x in tag_diff_stats['mean'].values]
                    
                    bars = ax.bar(range(len(tag_diff_stats)), tag_diff_stats['mean'].values,
                                 yerr=tag_diff_stats['sem'].values, capsize=5, 
                                 error_kw={'elinewidth': 2, 'capthick': 2}, color=colors)
                    ax.set_xticks(range(len(tag_diff_stats)))
                    ax.set_xticklabels(tag_diff_stats.index, rotation=45, ha='right')
                    ax.set_title('Mean Conviction Rate Change by Feedback Tag (Barplot)', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Feedback Tag', fontsize=12)
                    ax.set_ylabel('Mean Change in Conviction Rate', fontsize=12)
                    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
                    
                    # Add value labels on bars
                    for i, (idx, row) in enumerate(tag_diff_stats.iterrows()):
                        value = row['mean']
                        error = row['sem']
                        count = int(row['count'])
                        label_y = value + error + 0.2 if value >= 0 else value - error - 0.2
                        ax.text(i, label_y, f'{value:.2f}\n(n={count})', ha='center', 
                               va='bottom' if value >= 0 else 'top', fontsize=8, fontweight='bold')
                    
                    plt.tight_layout()
                    plt.savefig(FIGURES_DIR / "conviction_diff_by_tag_barplot.png", dpi=300, bbox_inches='tight')
                    print(f"Figure saved to: {FIGURES_DIR / 'conviction_diff_by_tag_barplot.png'}")
                    plt.close()
                else:
                    print("No tags with n>=5 found for conviction_diff_by_tag_barplot")
                    plt.close()
            else:
                print("No valid conviction rate differences found for tags")
                plt.close()
            
            # Bar plot of average conviction rate by tag
            fig, ax = plt.subplots(figsize=(14, 6))
            tag_conv_stats = df_exploded_plot.groupby('tag')['conviction_rate'].agg(['mean', 'sem', 'count']).sort_values('mean', ascending=False)
            # Filter to only tags with count >= 5
            tag_conv_stats = tag_conv_stats[tag_conv_stats['count'] >= 5]
            
            if len(tag_conv_stats) > 0:
            
                bars = ax.bar(range(len(tag_conv_stats)), tag_conv_stats['mean'].values,
                             yerr=tag_conv_stats['sem'].values, capsize=5, 
                             error_kw={'elinewidth': 2, 'capthick': 2}, color='steelblue')
                ax.set_xticks(range(len(tag_conv_stats)))
                ax.set_xticklabels(tag_conv_stats.index, rotation=45, ha='right')
                ax.set_title('Average Conviction Rate by Feedback Tag', fontsize=14, fontweight='bold')
                ax.set_xlabel('Feedback Tag', fontsize=12)
                ax.set_ylabel('Average Conviction Rate', fontsize=12)
                
                # Add count annotations
                for i, (tag, row) in enumerate(tag_conv_stats.iterrows()):
                    ax.text(i, row['mean'] + row['sem'] + 0.5, f'n={int(row["count"])}', 
                           ha='center', va='bottom', fontsize=8)
                
                plt.tight_layout()
                plt.savefig(FIGURES_DIR / "avg_conviction_by_tag.png", dpi=300, bbox_inches='tight')
                print(f"Figure saved to: {FIGURES_DIR / 'avg_conviction_by_tag.png'}")
                plt.close()
            else:
                print("No tags with n>=5 found for avg_conviction_by_tag")
                plt.close()
            
            # 14. Histogram of Fallacies Used
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Get all tags that are fallacies (common fallacy names)
            fallacy_keywords = ['fallacy', 'slippery', 'straw', 'ad hominem', 'false dilemma', 
                               'appeal to', 'causal', 'red herring', 'begging', 'equivocation']
            
            # Filter for fallacy tags
            fallacy_tags = df_exploded_clean[
                df_exploded_clean['tag'].astype(str).str.contains('|'.join(fallacy_keywords), 
                                                                  case=False, na=False)
            ]
            
            if len(fallacy_tags) > 0:
                fallacy_counts = fallacy_tags['tag'].value_counts()
                # Filter to only fallacies with count >= 5
                fallacy_counts_filtered = fallacy_counts[fallacy_counts >= 5]
                
                if len(fallacy_counts_filtered) > 0:
                    bars = ax.barh(range(len(fallacy_counts_filtered)), fallacy_counts_filtered.values,
                                  color='crimson', yerr=np.sqrt(fallacy_counts_filtered.values), 
                                  capsize=3, error_kw={'elinewidth': 1.5, 'capthick': 1.5})
                    ax.set_yticks(range(len(fallacy_counts_filtered)))
                    ax.set_yticklabels(fallacy_counts_filtered.index, fontsize=10)
                    ax.set_xlabel('Frequency', fontsize=12, fontweight='bold')
                    ax.set_ylabel('Fallacy Type', fontsize=12, fontweight='bold')
                    ax.set_title('Histogram of Fallacies Used in Debates', fontsize=14, fontweight='bold')
                    
                    # Add count labels on bars
                    for i, (tag, count) in enumerate(fallacy_counts_filtered.items()):
                        ax.text(count + 2, i, f'{int(count)}', va='center', fontsize=9, fontweight='bold')
                    
                    ax.invert_yaxis()  # Show most frequent at top
                    plt.tight_layout()
                    plt.savefig(FIGURES_DIR / "fallacies_histogram.png", dpi=300, bbox_inches='tight')
                    print(f"Figure saved to: {FIGURES_DIR / 'fallacies_histogram.png'}")
                    print(f"Total fallacy occurrences (n>=5): {len(fallacy_tags[fallacy_tags['tag'].isin(fallacy_counts_filtered.index)])}")
                else:
                    print("No fallacies with n>=5 found for fallacies_histogram")
            else:
                print("No fallacy tags found in the dataset")
            
            plt.close()
            
            # 15. Change in Conviction Rate for Each Fallacy
            # This is the key analysis: how much does conviction rate change when each fallacy is used?
            if len(fallacy_tags) > 0:
                # Filter for fallacy tags with valid conviction differences
                fallacy_diff = fallacy_tags[~pd.isna(fallacy_tags['conviction_diff'])].copy()
                
                if len(fallacy_diff) > 0:
                    # Get unique fallacies sorted by frequency, filter to n>=5
                    fallacy_counts_diff = fallacy_diff['tag'].value_counts()
                    fallacy_list = fallacy_counts_diff[fallacy_counts_diff >= 5].index.tolist()
                    
                    if len(fallacy_list) > 0:
                        # Boxplot: Change in conviction rate by fallacy type
                        fig, ax = plt.subplots(figsize=(14, 8))
                        sns.boxplot(data=fallacy_diff[fallacy_diff['tag'].isin(fallacy_list)], 
                                   x='tag', y='conviction_diff', ax=ax,
                                   palette='Set2', order=fallacy_list)
                        ax.set_title('Change in Conviction Rate for Each Fallacy Type', fontsize=14, fontweight='bold')
                        ax.set_xlabel('Fallacy Type', fontsize=12)
                        ax.set_ylabel('Change in Conviction Rate', fontsize=12)
                        ax.tick_params(axis='x', rotation=45, labelsize=9)
                        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='No Change')
                        ax.legend()
                        plt.tight_layout()
                        plt.savefig(FIGURES_DIR / "conviction_change_by_fallacy_boxplot.png", dpi=300, bbox_inches='tight')
                        print(f"Figure saved to: {FIGURES_DIR / 'conviction_change_by_fallacy_boxplot.png'}")
                        plt.close()
                    else:
                        print("No fallacies with n>=5 found for conviction_change_by_fallacy_boxplot")
                        plt.close()
                    
                    # Barplot: Mean change in conviction rate by fallacy type
                    fallacy_diff_stats = fallacy_diff.groupby('tag')['conviction_diff'].agg(['mean', 'sem', 'count']).sort_values('mean', ascending=False)
                    # Filter to only fallacies with count >= 5
                    fallacy_diff_stats = fallacy_diff_stats[fallacy_diff_stats['count'] >= 5]
                    
                    if len(fallacy_diff_stats) > 0:
                        fig, ax = plt.subplots(figsize=(14, 8))
                    
                        # Color bars based on positive/negative values
                        colors = ['#2ecc71' if x >= 0 else '#e74c3c' for x in fallacy_diff_stats['mean'].values]
                        
                        bars = ax.bar(range(len(fallacy_diff_stats)), fallacy_diff_stats['mean'].values,
                                     yerr=fallacy_diff_stats['sem'].values, capsize=5, 
                                     error_kw={'elinewidth': 2, 'capthick': 2}, color=colors)
                        ax.set_xticks(range(len(fallacy_diff_stats)))
                        ax.set_xticklabels(fallacy_diff_stats.index, rotation=45, ha='right', fontsize=10)
                        ax.set_title('Mean Change in Conviction Rate for Each Fallacy Type', fontsize=14, fontweight='bold')
                        ax.set_xlabel('Fallacy Type', fontsize=12)
                        ax.set_ylabel('Mean Change in Conviction Rate', fontsize=12)
                        ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
                        
                        # Add value labels on bars
                        for i, (fallacy, row) in enumerate(fallacy_diff_stats.iterrows()):
                            value = row['mean']
                            error = row['sem']
                            count = int(row['count'])
                            label_y = value + error + 0.15 if value >= 0 else value - error - 0.15
                            ax.text(i, label_y, f'{value:.2f}\n(n={count})', ha='center', 
                                   va='bottom' if value >= 0 else 'top', fontsize=9, fontweight='bold')
                        
                        plt.tight_layout()
                        plt.savefig(FIGURES_DIR / "conviction_change_by_fallacy_barplot.png", dpi=300, bbox_inches='tight')
                        print(f"Figure saved to: {FIGURES_DIR / 'conviction_change_by_fallacy_barplot.png'}")
                        print(f"Total fallacy instances with conviction changes (n>=5): {len(fallacy_diff[fallacy_diff['tag'].isin(fallacy_diff_stats.index)])}")
                        plt.close()
                        
                        # Summary statistics table for fallacies
                        print("\n" + "="*70)
                        print("FALLACY CONVICTION RATE CHANGE SUMMARY (n>=5)")
                        print("="*70)
                        print(f"{'Fallacy':<30} {'Mean Change':<15} {'SEM':<10} {'Count':<10}")
                        print("-"*70)
                        for fallacy, row in fallacy_diff_stats.iterrows():
                            print(f"{fallacy:<30} {row['mean']:>10.2f}     {row['sem']:>8.3f}   {int(row['count']):>6}")
                        print("="*70)
                    else:
                        print("No fallacies with n>=5 found for conviction_change_by_fallacy_barplot")
                        plt.close()
                else:
                    print("No valid conviction rate differences found for fallacies")
            else:
                print("No fallacy tags found for conviction change analysis")
        else:
            print("No valid feedback tags found in the dataset")
    else:
        print("Skipping tag/fallacy analyses (required columns missing)")
    
    # 16. Progression of Conviction Rate Over Trials (if conviction_rates_vector exists)
    if has_conviction_rates:
        # Create progression data from parsed conviction rates
        progression_data = []
        for idx, row in df.iterrows():
            conv_rates = row['parsed_conviction_rates']
            if len(conv_rates) > 0:
                for i, rate in enumerate(conv_rates):
                    if not pd.isna(rate):
                        progression_data.append({
                            'helper_type': row['helper_type'],
                            'finish_reason': row['finish_reason'],
                            'trial': i + 1,
                            'conviction_rate': rate
                        })
        
        df_progression = pd.DataFrame(progression_data)
        
        if len(df_progression) > 0:
            # Plot 1: Average progression by helper type
            fig, ax = plt.subplots(figsize=(12, 8))
            
            for helper in df_progression['helper_type'].unique():
                helper_data = df_progression[df_progression['helper_type'] == helper]
                progression_by_trial = helper_data.groupby('trial')['conviction_rate'].agg(['mean', 'sem', 'count'])
                
                # Only plot if we have at least 3 data points
                if len(progression_by_trial) >= 3:
                    ax.errorbar(progression_by_trial.index, progression_by_trial['mean'].values,
                               yerr=progression_by_trial['sem'].values, marker='o', linewidth=2,
                               markersize=6, capsize=4, capthick=2, label=helper, alpha=0.8)
            
            ax.set_xlabel('Trial Number', fontsize=12, fontweight='bold')
            ax.set_ylabel('Conviction Rate', fontsize=12, fontweight='bold')
            ax.set_title('Progression of Conviction Rate Over Trials by Helper Type', 
                        fontsize=14, fontweight='bold')
            ax.legend(title='Helper Type', fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "conviction_progression_by_helper.png", dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {FIGURES_DIR / 'conviction_progression_by_helper.png'}")
            plt.close()
            
            # Plot 2: Average progression by finish reason
            fig, ax = plt.subplots(figsize=(12, 8))
            
            finish_reasons_to_plot = ['Debater convinced', 'Max rounds reached']
            colors_map = {'Debater convinced': '#2ecc71', 'Max rounds reached': '#e74c3c'}
            
            for reason in finish_reasons_to_plot:
                reason_data = df_progression[df_progression['finish_reason'] == reason]
                if len(reason_data) > 0:
                    progression_by_trial = reason_data.groupby('trial')['conviction_rate'].agg(['mean', 'sem', 'count'])
                    
                    if len(progression_by_trial) >= 3:
                        ax.errorbar(progression_by_trial.index, progression_by_trial['mean'].values,
                                   yerr=progression_by_trial['sem'].values, marker='s', linewidth=2,
                                   markersize=6, capsize=4, capthick=2, label=reason, 
                                   color=colors_map.get(reason, 'blue'), alpha=0.8)
            
            ax.set_xlabel('Trial Number', fontsize=12, fontweight='bold')
            ax.set_ylabel('Conviction Rate', fontsize=12, fontweight='bold')
            ax.set_title('Progression of Conviction Rate Over Trials by Finish Reason', 
                        fontsize=14, fontweight='bold')
            ax.legend(title='Finish Reason', fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "conviction_progression_by_finish_reason.png", dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {FIGURES_DIR / 'conviction_progression_by_finish_reason.png'}")
            plt.close()
            
            # Plot 3: Overall average progression
            fig, ax = plt.subplots(figsize=(12, 8))
            
            overall_progression = df_progression.groupby('trial')['conviction_rate'].agg(['mean', 'sem', 'count'])
            
            ax.errorbar(overall_progression.index, overall_progression['mean'].values,
                       yerr=overall_progression['sem'].values, marker='o', linewidth=3,
                       markersize=8, capsize=5, capthick=2, color='steelblue', alpha=0.8,
                       label='Overall Average')
            
            ax.set_xlabel('Trial Number', fontsize=12, fontweight='bold')
            ax.set_ylabel('Conviction Rate', fontsize=12, fontweight='bold')
            ax.set_title('Overall Progression of Conviction Rate Over Trials', 
                        fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add sample size annotations
            for trial, row in overall_progression.iterrows():
                if row['count'] > 0:
                    ax.text(trial, row['mean'] + row['sem'] + 0.5, f'n={int(row["count"])}', 
                           ha='center', va='bottom', fontsize=8, alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "conviction_progression_overall.png", dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {FIGURES_DIR / 'conviction_progression_overall.png'}")
            print(f"Total progression data points: {len(df_progression)}")
            plt.close()
        else:
            print("No valid progression data found")
    else:
        print("Skipping conviction progression analyses (conviction_rates_vector column missing)")
    
    # Generate summary of all figures created
    print("\n" + "="*80)
    print(f"ANALYSIS COMPLETE - All plots generated for: {excel_file_path.name}")
    print("="*80)
    print(f"Figures directory: {FIGURES_DIR}")
    print("\nGenerated Figures:")
    print("-"*80)
    
    figure_list = [
        "1. rounds_distribution_comparison.png - Distribution of rounds by finish reason",
        "2. finish_reasons_distribution.png - Distribution of finish reasons",
        "3. helper_types_distribution.png - Distribution of helper types",
        "4. result_distribution.png - Distribution of results",
        "5. success_rate_by_helper.png - Success rate by helper type",
        "6. avg_rounds_by_helper.png - Average rounds by helper type",
        "7. rounds_boxplot_by_helper.png - Rounds distribution boxplot by helper",
        "8. result_boxplot_by_helper.png - Result distribution boxplot by helper",
        "9. finish_reason_by_helper_stacked.png - Finish reason distribution by helper (stacked)",
        "10. summary_statistics_table.png - Summary statistics table",
        "11. correlation_heatmap.png - Correlation heatmap of numeric variables",
        "12. last_conviction_rate_boxplot.png - Last conviction rate by finish reason",
        "13. convinced_maxrounds_ratio_by_helper.png - Ratio of convinced vs max rounds by helper",
        "14. conviction_diff_by_tag.png - Conviction rate change by feedback tag (boxplot)",
        "15. conviction_diff_by_tag_barplot.png - Mean conviction rate change by tag (barplot)",
        "16. avg_conviction_by_tag.png - Average conviction rate by feedback tag",
        "17. fallacies_histogram.png - Histogram of fallacies used",
        "18. conviction_change_by_fallacy_boxplot.png - Change in conviction rate for each fallacy (boxplot)",
        "19. conviction_change_by_fallacy_barplot.png - Mean change in conviction rate for each fallacy (barplot)",
        "20. conviction_progression_by_helper.png - Conviction rate progression by helper type",
        "21. conviction_progression_by_finish_reason.png - Conviction rate progression by finish reason",
        "22. conviction_progression_overall.png - Overall conviction rate progression"
    ]
    
    for fig_desc in figure_list:
        print(f"  {fig_desc}")
    
    print("-"*80)
    print(f"Total figures generated: {len(figure_list)}")
    print("="*80)


def run_multi_file_heatmap_analysis(excel_files, settings_files=None):
    """
    Create heatmaps showing success rate matrices for each helper type.
    Each heatmap shows: Rows = Debater models, Columns = Persuader models
    
    Args:
        excel_files: List of paths to Excel files
        settings_files: Optional list of settings file paths (same order as excel_files)
    """
    import seaborn as sns
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    
    print("\n" + "="*80)
    print("MULTI-FILE HEATMAP ANALYSIS: Success Rate by Helper Type")
    print("="*80)
    
    # Load all files and determine model roles
    file_data = []
    all_persuader_models = set()
    all_debater_models = set()
    
    for i, excel_file in enumerate(excel_files):
        excel_path = Path(excel_file)
        if not excel_path.exists():
            print(f"Warning: File not found: {excel_path}")
            continue
        
        print(f"\nLoading file {i+1}/{len(excel_files)}: {excel_path}")
        df = load_debates_summary(excel_path)
        
        # Try to find settings file
        settings_path = None
        if settings_files and i < len(settings_files):
            settings_path = Path(settings_files[i])
        else:
            # Try to find settings.yaml in the same directory
            possible_settings = excel_path.parent / "settings.yaml"
            if possible_settings.exists():
                settings_path = possible_settings
        
        # Parse model names from settings
        persuader_model = None
        debater_model = None
        
        print(f"  DEBUG: Processing file: {excel_path.name}")
        print(f"  DEBUG: Looking for settings file: {settings_path}")
        
        if settings_path and settings_path.exists():
            persuader_model, debater_model = parse_llm_names_from_settings(settings_path)
            print(f"  DEBUG: After settings parsing: P={persuader_model}, D={debater_model}")
        else:
            print(f"  DEBUG: Settings file not found or not provided")
        
        # If still not found, try to infer from filename
        if not persuader_model or not debater_model:
            filename_lower = excel_path.name.lower()
            print(f"  DEBUG: Inferring from filename: {excel_path.name}")
            
            # Check directory name for model info (e.g., "35turbovs41nano")
            dir_name_lower = excel_path.parent.name.lower()
            print(f"  DEBUG: Directory name: {excel_path.parent.name}")
            
            # Try to parse "XvsY" or "X_vs_Y" pattern from directory name
            if 'vs' in dir_name_lower:
                parts = dir_name_lower.split('vs')
                if len(parts) == 2:
                    model1_part = parts[0].strip()
                    model2_part = parts[1].strip()
                    print(f"  DEBUG: Parsed directory: {model1_part} vs {model2_part}")
                    
                    # Determine which model is which
                    if '35' in model1_part or '3.5' in model1_part:
                        if persuader_model is None:
                            persuader_model = 'gpt35_turbo'
                    if '41' in model1_part or '4.1' in model1_part:
                        if persuader_model is None:
                            persuader_model = 'gpt41_nano' if 'nano' in model1_part else 'gpt41_mini'
                    
                    if '35' in model2_part or '3.5' in model2_part:
                        if debater_model is None:
                            debater_model = 'gpt35_turbo'
                    if '41' in model2_part or '4.1' in model2_part:
                        if debater_model is None:
                            debater_model = 'gpt41_nano' if 'nano' in model2_part else 'gpt41_mini'
            
            # Fallback to filename-based inference
            if not persuader_model or not debater_model:
                if '35turbo' in filename_lower or 'gpt35' in filename_lower:
                    if persuader_model is None:
                        persuader_model = 'gpt35_turbo'
                    if debater_model is None:
                        debater_model = 'gpt35_turbo'
                if '41nano' in filename_lower or '41_mini' in filename_lower or '41mini' in filename_lower:
                    if '41nano' in filename_lower:
                        model_name = 'gpt41_nano'
                    else:
                        model_name = 'gpt41_mini'
                    if persuader_model is None:
                        persuader_model = model_name
                    if debater_model is None:
                        debater_model = model_name
            
            print(f"  DEBUG: After filename inference: P={persuader_model}, D={debater_model}")
        
        # Check for "opposite" in filename to swap roles
        is_opposite = 'opposite' in excel_path.name.lower()
        print(f"  DEBUG: Is 'opposite' in filename? {is_opposite}")
        if is_opposite:
            print(f"  DEBUG: BEFORE swap: P={persuader_model}, D={debater_model}")
            persuader_model, debater_model = debater_model, persuader_model
            print(f"  DEBUG: AFTER swap: P={persuader_model}, D={debater_model}")
        
        if not persuader_model or not debater_model:
            print(f"Warning: Could not determine model roles for {excel_path.name}")
            print(f"  Using defaults: persuader={persuader_model}, debater={debater_model}")
            continue
        
        all_persuader_models.add(persuader_model)
        all_debater_models.add(debater_model)
        
        file_data.append({
            'df': df,
            'persuader_model': persuader_model,
            'debater_model': debater_model,
            'file_path': excel_path
        })
        
        print(f"  Determined: {persuader_model} (persuader) vs {debater_model} (debater)")
    
    if len(file_data) == 0:
        print("ERROR: No valid files loaded!")
        return
    
    # Normalize model names (handle variations)
    model_name_mapping = {
        'gpt35_turbo': '3.5',
        'gpt-3.5-turbo': '3.5',
        'gpt41_nano': '4.1',
        'gpt-4.1-nano': '4.1',
        'gpt41_mini': '4.1',
        'gpt-4.1-mini': '4.1',
    }
    
    def normalize_model_name(model_name):
        """Normalize model name to short form."""
        if not model_name:
            return model_name
        # Try exact match first
        if model_name in model_name_mapping:
            return model_name_mapping[model_name]
        # Try partial match
        model_lower = model_name.lower()
        if '35' in model_lower or '3.5' in model_lower:
            return '3.5'
        if '41' in model_lower or '4.1' in model_lower:
            return '4.1'
        return model_name
    
    # Get unique normalized model names
    all_models = sorted(set([normalize_model_name(m) for m in list(all_persuader_models) + list(all_debater_models)]))
    print(f"\nUnique models found: {all_models}")
    
    # Create figures directory
    heatmap_dir = PROJECT_ROOT / "figures" / "heatmap_analysis"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nHeatmaps will be saved to: {heatmap_dir}")
    
    # Helper types to analyze
    helper_types = ['No Helper', 'Logical Helper', 'Fallacy Helper']
    helper_type_mapping = {
        'No_Helper': 'No Helper',
        'Default_No_Helper': 'No Helper',
        'Fallacy_Helper': 'Fallacy Helper',
        'Logical_Helper': 'Logical Helper'
    }
    
    # Calculate success rate (excluding 1-round successes)
    def calculate_success_rate(df_subset):
        """Calculate success rate excluding 1-round successes."""
        if len(df_subset) == 0:
            return {'success_rate': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan, 'count': 0}
        
        convinced_2plus = len(df_subset[(df_subset['result'] > 0) & (df_subset['rounds'] > 1)])
        not_convinced = len(df_subset[df_subset['result'] == 0])
        conclusive_2plus = convinced_2plus + not_convinced
        
        if conclusive_2plus > 0:
            p = convinced_2plus / conclusive_2plus
            # Calculate 95% confidence interval
            z = 1.96  # 95% CI
            se = np.sqrt(p * (1 - p) / conclusive_2plus)
            ci_lower = max(0, (p - z * se) * 100)
            ci_upper = min(100, (p + z * se) * 100)
            success_rate = p * 100
        else:
            success_rate = np.nan
            ci_lower = np.nan
            ci_upper = np.nan
        
        return {
            'success_rate': success_rate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'count': conclusive_2plus
        }
    
    # Build data structure: {helper_type: {persuader_model: {debater_model: stats}}}
    heatmap_data = {}
    
    # Debug: Print what combinations we're looking for
    print("\n" + "="*80)
    print("DEBUG: Available file combinations:")
    for file_info in file_data:
        persuader_orig = file_info['persuader_model']
        debater_orig = file_info['debater_model']
        persuader_norm = normalize_model_name(persuader_orig)
        debater_norm = normalize_model_name(debater_orig)
        print(f"  File: {file_info['file_path'].name}")
        print(f"    Original: {persuader_orig} (persuader) vs {debater_orig} (debater)")
        print(f"    Normalized: {persuader_norm} (persuader) vs {debater_norm} (debater)")
        print(f"    Helper types in file: {sorted(file_info['df']['helper_type'].unique())}")
    print("="*80)
    
    for helper_type in helper_types:
        heatmap_data[helper_type] = {}
        for persuader_norm in all_models:
            heatmap_data[helper_type][persuader_norm] = {}
            for debater_norm in all_models:
                # Find matching data
                matching_rows = []
                matching_files = []
                for file_info in file_data:
                    df = file_info['df']
                    persuader_orig = file_info['persuader_model']
                    debater_orig = file_info['debater_model']
                    
                    persuader_norm_check = normalize_model_name(persuader_orig)
                    debater_norm_check = normalize_model_name(debater_orig)
                    
                    if persuader_norm_check == persuader_norm and debater_norm_check == debater_norm:
                        # Find matching helper types
                        matching_helper_types = [ht for ht in df['helper_type'].unique() 
                                               if helper_type_mapping.get(ht, ht) == helper_type]
                        if matching_helper_types:
                            helper_df = df[df['helper_type'].isin(matching_helper_types)]
                            matching_rows.append(helper_df)
                            matching_files.append(file_info['file_path'].name)
                
                if matching_rows:
                    combined_df = pd.concat(matching_rows, ignore_index=True)
                    stats = calculate_success_rate(combined_df)
                    heatmap_data[helper_type][persuader_norm][debater_norm] = stats
                    print(f"DEBUG: {helper_type} | P:{persuader_norm} D:{debater_norm} | "
                          f"Success: {stats['success_rate']:.1f}% | Count: {stats['count']} | "
                          f"Files: {', '.join(matching_files)}")
                else:
                    heatmap_data[helper_type][persuader_norm][debater_norm] = {
                        'success_rate': np.nan,
                        'ci_lower': np.nan,
                        'ci_upper': np.nan,
                        'count': 0
                    }
                    print(f"DEBUG: {helper_type} | P:{persuader_norm} D:{debater_norm} | NO DATA")
    
    # Create heatmaps for each helper type
    for helper_type in helper_types:
        print(f"\nCreating heatmap for {helper_type}...")
        
        # Build matrix
        # Rows = Persuader models (Y-axis)
        # Columns = Debater models (X-axis)
        matrix = []
        matrix_text = []
        matrix_counts = []
        
        for persuader_norm in all_models:
            row = []
            row_text = []
            row_counts = []
            for debater_norm in all_models:
                stats = heatmap_data[helper_type][persuader_norm][debater_norm]
                row.append(stats['success_rate'])
                
                # Format text: mean ± CI margin (95% CI margin = distance from mean to upper bound)
                # Note: The ± value represents the margin of the 95% confidence interval.
                # The true success rate is likely between (mean - margin) and (mean + margin) with 95% confidence.
                if not np.isnan(stats['success_rate']) and stats['count'] > 0:
                    mean_val = stats['success_rate']
                    if not np.isnan(stats['ci_lower']) and not np.isnan(stats['ci_upper']):
                        # ci_margin is the distance from mean to upper bound
                        # For symmetric CIs, this equals half the CI width
                        ci_margin = stats['ci_upper'] - mean_val
                        # Also calculate lower margin for display (usually same as upper for proportions)
                        ci_lower_margin = mean_val - stats['ci_lower']
                        # Use the larger margin for display (or average if they differ significantly)
                        display_margin = max(ci_margin, ci_lower_margin)
                        text = f"{mean_val:.1f}% ± {display_margin:.1f}%"
                    else:
                        text = f"{mean_val:.1f}%\n(n={stats['count']})"
                else:
                    text = "N/A"
                
                row_text.append(text)
                row_counts.append(stats['count'])
            
            matrix.append(row)
            matrix_text.append(row_text)
            matrix_counts.append(row_counts)
        
        matrix = np.array(matrix)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        
        # Set ticks and labels
        # X-axis = Debater models, Y-axis = Persuader models
        ax.set_xticks(np.arange(len(all_models)))
        ax.set_yticks(np.arange(len(all_models)))
        ax.set_xticklabels(all_models)  # X-axis: Debater model versions
        ax.set_yticklabels(all_models)  # Y-axis: Persuader model versions
        
        # Add text annotations
        for i in range(len(all_models)):
            for j in range(len(all_models)):
                value = matrix[i, j]
                text = matrix_text[i][j]
                count = matrix_counts[i][j]
                
                if not np.isnan(value):
                    # Choose text color based on background
                    text_color = 'black' if 30 < value < 70 else 'white'
                    ax.text(j, i, text, ha="center", va="center", 
                           color=text_color, fontweight='bold', fontsize=11)
                else:
                    ax.text(j, i, "N/A", ha="center", va="center", 
                           color="gray", fontsize=10)
        
        # Add labels
        # X-axis = Debater, Y-axis = Persuader
        ax.set_xlabel("Debater model version", fontsize=14, fontweight='bold')
        ax.set_ylabel("Persuader model version", fontsize=14, fontweight='bold')
        ax.set_title(f'Success Rate Heatmap: {helper_type}', fontsize=16, fontweight='bold', pad=20)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Success Rate (%)', fontsize=12)
        
        plt.tight_layout()
        
        # Save figure
        helper_name_safe = helper_type.replace(' ', '_').lower()
        output_path = heatmap_dir / f"heatmap_{helper_name_safe}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap: {output_path}")
        plt.close()
    
    # Print summary of data availability
    print("\n" + "="*80)
    print("DATA AVAILABILITY SUMMARY")
    print("="*80)
    for helper_type in helper_types:
        print(f"\n{helper_type}:")
        for persuader_norm in all_models:
            for debater_norm in all_models:
                stats = heatmap_data[helper_type][persuader_norm][debater_norm]
                if not np.isnan(stats['success_rate']) and stats['count'] > 0:
                    print(f"  ✓ P:{persuader_norm} D:{debater_norm} - Success: {stats['success_rate']:.1f}% (n={stats['count']})")
                else:
                    print(f"  ✗ P:{persuader_norm} D:{debater_norm} - NO DATA")
    print("\n" + "="*80)
    print("HEATMAP ANALYSIS COMPLETE")
    print("="*80)
    print(f"All heatmaps saved to: {heatmap_dir}")
    print("\nNote: The ± value represents the margin of the 95% confidence interval.")
    print("      The true success rate is likely within [mean - margin, mean + margin] with 95% confidence.")
    print("="*80)


def run_comparison_analysis(file1_path, file2_path, model1_name=None, model2_name=None):
    """
    Run comparison analysis between two Excel files representing different model roles.
    Creates matrices grouped by helper type showing performance for both LLM scenarios.
    
    Args:
        file1_path: Path to Excel file where model1 is persuader vs model2 as debater
        file2_path: Path to Excel file where model2 is persuader vs model1 as debater
        model1_name: Optional name for model 1 (defaults to helper_type from file1)
        model2_name: Optional name for model 2 (defaults to helper_type from file2)
    """
    import seaborn as sns
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    
    print("\n" + "="*80)
    print("COMPARISON MODE: Model Performance Matrix by Helper Type")
    print("="*80)
    
    # Load both files
    print(f"\nLoading File 1: {file1_path}")
    df1 = load_debates_summary(file1_path)
    
    print(f"\nLoading File 2: {file2_path}")
    df2 = load_debates_summary(file2_path)
    
    # Determine model names from helper_type if not provided
    # Try to infer from the data - look for common patterns
    if model1_name is None:
        # Try to extract model names from helper_type or use defaults
        helper_types_1 = df1['helper_type'].unique() if 'helper_type' in df1.columns else []
        model1_name = "LLM1"  # Default
    
    if model2_name is None:
        helper_types_2 = df2['helper_type'].unique() if 'helper_type' in df2.columns else []
        model2_name = "LLM2"  # Default
    
    print(f"\nModel 1 (Persuader in File 1): {model1_name}")
    print(f"Model 2 (Persuader in File 2): {model2_name}")
    
    # Create figures directory for comparison
    comparison_dir = PROJECT_ROOT / "figures" / "model_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nComparison figures will be saved to: {comparison_dir}")
    
    # Get all unique helper types from both files
    if 'helper_type' not in df1.columns or 'helper_type' not in df2.columns:
        print("ERROR: 'helper_type' column not found in one or both files!")
        return
    
    # Show helper types in each file
    helper_types_file1 = sorted(df1['helper_type'].unique())
    helper_types_file2 = sorted(df2['helper_type'].unique())
    print(f"\nHelper types in File 1 (LLM1→LLM2): {helper_types_file1}")
    print(f"Helper types in File 2 (LLM2→LLM1): {helper_types_file2}")
    
    all_helper_types = sorted(set(helper_types_file1) | set(helper_types_file2))
    print(f"All unique helper types: {all_helper_types}")
    
    # Normalize helper type names (handle variations)
    helper_type_mapping = {
        'No_Helper': 'No Helper',
        'Default_No_Helper': 'No Helper',
        'Fallacy_Helper': 'Fallacy Helper',
        'Logical_Helper': 'Logical Helper'
    }
    
    normalized_helper_types = []
    for ht in all_helper_types:
        normalized = helper_type_mapping.get(ht, ht)
        if normalized not in normalized_helper_types:
            normalized_helper_types.append(normalized)
    
    # Ensure we have the three main types
    expected_types = ['No Helper', 'Fallacy Helper', 'Logical Helper']
    helper_types_to_use = [ht for ht in expected_types if ht in normalized_helper_types]
    if len(helper_types_to_use) == 0:
        helper_types_to_use = normalized_helper_types
    
    print(f"Using helper types: {helper_types_to_use}")
    
    # Calculate metrics for each scenario and helper type
    def calculate_metrics(df_subset):
        """Calculate key metrics from a dataframe subset."""
        metrics = {}
        
        if len(df_subset) == 0:
            return metrics
        
        # Success rate (excluding 1-round successes for more valid statistics)
        # Count convinced debates with 2+ rounds
        convinced_2plus = len(df_subset[(df_subset['result'] > 0) & (df_subset['rounds'] > 1)])
        # Count not convinced debates
        not_convinced = len(df_subset[df_subset['result'] == 0])
        # Total conclusive debates (excluding 1-round successes and errors)
        conclusive_2plus = convinced_2plus + not_convinced
        
        if conclusive_2plus > 0:
            metrics['success_rate'] = (convinced_2plus / conclusive_2plus) * 100
        else:
            metrics['success_rate'] = 0
        
        # Average rounds
        metrics['avg_rounds'] = df_subset['rounds'].mean() if 'rounds' in df_subset.columns else np.nan
        
        # Average result
        metrics['avg_result'] = df_subset['result'].mean() if 'result' in df_subset.columns else np.nan
        
        # Conviction rate (if finish_reason exists)
        if 'finish_reason' in df_subset.columns:
            metrics['conviction_rate'] = (df_subset['finish_reason'] == 'Debater convinced').sum() / len(df_subset) * 100
        else:
            metrics['conviction_rate'] = np.nan
        
        # Standard errors (for success rate excluding 1-round successes)
        if conclusive_2plus > 1:
            p = convinced_2plus / conclusive_2plus
            metrics['success_rate_se'] = np.sqrt(p * (1 - p) / conclusive_2plus) * 100
        else:
            metrics['success_rate_se'] = 0
        
        metrics['avg_rounds_se'] = df_subset['rounds'].sem() if 'rounds' in df_subset.columns else np.nan
        metrics['avg_result_se'] = df_subset['result'].sem() if 'result' in df_subset.columns else np.nan
        
        metrics['count'] = len(df_subset)
        
        return metrics
    
    # Build metrics dictionary: {helper_type: {scenario: metrics}}
    metrics_by_helper = {}
    
    for helper_type in helper_types_to_use:
        # Find matching helper type in original data (handle name variations)
        matching_types_1 = [ht for ht in df1['helper_type'].unique() 
                           if helper_type_mapping.get(ht, ht) == helper_type]
        matching_types_2 = [ht for ht in df2['helper_type'].unique() 
                           if helper_type_mapping.get(ht, ht) == helper_type]
        
        if not matching_types_1 and not matching_types_2:
            print(f"Warning: No data found for helper type '{helper_type}' in either file")
            continue
        
        # Get data for this helper type from both files
        # File 1: LLM1 as persuader vs LLM2 as debater
        df1_helper = df1[df1['helper_type'].isin(matching_types_1)] if matching_types_1 else pd.DataFrame()
        # File 2: LLM2 as persuader vs LLM1 as debater
        df2_helper = df2[df2['helper_type'].isin(matching_types_2)] if matching_types_2 else pd.DataFrame()
        
        print(f"\nProcessing {helper_type}:")
        print(f"  File 1 ({model1_name}→{model2_name}): {len(df1_helper)} debates")
        print(f"  File 2 ({model2_name}→{model1_name}): {len(df2_helper)} debates")
        
        metrics_by_helper[helper_type] = {
            f'{model1_name}→{model2_name}': calculate_metrics(df1_helper),
            f'{model2_name}→{model1_name}': calculate_metrics(df2_helper)
        }
    
    # Print summary
    print("\n" + "="*80)
    print("METRICS SUMMARY BY HELPER TYPE")
    print("="*80)
    for helper_type, scenarios in metrics_by_helper.items():
        print(f"\n{helper_type}:")
        for scenario, metrics in scenarios.items():
            print(f"  {scenario}:")
            print(f"    Success Rate: {metrics.get('success_rate', 0):.2f}% ± {metrics.get('success_rate_se', 0):.2f}%")
            print(f"    Avg Rounds: {metrics.get('avg_rounds', np.nan):.2f} ± {metrics.get('avg_rounds_se', np.nan):.2f}")
            print(f"    Avg Result: {metrics.get('avg_result', np.nan):.2f} ± {metrics.get('avg_result_se', np.nan):.2f}")
            print(f"    Conviction Rate: {metrics.get('conviction_rate', np.nan):.2f}%")
            print(f"    Sample Size: {metrics.get('count', 0)}")
    
    # Create matrix data structure
    # Rows: Helper types, Columns: LLM scenarios (LLM1→LLM2, LLM2→LLM1)
    scenario_cols = [f'{model1_name}→{model2_name}', f'{model2_name}→{model1_name}']
    
    matrix_data = {}
    for metric_name in ['success_rate', 'avg_rounds', 'avg_result', 'conviction_rate']:
        matrix = []
        for helper_type in helper_types_to_use:
            if helper_type in metrics_by_helper:
                row = [
                    metrics_by_helper[helper_type][scenario_cols[0]].get(metric_name, np.nan),
                    metrics_by_helper[helper_type][scenario_cols[1]].get(metric_name, np.nan)
                ]
            else:
                row = [np.nan, np.nan]
            matrix.append(row)
        matrix_data[metric_name] = np.array(matrix)
    
    row_labels = helper_types_to_use
    col_labels = scenario_cols
    
    # Create matrix visualizations for each metric
    metrics_to_plot = {
        'success_rate': {'title': 'Success Rate (%)', 'fmt': '.1f', 'vmin': 0, 'vmax': 100},
        'avg_rounds': {'title': 'Average Rounds', 'fmt': '.2f', 'vmin': None, 'vmax': None},
        'avg_result': {'title': 'Average Result', 'fmt': '.2f', 'vmin': None, 'vmax': None},
        'conviction_rate': {'title': 'Conviction Rate (%)', 'fmt': '.1f', 'vmin': 0, 'vmax': 100}
    }
    
    for metric_name, plot_config in metrics_to_plot.items():
        if metric_name not in matrix_data:
            continue
        
        matrix = matrix_data[metric_name]
        
        # Skip if all values are NaN
        if np.all(np.isnan(matrix)):
            print(f"\nSkipping {metric_name} matrix (no valid data)")
            continue
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', 
                      vmin=plot_config.get('vmin'), vmax=plot_config.get('vmax'))
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                value = matrix[i, j]
                if not np.isnan(value):
                    text = ax.text(j, i, f"{value:{plot_config['fmt']}}",
                                 ha="center", va="center", color="black", fontweight='bold', fontsize=14)
                else:
                    text = ax.text(j, i, "N/A",
                                 ha="center", va="center", color="gray", fontsize=12)
        
        # Add labels
        ax.set_xlabel("LLM Scenario", fontsize=14, fontweight='bold')
        ax.set_ylabel("Helper Type", fontsize=14, fontweight='bold')
        ax.set_title(f"{plot_config['title']} by Helper Type", fontsize=16, fontweight='bold', pad=20)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(plot_config['title'], fontsize=12)
        
        plt.tight_layout()
        
        # Save figure
        output_path = comparison_dir / f"matrix_{metric_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved matrix: {output_path}")
        plt.close()
    
    # Create a combined summary matrix with multiple metrics
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    plot_idx = 0
    for metric_name, plot_config in metrics_to_plot.items():
        if metric_name not in matrix_data:
            continue
        
        matrix = matrix_data[metric_name]
        
        if np.all(np.isnan(matrix)):
            continue
        
        ax = axes[plot_idx]
        
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto',
                      vmin=plot_config.get('vmin'), vmax=plot_config.get('vmax'))
        
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                value = matrix[i, j]
                if not np.isnan(value):
                    ax.text(j, i, f"{value:{plot_config['fmt']}}",
                           ha="center", va="center", color="black", fontweight='bold', fontsize=11)
                else:
                    ax.text(j, i, "N/A",
                           ha="center", va="center", color="gray", fontsize=10)
        
        ax.set_xlabel("LLM Scenario", fontsize=11, fontweight='bold')
        ax.set_ylabel("Helper Type", fontsize=11, fontweight='bold')
        ax.set_title(plot_config['title'], fontsize=12, fontweight='bold')
        
        plt.colorbar(im, ax=ax)
        
        plot_idx += 1
    
    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Model Performance Comparison Matrix by Helper Type', fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = comparison_dir / "matrix_combined.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined matrix: {output_path}")
    plt.close()
    
    # Create a detailed comparison table
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Create detailed comparison table by helper type
    comparison_table_data = []
    header = ['Helper Type', 'Metric'] + scenario_cols
    comparison_table_data.append(header)
    
    metrics_list = [
        ('Success Rate (%)', 'success_rate', 'success_rate_se', True),
        ('Avg Rounds', 'avg_rounds', 'avg_rounds_se', True),
        ('Avg Result', 'avg_result', 'avg_result_se', True),
        ('Conviction Rate (%)', 'conviction_rate', None, False),
        ('Sample Size', 'count', None, False)
    ]
    
    for helper_type in helper_types_to_use:
        if helper_type not in metrics_by_helper:
            continue
        
        for metric_display, metric_key, se_key, has_se in metrics_list:
            row = [helper_type if metric_display == metrics_list[0][0] else '', metric_display]
            
            for scenario in scenario_cols:
                if scenario in metrics_by_helper[helper_type]:
                    metrics = metrics_by_helper[helper_type][scenario]
                    value = metrics.get(metric_key, np.nan)
                    
                    if has_se and se_key and se_key in metrics:
                        se = metrics.get(se_key, 0)
                        if not np.isnan(value):
                            row.append(f"{value:.2f} ± {se:.2f}")
                        else:
                            row.append("N/A")
                    else:
                        if not np.isnan(value):
                            row.append(f"{value:.2f}")
                        else:
                            row.append("N/A")
                else:
                    row.append("N/A")
            
            comparison_table_data.append(row)
    
    table = ax.table(cellText=comparison_table_data,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    
    # Style header
    num_cols = len(header)
    for i in range(num_cols):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style helper type column and metric column
    for i in range(1, len(comparison_table_data)):
        table[(i, 0)].set_facecolor('#E8F5E9')  # Helper type column
        table[(i, 0)].set_text_props(weight='bold')
        table[(i, 1)].set_facecolor('#F1F8E9')  # Metric column
        table[(i, 1)].set_text_props(weight='bold')
    
    ax.set_title('Detailed Comparison Table', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(comparison_dir / "comparison_table.png", dpi=300, bbox_inches='tight')
    print(f"Saved comparison table: {comparison_dir / 'comparison_table.png'}")
    plt.close()
    
    print("\n" + "="*80)
    print("COMPARISON ANALYSIS COMPLETE")
    print("="*80)
    print(f"All comparison figures saved to: {comparison_dir}")
    print("="*80)


#%%
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Analyze debate summaries and generate visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard mode: analyze all Excel files found in data folder
  python utils/main_data_analisys.py
  
  # Regular mode: analyze a single specific file
  python utils/main_data_analisys.py --file path/to/all_debates_summary.xlsx
  
  # Comparison mode: compare two specific files
  python utils/main_data_analisys.py --compare file1.xlsx file2.xlsx
  
  # Comparison mode with settings file to extract LLM names
  python utils/main_data_analisys.py --compare file1.xlsx file2.xlsx --settings settings.yaml
  
  # Comparison mode with custom model names (overrides settings)
  python utils/main_data_analisys.py --compare file1.xlsx file2.xlsx --settings settings.yaml --model1 "GPT-4" --model2 "Claude"
        """
    )
    
    parser.add_argument('--file', type=str, default=None,
                       help='Single Excel file to analyze (regular mode, single file)')
    parser.add_argument('--compare', nargs=2, metavar=('FILE1', 'FILE2'),
                       help='Comparison mode: provide two Excel file paths. '
                            'File1: Model1 as persuader vs Model2 as debater. '
                            'File2: Model2 as persuader vs Model1 as debater.')
    parser.add_argument('--heatmap', nargs='+', metavar='FILE',
                       help='Heatmap mode: provide multiple Excel file paths to create '
                            'success rate heatmaps by helper type (rows=debater, cols=persuader)')
    parser.add_argument('--settings', type=str, nargs='+', default=None,
                       help='Path(s) to settings YAML file(s) to extract LLM model names from '
                            '(one per Excel file in --heatmap mode)')
    parser.add_argument('--model1', type=str, default=None,
                       help='Name for Model 1 (overrides settings file if provided)')
    parser.add_argument('--model2', type=str, default=None,
                       help='Name for Model 2 (overrides settings file if provided)')
    
    args = parser.parse_args()
    
    # Check if heatmap mode is requested
    if args.heatmap:
        excel_files = [Path(f) for f in args.heatmap]
        
        # Validate files exist
        for excel_file in excel_files:
            if not excel_file.exists():
                print(f"ERROR: File not found: {excel_file}")
                sys.exit(1)
        
        # Get settings files if provided
        settings_files = None
        if args.settings:
            settings_files = [Path(f) for f in args.settings]
            if len(settings_files) != len(excel_files):
                print(f"Warning: Number of settings files ({len(settings_files)}) doesn't match "
                      f"number of Excel files ({len(excel_files)}). Using available settings files.")
        
        try:
            run_multi_file_heatmap_analysis(excel_files, settings_files)
        except Exception as e:
            print(f"\nERROR in heatmap analysis:")
            print(f"  {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    # Check if comparison mode is requested
    elif args.compare:
        file1_path = Path(args.compare[0])
        file2_path = Path(args.compare[1])
        
        if not file1_path.exists():
            print(f"ERROR: File 1 not found: {file1_path}")
            sys.exit(1)
        
        if not file2_path.exists():
            print(f"ERROR: File 2 not found: {file2_path}")
            sys.exit(1)
        
        # Parse LLM names from settings file if provided
        model1_name = args.model1
        model2_name = args.model2
        
        if args.settings:
            settings_path = Path(args.settings)
            if settings_path.exists():
                parsed_model1, parsed_model2 = parse_llm_names_from_settings(settings_path)
                # Use parsed names only if not explicitly provided
                if model1_name is None:
                    model1_name = parsed_model1
                if model2_name is None:
                    model2_name = parsed_model2
            else:
                print(f"Warning: Settings file not found: {settings_path}")
        
        try:
            run_comparison_analysis(file1_path, file2_path, model1_name, model2_name)
        except Exception as e:
            print(f"\nERROR in comparison analysis:")
            print(f"  {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    elif args.file:
        # Single file mode: analyze the specified file
        file_path = Path(args.file)
        
        if not file_path.exists():
            print(f"ERROR: File not found: {file_path}")
            sys.exit(1)
        
        try:
            print(f"\nAnalyzing single file: {file_path}")
            run_analysis_for_file(file_path)
            print("\n" + "="*80)
            print("Analysis complete!")
            print("="*80)
        except Exception as e:
            print(f"\nERROR processing {file_path}:")
            print(f"  {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        # Standard mode: Find all all_debates_summary.xlsx files in the data folder
        data_folder = PROJECT_ROOT / "data"
        excel_files = list(data_folder.rglob("all_debates_summary.xlsx"))
        
        if len(excel_files) == 0:
            print("No all_debates_summary.xlsx files found in the data folder.")
            print("Using default file...")
            excel_files = [DEFAULT_EXCEL_FILE_PATH] if DEFAULT_EXCEL_FILE_PATH.exists() else []
        
        if len(excel_files) == 0:
            print("ERROR: No Excel files found to analyze!")
            sys.exit(1)
        
        print(f"\nFound {len(excel_files)} Excel file(s) to analyze:")
        for i, file_path in enumerate(excel_files, 1):
            print(f"  {i}. {file_path}")
        
        # Process each file
        for i, excel_file_path in enumerate(excel_files, 1):
            try:
                print(f"\n\n{'='*80}")
                print(f"Processing file {i}/{len(excel_files)}")
                print(f"{'='*80}")
                run_analysis_for_file(excel_file_path)
            except Exception as e:
                print(f"\nERROR processing {excel_file_path}:")
                print(f"  {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        print("\n" + "="*80)
        print("All files processed!")
        print("="*80)

#%%