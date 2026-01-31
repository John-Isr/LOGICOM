"""
Script to analyze debate results from Excel file
Generates tables and graphs showing statistics for each helper type, including:
- Success rates and efficiency metrics
- Conviction rate progression over rounds
- Feedback tag frequency and effectiveness
- Comparative analysis across helper types
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from datetime import datetime
import json
import numpy as np
from collections import Counter

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def load_and_prepare_data(excel_path):
    """Load Excel file and prepare data for analysis."""
    try:
        # Try to load Summary sheet
        df = pd.read_excel(excel_path, sheet_name='Summary')
        print(f"✓ Loaded {len(df)} debates from Summary sheet")
        
        # Parse JSON vectors
        if 'conviction_rates_vector' in df.columns:
            df['conviction_rates'] = df['conviction_rates_vector'].apply(
                lambda x: json.loads(x) if pd.notna(x) and x else []
            )
        else:
            df['conviction_rates'] = [[] for _ in range(len(df))]
            
        if 'feedback_tags_vector' in df.columns:
            df['feedback_tags'] = df['feedback_tags_vector'].apply(
                lambda x: json.loads(x) if pd.notna(x) and x else []
            )
        else:
            df['feedback_tags'] = [[] for _ in range(len(df))]
        
        if 'argument_quality_rates_vector' in df.columns:
            df['argument_quality_rates'] = df['argument_quality_rates_vector'].apply(
                lambda x: json.loads(x) if pd.notna(x) and x else []
            )
        else:
            df['argument_quality_rates'] = [[] for _ in range(len(df))]
        
        # Parse debate quality rating (should be numeric)
        if 'debate_quality_rating' in df.columns:
            df['debate_quality_rating'] = pd.to_numeric(df['debate_quality_rating'], errors='coerce')
        else:
            df['debate_quality_rating'] = None
        
        print(f"\nColumns: {df.columns.tolist()}")
        return df

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)


def calculate_basic_stats(df):
    """
    Calculate basic statistics per helper type.
    Result codes: 1=convinced, 0=not convinced, 2=inconclusive, -1=error
    Note: Success rate excludes debates convinced in 1 round for more valid statistics.
    """
    stats = []
    
    for helper_type in df['helper_type'].unique():
        helper_df = df[df['helper_type'] == helper_type]
        
        total = len(helper_df)
        convinced = len(helper_df[helper_df['result'] == 1])
        not_convinced = len(helper_df[helper_df['result'] == 0])
        inconclusive = len(helper_df[helper_df['result'] == 2])
        errors = len(helper_df[helper_df['result'] == -1])
        
        # Count debates convinced in 1 round
        convinced_in_1_round = len(helper_df[(helper_df['result'] == 1) & (helper_df['rounds'] == 1)])
        one_round_rate = (convinced_in_1_round / convinced * 100) if convinced > 0 else 0
        
        # Calculate success rate EXCLUDING 1-round debates (for more valid statistics)
        convinced_2plus_rounds = convinced - convinced_in_1_round
        conclusive_2plus_rounds = convinced_2plus_rounds + not_convinced
        success_rate = (convinced_2plus_rounds / conclusive_2plus_rounds * 100) if conclusive_2plus_rounds > 0 else 0
        
        # Average rounds (excluding errors)
        valid_debates = helper_df[helper_df['result'] != -1]
        avg_rounds = valid_debates['rounds'].mean() if len(valid_debates) > 0 else 0
        
        stats.append({
            'Helper Type': helper_type,
            'Total Debates': total,
            'Convinced': convinced,
            'Convinced in 1 Round': convinced_in_1_round,
            '1-Round Rate (%)': round(one_round_rate, 2),
            'Not Convinced': not_convinced,
            'Inconclusive': inconclusive,
            'Errors': errors,
            'Success Rate (%)': round(success_rate, 2),
            'Avg Rounds': round(avg_rounds, 2)
        })
    
    return pd.DataFrame(stats)


def analyze_conviction_progression(df):
    """
    Analyze how conviction rates progress over rounds for each helper type.
    Returns stats about initial, final, and average change in conviction rates.
    """
    stats = []
    
    for helper_type in df['helper_type'].unique():
        helper_df = df[df['helper_type'] == helper_type]
        
        # Get all conviction rate vectors (exclude debates with no conviction data)
        # Filter out None values (and -1 for backward compatibility with old data)
        conviction_vectors = [
            rates for rates in helper_df['conviction_rates'] 
            if rates and len(rates) > 0 and any(r is not None and r != -1 for r in rates)
        ]
        
        if not conviction_vectors:
            continue
        
        # Calculate stats (exclude None and -1 for backward compatibility)
        initial_rates = [v[0] for v in conviction_vectors if v[0] is not None and v[0] != -1]
        final_rates = [v[-1] for v in conviction_vectors if v[-1] is not None and v[-1] != -1]
        
        # Calculate average change
        changes = [
            v[-1] - v[0] for v in conviction_vectors 
            if v[0] is not None and v[0] != -1 and v[-1] is not None and v[-1] != -1
        ]
        
        stats.append({
            'Helper Type': helper_type,
            'Avg Initial Rate': round(np.mean(initial_rates), 2) if initial_rates else -1,
            'Avg Final Rate': round(np.mean(final_rates), 2) if final_rates else -1,
            'Avg Change': round(np.mean(changes), 2) if changes else 0,
            'Max Improvement': max(changes) if changes else 0,
            'Min Change': min(changes) if changes else 0,
            'Num Tracked': len(conviction_vectors)
        })
    
    return pd.DataFrame(stats)


def analyze_feedback_tags(df):
    """Analyze feedback tag usage and effectiveness per helper type (fallacy helpers only)."""
    stats = []
    
    # Filter to only include fallacy-related helper types
    fallacy_helpers = [ht for ht in df['helper_type'].unique() if 'fallacy' in ht.lower()]
    
    if not fallacy_helpers:
        print("⚠ No fallacy helpers found in data, skipping feedback tag analysis")
        return pd.DataFrame(), Counter()
    
    for helper_type in fallacy_helpers:
        helper_df = df[df['helper_type'] == helper_type]
        
        # Flatten all feedback tags (exclude None), normalize to title case for case-insensitive counting
        all_tags = [
            tag.title() for tags in helper_df['feedback_tags'] 
            if tags for tag in tags if tag
        ]
        
        if not all_tags:
            stats.append({
                'Helper Type': helper_type,
                'Total Tags Used': 0,
                'Unique Tags': 0,
                'Most Common Tag': 'N/A',
                'Tag Count': 0
            })
            continue
        
        tag_counter = Counter(all_tags)
        most_common = tag_counter.most_common(1)[0]
        
        stats.append({
            'Helper Type': helper_type,
            'Total Tags Used': len(all_tags),
            'Unique Tags': len(tag_counter),
            'Most Common Tag': most_common[0],
            'Tag Count': most_common[1]
        })
    
    return pd.DataFrame(stats), tag_counter


def analyze_argument_quality(df):
    """Analyze argument quality rates per helper type."""
    stats = []
    
    for helper_type in df['helper_type'].unique():
        helper_df = df[df['helper_type'] == helper_type]
        
        # Get all argument quality rate vectors (exclude debates with no data)
        arg_quality_vectors = [
            rates for rates in helper_df['argument_quality_rates'] 
            if rates and len(rates) > 0 and any(r is not None and r != -1 for r in rates)
        ]
        
        if not arg_quality_vectors:
            stats.append({
                'Helper Type': helper_type,
                'Avg Argument Quality': -1,
                'Avg Initial Quality': -1,
                'Avg Final Quality': -1,
                'Avg Change': 0,
                'Max Quality': -1,
                'Min Quality': -1,
                'Num Tracked': 0
            })
            continue
        
        # Flatten all rates (excluding None and -1)
        all_rates = [
            r for rates in arg_quality_vectors 
            for r in rates if r is not None and r != -1
        ]
        
        # Calculate initial and final rates
        initial_rates = [v[0] for v in arg_quality_vectors if v[0] is not None and v[0] != -1]
        final_rates = [v[-1] for v in arg_quality_vectors if v[-1] is not None and v[-1] != -1]
        
        # Calculate average change
        changes = [
            v[-1] - v[0] for v in arg_quality_vectors 
            if v[0] is not None and v[0] != -1 and v[-1] is not None and v[-1] != -1
        ]
        
        stats.append({
            'Helper Type': helper_type,
            'Avg Argument Quality': round(np.mean(all_rates), 2) if all_rates else -1,
            'Avg Initial Quality': round(np.mean(initial_rates), 2) if initial_rates else -1,
            'Avg Final Quality': round(np.mean(final_rates), 2) if final_rates else -1,
            'Avg Change': round(np.mean(changes), 2) if changes else 0,
            'Max Quality': max(all_rates) if all_rates else -1,
            'Min Quality': min(all_rates) if all_rates else -1,
            'Num Tracked': len(arg_quality_vectors)
        })
    
    return pd.DataFrame(stats)


def analyze_debate_quality(df):
    """Analyze overall debate quality ratings per helper type."""
    stats = []
    
    for helper_type in df['helper_type'].unique():
        helper_df = df[df['helper_type'] == helper_type]
        
        # Get valid debate quality ratings (exclude None and NaN)
        quality_ratings = [
            r for r in helper_df['debate_quality_rating'] 
            if pd.notna(r) and r is not None
        ]
        
        if not quality_ratings:
            stats.append({
                'Helper Type': helper_type,
                'Avg Debate Quality': -1,
                'Max Quality': -1,
                'Min Quality': -1,
                'Std Dev': -1,
                'Num Rated': 0
            })
            continue
        
        stats.append({
            'Helper Type': helper_type,
            'Avg Debate Quality': round(np.mean(quality_ratings), 2),
            'Max Quality': max(quality_ratings),
            'Min Quality': min(quality_ratings),
            'Std Dev': round(np.std(quality_ratings), 2),
            'Num Rated': len(quality_ratings)
        })
    
    return pd.DataFrame(stats)


def plot_success_rates(stats_df, output_dir):
    """Create bar chart comparing success rates across helper types."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort by success rate
    stats_df = stats_df.sort_values('Success Rate (%)', ascending=False)
    
    colors = sns.color_palette("husl", len(stats_df))
    bars = ax.bar(stats_df['Helper Type'], stats_df['Success Rate (%)'], color=colors)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Helper Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Success Rate Comparison Across Helper Types', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'success_rate_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: success_rate_comparison.png")
    plt.close()


def plot_rounds_distribution(df, output_dir):
    """Create box plot showing distribution of rounds for convinced debates (excluding 1-round debates)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Only include convinced results with 2+ rounds (exclude 1-round debates for valid statistics)
    convinced_df = df[(df['result'] == 1) & (df['rounds'] > 1)]
    
    helper_types = sorted(convinced_df['helper_type'].unique())
    
    # Prepare data: rounds for each helper type
    data_to_plot = [convinced_df[convinced_df['helper_type'] == ht]['rounds'].values 
                    for ht in helper_types]
    
    if not data_to_plot or all(len(d) == 0 for d in data_to_plot):
        print("⚠ No convinced debates with 2+ rounds found, skipping rounds distribution plot")
        return
    
    # Create box plot
    bp = ax.boxplot(data_to_plot, labels=helper_types, patch_artist=True)
    
    # Color the boxes
    colors = sns.color_palette("husl", len(helper_types))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Helper Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Rounds', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Rounds to Completion (Excluding 1-Round Debates)', fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'rounds_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: rounds_distribution.png")
    plt.close()


def plot_conviction_progression(df, output_dir):
    """Plot average conviction rate progression over rounds in two separate graphs."""
    
    # Define outcome categories
    outcomes = [
        (1, 'Convinced', 'conviction_progression_convinced.png'),
        (0, 'Not Convinced', 'conviction_progression_not_convinced.png')
    ]
    
    colors = sns.color_palette("husl", len(df['helper_type'].unique()))
    color_map = {ht: colors[i] for i, ht in enumerate(df['helper_type'].unique())}
    
    for result_code, outcome_label, filename in outcomes:
        fig, ax = plt.subplots(figsize=(14, 7))
        
        has_data = False
        for helper_type in df['helper_type'].unique():
            # Filter by helper type and outcome
            helper_outcome_df = df[(df['helper_type'] == helper_type) & (df['result'] == result_code)]
            
            # Get conviction rate vectors (exclude None and -1 for backward compatibility)
            conviction_vectors = [
                rates for rates in helper_outcome_df['conviction_rates'] 
                if rates and len(rates) > 0 and any(r is not None and r != -1 for r in rates)
            ]
            
            if not conviction_vectors:
                continue
            
            has_data = True
            
            # Find max rounds
            max_rounds = max(len(v) for v in conviction_vectors)
            
            # Calculate average conviction rate per round
            avg_rates = []
            for round_idx in range(max_rounds):
                rates_at_round = [
                    v[round_idx] for v in conviction_vectors 
                    if round_idx < len(v) and v[round_idx] is not None and v[round_idx] != -1
                ]
                if rates_at_round:
                    avg_rates.append(np.mean(rates_at_round))
                else:
                    avg_rates.append(None)
            
            # Plot line
            rounds = list(range(1, len(avg_rates) + 1))
            ax.plot(rounds, avg_rates, marker='o', linewidth=2, label=helper_type, 
                   color=color_map[helper_type], markersize=8)
        
        if not has_data:
            print(f"⚠ No data for {outcome_label}, skipping plot")
            plt.close()
            continue
        
        ax.set_xlabel('Round Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Conviction Rate (1-10)', fontsize=12, fontweight='bold')
        ax.set_title(f'Conviction Rate Progression - {outcome_label} Debates', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 10.5)
        
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()


def plot_tag_frequency(df, output_dir):
    """Plot most common feedback tags across fallacy helper types only (case-insensitive)."""
    # Filter to only include fallacy-related helper types
    fallacy_helpers = [ht for ht in df['helper_type'].unique() if 'fallacy' in ht.lower()]
    
    if not fallacy_helpers:
        print("⚠ No fallacy helpers found, skipping tag frequency plot")
        return
    
    fallacy_df = df[df['helper_type'].isin(fallacy_helpers)]
    
    # Collect all tags from fallacy helpers only, normalize to title case for case-insensitive counting
    all_tags = []
    for tags in fallacy_df['feedback_tags']:
        if tags:
            all_tags.extend([tag.title() for tag in tags if tag])
    
    if not all_tags:
        print("⚠ No feedback tags found for fallacy helpers, skipping tag frequency plot")
        return
    
    # Get top 15 most common tags
    tag_counter = Counter(all_tags)
    top_tags = tag_counter.most_common(15)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    tags, counts = zip(*top_tags)
    colors = sns.color_palette("viridis", len(tags))
    bars = ax.barh(range(len(tags)), counts, color=colors)
    
    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(count, i, f' {count}', va='center', fontsize=10, fontweight='bold')
    
    ax.set_yticks(range(len(tags)))
    ax.set_yticklabels(tags)
    ax.set_xlabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Top 15 Most Common Feedback Tags (Fallacy Helpers Only)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'tag_frequency.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: tag_frequency.png")
    plt.close()


def plot_finish_reasons(df, output_dir):
    """Plot distribution of finish reasons per helper type (grouped into 3 categories)."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get helper types
    helper_types = df['helper_type'].unique()
    
    # Define categorization function
    def categorize_finish_reason(reason):
        if pd.isna(reason):
            return 'Other'
        reason_str = str(reason).lower()
        if 'convinced' in reason_str:
            return 'Debater Convinced'
        elif 'max' in reason_str and 'round' in reason_str:
            return 'Max Rounds Reached'
        else:
            return 'Other'
    
    # Add categorized finish reason column
    df['finish_reason_category'] = df['finish_reason'].apply(categorize_finish_reason)
    
    # Define the three categories in order
    categories = ['Debater Convinced', 'Max Rounds Reached', 'Other']
    
    # Create matrix for stacked bar chart
    data = []
    for category in categories:
        counts = [len(df[(df['helper_type'] == ht) & (df['finish_reason_category'] == category)]) 
                  for ht in helper_types]
        data.append(counts)
    
    # Create stacked bar chart
    x = np.arange(len(helper_types))
    width = 0.6
    bottom = np.zeros(len(helper_types))
    
    colors = ['#90EE90', '#FFB6C1', '#D3D3D3']  # Green, Pink, Gray
    
    for i, (category, counts) in enumerate(zip(categories, data)):
        ax.bar(x, counts, width, label=category, bottom=bottom, color=colors[i])
        bottom += counts
    
    ax.set_xlabel('Helper Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Debates', fontsize=12, fontweight='bold')
    ax.set_title('Finish Reason Distribution by Helper Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(helper_types, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'finish_reasons.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: finish_reasons.png")
    plt.close()


def plot_argument_quality_progression(df, output_dir):
    """Plot average argument quality progression over rounds."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = sns.color_palette("husl", len(df['helper_type'].unique()))
    color_map = {ht: colors[i] for i, ht in enumerate(df['helper_type'].unique())}
    
    has_data = False
    for helper_type in df['helper_type'].unique():
        helper_df = df[df['helper_type'] == helper_type]
        
        # Get argument quality rate vectors
        arg_quality_vectors = [
            rates for rates in helper_df['argument_quality_rates'] 
            if rates and len(rates) > 0 and any(r is not None and r != -1 for r in rates)
        ]
        
        if not arg_quality_vectors:
            continue
        
        has_data = True
        
        # Find max rounds
        max_rounds = max(len(v) for v in arg_quality_vectors)
        
        # Calculate average argument quality rate per round
        avg_rates = []
        for round_idx in range(max_rounds):
            rates_at_round = [
                v[round_idx] for v in arg_quality_vectors 
                if round_idx < len(v) and v[round_idx] is not None and v[round_idx] != -1
            ]
            if rates_at_round:
                avg_rates.append(np.mean(rates_at_round))
            else:
                avg_rates.append(None)
        
        # Plot line
        rounds = list(range(1, len(avg_rates) + 1))
        ax.plot(rounds, avg_rates, marker='o', linewidth=2, label=helper_type, 
               color=color_map[helper_type], markersize=8)
    
    if not has_data:
        print("⚠ No argument quality data found, skipping argument quality progression plot")
        plt.close()
        return
    
    ax.set_xlabel('Round Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Argument Quality Rate (1-10)', fontsize=12, fontweight='bold')
    ax.set_title('Argument Quality Progression Over Rounds', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 10.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'argument_quality_progression.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: argument_quality_progression.png")
    plt.close()


def plot_debate_quality_comparison(df, output_dir):
    """Create bar chart comparing average debate quality ratings across helper types."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    stats = []
    for helper_type in df['helper_type'].unique():
        helper_df = df[df['helper_type'] == helper_type]
        quality_ratings = [
            r for r in helper_df['debate_quality_rating'] 
            if pd.notna(r) and r is not None
        ]
        if quality_ratings:
            stats.append({
                'Helper Type': helper_type,
                'Avg Quality': np.mean(quality_ratings)
            })
    
    if not stats:
        print("⚠ No debate quality ratings found, skipping debate quality comparison plot")
        plt.close()
        return
    
    stats_df = pd.DataFrame(stats).sort_values('Avg Quality', ascending=False)
    
    colors = sns.color_palette("husl", len(stats_df))
    bars = ax.bar(stats_df['Helper Type'], stats_df['Avg Quality'], color=colors)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Helper Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Debate Quality (1-10)', fontsize=12, fontweight='bold')
    ax.set_title('Average Debate Quality Comparison Across Helper Types', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 10.5)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'debate_quality_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: debate_quality_comparison.png")
    plt.close()


def generate_summary_report(stats_df, conviction_df, tags_df, arg_quality_df, debate_quality_df, output_dir):
    """Generate a text summary report."""
    report_path = output_dir / 'analysis_summary.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("DEBATE ANALYSIS SUMMARY REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        # Basic Statistics
        f.write("BASIC STATISTICS BY HELPER TYPE\n")
        f.write("-" * 80 + "\n")
        f.write(stats_df.to_string(index=False))
        f.write("\n\n")
        
        # Conviction Progression
        if not conviction_df.empty:
            f.write("CONVICTION RATE PROGRESSION\n")
            f.write("-" * 80 + "\n")
            f.write(conviction_df.to_string(index=False))
            f.write("\n\n")
        
        # Argument Quality
        if not arg_quality_df.empty:
            f.write("ARGUMENT QUALITY ANALYSIS\n")
            f.write("-" * 80 + "\n")
            f.write(arg_quality_df.to_string(index=False))
            f.write("\n\n")
        
        # Debate Quality
        if not debate_quality_df.empty:
            f.write("DEBATE QUALITY ANALYSIS\n")
            f.write("-" * 80 + "\n")
            f.write(debate_quality_df.to_string(index=False))
            f.write("\n\n")
        
        # Feedback Tags
        if not tags_df.empty:
            f.write("FEEDBACK TAG ANALYSIS\n")
            f.write("-" * 80 + "\n")
            f.write(tags_df.to_string(index=False))
            f.write("\n\n")
        
        # Key Findings
        f.write("KEY FINDINGS\n")
        f.write("-" * 80 + "\n")
        
        # Best success rate
        best_helper = stats_df.loc[stats_df['Success Rate (%)'].idxmax()]
        f.write(f"✓ Highest Success Rate: {best_helper['Helper Type']} ({best_helper['Success Rate (%)']}%)\n")
        
        # Most efficient (fewest rounds)
        efficient_helper = stats_df.loc[stats_df['Avg Rounds'].idxmin()]
        f.write(f"✓ Most Efficient: {efficient_helper['Helper Type']} ({efficient_helper['Avg Rounds']} avg rounds)\n")
        
        # Best improvement
        if not conviction_df.empty:
            best_improvement = conviction_df.loc[conviction_df['Avg Change'].idxmax()]
            f.write(f"✓ Best Conviction Improvement: {best_improvement['Helper Type']} (+{best_improvement['Avg Change']} avg)\n")
        
        # Best argument quality
        if not arg_quality_df.empty and (arg_quality_df['Avg Argument Quality'] > 0).any():
            best_arg_quality = arg_quality_df.loc[arg_quality_df['Avg Argument Quality'].idxmax()]
            f.write(f"✓ Best Argument Quality: {best_arg_quality['Helper Type']} ({best_arg_quality['Avg Argument Quality']} avg)\n")
        
        # Best debate quality
        if not debate_quality_df.empty:
            best_debate_quality = debate_quality_df.loc[debate_quality_df['Avg Debate Quality'].idxmax()]
            f.write(f"✓ Best Debate Quality: {best_debate_quality['Helper Type']} ({best_debate_quality['Avg Debate Quality']} avg)\n")
        
        f.write("\n")
    
    print(f"✓ Saved: analysis_summary.txt")


def main():
    """Main analysis pipeline."""
    # Get path to Excel file
    if len(sys.argv) > 1:
        excel_path = Path(sys.argv[1])
    else:
        excel_path = Path('all_debates_summary.xlsx')
    
    if not excel_path.exists():
        print(f"❌ File not found: {excel_path}")
        print("Usage: python analyze_debate_results.py [path_to_excel_file] [output_directory]")
        sys.exit(1)
    
    # Get output directory (optional - defaults to debate_analysis_output/timestamp)
    if len(sys.argv) > 2:
        output_dir = Path(sys.argv[2])
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path('debate_analysis_output') / timestamp
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"ANALYZING DEBATE RESULTS")
    print(f"{'='*80}\n")
    print(f"Output directory: {output_dir}\n")
    
    # Load data
    print("Loading data...")
    df = load_and_prepare_data(excel_path)
    
    # Calculate statistics
    print("\nCalculating statistics...")
    stats_df = calculate_basic_stats(df)
    conviction_df = analyze_conviction_progression(df)
    tags_df, tag_counter = analyze_feedback_tags(df)
    arg_quality_df = analyze_argument_quality(df)
    debate_quality_df = analyze_debate_quality(df)
    
    # Generate plots
    print("\nGenerating visualizations...")
    plot_success_rates(stats_df, output_dir)
    plot_rounds_distribution(df, output_dir)
    plot_conviction_progression(df, output_dir)
    plot_argument_quality_progression(df, output_dir)
    plot_debate_quality_comparison(df, output_dir)
    plot_tag_frequency(df, output_dir)
    plot_finish_reasons(df, output_dir)
    
    # Save tables to Excel
    print("\nSaving analysis tables...")
    analysis_excel = output_dir / 'analysis_tables.xlsx'
    with pd.ExcelWriter(analysis_excel, engine='openpyxl') as writer:
        stats_df.to_excel(writer, sheet_name='Basic_Stats', index=False)
        conviction_df.to_excel(writer, sheet_name='Conviction_Progress', index=False)
        arg_quality_df.to_excel(writer, sheet_name='Argument_Quality', index=False)
        debate_quality_df.to_excel(writer, sheet_name='Debate_Quality', index=False)
        tags_df.to_excel(writer, sheet_name='Feedback_Tags', index=False)
    print(f"✓ Saved: analysis_tables.xlsx")
    
    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(stats_df, conviction_df, tags_df, arg_quality_df, debate_quality_df, output_dir)
    
    print(f"\n{'='*80}")
    print(f"✅ ANALYSIS COMPLETE!")
    print(f"All outputs saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
