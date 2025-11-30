"""
Statistical Analysis Module
Built-in statistical tests and visualization for cell tracking data
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, f_oneway, kruskal, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns


class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis for cell tracking experiments
    
    Capabilities:
    - Parametric and non-parametric tests
    - Effect size calculations
    - Multiple comparison corrections
    - Survival analysis
    - Publication-ready plots
    """
    
    def __init__(self):
        """Initialize statistical analyzer"""
        self.results = {}
        sns.set_style("whitegrid")
    
    def compare_two_conditions(self,
                               control: List[float],
                               treated: List[float],
                               metric_name: str = "speed",
                               parametric: bool = True,
                               paired: bool = False) -> Dict:
        """
        Compare two experimental conditions
        
        Args:
            control: Measurements from control condition
            treated: Measurements from treated condition
            metric_name: Name of the metric being compared
            parametric: Use parametric (t-test) or non-parametric (Mann-Whitney)
            paired: Whether data is paired
            
        Returns:
            Dictionary with test results
        """
        control = np.array(control)
        treated = np.array(treated)
        
        # Descriptive statistics
        control_mean = np.mean(control)
        control_std = np.std(control, ddof=1)
        control_sem = control_std / np.sqrt(len(control))
        
        treated_mean = np.mean(treated)
        treated_std = np.std(treated, ddof=1)
        treated_sem = treated_std / np.sqrt(len(treated))
        
        # Statistical test
        if parametric:
            if paired:
                statistic, p_value = stats.ttest_rel(control, treated)
                test_name = "Paired t-test"
            else:
                statistic, p_value = stats.ttest_ind(control, treated)
                test_name = "Independent t-test"
        else:
            if paired:
                statistic, p_value = stats.wilcoxon(control, treated)
                test_name = "Wilcoxon signed-rank test"
            else:
                statistic, p_value = stats.mannwhitneyu(control, treated)
                test_name = "Mann-Whitney U test"
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((control_std**2 + treated_std**2) / 2)
        cohens_d = (treated_mean - control_mean) / max(pooled_std, 1e-10)
        
        # Interpret effect size
        if abs(cohens_d) < 0.2:
            effect_interpretation = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_interpretation = "small"
        elif abs(cohens_d) < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"
        
        # Percent change
        if control_mean != 0:
            percent_change = ((treated_mean - control_mean) / control_mean) * 100
        else:
            percent_change = float('inf')
        
        results = {
            'metric': metric_name,
            'test': test_name,
            'control_n': len(control),
            'control_mean': control_mean,
            'control_std': control_std,
            'control_sem': control_sem,
            'treated_n': len(treated),
            'treated_mean': treated_mean,
            'treated_std': treated_std,
            'treated_sem': treated_sem,
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'cohens_d': cohens_d,
            'effect_size': effect_interpretation,
            'percent_change': percent_change
        }
        
        self.results[f"{metric_name}_comparison"] = results
        return results
    
    def compare_multiple_conditions(self,
                                    groups: Dict[str, List[float]],
                                    metric_name: str = "speed",
                                    parametric: bool = True) -> Dict:
        """
        Compare multiple experimental conditions (ANOVA or Kruskal-Wallis)
        
        Args:
            groups: Dictionary mapping condition name to measurements
            metric_name: Name of metric
            parametric: Use ANOVA (True) or Kruskal-Wallis (False)
            
        Returns:
            Test results including post-hoc comparisons
        """
        group_names = list(groups.keys())
        group_data = [np.array(groups[name]) for name in group_names]
        
        # Overall test
        if parametric:
            statistic, p_value = f_oneway(*group_data)
            test_name = "One-way ANOVA"
        else:
            statistic, p_value = kruskal(*group_data)
            test_name = "Kruskal-Wallis H test"
        
        # Descriptive statistics
        group_stats = {}
        for name, data in groups.items():
            group_stats[name] = {
                'n': len(data),
                'mean': np.mean(data),
                'std': np.std(data, ddof=1),
                'sem': np.std(data, ddof=1) / np.sqrt(len(data))
            }
        
        # Post-hoc pairwise comparisons (if significant)
        posthoc = {}
        if p_value < 0.05:
            for i, name1 in enumerate(group_names):
                for name2 in group_names[i+1:]:
                    data1 = np.array(groups[name1])
                    data2 = np.array(groups[name2])
                    
                    if parametric:
                        _, pairwise_p = ttest_ind(data1, data2)
                    else:
                        _, pairwise_p = mannwhitneyu(data1, data2)
                    
                    # Bonferroni correction
                    n_comparisons = len(group_names) * (len(group_names) - 1) / 2
                    corrected_p = min(pairwise_p * n_comparisons, 1.0)
                    
                    posthoc[f"{name1}_vs_{name2}"] = {
                        'p_value': pairwise_p,
                        'corrected_p': corrected_p,
                        'significant': corrected_p < 0.05
                    }
        
        results = {
            'metric': metric_name,
            'test': test_name,
            'n_groups': len(groups),
            'group_stats': group_stats,
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'posthoc_comparisons': posthoc
        }
        
        self.results[f"{metric_name}_anova"] = results
        return results
    
    def survival_analysis(self,
                         tracks: List,
                         max_time: Optional[int] = None) -> Dict:
        """
        Kaplan-Meier survival analysis for cell tracking
        
        Args:
            tracks: List of Track objects
            max_time: Maximum observation time
            
        Returns:
            Survival statistics and curve
        """
        if max_time is None:
            max_time = max(max(t.frame_indices) for t in tracks)
        
        # Extract survival times
        survival_times = []
        censored = []
        
        for track in tracks:
            # Time is last frame index
            time = max(track.frame_indices)
            survival_times.append(time)
            
            # Censored if track reaches end of observation
            censored.append(time >= max_time - 1)
        
        survival_times = np.array(survival_times)
        censored = np.array(censored)
        
        # Calculate Kaplan-Meier curve
        unique_times = np.unique(survival_times)
        survival_prob = []
        
        n_at_risk = len(survival_times)
        cumulative_survival = 1.0
        
        for t in unique_times:
            # Number of events at this time
            n_events = np.sum((survival_times == t) & ~censored)
            
            # Number censored at this time
            n_censored = np.sum((survival_times == t) & censored)
            
            # Update survival probability
            if n_at_risk > 0:
                survival_rate = 1 - (n_events / n_at_risk)
                cumulative_survival *= survival_rate
            
            survival_prob.append(cumulative_survival)
            
            # Update at-risk population
            n_at_risk -= (n_events + n_censored)
        
        # Calculate median survival
        median_survival = None
        for t, prob in zip(unique_times, survival_prob):
            if prob <= 0.5:
                median_survival = t
                break
        
        results = {
            'n_tracks': len(tracks),
            'n_events': np.sum(~censored),
            'n_censored': np.sum(censored),
            'median_survival': median_survival,
            'times': unique_times.tolist(),
            'survival_probability': survival_prob
        }
        
        self.results['survival_analysis'] = results
        return results
    
    def calculate_power_analysis(self,
                                effect_size: float,
                                n_samples: int,
                                alpha: float = 0.05) -> Dict:
        """
        Calculate statistical power for given parameters
        
        Args:
            effect_size: Expected Cohen's d
            n_samples: Sample size per group
            alpha: Significance level
            
        Returns:
            Power analysis results
        """
        from scipy.stats import norm
        
        # Calculate power for two-sample t-test
        # This is a simplified version
        ncp = effect_size * np.sqrt(n_samples / 2)  # Non-centrality parameter
        critical_value = norm.ppf(1 - alpha/2)
        power = 1 - norm.cdf(critical_value - ncp) + norm.cdf(-critical_value - ncp)
        
        # Minimum detectable effect
        mde = critical_value * np.sqrt(2 / n_samples)
        
        results = {
            'effect_size': effect_size,
            'n_per_group': n_samples,
            'alpha': alpha,
            'power': power,
            'minimum_detectable_effect': mde,
            'adequate_power': power >= 0.8
        }
        
        return results
    
    def plot_comparison(self,
                       groups: Dict[str, List[float]],
                       metric_name: str = "Speed",
                       plot_type: str = "violin",
                       save_path: Optional[str] = None):
        """
        Create publication-quality comparison plots
        
        Args:
            groups: Dictionary mapping condition to measurements
            metric_name: Name for y-axis
            plot_type: 'violin', 'box', 'bar', or 'swarm'
            save_path: Path to save figure
        """
        # Prepare data
        data_list = []
        for condition, values in groups.items():
            for value in values:
                data_list.append({'Condition': condition, metric_name: value})
        
        df = pd.DataFrame(data_list)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if plot_type == "violin":
            sns.violinplot(data=df, x='Condition', y=metric_name, ax=ax)
            sns.swarmplot(data=df, x='Condition', y=metric_name, 
                         color='black', alpha=0.3, size=3, ax=ax)
        
        elif plot_type == "box":
            sns.boxplot(data=df, x='Condition', y=metric_name, ax=ax)
            sns.swarmplot(data=df, x='Condition', y=metric_name,
                         color='black', alpha=0.3, size=3, ax=ax)
        
        elif plot_type == "bar":
            means = df.groupby('Condition')[metric_name].mean()
            sems = df.groupby('Condition')[metric_name].sem()
            
            ax.bar(range(len(means)), means.values, yerr=sems.values,
                  capsize=5, alpha=0.7)
            ax.set_xticks(range(len(means)))
            ax.set_xticklabels(means.index)
            ax.set_ylabel(metric_name)
        
        elif plot_type == "swarm":
            sns.swarmplot(data=df, x='Condition', y=metric_name, ax=ax)
        
        # Add significance markers if comparison exists
        comparison_key = f"{metric_name.lower()}_anova"
        if comparison_key in self.results:
            results = self.results[comparison_key]
            if results['significant']:
                ax.text(0.5, 0.95, f"p = {results['p_value']:.4f} *",
                       transform=ax.transAxes, ha='center', fontweight='bold')
        
        ax.set_title(f'{metric_name} by Condition')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_survival_curve(self,
                           survival_results: Dict,
                           save_path: Optional[str] = None):
        """
        Plot Kaplan-Meier survival curve
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        times = survival_results['times']
        survival_prob = survival_results['survival_probability']
        
        # Plot step function
        ax.step(times, survival_prob, where='post', linewidth=2)
        ax.set_xlabel('Time (frames)')
        ax.set_ylabel('Survival Probability')
        ax.set_title('Kaplan-Meier Survival Curve')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        # Add median survival line
        if survival_results['median_survival'] is not None:
            ax.axvline(survival_results['median_survival'], 
                      color='r', linestyle='--', alpha=0.7,
                      label=f"Median: {survival_results['median_survival']:.1f}")
            ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
        
        # Add n at risk
        info_text = (f"n = {survival_results['n_tracks']}\n"
                    f"Events = {survival_results['n_events']}\n"
                    f"Censored = {survival_results['n_censored']}")
        ax.text(0.98, 0.98, info_text, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Survival curve saved to {save_path}")
        
        plt.show()
    
    def export_report(self, output_path: str):
        """
        Export comprehensive statistical report
        """
        with open(output_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("STATISTICAL ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            for analysis_name, results in self.results.items():
                f.write(f"\n{analysis_name.upper()}\n")
                f.write("-" * 70 + "\n")
                
                for key, value in results.items():
                    if isinstance(value, dict):
                        f.write(f"{key}:\n")
                        for k, v in value.items():
                            f.write(f"  {k}: {v}\n")
                    else:
                        f.write(f"{key}: {value}\n")
                
                f.write("\n")
        
        print(f"Report saved to {output_path}")


def example_usage():
    """Example statistical analysis"""
    from pipeline import Pipeline
    
    # Simulate two experimental conditions
    # In practice, you'd run the pipeline on different datasets
    
    # Control condition
    control_speeds = np.random.normal(10, 2, 50)  # Mean=10, std=2
    
    # Treated condition (20% increase)
    treated_speeds = np.random.normal(12, 2, 50)  # Mean=12, std=2
    
    # Initialize analyzer
    analyzer = StatisticalAnalyzer()
    
    # Compare two conditions
    results = analyzer.compare_two_conditions(
        control_speeds,
        treated_speeds,
        metric_name="cell_speed",
        parametric=True
    )
    
    print("\nTwo-Sample Comparison:")
    print(f"  Test: {results['test']}")
    print(f"  Control: {results['control_mean']:.2f} ± {results['control_sem']:.2f}")
    print(f"  Treated: {results['treated_mean']:.2f} ± {results['treated_sem']:.2f}")
    print(f"  p-value: {results['p_value']:.4f}")
    print(f"  Cohen's d: {results['cohens_d']:.3f} ({results['effect_size']})")
    print(f"  Significant: {'YES' if results['significant'] else 'NO'}")
    
    # Multiple conditions
    groups = {
        'Control': control_speeds.tolist(),
        'Treatment_A': treated_speeds.tolist(),
        'Treatment_B': np.random.normal(15, 2, 50).tolist()
    }
    
    anova_results = analyzer.compare_multiple_conditions(
        groups,
        metric_name="cell_speed"
    )
    
    print(f"\nANOVA Results:")
    print(f"  p-value: {anova_results['p_value']:.4f}")
    print(f"  Significant: {'YES' if anova_results['significant'] else 'NO'}")
    
    # Visualize
    analyzer.plot_comparison(groups, "Cell Speed (µm/min)", plot_type="violin",
                            save_path="comparison_plot.png")
    
    # Power analysis
    power = analyzer.calculate_power_analysis(
        effect_size=0.5,
        n_samples=50,
        alpha=0.05
    )
    
    print(f"\nPower Analysis:")
    print(f"  Power: {power['power']:.3f}")
    print(f"  Adequate: {'YES' if power['adequate_power'] else 'NO'}")
    
    # Export report
    analyzer.export_report("statistical_report.txt")


if __name__ == "__main__":
    example_usage()

