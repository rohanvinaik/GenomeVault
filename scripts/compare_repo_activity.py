#!/usr/bin/env python3
"""
Multi-Repository Activity Comparison

Compares activity metrics across multiple GitHub repositories to determine
if GenomeVault's activity is statistically significant.

Requires GitHub personal access token with repo:public_repo scope.
"""

import os
import sys
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import numpy as np
from scipy import stats


class RepoActivityComparator:
    """Compare activity metrics across GitHub repositories"""

    def __init__(self, github_token: Optional[str] = None):
        self.github_token = github_token or os.getenv('GITHUB_TOKEN')
        if not self.github_token:
            raise ValueError("GitHub token required. Set GITHUB_TOKEN environment variable.")

        self.headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        self.base_url = 'https://api.github.com'

    def get_repo_traffic(self, owner: str, repo: str) -> Dict:
        """Fetch traffic data for a repository"""
        traffic_data = {}

        # Get views
        response = requests.get(
            f'{self.base_url}/repos/{owner}/{repo}/traffic/views',
            headers=self.headers
        )
        if response.status_code == 200:
            traffic_data['views'] = response.json()
        else:
            print(f"Warning: Could not fetch views for {repo} (status {response.status_code})")
            traffic_data['views'] = {'count': 0, 'uniques': 0, 'views': []}

        # Get clones
        response = requests.get(
            f'{self.base_url}/repos/{owner}/{repo}/traffic/clones',
            headers=self.headers
        )
        if response.status_code == 200:
            traffic_data['clones'] = response.json()
        else:
            print(f"Warning: Could not fetch clones for {repo} (status {response.status_code})")
            traffic_data['clones'] = {'count': 0, 'uniques': 0, 'clones': []}

        # Get repository metadata
        response = requests.get(
            f'{self.base_url}/repos/{owner}/{repo}',
            headers=self.headers
        )
        if response.status_code == 200:
            repo_data = response.json()
            traffic_data['metadata'] = {
                'stars': repo_data['stargazers_count'],
                'forks': repo_data['forks_count'],
                'watchers': repo_data['watchers_count'],
                'open_issues': repo_data['open_issues_count'],
                'created_at': repo_data['created_at'],
                'updated_at': repo_data['updated_at'],
                'size': repo_data['size'],
                'language': repo_data['language']
            }

        return traffic_data

    def calculate_statistics(self, repos_data: Dict[str, Dict]) -> Dict:
        """Calculate comparative statistics across repositories"""
        stats_summary = {}

        # Extract metrics
        repo_names = list(repos_data.keys())
        view_counts = [repos_data[repo]['views']['count'] for repo in repo_names]
        unique_viewers = [repos_data[repo]['views']['uniques'] for repo in repo_names]
        clone_counts = [repos_data[repo]['clones']['count'] for repo in repo_names]
        unique_cloners = [repos_data[repo]['clones']['uniques'] for repo in repo_names]

        # Calculate descriptive statistics
        stats_summary['views'] = {
            'mean': np.mean(view_counts),
            'median': np.median(view_counts),
            'std': np.std(view_counts),
            'min': np.min(view_counts),
            'max': np.max(view_counts)
        }

        stats_summary['clones'] = {
            'mean': np.mean(clone_counts),
            'median': np.median(clone_counts),
            'std': np.std(clone_counts),
            'min': np.min(clone_counts),
            'max': np.max(clone_counts)
        }

        # Calculate z-scores for GenomeVault if present
        if 'genomevault' in repos_data:
            gv_views = repos_data['genomevault']['views']['count']
            gv_clones = repos_data['genomevault']['clones']['count']

            if stats_summary['views']['std'] > 0:
                stats_summary['genomevault_view_zscore'] = (
                    (gv_views - stats_summary['views']['mean']) / stats_summary['views']['std']
                )
            else:
                stats_summary['genomevault_view_zscore'] = 0

            if stats_summary['clones']['std'] > 0:
                stats_summary['genomevault_clone_zscore'] = (
                    (gv_clones - stats_summary['clones']['mean']) / stats_summary['clones']['std']
                )
            else:
                stats_summary['genomevault_clone_zscore'] = 0

            # Percentile ranking
            stats_summary['genomevault_view_percentile'] = (
                stats.percentileofscore(view_counts, gv_views)
            )
            stats_summary['genomevault_clone_percentile'] = (
                stats.percentileofscore(clone_counts, gv_clones)
            )

        return stats_summary

    def generate_comparison_report(
        self,
        owner: str,
        repos: List[str],
        output_file: Optional[str] = None
    ) -> str:
        """Generate comprehensive comparison report"""
        print(f"Fetching traffic data for {len(repos)} repositories...")

        repos_data = {}
        for repo in repos:
            print(f"  Fetching {repo}...")
            try:
                repos_data[repo.lower()] = self.get_repo_traffic(owner, repo)
            except Exception as e:
                print(f"  Error fetching {repo}: {e}")

        print(f"\nSuccessfully fetched data for {len(repos_data)} repositories")

        # Save raw data to JSON for archival
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        raw_data_file = Path('repository_insights') / f'comparison_raw_data_{timestamp}.json'
        raw_data_file.parent.mkdir(parents=True, exist_ok=True)

        with open(raw_data_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'owner': owner,
                'repositories': repos_data
            }, f, indent=2)
        print(f"Raw data saved to: {raw_data_file}")

        # Calculate statistics
        stats_summary = self.calculate_statistics(repos_data)

        # Generate report
        output = []
        output.append("=" * 80)
        output.append("MULTI-REPOSITORY ACTIVITY COMPARISON")
        output.append("=" * 80)
        output.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output.append(f"Repositories Analyzed: {len(repos_data)}")
        output.append("")

        # Overall statistics
        output.append("OVERALL STATISTICS (14-day window)")
        output.append("-" * 80)
        output.append("")
        output.append("Page Views:")
        output.append(f"  Mean:   {stats_summary['views']['mean']:.1f}")
        output.append(f"  Median: {stats_summary['views']['median']:.1f}")
        output.append(f"  Std:    {stats_summary['views']['std']:.1f}")
        output.append(f"  Range:  {stats_summary['views']['min']:.0f} - {stats_summary['views']['max']:.0f}")
        output.append("")
        output.append("Repository Clones:")
        output.append(f"  Mean:   {stats_summary['clones']['mean']:.1f}")
        output.append(f"  Median: {stats_summary['clones']['median']:.1f}")
        output.append(f"  Std:    {stats_summary['clones']['std']:.1f}")
        output.append(f"  Range:  {stats_summary['clones']['min']:.0f} - {stats_summary['clones']['max']:.0f}")
        output.append("")

        # Individual repository details
        output.append("INDIVIDUAL REPOSITORY METRICS")
        output.append("-" * 80)
        output.append("")

        for repo_name, data in sorted(repos_data.items()):
            is_genomevault = repo_name == 'genomevault'
            marker = " 🎯" if is_genomevault else ""

            output.append(f"{repo_name.upper()}{marker}")
            output.append("-" * 40)

            # Traffic metrics
            views = data['views']
            clones = data['clones']
            metadata = data.get('metadata', {})

            output.append(f"  Page Views:     {views['count']} (from {views['uniques']} unique visitors)")
            output.append(f"  Clones:         {clones['count']} (from {clones['uniques']} unique sources)")
            output.append(f"  Stars:          {metadata.get('stars', 'N/A')}")
            output.append(f"  Forks:          {metadata.get('forks', 'N/A')}")
            output.append(f"  Language:       {metadata.get('language', 'N/A')}")
            output.append(f"  Size:           {metadata.get('size', 0)} KB")
            output.append("")

        # Statistical significance for GenomeVault
        if 'genomevault' in repos_data:
            output.append("")
            output.append("GENOMEVAULT STATISTICAL ANALYSIS")
            output.append("=" * 80)
            output.append("")

            view_zscore = stats_summary['genomevault_view_zscore']
            clone_zscore = stats_summary['genomevault_clone_zscore']
            view_percentile = stats_summary['genomevault_view_percentile']
            clone_percentile = stats_summary['genomevault_clone_percentile']

            output.append("Z-Scores (standard deviations from mean):")
            output.append(f"  Views:  {view_zscore:+.2f}σ")
            output.append(f"  Clones: {clone_zscore:+.2f}σ")
            output.append("")

            output.append("Percentile Rankings:")
            output.append(f"  Views:  {view_percentile:.1f}th percentile")
            output.append(f"  Clones: {clone_percentile:.1f}th percentile")
            output.append("")

            # Interpretation
            output.append("Statistical Interpretation:")
            output.append("-" * 80)

            # Views interpretation
            if abs(view_zscore) < 1:
                view_sig = "NOT statistically significant (within 1σ of mean)"
            elif abs(view_zscore) < 2:
                view_sig = "MODERATELY significant (1-2σ from mean)"
            elif abs(view_zscore) < 3:
                view_sig = "HIGHLY significant (2-3σ from mean)"
            else:
                view_sig = "EXTREMELY significant (>3σ from mean)"

            output.append(f"  Views:  {view_sig}")

            # Clones interpretation
            if abs(clone_zscore) < 1:
                clone_sig = "NOT statistically significant (within 1σ of mean)"
            elif abs(clone_zscore) < 2:
                clone_sig = "MODERATELY significant (1-2σ from mean)"
            elif abs(clone_zscore) < 3:
                clone_sig = "HIGHLY significant (2-3σ from mean)"
            else:
                clone_sig = "EXTREMELY significant (>3σ from mean)"

            output.append(f"  Clones: {clone_sig}")
            output.append("")

            # Overall assessment
            if clone_zscore > 2 or view_zscore > 2:
                output.append("🔴 CONCLUSION: GenomeVault shows SIGNIFICANTLY HIGHER activity than")
                output.append("   typical repositories in your portfolio.")
            elif clone_zscore > 1 or view_zscore > 1:
                output.append("🟡 CONCLUSION: GenomeVault shows MODERATELY HIGHER activity than")
                output.append("   typical repositories in your portfolio.")
            else:
                output.append("🟢 CONCLUSION: GenomeVault activity is SIMILAR to other repositories")
                output.append("   in your portfolio.")

        output.append("")
        output.append("=" * 80)

        report = '\n'.join(output)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"\nReport saved to: {output_file}")

        return report


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare GitHub repository activity metrics"
    )
    parser.add_argument(
        '--owner',
        default='rohanvinaik',
        help='GitHub repository owner'
    )
    parser.add_argument(
        '--repos',
        nargs='+',
        default=['GenomeVault', 'REV', 'AuDHD_Correlation_Study', 'COEC-Framework', 'VintageOptics'],
        help='List of repositories to compare'
    )
    parser.add_argument(
        '--output',
        help='Output file for report'
    )
    parser.add_argument(
        '--token',
        help='GitHub personal access token (or set GITHUB_TOKEN env var)'
    )

    args = parser.parse_args()

    # Set token if provided
    if args.token:
        os.environ['GITHUB_TOKEN'] = args.token

    comparator = RepoActivityComparator()
    report = comparator.generate_comparison_report(
        owner=args.owner,
        repos=args.repos,
        output_file=args.output
    )

    print(report)


if __name__ == '__main__':
    main()
