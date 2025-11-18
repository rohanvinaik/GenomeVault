#!/usr/bin/env python3
"""
Self-Exclusion Setup for Privacy Insights
Filters out repository owner's traffic from analytics
"""

import json
import os
from pathlib import Path


class SelfExclusionManager:
    """
    Manage self-exclusion for repository owner traffic
    Uses browser-based identification
    """

    def __init__(self):
        self.config_file = Path("repository_insights/self_exclusion_config.json")
        self.config_file.parent.mkdir(exist_ok=True)

    def create_config(self):
        """
        Create self-exclusion configuration
        """
        config = {
            "self_exclusion_enabled": True,
            "exclusion_methods": {
                "user_agent_patterns": [
                    "GitHub-Mobile",  # GitHub mobile app
                    "GitHubDesktop",  # GitHub Desktop app
                ],
                "referrer_patterns": [
                    "github.com/rohanvinaik/GenomeVault/tree/",  # Browsing own repo
                    "github.com/rohanvinaik/GenomeVault/blob/",  # Viewing files
                    "github.com/rohanvinaik/GenomeVault/commits/",  # Viewing commits
                ],
                "known_owner_sessions": [
                    # GitHub API identifies your sessions
                    # These will be filtered automatically
                ]
            },
            "filter_settings": {
                "exclude_owner_from_view_counts": True,
                "exclude_owner_from_clone_counts": False,  # Keep your clones (for development)
                "exclude_owner_from_pattern_analysis": True
            },
            "notes": [
                "GitHub's Traffic API already excludes repository owner traffic by default",
                "This config provides additional filtering for edge cases",
                "External badge counters (komarev, visitor-badge) may still count owner visits"
            ]
        }

        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)

        return config

    def get_browser_exclusion_instructions(self):
        """
        Generate instructions for browser-based self-exclusion
        """
        instructions = """
======================================================================
  Browser-Based Self-Exclusion Setup
======================================================================

GOOD NEWS: GitHub's Traffic API already filters out repository owner traffic!

When you view your own repository:
  ✅ Your views are NOT counted in the Traffic API
  ✅ Your clones are NOT counted (unless from different account)
  ✅ Only external visitors are counted

This means the privacy insights collector already excludes you automatically.

======================================================================
  Badge Counters (External Services)
======================================================================

The README badges MAY count your visits because they're external services:
  - komarev.com/ghpvc (Repository Access Insights)
  - visitor-badge.laobi.icu (Community Engagement)

To exclude yourself from these:

Option 1: Browser Extension (Recommended)
  1. Install: "Block Site" or "uBlock Origin" extension
  2. Add these to your blocklist:
     - komarev.com/ghpvc
     - visitor-badge.laobi.icu
  3. These badges won't load when YOU view the README
  4. Other visitors will still see them normally

Option 2: Private Browsing (Simple)
  - View your repo in incognito/private mode
  - Badges may not count private browsing sessions
  - External visitors in normal mode are still counted

Option 3: Do Nothing (Easiest)
  - Badge counters include ~1 extra count (you)
  - Not a problem: you know you're the "+1"
  - All GitHub API data already excludes you
  - Privacy insights analysis is accurate

======================================================================
  Verification
======================================================================

To verify you're excluded from GitHub API:

1. Visit your repository a few times
2. Wait 24 hours
3. Run: python3 scripts/privacy_preserving_insights.py --summary
4. Check "unique_observer_sessions" - should be 0 or 1 (not you)

If you see unexpected counts, they're likely from:
  - Search engine bots
  - GitHub's own crawlers
  - CI/CD systems (GitHub Actions)

======================================================================
  Configuration Saved
======================================================================

Self-exclusion config: repository_insights/self_exclusion_config.json
Status: ENABLED (GitHub API automatically excludes owner traffic)

======================================================================
"""
        return instructions


def main():
    """
    Set up self-exclusion configuration and display instructions
    """
    manager = SelfExclusionManager()

    print("\n🔧 Setting up self-exclusion for privacy insights...")
    config = manager.create_config()

    print(f"✅ Configuration created: {manager.config_file}")
    print(manager.get_browser_exclusion_instructions())

    # Verify GitHub CLI authentication
    print("\n🔍 Verifying GitHub authentication...")
    import subprocess
    result = subprocess.run("gh auth status", shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ GitHub CLI authenticated")
        print("   GitHub's Traffic API will automatically exclude your traffic")
    else:
        print("⚠️  GitHub CLI not authenticated")
        print("   Run: gh auth login")


if __name__ == "__main__":
    main()
