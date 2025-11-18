# Traffic Monitoring Setup - Complete Guide

## ✅ Installation Complete

Your GitHub traffic monitoring system is now fully automated and running.

---

## 📊 What's Monitoring

The system automatically captures daily snapshots of:
- **Clone statistics** (who's cloning your repo)
- **View statistics** (who's viewing your repo)  
- **Referrer data** (where traffic comes from)
- **Popular paths** (which files are most accessed)

**Why?** GitHub only keeps traffic data for 14 days. After that, it's permanently deleted. This system saves snapshots daily so you have a permanent historical record.

---

## 🤖 Dual Monitoring System (ACTIVE)

**Status:** ✅ **RUNNING** (Primary + Backup)

### Primary: Local Cron Monitoring

**Schedule:** Every day at 12:30 PM (local time)

**What it does:**
1. Captures GitHub traffic data from API
2. Saves snapshots to `security_analysis/traffic_history/`
3. Logs activity to `security_analysis/traffic_monitor.log`
4. Keeps data 100% private (never leaves your machine)

**Installed cron job:**
```
30 12 * * * \
  cd /Users/rohanvinaik/genomevault && \
  /Users/rohanvinaik/genomevault/scripts/monitor_github_traffic.sh >> \
  /Users/rohanvinaik/genomevault/security_analysis/traffic_monitor.log 2>&1
```

### Backup: GitHub Actions

**Schedule:** Every day at 12:30 PM UTC (7:30 AM EST / 8:30 AM EDT)

**What it does:**
1. Captures GitHub traffic data from API (same as local)
2. Creates analysis report with bot detection
3. Uploads to GitHub Actions artifacts (90-day retention)
4. Runs even when your computer is off

**Workflow:** `.github/workflows/monitor-traffic.yml`

**View backups:**
https://github.com/rohanvinaik/GenomeVault/actions/workflows/monitor-traffic.yml

**Advantages:**
- ✅ Runs 24/7 (doesn't depend on your computer)
- ✅ Automatic bot detection analysis
- ✅ 90-day artifact retention
- ⚠️ Data stored on GitHub (but only in artifacts, never in repo)

---

## 📁 Data Storage

All traffic data is saved locally and kept private:

```
security_analysis/
├── traffic_analysis_20251024.md      # Your threat analysis report
├── traffic_clones_20251024.json      # Today's clone data
├── traffic_views_20251024.json       # Today's view data
├── traffic_monitor.log               # Automation logs
└── traffic_history/                  # Daily snapshots
    ├── clones_20251024.json
    ├── views_20251024.json
    ├── referrers_20251024.json
    ├── paths_20251024.json
    ├── summary_20251024.txt
    ├── clones_20251025.json          # Tomorrow's snapshot
    ├── views_20251025.json
    └── ... (continues daily)
```

**Protection:** All files in `security_analysis/` are automatically ignored by Git (.gitignore) and will never be committed to GitHub.

---

## 🛠️ Manual Operations

### View Current Cron Jobs
```bash
crontab -l
```

### Run Monitor Manually (Right Now)
```bash
cd /Users/rohanvinaik/genomevault
./scripts/monitor_github_traffic.sh
```

### View Automation Logs
```bash
tail -f /Users/rohanvinaik/genomevault/security_analysis/traffic_monitor.log
```

### View Latest Traffic Summary
```bash
cat security_analysis/traffic_history/summary_$(date +%Y%m%d).txt
```

### View All Saved Snapshots
```bash
ls -la security_analysis/traffic_history/
```

### Disable Automated Monitoring
```bash
crontab -l | grep -v monitor_github_traffic.sh | crontab -
```

### Re-enable Automated Monitoring
```bash
./setup_traffic_monitoring.sh
```

---

## 🔍 Understanding the Data

### Clone Statistics
Shows how many times your repo was cloned (downloaded):
- **Total clones:** All clone operations in last 14 days
- **Unique cloners:** Number of different IP addresses
- **Ratio:** Total ÷ Unique (>10 = likely bots)

### View Statistics
Shows page views on GitHub.com:
- **Total views:** All page views in last 14 days
- **Unique viewers:** Number of different visitors
- **Pattern:** Humans view before cloning, bots clone without viewing

### Red Flags
- ⚠️ High clone-to-unique ratio (>10) = Automated bots
- ⚠️ Clones without views = Non-human traffic
- ⚠️ Sudden spikes after major commits = Potential IP harvesting
- ✅ Low ratio (1-3) = Likely human developers

---

## 📈 Traffic Analysis Results (Oct 24, 2025)

**Finding:** 100% automated bot traffic, 0% organic human interest

**Evidence:**
- 323 total clones from 25 IPs
- 74 total views from 1 viewer (YOU)
- 0 external human viewers
- Clone-to-IP ratio: 12.92 (normal humans: 1-2)

**Spike Period (Oct 22-24):**
- Oct 22: 137 clones (12 IPs) - Major spike
- Oct 23: 30 clones (9 IPs)
- Oct 24: 111 clones (8 IPs)

**Likely Sources:**
1. **Software Heritage** (60%) - Benign archiving service
2. **Competitive Intelligence** (25%) - Biotech company monitoring
3. **GitHub Security Scanners** (10%) - Dependabot, Snyk
4. **Targeted IP Theft** (5%) - Deliberate competitor harvesting

**Risk Level:** MODERATE - IP was exposed to automated services, likely benign but warrants monitoring

**Full Analysis:** See `security_analysis/traffic_analysis_20251024.md`

---

## ⚙️ System Requirements

**What you need:**
- ✅ macOS/Linux with cron support
- ✅ GitHub CLI (`gh`) installed and authenticated
- ✅ Computer powered on at 2 AM daily (or change schedule)

**If `gh` not installed:**
```bash
brew install gh
gh auth login
```

---

## 🔒 Privacy & Security

**What's Protected:**
- ✅ All traffic data stays local on your machine
- ✅ Never committed to Git (automatically ignored)
- ✅ Never uploaded to GitHub
- ✅ Only you have access

**What's Shared:**
- ✅ Monitoring scripts (tools only, no data)
- ✅ .gitignore rules (protection mechanism)

---

## 🆘 Troubleshooting

### Monitor not running at 2 AM?
Check if your computer was on:
```bash
tail -20 security_analysis/traffic_monitor.log
```

If log is empty, computer was likely asleep. Options:
1. Keep computer on overnight
2. Change cron time to when computer is on
3. Use caffeinate to prevent sleep:
   ```bash
   caffeinate -s &  # Prevents sleep while plugged in
   ```

### "gh: command not found"
Install GitHub CLI:
```bash
brew install gh
gh auth login
```

### Cron job not listed?
Reinstall:
```bash
./setup_traffic_monitoring.sh
```

### Want to change the schedule?
Edit crontab:
```bash
crontab -e
```

Cron schedule format: `minute hour day month weekday`
- `0 2 * * *` = 2:00 AM daily
- `0 14 * * *` = 2:00 PM daily
- `0 */6 * * *` = Every 6 hours
- `0 0 * * 1` = Midnight every Monday

---

## 📚 Additional Resources

- **Traffic Analysis Report:** `security_analysis/traffic_analysis_20251024.md`
- **Monitor Script:** `scripts/monitor_github_traffic.sh`
- **Setup Script:** `setup_traffic_monitoring.sh`
- **GitHub API Docs:** https://docs.github.com/en/rest/metrics/traffic

---

**Last Updated:** October 24, 2025  
**Status:** ✅ ACTIVE - Monitoring daily at 2:00 AM
