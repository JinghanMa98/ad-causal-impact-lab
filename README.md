# Ad Causal Impact Lab

An open-source toolkit for measuring the causal impact of advertising campaigns.  
It integrates **Difference-in-Differences (DiD)** and **Synthetic Control** — two widely used causal inference methods — making it suitable for A/B testing, policy change evaluation, and iPinYou / RTB ad data analysis.

## ✨ Features
- **Difference-in-Differences (DiD)**  
  Ideal when you have a clearly defined treatment and control group to estimate the Average Treatment Effect (ATE).
- **Synthetic Control**  
  Works when no natural control group exists. Constructs a synthetic control by combining multiple untreated units.
- **Visualization**  
  Automatically generates a comparison plot of the treated unit vs. synthetic control, along with numerical effect estimates.
- **Extensible**  
  Works with any structured dataset (CSV / Parquet) containing timestamps, unit identifiers, and outcome metrics.

## 📦 Installation
```bash
git clone https://github.com/yourusername/ad-causal-impact-lab.git
cd ad-causal-impact-lab
pip install -r requirements.txt
