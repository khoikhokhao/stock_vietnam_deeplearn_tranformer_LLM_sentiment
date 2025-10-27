
# 🇻🇳 Vietnam Stock Portfolio Optimization using Transformer + Monte Carlo

## 📘 Overview

This project builds an **AI-driven portfolio optimization pipeline** for the Vietnamese stock market (HOSE), integrating **deep learning**, **macro-financial features**, **sentiment analysis**, and **Monte Carlo simulation** for risk-aware portfolio construction.

The workflow consists of two main stages:

1. **Prediction Model (Transformer Encoder)**
   - Predicts the probability that a stock’s price will increase after 20 trading days.
   - Input: 60-day sequence of technical, macro, and sentiment features.
   - Output: Probability `p_long` (0–1) representing confidence in price increase.

2. **Portfolio Optimization (Monte Carlo 3-Stage)**
   - Uses the predicted probabilities as expected-return proxies.
   - Generates 10,000 multi-asset market scenarios (block-bootstrap 5-day blocks).
   - Performs **3-stage weight optimization** to maximize Sharpe ratio while controlling drawdown.

---

## 🧠 Model Architecture

### Transformer Encoder
- Input: 60 × 10 feature matrix per stock
- Features include:
  - Technical: Return, RSI, Volume Ratio, Volatility
  - Macro: CPI YoY, USD/VND, Interbank Rate, Market Regime
  - Sentiment: Aggregated 3-day positive/neutral/negative score from financial news
- Architecture:
  - Linear embedding + Positional encoding
  - 2–3 Transformer encoder layers (multi-head attention, FFN)
  - Global pooling → Dense → Sigmoid output

### Loss & Metrics
- **Loss:** Binary Cross-Entropy
- **Metrics:** Accuracy, ROC-AUC, PR-AUC, F1-score, Brier score

---

## 📊 Portfolio Simulation

### Monte Carlo 3-Stage Optimization
**Stage A:** Coarse grid search across all weight combinations (5% step, cap 40%)  
**Stage B:** Zoom-in exploration near top 20 promising weights  
**Stage C:** Fine-tuning and validation on new random seeds (10,000 new scenarios)

### Backtest Summary (Optimal Config)
| Metric | Value |
|:--|--:|
| Rebalances | 19 |
| Total Trades | 131 |
| Win Rate | 61.07% |
| CAGR | 28.79% |
| Volatility | 21.88% |
| Sharpe | **1.32** |
| Max Drawdown | -13.95% |
| Final Equity | 1.33× initial capital |

---

## 🧩 Folder Structure

```
├── data/
│   ├── raw/                # raw historical stock, macro, sentiment data
│   ├── processed/          # merged + cleaned datasets
│
├── models/
│   ├── transformer_tech_macro_sent_best.keras
│   ├── montecarlo_weights.pkl
│
├── scripts/
│   ├── train_transformer.py
│   ├── backtest_montecarlo.py
│
├── notebooks/
│   ├── main_transformer.ipynb
│   ├── backtest_portfolio.ipynb
│
├── outputs/
│   ├── charts/
│   ├── results_summary.csv
│
└── README.md
```

---

## ⚙️ Requirements

```
python>=3.10
tensorflow>=2.15
numpy
pandas
scikit-learn
matplotlib
tqdm
seaborn
scipy
```

Install via:

```bash
pip install -r requirements.txt
```

---

## 📈 Highlights

✅ 10-year HOSE dataset (2015–2025) with 25 liquid tickers  
✅ Integration of macro-financial signals (CPI, FX, interbank)  
✅ Sentiment fine-tuned from **PhoBERT-large** on 5,600 CafeF articles  
✅ Robust Monte Carlo risk-tilting strategy  
✅ Sharpe > 1.3 with controlled drawdown < -15%  

---

## 👤 Author

**Phạm Minh Khôi (HE190065)**  
AI Major @ FPT University Hanoi  
Aiming to bridge **AI research and quantitative finance**.  
Contact: [linkedin.com/in/minhkhoi-ai-finance](https://www.linkedin.com/in/minhkhoi-ai-finance/)


