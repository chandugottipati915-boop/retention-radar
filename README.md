# Retention Radar

Churn prediction and RFM segmentation for an online retailer, wrapped in a Streamlit app
that turns a model score into a decision: **who do we contact, and what do we send them?**

Score a customer on Recency / Frequency / Monetary and the app returns their probability of
churning in the next 180 days, the segment they belong to, and the retention play for that
segment — alongside the precision, recall, and campaign size implied by whatever cutoff you pick.

---

## The problem

A churn model that outputs `0.73` is not actionable. Two things are missing:

1. **What does flagging cost?** A 50% cutoff and a 30% cutoff imply very different campaigns —
   different numbers of customers contacted, different amounts of budget wasted on people who
   were never going to leave. The app makes that tradeoff explicit and adjustable.
2. **What do we actually do?** A probability is not a plan. Segmentation supplies the action:
   a Champion and a Hibernating customer at the same 70% risk warrant completely different spend.

---

## Data

[UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) — ~1M transactions
from a UK gift retailer, Dec 2009 – Dec 2011 (`data/raw/online_retail_II.xlsx`).

Cleaning drops rows with no `Customer ID`, cancellation invoices (`Invoice` starting with `C`),
and non-positive quantities. **4,979 customers** survive with a usable purchase history.

### Labeling churn

The last **180 days** of the timeline are held out as a performance window. RFM features are built
only from transactions *before* the cutoff; a customer is labeled churned if they never purchase
*after* it. This keeps the label strictly in the future relative to every feature — no leakage.

180 days was chosen empirically. Windows of 90/120/180/270/365 were tested: 180 maximizes ROC-AUC
(0.81 vs 0.80 at 90d). Beyond 180 the label gets cleaner but the observation history shrinks too far
and accuracy falls back off. The resulting classes are close to balanced — **48.2% churn**.

---

## Modeling

### Segmentation — K-Means on log-scaled RFM

RFM is heavily right-skewed, so features go through `log1p` → `StandardScaler` before clustering.
K=4 was selected on the elbow curve and silhouette score. The four centroids:

| Segment | Recency (d) | Frequency | Monetary (£) | Customers |
|---|---:|---:|---:|---:|
| Champions | 32 | 18.5 | 10,735 | 778 |
| New Customers | 21 | 3.5 | 1,070 | 708 |
| At-Risk | 180 | 4.6 | 1,792 | 1,493 |
| Hibernating | 275 | 1.3 | 326 | 2,000 |

Cluster **IDs are not stable** — re-fitting reshuffles them, so a hardcoded `{0: 'Champions', ...}`
map silently mislabels everyone. Names are instead derived from each centroid's actual R/F profile
(recent/frequent quadrants) and written to `models/segment_names.json` at fit time, which is what
the app loads.

### Churn — XGBoost on raw RFM

SMOTE balances the training fold; the model trains on raw (unscaled) R/F/M. Hyperparameters come
from 5-fold `RandomizedSearchCV`. XGBoost's defaults (depth 6, lr 0.3, 100 trees) are tuned for wide
feature sets and badly overfit three columns — shallow trees, a slow learning rate, and
regularization lift ROC-AUC from ~0.76 to **0.81**.

| Metric | Value |
|---|---|
| ROC-AUC (holdout) | 0.813 |
| ROC-AUC (5-fold CV) | **0.814 ± 0.012** |
| Accuracy @ 0.5 | 0.73 |
| Best-F1 threshold | 0.38 (P 0.65 / R 0.88) |

### The threshold curve

Precision, recall, F1, and campaign size are swept across every cutoff from 0.20 to 0.90 and saved to
`models/threshold_curve.json`. These come from **out-of-fold predictions** — every one of the 4,979
customers scored by a model that never trained on them — not from a single 25% split, which is too
noisy at this sample size to quote honestly.

| Threshold | Precision | Recall | Customers flagged |
|---:|---:|---:|---:|
| 0.30 | 62% | 92% | 3,559 |
| 0.40 | 66% | 87% | 3,159 |
| 0.50 | 70% | 78% | 2,716 |
| 0.60 | 75% | 65% | 2,060 |
| 0.70 | 80% | 46% | 1,398 |

Reading the 0.50 row: contact 2,716 of 4,979 customers, ~30% of that budget lands on customers who
would have stayed anyway, and ~22% of real churners are missed.

---

## Retention plays

| Segment | Action |
|---|---|
| Champions | Reward loyalty — early access, VIP perks |
| New Customers | Onboard warmly — welcome series, first-repeat nudge |
| At-Risk | Win them back — personalized offer, we-miss-you email |
| Hibernating | Low-cost re-engagement — automated reactivation email |

---

## Running it

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync
uv run streamlit run app.py
```

Opens at http://localhost:8501. The app loads the pre-built artifacts in `models/`, so it runs
without re-executing the notebooks.

### Rebuilding from scratch

```bash
uv run jupyter lab
```

Run the notebooks in order — each writes the inputs the next one reads:

| Notebook | Produces |
|---|---|
| `01_eda.ipynb` | cleaning, RFM + churn label → `data/processed/customer_rfm.parquet` |
| `02_segmentation.ipynb` | K-Means → `customer_segments.parquet`, `rfm_scaler.pkl`, `kmeans.pkl`, `segment_names.json` |
| `03_churn.ipynb` | XGBoost → `churn_model.pkl`, `threshold_curve.json`, `reports/feature_importance.png` |

---

## Layout

```
app.py                      Streamlit app — scoring + threshold explorer
notebooks/                  01 EDA · 02 segmentation · 03 churn
models/                     churn_model.pkl · kmeans.pkl · rfm_scaler.pkl
                            segment_names.json · threshold_curve.json
data/raw/                   online_retail_II.xlsx
data/processed/             customer_rfm.parquet · customer_segments.parquet
reports/                    feature_importance.png
```

Two preprocessing paths must stay in sync between notebooks and app: the churn model consumes
**raw** RFM, while segmentation requires `log1p` → `scaler.transform` → `kmeans.predict`. Applying
the wrong one produces a plausible-looking but wrong answer.

---

## Limitations

- Trained on a single retailer over one two-year window; seasonality is not modeled.
- Three features only. Product mix, tenure, and channel would likely push AUC past 0.81.
- The 180-day label is a proxy for churn — a customer may simply have a long purchase cycle.
- Segment strategies are heuristics, not the output of an uplift model; they haven't been A/B tested.
