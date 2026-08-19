import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# load the saved artifacts (paths are from project root, no ../)
churn_model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/rfm_scaler.pkl")
kmeans = joblib.load("models/kmeans.pkl")

# Written by Phase 2 so it always matches the K-Means fit that's loaded above.
# Hardcoding this breaks silently whenever the clusters are re-fit.
with open("models/segment_names.json") as fh:
    segment_names = {int(k): v for k, v in json.load(fh).items()}

# precision/recall at every cutoff, from Phase 3's out-of-fold predictions
# (every customer scored by a model that never trained on them)
with open("models/threshold_curve.json") as fh:
    threshold_curve = {float(k): v for k, v in json.load(fh).items()}


def curve_at(t):
    """Nearest measured point on the precision/recall curve."""
    return threshold_curve[min(threshold_curve, key=lambda k: abs(k - t))]

# the business action for each segment (your Phase 5 strategy)
strategies = {
    'Champions':     "Reward loyalty — early access, VIP perks.",
    'New Customers': "Onboard warmly — welcome series, first-repeat nudge.",
    'At-Risk':       "Win them back — personalized offer, we-miss-you email.",
    'Hibernating':   "Low-cost re-engagement — automated reactivation email."
}

st.title("Retention Radar")
st.write("Set a customer's RFM profile to see their churn risk and segment.")
st.caption("Churn = no purchase in the next 180 days.")

recency   = st.slider("Recency — days since last purchase", 0, 400, 30)
frequency = st.slider("Frequency — number of orders", 1, 50, 5)
monetary  = st.slider("Monetary — total spend (£)", 0, 15000, 1000)

st.divider()
threshold = st.slider(
    "Flag as at-risk above this probability",
    0.05, 0.95, 0.50, step=0.01,
    help="Lower = catch more churners but waste more offers on customers who "
         "would have stayed. Higher = only chase the surest bets."
)

# what that cutoff actually costs, measured across the whole customer base
stats = curve_at(threshold)
c1, c2, c3 = st.columns(3)
c1.metric("Precision", f"{stats['precision']:.0%}",
          help="Of the customers we flag, this share really do churn.")
c2.metric("Recall", f"{stats['recall']:.0%}",
          help="Of everyone who really churns, this share gets flagged.")
c3.metric("Customers flagged", f"{stats['n_flagged']:,.0f}",
          help="How many of your customers this cutoff would put in the campaign.")
st.caption(
    f"At a {threshold:.0%} cutoff you'd contact "
    f"{stats['n_flagged']:,.0f} of {stats['n_total']:,.0f} customers. Roughly "
    f"{1 - stats['precision']:.0%} of that spend goes to customers who would have "
    f"stayed anyway, and {1 - stats['recall']:.0%} of real churners slip through."
)
st.divider()

if st.button("Score customer"):
    customer = pd.DataFrame(
        [[recency, frequency, monetary]],
        columns=['Recency', 'Frequency', 'Monetary']
    )

    # churn model takes RAW rfm
    churn_prob = churn_model.predict_proba(customer)[0, 1]

    # segment needs the SAME prep as Phase 2: log -> scale -> cluster
    customer_scaled = scaler.transform(np.log1p(customer))
    cluster = kmeans.predict(customer_scaled)[0]
    segment = segment_names[cluster]

    st.metric("Churn probability (next 180 days)", f"{churn_prob:.0%}")

    if churn_prob >= threshold:
        st.error(f"AT RISK — flagged (above your {threshold:.0%} cutoff)")
    else:
        st.success(f"Retained — not flagged (below your {threshold:.0%} cutoff)")

    st.subheader(f"Segment: {segment}")
    st.info(strategies[segment])