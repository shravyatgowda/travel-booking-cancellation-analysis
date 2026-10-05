"""Clean the booking data, segment by lead time, and model cancellation risk."""
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

OUT = Path("output")
OUT.mkdir(exist_ok=True)

# 1. Clean ---------------------------------------------------------------
df = pd.read_csv("data/bookings_raw.csv", parse_dates=["booking_date", "travel_date"])
raw_rows = len(df)
df = df.drop_duplicates(subset="booking_id")
df["customer_age"] = df["customer_age"].fillna(df["customer_age"].median())
df["payment_method"] = df["payment_method"].fillna("Unknown")
df = df[df["lead_time_days"] >= 0]
print(f"Cleaned: {raw_rows:,} -> {len(df):,} rows")

# 2. SQL segmentation (window functions) --------------------------------
con = sqlite3.connect("data/bookings.db")
df.to_sql("bookings", con, if_exists="replace", index=False)
seg = pd.read_sql(Path("sql/lead_time_segments.sql").read_text(), con)
seg.to_csv(OUT / "segments_for_powerbi.csv", index=False)
print("\nCancellation rate by lead-time band and reminder:\n", seg.to_string(index=False))

# 3. Headline comparison: 60+ days, reminder vs none --------------------
long = df[df.lead_time_days > 60]
rate = long.groupby("reminder_sent")["cancelled"].mean()
uplift = rate[0] / rate[1] - 1
print(f"\n60+ day bookings: {rate[0]:.1%} cancel without a reminder vs "
      f"{rate[1]:.1%} with one ({uplift:+.0%} relative difference)")

# 4. Logistic regression: probability of cancellation -------------------
X = pd.get_dummies(df[["lead_time_days", "reminder_sent", "customer_age", "payment_method"]],
                   drop_first=True).astype(float)
y = df["cancelled"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)
model = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
print(f"Logistic regression ROC-AUC: {roc_auc_score(yte, model.predict_proba(Xte)[:, 1]):.3f}")
df["cancel_probability"] = model.predict_proba(X)[:, 1]
df.to_csv(OUT / "bookings_scored_for_powerbi.csv", index=False)

# 5. Chart ---------------------------------------------------------------
pivot = seg.pivot(index="lead_band", columns="reminder_sent", values="cancellation_rate_pct")
pivot.columns = ["No reminder", "Reminder sent"]
ax = pivot.plot(kind="bar", figsize=(8, 4.5), rot=0)
ax.set_ylabel("Cancellation rate (%)")
ax.set_xlabel("Lead time band")
ax.set_title("Cancellation rate by lead time and reminder (synthetic data)")
plt.tight_layout()
Path("images").mkdir(exist_ok=True)
plt.savefig("images/cancellation_by_lead_time.png", dpi=150)
print("\nSaved images/cancellation_by_lead_time.png")
