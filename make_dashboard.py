"""Create a static summary dashboard image from the analysis outputs.

This is a Python (matplotlib) visual, NOT a Power BI report. Run analysis.py first.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("output/bookings_scored_for_powerbi.csv", parse_dates=["booking_date"])
Path("images").mkdir(exist_ok=True)

fig = plt.figure(figsize=(13, 8))
fig.suptitle("Travel Booking Cancellations: Summary Dashboard (synthetic data, Python-generated)",
             fontsize=14, fontweight="bold")
gs = fig.add_gridspec(3, 3, height_ratios=[0.6, 1.5, 1.5], hspace=0.7, wspace=0.35)

kpis = [("Bookings", f"{len(df):,}"),
        ("Cancellation rate", f"{df.cancelled.mean():.1%}"),
        ("Avg lead time", f"{df.lead_time_days.mean():.0f} days")]
for i, (label, val) in enumerate(kpis):
    ax = fig.add_subplot(gs[0, i]); ax.axis("off")
    ax.text(.5, .62, val, ha="center", fontsize=24, fontweight="bold", color="#1f4e79")
    ax.text(.5, .12, label, ha="center", fontsize=11, color="#555")

bands = pd.cut(df.lead_time_days, [-1, 14, 30, 60, 400], labels=["0-14", "15-30", "31-60", "60+"])
t = df.groupby([bands, "reminder_sent"], observed=True).cancelled.mean().unstack() * 100
ax = fig.add_subplot(gs[1, :2])
t.rename(columns={0: "No reminder", 1: "Reminder sent"}).plot(kind="bar", ax=ax, rot=0, color=["#c0504d", "#4f81bd"])
ax.set_title("Cancellation rate by lead time (days)"); ax.set_ylabel("%"); ax.set_xlabel("")

ax = fig.add_subplot(gs[1, 2])
(df.groupby("payment_method").cancelled.mean().sort_values() * 100).plot(kind="barh", ax=ax, color="#4f81bd")
ax.set_title("Cancellation rate by payment method"); ax.set_xlabel("%")

ax = fig.add_subplot(gs[2, :2])
m = df.set_index("booking_date").cancelled.resample("ME").mean() * 100
m.plot(ax=ax, marker="o", color="#1f4e79")
ax.set_title("Monthly cancellation rate"); ax.set_ylabel("%"); ax.set_xlabel("")

ax = fig.add_subplot(gs[2, 2])
ax.hist(df.cancel_probability, bins=30, color="#9bbb59")
ax.set_title("Predicted cancellation probability"); ax.set_xlabel("probability")

plt.savefig("images/dashboard_summary.png", dpi=130, bbox_inches="tight")
print("Saved images/dashboard_summary.png")
