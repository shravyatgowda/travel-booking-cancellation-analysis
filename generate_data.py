"""Generate a synthetic travel booking dataset (~45,000 rows).

The data is SYNTHETIC. It is built so that long lead times and missing
reminders raise cancellation risk, which lets the analysis pipeline be
demonstrated end to end without using any client data.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 45_000

lead = np.clip(rng.gamma(shape=2.2, scale=28, size=N), 1, 365).astype(int)
booking_date = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 540, N), unit="D")
travel_date = booking_date + pd.to_timedelta(lead, unit="D")

df = pd.DataFrame({
    "booking_id": np.arange(100001, 100001 + N),
    "customer_age": rng.integers(18, 75, N).astype(float),
    "country": rng.choice(["India", "UK", "Germany", "UAE", "USA"], N, p=[.4, .2, .15, .15, .1]),
    "booking_date": booking_date.date,
    "travel_date": travel_date.date,
    "lead_time_days": lead,
    "payment_method": rng.choice(["Card", "UPI", "NetBanking", "Wallet"], N, p=[.45, .3, .15, .1]),
    "reminder_sent": rng.choice([0, 1], N, p=[.45, .55]),
})

logit = -2.3 + 0.012 * df.lead_time_days - 0.9 * df.reminder_sent
logit += 0.012 * np.maximum(df.lead_time_days - 60, 0) * (1 - df.reminder_sent)
logit += np.where(df.payment_method == "Wallet", 0.3, 0)
df["cancelled"] = (rng.random(N) < 1 / (1 + np.exp(-logit))).astype(int)

# Add realistic mess for the cleaning step
df.loc[rng.choice(N, 900, replace=False), "customer_age"] = np.nan
df.loc[rng.choice(N, 600, replace=False), "payment_method"] = None
df = pd.concat([df, df.sample(250, random_state=1)], ignore_index=True)  # duplicates

df.to_csv("data/bookings_raw.csv", index=False)
print(f"Wrote data/bookings_raw.csv with {len(df):,} rows")
