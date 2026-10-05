-- Segment bookings by lead time and compare cancellation rates
-- with and without an automated reminder. SQLite-compatible.
WITH banded AS (
    SELECT
        booking_id,
        reminder_sent,
        cancelled,
        CASE
            WHEN lead_time_days <= 14  THEN '1. 0-14 days'
            WHEN lead_time_days <= 30  THEN '2. 15-30 days'
            WHEN lead_time_days <= 60  THEN '3. 31-60 days'
            ELSE                            '4. 60+ days'
        END AS lead_band
    FROM bookings
)
SELECT
    lead_band,
    reminder_sent,
    COUNT(*)                                      AS bookings,
    ROUND(100.0 * AVG(cancelled), 2)              AS cancellation_rate_pct,
    ROUND(100.0 * COUNT(*) /
          SUM(COUNT(*)) OVER (PARTITION BY lead_band), 1) AS share_of_band_pct
FROM banded
GROUP BY lead_band, reminder_sent
ORDER BY lead_band, reminder_sent;
