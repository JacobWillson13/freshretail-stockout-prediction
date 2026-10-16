-- int_target: next-day stockout label.
-- Grain: (store_id, product_id, dt) where dt is the prediction day D (cutoff = end of D).
-- label = 1 when D+1 has more than 2 out-of-stock hours in the 06:00 to 22:00 window.
-- Overnight out-of-stock hours of D+1 are kept for reporting only, never in the label.
WITH nxt AS (
  SELECT
    store_id,
    product_id,
    dt,
    LEAD(dt) OVER w AS next_dt,
    LEAD(oos_hours) OVER w AS next_oos_hours,
    LEAD(oos_hours_overnight) OVER w AS next_oos_hours_overnight
  FROM int_daily
  WINDOW w AS (PARTITION BY store_id, product_id ORDER BY dt)
)
SELECT
  store_id,
  product_id,
  dt,
  next_dt AS label_dt,
  next_oos_hours,
  next_oos_hours_overnight,
  CASE WHEN next_oos_hours > 2 THEN 1 ELSE 0 END AS label
FROM nxt
WHERE next_dt = CAST(DATE_ADD(dt, INTERVAL 1 DAY) AS DATE)
