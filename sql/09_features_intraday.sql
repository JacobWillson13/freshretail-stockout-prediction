-- features_intraday: timing of out-of-stock hours within day D, from the hourly grain.
-- Grain: (store_id, product_id, dt). Every hour of D (0..23) is known at the end of D,
-- including the late-night hours 22 and 23, which the target window (6..21) excludes.
-- Hour 21 (21:00 to 22:00) is the last operating hour.
WITH day_hours AS (
  SELECT
    store_id,
    product_id,
    dt,
    MAX(CASE WHEN hour = 21 THEN stock_status END) AS oos_at_close,
    SUM(CASE WHEN hour BETWEEN 19 AND 21 THEN stock_status ELSE 0 END) AS oos_hours_last3,
    SUM(CASE WHEN hour IN (22, 23) THEN stock_status ELSE 0 END) AS oos_hours_late_night,
    MAX(CASE WHEN hour = 23 THEN stock_status END) AS ended_day_oos,
    -- window hours only; NULL when the event never happens in the window
    MIN(CASE WHEN in_window AND stock_status = 1 THEN hour END) AS first_oos_hour,
    MAX(CASE WHEN in_window AND stock_status = 1 THEN hour END) AS last_oos_hour,
    MAX(CASE WHEN in_window AND stock_status = 0 THEN hour END) AS last_in_stock_hour,
    MIN(CASE WHEN in_window AND sale > 0 THEN hour END) AS first_sale_hour
  FROM int_hourly
  GROUP BY store_id, product_id, dt
),
flags AS (
  SELECT
    *,
    -- sell-out signal: an out-of-stock window hour after the first window hour with a sale
    CASE WHEN last_oos_hour > first_sale_hour THEN 1 ELSE 0 END AS oos_after_sale
  FROM day_hours
)
SELECT
  store_id,
  product_id,
  dt,
  oos_at_close,
  oos_hours_last3,
  oos_hours_late_night,
  first_oos_hour,
  last_in_stock_hour,
  oos_after_sale,
  ended_day_oos,
  LAG(oos_at_close, 1) OVER w AS oos_at_close_lag1,
  -- 7-day rolling (D-6 .. D); series are contiguous by day, so ROWS frames equal day frames
  AVG(oos_at_close) OVER (w ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS oos_at_close_mean_7d
FROM flags
WINDOW w AS (PARTITION BY store_id, product_id ORDER BY dt)
