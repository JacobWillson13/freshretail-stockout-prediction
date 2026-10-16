-- features_daily: store-product level signals known at the end of day D.
-- Grain: (store_id, product_id, dt). All windows end at the current row (day D).
-- Series are contiguous by day (checked in the data checks), so ROWS frames equal day frames.
WITH base AS (
  SELECT
    *,
    CASE WHEN oos_hours > 2 THEN 1 ELSE 0 END AS oos_day
  FROM int_daily
)
SELECT
  store_id,
  product_id,
  dt,
  -- today
  oos_hours,
  oos_fraction,
  longest_oos_run,
  in_stock_sales,
  sale_amount,
  sales_per_available_hour,
  discount,
  activity_flag,
  precpt,
  avg_temperature,
  avg_humidity,
  avg_wind_level,
  -- calendar
  day_of_week,
  holiday_flag,
  holiday_tomorrow,
  -- lags
  LAG(oos_hours, 1) OVER w AS oos_hours_lag1,
  LAG(oos_hours, 2) OVER w AS oos_hours_lag2,
  LAG(oos_hours, 7) OVER w AS oos_hours_lag7,
  LAG(sale_amount, 1) OVER w AS sale_amount_lag1,
  LAG(sale_amount, 7) OVER w AS sale_amount_lag7,
  LAG(discount, 1) OVER w AS discount_lag1,
  -- 7-day rolling (D-6 .. D)
  AVG(oos_hours) OVER w7 AS oos_hours_mean_7d,
  SUM(oos_day) OVER w7 AS oos_days_7d,
  MAX(longest_oos_run) OVER w7 AS longest_oos_run_max_7d,
  AVG(sale_amount) OVER w7 AS sales_mean_7d,
  STDDEV_SAMP(sale_amount) OVER w7 AS sales_std_7d,
  AVG(discount) OVER w7 AS discount_mean_7d,
  SUM(activity_flag) OVER w7 AS activity_days_7d,
  -- 14-day rolling (D-13 .. D)
  AVG(oos_hours) OVER w14 AS oos_hours_mean_14d,
  SUM(oos_day) OVER w14 AS oos_days_14d,
  AVG(sale_amount) OVER w14 AS sales_mean_14d,
  STDDEV_SAMP(sale_amount) OVER w14 AS sales_std_14d,
  -- trend: recent week versus the two-week mean (BigQuery: SAFE_DIVIDE)
  AVG(sale_amount) OVER w7 / NULLIF(AVG(sale_amount) OVER w14, 0) AS sales_trend_7v14,
  COUNT(*) OVER w14 AS history_days
FROM base
WINDOW
  w AS (PARTITION BY store_id, product_id ORDER BY dt),
  w7 AS (PARTITION BY store_id, product_id ORDER BY dt ROWS BETWEEN 6 PRECEDING AND CURRENT ROW),
  w14 AS (PARTITION BY store_id, product_id ORDER BY dt ROWS BETWEEN 13 PRECEDING AND CURRENT ROW)
