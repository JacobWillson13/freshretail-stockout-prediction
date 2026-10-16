-- features_history: long-memory statistics per store-product, expanding from the first
-- observed day up to and including day D. No days-observed count: every series starts on the
-- same day, so it would act as a date index with test values outside the training range.
-- Grain: (store_id, product_id, dt).
SELECT
  store_id,
  product_id,
  dt,
  AVG(oos_hours) OVER w AS hist_oos_hours_mean,
  AVG(CASE WHEN oos_hours > 2 THEN 1.0 ELSE 0.0 END) OVER w AS hist_oos_day_share
FROM int_daily
WINDOW w AS (
  PARTITION BY store_id, product_id ORDER BY dt
  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
)
