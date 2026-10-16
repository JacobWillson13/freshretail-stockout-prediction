-- model_dataset: one labeled prediction row per store-product-day with all feature sets.
-- Grain: (store_id, product_id, dt). Split is assigned by the LABEL date (D+1):
--   test  = label dates inside eval,
--   valid = label dates in the last 14 days of train,
--   train = earlier label dates.
-- Features come from features_all, which already drops the 13-day warm-up.
WITH bounds AS (
  SELECT
    MAX(CASE WHEN source = 'train' THEN dt END) AS train_end,
    MIN(CASE WHEN source = 'eval' THEN dt END) AS eval_start
  FROM stg_sales
)
SELECT
  t.store_id,
  t.product_id,
  t.dt,
  t.label_dt,
  t.label,
  t.next_oos_hours,
  t.next_oos_hours_overnight,
  CASE
    WHEN t.label_dt >= b.eval_start THEN 'test'
    -- BigQuery: DATE_SUB(b.train_end, INTERVAL 14 DAY)
    WHEN t.label_dt > CAST(b.train_end - INTERVAL 14 DAY AS DATE) THEN 'valid'
    ELSE 'train'
  END AS split,
  f.* EXCLUDE (store_id, product_id, dt)
  -- BigQuery: SELECT * EXCEPT (store_id, product_id, dt)
FROM int_target AS t
CROSS JOIN bounds AS b
JOIN features_all AS f
  ON t.store_id = f.store_id AND t.product_id = f.product_id AND t.dt = f.dt
