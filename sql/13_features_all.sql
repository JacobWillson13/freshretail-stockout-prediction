-- features_all: every feature set joined on (store_id, product_id, dt), for every day with
-- 14 days of history, whether or not the next day (the label) is observed yet.
-- model_dataset adds the label to these rows; src/predict.py scores this table directly,
-- so the last observed day can be scored before its label exists.
-- Rows need 14 days of history (warm-up), so the first 13 days of each series are dropped.
WITH bounds AS (
  SELECT MIN(dt) AS first_dt FROM stg_sales
)
SELECT
  f.*,
  w.* EXCLUDE (store_id, product_id, dt),
  c.* EXCLUDE (store_id, product_id, dt),
  i.* EXCLUDE (store_id, product_id, dt),
  h.* EXCLUDE (store_id, product_id, dt),
  e.* EXCLUDE (store_id, product_id, dt),
  m.* EXCLUDE (store_id, product_id, dt)
  -- BigQuery: SELECT * EXCEPT (store_id, product_id, dt)
FROM features_daily AS f
CROSS JOIN bounds AS b
JOIN features_weekly AS w
  ON f.store_id = w.store_id AND f.product_id = w.product_id AND f.dt = w.dt
JOIN features_category AS c
  ON f.store_id = c.store_id AND f.product_id = c.product_id AND f.dt = c.dt
JOIN features_intraday AS i
  ON f.store_id = i.store_id AND f.product_id = i.product_id AND f.dt = i.dt
JOIN features_history AS h
  ON f.store_id = h.store_id AND f.product_id = h.product_id AND f.dt = h.dt
JOIN features_entity AS e
  ON f.store_id = e.store_id AND f.product_id = e.product_id AND f.dt = e.dt
JOIN features_demand AS m
  ON f.store_id = m.store_id AND f.product_id = m.product_id AND f.dt = m.dt
WHERE f.dt >= CAST(DATE_ADD(b.first_dt, INTERVAL 13 DAY) AS DATE)
