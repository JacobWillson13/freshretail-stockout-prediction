-- features_entity: stockout context from related series on day D and the trailing 7 days.
-- Stockout rate = share of series with more than 2 out-of-stock window hours.
--   product: the same product in all OTHER stores (leave-one-out: the row's own store is
--            excluded, so the rate does not contain the row's own stockout of day D);
--   store: all products of the store (own product included);
--   store x second category: the store's OTHER products in the same second-level category
--            (leave-one-out: the row's own product is excluded).
-- Leave-one-out rate = (group sum - own value) / (group count - 1), NULL when the row is the
-- only member of its group. The 7-day means average the row's own leave-one-out daily rates
-- over D-6 .. D (AVG skips NULL days).
-- Grain: (store_id, product_id, dt). Every series is observed every day (data checks), so
-- ROWS frames equal day frames.
WITH base AS (
  SELECT
    store_id,
    product_id,
    second_category_id,
    third_category_id,
    dt,
    CASE WHEN oos_hours > 2 THEN 1.0 ELSE 0.0 END AS oos_day
  FROM int_daily
),
groups AS (
  SELECT
    *,
    SUM(oos_day) OVER (PARTITION BY product_id, dt) AS prod_sum,
    COUNT(*) OVER (PARTITION BY product_id, dt) AS prod_n,
    AVG(oos_day) OVER (PARTITION BY store_id, dt) AS store_oos_rate,
    SUM(oos_day) OVER (PARTITION BY store_id, second_category_id, dt) AS cat2_sum,
    COUNT(*) OVER (PARTITION BY store_id, second_category_id, dt) AS cat2_n
  FROM base
),
loo AS (
  SELECT
    *,
    -- BigQuery: SAFE_DIVIDE(prod_sum - oos_day, prod_n - 1)
    (prod_sum - oos_day) / NULLIF(prod_n - 1, 0) AS prod_oos_rate,
    (cat2_sum - oos_day) / NULLIF(cat2_n - 1, 0) AS store_cat2_oos_rate
  FROM groups
),
store_daily AS (
  -- one row per store and day for the store's own 7-day mean
  SELECT DISTINCT store_id, dt, store_oos_rate FROM loo
),
store_feat AS (
  SELECT
    *,
    AVG(store_oos_rate) OVER (
      PARTITION BY store_id ORDER BY dt ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS store_oos_rate_mean_7d
  FROM store_daily
)
SELECT
  l.store_id,
  l.product_id,
  l.dt,
  l.second_category_id,
  l.third_category_id,
  l.prod_oos_rate,
  AVG(l.prod_oos_rate) OVER w7 AS prod_oos_rate_mean_7d,
  l.store_oos_rate,
  s.store_oos_rate_mean_7d,
  l.store_cat2_oos_rate,
  AVG(l.store_cat2_oos_rate) OVER w7 AS store_cat2_oos_rate_mean_7d
FROM loo AS l
JOIN store_feat AS s
  ON l.store_id = s.store_id AND l.dt = s.dt
WINDOW w7 AS (
  PARTITION BY l.store_id, l.product_id ORDER BY l.dt ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
)
