-- features_category: store x first_category signals for day D.
-- Grain: (store_id, product_id, dt), joined from the (store_id, first_category_id, dt) aggregate.
WITH cat_daily AS (
  SELECT
    store_id,
    first_category_id,
    dt,
    COUNT(*) AS cat_n_products,
    SUM(oos_hours) / (16.0 * COUNT(*)) AS cat_oos_rate,
    AVG(CASE WHEN oos_hours > 2 THEN 1.0 ELSE 0.0 END) AS cat_share_affected,
    SUM(sale_amount) AS cat_sales
  FROM int_daily
  GROUP BY store_id, first_category_id, dt
),
cat_feat AS (
  SELECT
    *,
    LAG(cat_oos_rate, 1) OVER w AS cat_oos_rate_lag1,
    AVG(cat_oos_rate) OVER w7 AS cat_oos_rate_mean_7d,
    AVG(cat_share_affected) OVER w7 AS cat_share_affected_mean_7d,
    AVG(cat_sales) OVER w7 AS cat_sales_mean_7d,
    -- trend: last 7 days versus the 7 days before (BigQuery: SAFE_DIVIDE)
    AVG(cat_sales) OVER w7 / NULLIF(AVG(cat_sales) OVER wprev7, 0) AS cat_sales_trend
  FROM cat_daily
  WINDOW
    w AS (PARTITION BY store_id, first_category_id ORDER BY dt),
    w7 AS (PARTITION BY store_id, first_category_id ORDER BY dt ROWS BETWEEN 6 PRECEDING AND CURRENT ROW),
    wprev7 AS (PARTITION BY store_id, first_category_id ORDER BY dt ROWS BETWEEN 13 PRECEDING AND 7 PRECEDING)
)
SELECT
  d.store_id,
  d.product_id,
  d.dt,
  d.first_category_id,
  c.cat_n_products,
  c.cat_oos_rate,
  c.cat_share_affected,
  c.cat_sales,
  c.cat_oos_rate_lag1,
  c.cat_oos_rate_mean_7d,
  c.cat_share_affected_mean_7d,
  c.cat_sales_mean_7d,
  c.cat_sales_trend
FROM int_daily AS d
JOIN cat_feat AS c
  ON d.store_id = c.store_id AND d.first_category_id = c.first_category_id AND d.dt = c.dt
