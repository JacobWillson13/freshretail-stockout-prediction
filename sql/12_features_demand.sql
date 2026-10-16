-- features_demand: censored-demand estimates for day D.
-- Sales are only observed while the product is in stock, so daily sales understate demand
-- on stockout days. demand_est scales the in-stock sales rate to the 16-hour window;
-- lost_sales_est is the part of that estimate not realized as window sales.
-- Grain: (store_id, product_id, dt).
WITH est AS (
  SELECT
    store_id,
    product_id,
    dt,
    -- NULL when no window hour was available (sales_per_available_hour is NULL)
    sales_per_available_hour * 16 AS demand_est,
    window_sales
  FROM int_daily
),
lost AS (
  SELECT
    *,
    -- floored at 0 and NULL when demand_est is NULL. A CASE instead of GREATEST because
    -- DuckDB GREATEST skips NULL arguments while BigQuery GREATEST returns NULL.
    CASE
      WHEN demand_est - window_sales < 0 THEN 0.0
      ELSE demand_est - window_sales
    END AS lost_sales_est
  FROM est
)
SELECT
  store_id,
  product_id,
  dt,
  demand_est,
  lost_sales_est,
  -- 7-day rolling (D-6 .. D); AVG skips NULL days
  AVG(demand_est) OVER w7 AS demand_est_mean_7d,
  AVG(lost_sales_est) OVER w7 AS lost_sales_est_mean_7d
FROM lost
WINDOW w7 AS (PARTITION BY store_id, product_id ORDER BY dt ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)
