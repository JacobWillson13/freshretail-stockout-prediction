-- int_daily_stock: daily stock and sales metrics derived from the hourly grain.
-- Grain: (store_id, product_id, dt). Operating window = hours 6..21 (16 hours).
WITH runs AS (
  -- gaps and islands: consecutive out-of-stock hours inside the window share a grp
  SELECT
    store_id,
    product_id,
    dt,
    hour - ROW_NUMBER() OVER (
      PARTITION BY store_id, product_id, dt ORDER BY hour
    ) AS grp
  FROM int_hourly
  WHERE in_window AND stock_status = 1
),
longest AS (
  SELECT store_id, product_id, dt, MAX(run_len) AS longest_oos_run
  FROM (
    SELECT store_id, product_id, dt, grp, COUNT(*) AS run_len
    FROM runs
    GROUP BY store_id, product_id, dt, grp
  ) AS r
  GROUP BY store_id, product_id, dt
),
agg AS (
  SELECT
    store_id,
    product_id,
    dt,
    SUM(CASE WHEN in_window AND stock_status = 1 THEN 1 ELSE 0 END) AS oos_hours,
    SUM(CASE WHEN in_window AND stock_status = 0 THEN 1 ELSE 0 END) AS available_hours,
    SUM(CASE WHEN NOT in_window AND stock_status = 1 THEN 1 ELSE 0 END) AS oos_hours_overnight,
    SUM(CASE WHEN in_window AND stock_status = 0 THEN sale ELSE 0 END) AS in_stock_sales,
    SUM(CASE WHEN in_window THEN sale ELSE 0 END) AS window_sales
  FROM int_hourly
  GROUP BY store_id, product_id, dt
)
SELECT
  a.store_id,
  a.product_id,
  a.dt,
  a.oos_hours,
  a.oos_hours / 16.0 AS oos_fraction,
  a.available_hours,
  a.oos_hours_overnight,
  COALESCE(l.longest_oos_run, 0) AS longest_oos_run,
  a.in_stock_sales,
  a.window_sales,
  -- NULL when no hours were available (BigQuery: SAFE_DIVIDE)
  a.in_stock_sales / NULLIF(a.available_hours, 0) AS sales_per_available_hour
FROM agg AS a
LEFT JOIN longest AS l
  ON a.store_id = l.store_id AND a.product_id = l.product_id AND a.dt = l.dt
