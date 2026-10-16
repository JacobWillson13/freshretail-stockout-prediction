-- features_weekly: store-product totals for the last COMPLETED calendar week (Monday to Sunday).
-- Grain: (store_id, product_id, dt). For day D the last completed week is the latest
-- week whose Sunday is on or before D: week_start = monday_of(D + 1 day) - 7 days.
-- Incomplete weeks (fewer than 7 observed days, e.g. the first partial week) give NULL.
WITH weekly AS (
  SELECT
    store_id,
    product_id,
    -- BigQuery: DATE_TRUNC(dt, WEEK(MONDAY))
    CAST(DATE_TRUNC('week', dt) AS DATE) AS week_start,
    COUNT(*) AS days_observed,
    SUM(oos_hours) AS oos_hours_sum,
    SUM(CASE WHEN oos_hours > 2 THEN 1 ELSE 0 END) AS oos_days,
    SUM(CASE WHEN oos_hours > 0 THEN 1 ELSE 0 END) AS any_oos_days,
    SUM(sale_amount) AS sales_sum,
    STDDEV_SAMP(sale_amount) AS sales_std,
    AVG(discount) AS discount_mean,
    SUM(activity_flag) AS activity_days
  FROM int_daily
  GROUP BY 1, 2, 3
),
complete AS (
  SELECT
    *,
    oos_hours_sum - LAG(oos_hours_sum) OVER w AS oos_hours_change,
    oos_days - LAG(oos_days) OVER w AS oos_days_change,
    sales_sum / NULLIF(LAG(sales_sum) OVER w, 0) - 1 AS sales_change_pct
  FROM weekly
  WHERE days_observed = 7
  WINDOW w AS (PARTITION BY store_id, product_id ORDER BY week_start)
),
complete_checked AS (
  -- change vs the week before only when that week is the adjacent complete week
  SELECT
    c.*,
    LAG(week_start) OVER (PARTITION BY store_id, product_id ORDER BY week_start) AS prev_week_start
  FROM complete AS c
)
SELECT
  d.store_id,
  d.product_id,
  d.dt,
  w.week_start AS wk_week_start,
  w.oos_hours_sum AS wk_oos_hours_sum,
  w.oos_days AS wk_oos_days,
  w.any_oos_days AS wk_any_oos_days,
  w.sales_sum AS wk_sales_sum,
  w.sales_std AS wk_sales_std,
  w.sales_std / NULLIF(w.sales_sum / 7.0, 0) AS wk_sales_cv,
  w.discount_mean AS wk_discount_mean,
  w.activity_days AS wk_activity_days,
  CASE WHEN w.prev_week_start = CAST(w.week_start - INTERVAL 7 DAY AS DATE)
    THEN w.oos_hours_change END AS wk_oos_hours_change,
  CASE WHEN w.prev_week_start = CAST(w.week_start - INTERVAL 7 DAY AS DATE)
    THEN w.oos_days_change END AS wk_oos_days_change,
  CASE WHEN w.prev_week_start = CAST(w.week_start - INTERVAL 7 DAY AS DATE)
    THEN w.sales_change_pct END AS wk_sales_change_pct
FROM int_daily AS d
LEFT JOIN complete_checked AS w
  ON d.store_id = w.store_id
  AND d.product_id = w.product_id
  -- BigQuery: DATE_SUB(DATE_TRUNC(DATE_ADD(d.dt, INTERVAL 1 DAY), WEEK(MONDAY)), INTERVAL 7 DAY)
  AND w.week_start = CAST(DATE_TRUNC('week', DATE_ADD(d.dt, INTERVAL 1 DAY)) AS DATE) - INTERVAL 7 DAY
