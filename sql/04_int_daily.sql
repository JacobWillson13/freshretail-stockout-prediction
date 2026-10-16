-- int_daily: daily attributes, stock metrics and calendar fields on one row.
-- Grain: (store_id, product_id, dt).
SELECT
  s.source,
  s.city_id,
  s.store_id,
  s.product_id,
  s.management_group_id,
  s.first_category_id,
  s.second_category_id,
  s.third_category_id,
  s.dt,
  -- BigQuery: EXTRACT(DAYOFWEEK FROM dt) is 1 = Sunday .. 7 = Saturday; DuckDB dayofweek is 0 = Sunday
  EXTRACT(DAYOFWEEK FROM s.dt) AS day_of_week,
  s.sale_amount,
  s.stock_hour6_22_cnt,
  s.discount,
  s.holiday_flag,
  s.activity_flag,
  s.precpt,
  s.avg_temperature,
  s.avg_humidity,
  s.avg_wind_level,
  d.oos_hours,
  d.oos_fraction,
  d.available_hours,
  d.oos_hours_overnight,
  d.longest_oos_run,
  d.in_stock_sales,
  d.window_sales,
  d.sales_per_available_hour,
  -- tomorrow's holiday flag is a calendar fact, known in advance
  CASE
    WHEN LEAD(s.dt) OVER w = CAST(DATE_ADD(s.dt, INTERVAL 1 DAY) AS DATE)
      THEN LEAD(s.holiday_flag) OVER w
  END AS holiday_tomorrow
FROM stg_sales AS s
JOIN int_daily_stock AS d
  ON s.store_id = d.store_id AND s.product_id = d.product_id AND s.dt = d.dt
WINDOW w AS (PARTITION BY s.store_id, s.product_id ORDER BY s.dt)
