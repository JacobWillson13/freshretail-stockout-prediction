-- int_hourly: the only model that unnests the 24-element hourly arrays.
-- Grain: (store_id, product_id, dt, hour), hour = 0..23 (array position, 0-based).
-- Coding verified in the data checks: stock_status = 1 means out of stock, and
-- stock_hour6_22_cnt equals the count of 1s over hours 6..21 (06:00 to 22:00).
--
-- BigQuery equivalent of the unnest:
--   FROM stg_sales AS s
--   CROSS JOIN UNNEST(s.hours_stock_status) AS stock_status WITH OFFSET AS hour
--   -- and s.hours_sale[OFFSET(hour)] AS sale
-- DuckDB arrays are 1-based, hence hour + 1 below.
SELECT
  s.store_id,
  s.product_id,
  s.first_category_id,
  s.dt,
  h.hour,
  s.hours_stock_status[h.hour + 1] AS stock_status,
  s.hours_sale[h.hour + 1] AS sale,
  (h.hour BETWEEN 6 AND 21) AS in_window
FROM stg_sales AS s
CROSS JOIN range(0, 24) AS h(hour)
