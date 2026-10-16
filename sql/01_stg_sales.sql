-- stg_sales: one row per store-product-day from train and eval, typed and tagged by source.
-- Grain: (store_id, product_id, dt).
SELECT
  'train' AS source,
  city_id,
  store_id,
  management_group_id,
  first_category_id,
  second_category_id,
  third_category_id,
  product_id,
  CAST(dt AS DATE) AS dt,
  sale_amount,
  hours_sale,
  stock_hour6_22_cnt,
  hours_stock_status,
  discount,
  holiday_flag,
  activity_flag,
  precpt,
  avg_temperature,
  avg_humidity,
  avg_wind_level
FROM raw_train
UNION ALL
SELECT
  'eval' AS source,
  city_id,
  store_id,
  management_group_id,
  first_category_id,
  second_category_id,
  third_category_id,
  product_id,
  CAST(dt AS DATE) AS dt,
  sale_amount,
  hours_sale,
  stock_hour6_22_cnt,
  hours_stock_status,
  discount,
  holiday_flag,
  activity_flag,
  precpt,
  avg_temperature,
  avg_humidity,
  avg_wind_level
FROM raw_eval
