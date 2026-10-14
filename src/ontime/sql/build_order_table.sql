-- One row per order (ALL 99,441 orders; population filters are applied afterwards in Python so
-- every step of the funnel can be counted and logged).
-- Source tables are registered by build_orders.py as: orders, order_items, products, sellers,
-- customers, geolocation, payments, category_translation.
-- Placeholders ${lat_min} etc. are filled from configs/config.yaml.
-- NOTE: order_delivered_carrier_date and review data are deliberately NOT selected (forbidden
-- as features; keeping them out of the table makes accidental use impossible).

-- 1. Geolocation: drop points outside Brazil, then ONE centroid per zip prefix (median is robust
--    to the remaining GPS noise).
CREATE OR REPLACE TEMP TABLE geo_points AS
SELECT geolocation_zip_code_prefix AS zip, geolocation_state AS state,
       geolocation_lat AS lat, geolocation_lng AS lng
FROM geolocation
WHERE geolocation_lat BETWEEN ${lat_min} AND ${lat_max}
  AND geolocation_lng BETWEEN ${lng_min} AND ${lng_max};

CREATE OR REPLACE TEMP TABLE zip_centroid AS
SELECT zip, median(lat) AS lat, median(lng) AS lng
FROM geo_points GROUP BY zip;

-- Fallback for prefixes absent from geolocation: centroid of the whole state.
CREATE OR REPLACE TEMP TABLE state_centroid AS
SELECT state, median(lat) AS lat, median(lng) AS lng
FROM geo_points GROUP BY state;

CREATE OR REPLACE TEMP TABLE seller_geo AS
SELECT s.seller_id, s.seller_state,
       COALESCE(z.lat, st.lat) AS seller_lat,
       COALESCE(z.lng, st.lng) AS seller_lng,
       (z.zip IS NULL) AS seller_geo_imputed
FROM sellers s
LEFT JOIN zip_centroid z ON z.zip = s.seller_zip_code_prefix
LEFT JOIN state_centroid st ON st.state = s.seller_state;

CREATE OR REPLACE TEMP TABLE customer_geo AS
SELECT c.customer_id, c.customer_unique_id, c.customer_state,
       COALESCE(z.lat, st.lat) AS customer_lat,
       COALESCE(z.lng, st.lng) AS customer_lng,
       (z.zip IS NULL) AS customer_geo_imputed
FROM customers c
LEFT JOIN zip_centroid z ON z.zip = c.customer_zip_code_prefix
LEFT JOIN state_centroid st ON st.state = c.customer_state;

-- 2. Items joined to product attributes + English category.
CREATE OR REPLACE TEMP TABLE items_enriched AS
SELECT i.order_id, i.order_item_id, i.seller_id, i.price, i.freight_value, i.shipping_limit_date,
       p.product_weight_g,
       p.product_length_cm * p.product_height_cm * p.product_width_cm AS volume_cm3,
       p.product_photos_qty,
       COALESCE(t.product_category_name_english, p.product_category_name) AS category_en
FROM order_items i
LEFT JOIN products p USING (product_id)
LEFT JOIN category_translation t ON t.product_category_name = p.product_category_name;

-- 3. Order-level item aggregates.
CREATE OR REPLACE TEMP TABLE item_agg AS
SELECT order_id,
       count(*)                          AS n_items,
       count(DISTINCT seller_id)         AS n_distinct_sellers,
       sum(price)                        AS total_price,
       sum(freight_value)                AS total_freight,
       sum(product_weight_g)             AS total_weight_g,
       max(product_weight_g)             AS max_item_weight_g,
       sum(volume_cm3)                   AS total_volume_cm3,
       avg(product_photos_qty)           AS product_photos_qty_mean,
       max(shipping_limit_date)          AS max_shipping_limit_date
FROM items_enriched GROUP BY order_id;

-- 4. Primary seller = seller with the highest item value in the order; ties broken by seller_id
--    so the choice is deterministic.
CREATE OR REPLACE TEMP TABLE primary_seller AS
SELECT order_id, seller_id AS primary_seller_id
FROM (
    SELECT order_id, seller_id,
           row_number() OVER (PARTITION BY order_id ORDER BY sum(price) DESC, seller_id) AS rn
    FROM items_enriched GROUP BY order_id, seller_id
) WHERE rn = 1;

-- 5. Category of the single most expensive item (ties: lowest order_item_id).
CREATE OR REPLACE TEMP TABLE top_item_category AS
SELECT order_id, category_en AS top_item_category
FROM (
    SELECT order_id, category_en,
           row_number() OVER (PARTITION BY order_id ORDER BY price DESC, order_item_id) AS rn
    FROM items_enriched
) WHERE rn = 1;

-- 6. Payments: total value, max installments, dominant payment type by value.
CREATE OR REPLACE TEMP TABLE payment_agg AS
SELECT a.order_id, a.total_payment_value, a.payment_installments_max, d.payment_type_dominant
FROM (
    SELECT order_id, sum(payment_value) AS total_payment_value,
           max(payment_installments) AS payment_installments_max
    FROM payments GROUP BY order_id
) a
JOIN (
    SELECT order_id, payment_type AS payment_type_dominant
    FROM (
        SELECT order_id, payment_type,
               row_number() OVER (PARTITION BY order_id
                                  ORDER BY sum(payment_value) DESC, payment_type) AS rn
        FROM payments GROUP BY order_id, payment_type
    ) WHERE rn = 1
) d USING (order_id);

-- 7. Final order table (all orders). Target is defined on DATES, not timestamps, because the
--    estimate is a date: late = delivered date strictly after estimated date.
CREATE OR REPLACE TABLE orders_all AS
SELECT o.order_id, o.customer_id, c.customer_unique_id, o.order_status,
       o.order_purchase_timestamp, o.order_approved_at,
       o.order_delivered_customer_date, o.order_estimated_delivery_date,
       CASE WHEN o.order_delivered_customer_date IS NULL THEN NULL
            WHEN CAST(o.order_delivered_customer_date AS DATE)
                 > CAST(o.order_estimated_delivery_date AS DATE) THEN 1 ELSE 0 END AS is_late,
       date_diff('second', o.order_approved_at, o.order_delivered_customer_date) / 86400.0
           AS delivery_days,
       ps.primary_seller_id, sg.seller_state, sg.seller_lat, sg.seller_lng, sg.seller_geo_imputed,
       c.customer_state, c.customer_lat, c.customer_lng, c.customer_geo_imputed,
       ia.n_items, ia.n_distinct_sellers, ia.total_price, ia.total_freight, ia.total_weight_g,
       ia.max_item_weight_g, ia.total_volume_cm3, ia.product_photos_qty_mean,
       ia.max_shipping_limit_date, tc.top_item_category,
       pa.total_payment_value, pa.payment_installments_max, pa.payment_type_dominant
FROM orders o
LEFT JOIN customer_geo c USING (customer_id)
LEFT JOIN item_agg ia USING (order_id)
LEFT JOIN primary_seller ps USING (order_id)
LEFT JOIN seller_geo sg ON sg.seller_id = ps.primary_seller_id
LEFT JOIN top_item_category tc USING (order_id)
LEFT JOIN payment_agg pa USING (order_id);
