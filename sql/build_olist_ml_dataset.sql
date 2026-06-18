WITH step1 AS (
    SELECT
        c.customer_id,
        c.customer_city,
        c.customer_state,

        o.C1 AS order_id,
        o.C3 AS order_status,
        o.C4 AS order_purchase_timestamp,
        o.C5 AS order_approved_at,
        o.C7 AS order_delivered_customer_date,
        o.C8 AS order_estimated_delivery_date,

        TIMESTAMPDIFF(DAY, o.C4, o.C5) AS approval_delay_days,
        TIMESTAMPDIFF(DAY, o.C4, o.C7) AS delivery_days,
        TIMESTAMPDIFF(DAY, o.C4, o.C8) AS estimated_delivery_days,
        TIMESTAMPDIFF(DAY, o.C8, o.C7) AS delay_vs_estimated_days,

        CASE
            WHEN o.C7 IS NULL THEN NULL
            WHEN o.C7 > o.C8 THEN 1
            ELSE 0
        END AS is_late_delivery,

        MONTH(o.C4) AS order_month,
        DAYOFWEEK(o.C4) AS order_day_of_week,
        HOUR(o.C4) AS order_hour

    FROM olist_orders_dataset o
    LEFT JOIN olist_customers_dataset c
        ON c.customer_id = o.C2
),

item_agg AS (
    SELECT
        items.order_id,

        COUNT(*) AS items_count,
        COUNT(DISTINCT items.product_id) AS unique_products_count,
        COUNT(DISTINCT items.seller_id) AS unique_sellers_count,

        SUM(items.price) AS total_price,
        SUM(items.freight_value) AS total_freight,
        SUM(items.price + items.freight_value) AS total_sum,

        AVG(items.price) AS avg_item_price,
        MAX(items.price) AS max_item_price,

        SUM(items.freight_value) / NULLIF(SUM(items.price), 0) AS freight_ratio

    FROM olist_order_items_dataset items
    GROUP BY items.order_id
),

payment_agg AS (
    SELECT
        order_id,

        COUNT(payment_sequential) AS payments_count,
        SUM(payment_value) AS total_payment_value,
        MAX(payment_installments) AS max_installments,

        CASE
            WHEN COUNT(payment_sequential) > 1 THEN 1
            ELSE 0
        END AS has_multiple_payments

    FROM olist_order_payments_dataset
    GROUP BY order_id
),

ranked_payments AS (
    SELECT
        order_id,
        payment_type,
        payment_value,
        payment_sequential,

        ROW_NUMBER() OVER (
            PARTITION BY order_id
            ORDER BY payment_value DESC, payment_sequential ASC
        ) AS rn

    FROM olist_order_payments_dataset
),

main_payment AS (
    SELECT
        order_id,
        payment_type AS main_payment_type
    FROM ranked_payments
    WHERE rn = 1
),

items_products AS (
    SELECT
        items.order_id,
        items.order_item_id,
        items.product_id,
        items.seller_id,
        items.price,

        COALESCE(trans.C2, prod.product_category_name) AS product_category_name,

        prod.product_weight_g,
        prod.product_length_cm,
        prod.product_height_cm,
        prod.product_width_cm,
        prod.product_photos_qty,
        prod.product_name_lenght,
        prod.product_description_lenght,

        prod.product_length_cm * prod.product_height_cm * prod.product_width_cm AS product_volume_cm3

    FROM olist_order_items_dataset items
    LEFT JOIN olist_products_dataset prod
        ON items.product_id = prod.product_id
    LEFT JOIN product_category_name_translation trans
        ON prod.product_category_name = trans.C1
),

ranked_products_categories AS (
    SELECT
        order_id,

        seller_id AS main_seller_id,
        product_category_name AS main_product_category,

        ROW_NUMBER() OVER (
            PARTITION BY order_id
            ORDER BY price DESC, order_item_id ASC
        ) AS rn

    FROM items_products
),

product_agg AS (
    SELECT
        order_id,

        COUNT(DISTINCT product_category_name) AS categories_count,

        SUM(product_weight_g) AS total_products_weight,
        SUM(product_volume_cm3) AS total_products_volume,

        AVG(product_weight_g) AS avg_product_weight,
        MAX(product_weight_g) AS max_product_weight,

        AVG(product_volume_cm3) AS avg_product_volume,
        MAX(product_volume_cm3) AS max_product_volume,

        AVG(product_weight_g / NULLIF(product_volume_cm3, 0)) AS avg_product_density,

        AVG(product_photos_qty) AS avg_photos_qty,
        MAX(product_photos_qty) AS max_photos_qty,

        AVG(product_name_lenght) AS avg_product_name_length,
        AVG(product_description_lenght) AS avg_product_description_length

    FROM items_products
    GROUP BY order_id
),

seller_agg AS (
    SELECT
        order_id,

        COUNT(DISTINCT seller_id) AS sellers_count_check,

        CASE
            WHEN COUNT(DISTINCT seller_id) > 1 THEN 1
            ELSE 0
        END AS has_multiple_sellers

    FROM olist_order_items_dataset
    GROUP BY order_id
),

main_seller AS (
    SELECT
        rpc.order_id,
        rpc.main_seller_id,

        seller.seller_city AS main_seller_city,
        seller.seller_state AS main_seller_state

    FROM ranked_products_categories rpc
    LEFT JOIN olist_sellers_dataset seller
        ON rpc.main_seller_id = seller.seller_id
    WHERE rpc.rn = 1
),

review_agg AS (
    SELECT
        order_id,

        AVG(review_score) AS avg_review_score,
        COUNT(*) AS reviews_count,

        CASE
            WHEN AVG(review_score) <= 3 THEN 1
            WHEN AVG(review_score) >= 4 THEN 0
            ELSE NULL
        END AS is_negative_review

    FROM olist_order_reviews_dataset
    GROUP BY order_id
),

final_dataset AS (
    SELECT
        s1.order_id,

        s1.customer_city,
        s1.customer_state,

        s1.order_status,

        s1.approval_delay_days,
        s1.delivery_days,
        s1.estimated_delivery_days,
        s1.delay_vs_estimated_days,
        s1.is_late_delivery,

        s1.order_month,
        s1.order_day_of_week,
        s1.order_hour,

        ia.items_count,
        ia.unique_products_count,
        ia.unique_sellers_count,
        ia.total_price,
        ia.total_freight,
        ia.total_sum,
        ia.avg_item_price,
        ia.max_item_price,
        ia.freight_ratio,

        pa.payments_count,
        pa.total_payment_value,
        pa.max_installments,
        pa.has_multiple_payments,

        mp.main_payment_type,

        rpc.main_product_category,
        pag.categories_count,
        pag.total_products_weight,
        pag.total_products_volume,
        pag.avg_product_weight,
        pag.max_product_weight,
        pag.avg_product_volume,
        pag.max_product_volume,
        pag.avg_product_density,
        pag.avg_photos_qty,
        pag.max_photos_qty,
        pag.avg_product_name_length,
        pag.avg_product_description_length,

        ms.main_seller_city,
        ms.main_seller_state,
        sa.has_multiple_sellers,

        CASE
            WHEN s1.customer_state = ms.main_seller_state THEN 1
            WHEN s1.customer_state IS NULL OR ms.main_seller_state IS NULL THEN NULL
            ELSE 0
        END AS same_state_customer_seller,

        ra.is_negative_review

    FROM step1 s1

    LEFT JOIN item_agg ia
        ON s1.order_id = ia.order_id

    LEFT JOIN payment_agg pa
        ON s1.order_id = pa.order_id

    LEFT JOIN main_payment mp
        ON s1.order_id = mp.order_id

    LEFT JOIN ranked_products_categories rpc
        ON s1.order_id = rpc.order_id
        AND rpc.rn = 1

    LEFT JOIN product_agg pag
        ON s1.order_id = pag.order_id

    LEFT JOIN seller_agg sa
        ON s1.order_id = sa.order_id

    LEFT JOIN main_seller ms
        ON s1.order_id = ms.order_id

    INNER JOIN review_agg ra
        ON s1.order_id = ra.order_id
    WHERE ra.is_negative_review IS NOT NULL
)

SELECT *
FROM final_dataset;