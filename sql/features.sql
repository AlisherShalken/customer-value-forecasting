-- Point-in-time customer features.
--
-- One row per (snapshot_date, customer_id) for customers with a purchase in
-- the `active_window_days` before the snapshot. Every aggregate is computed
-- from orders strictly before snapshot_date, so a row contains exactly what
-- was known at that moment. The same template serves the batch feature store
-- and on-demand scoring of a single customer (online/offline parity).
--
-- Template params: snapshots (relation with a snapshot_date column),
-- windows (list of day windows), horizon_days, active_window_days,
-- single_customer.
-- Bound params when single_customer: $customer_id.

WITH snapshots AS (
    SELECT DISTINCT CAST(snapshot_date AS DATE) AS snapshot_date
    FROM {{ snapshots }}
),

orders AS (
    SELECT *
    FROM stg.orders
    {% if single_customer %}WHERE customer_id = $customer_id{% endif %}
),

history AS (
    SELECT
        s.snapshot_date,
        o.customer_id,
        o.invoice_ts,
        o.is_cancellation,
        o.amount,
        o.units,
        o.n_products,
        o.country,
        date_diff('day', CAST(o.invoice_ts AS DATE), s.snapshot_date) AS age_days
    FROM snapshots AS s
    JOIN orders AS o
      ON o.invoice_ts < s.snapshot_date
),

order_stats AS (
    SELECT
        snapshot_date,
        customer_id,
        min(age_days) FILTER (WHERE NOT is_cancellation)                AS recency_days,
        max(age_days)                                                   AS tenure_days,
        count(*) FILTER (WHERE NOT is_cancellation)                     AS n_orders,
        {%- for w in windows %}
        count(*) FILTER (WHERE NOT is_cancellation AND age_days <= {{ w }})
                                                                        AS n_orders_{{ w }}d,
        coalesce(sum(amount) FILTER (WHERE NOT is_cancellation AND age_days <= {{ w }}), 0)
                                                                        AS revenue_{{ w }}d,
        {%- endfor %}
        -- The same `horizon` a year earlier: captures seasonal (e.g. Q4) buyers.
        count(*) FILTER (WHERE NOT is_cancellation
                         AND age_days BETWEEN 366 - {{ horizon_days }} AND 365)
                                                                        AS n_orders_same_period_ly,
        coalesce(sum(amount) FILTER (WHERE NOT is_cancellation
                         AND age_days BETWEEN 366 - {{ horizon_days }} AND 365), 0)
                                                                        AS revenue_same_period_ly,
        sum(amount) FILTER (WHERE NOT is_cancellation)                  AS gross_revenue,
        coalesce(-sum(amount) FILTER (WHERE is_cancellation), 0)        AS cancelled_amount,
        count(*) FILTER (WHERE is_cancellation)                         AS n_cancellations,
        avg(amount) FILTER (WHERE NOT is_cancellation)                  AS avg_order_value,
        max(amount) FILTER (WHERE NOT is_cancellation)                  AS max_order_value,
        avg(n_products) FILTER (WHERE NOT is_cancellation)              AS avg_products_per_order,
        avg(units) FILTER (WHERE NOT is_cancellation)                   AS avg_units_per_order,
        -- Rolling ~30-day periods (12 per year) rather than calendar months,
        -- which a 365-day window can intersect 13 of.
        count(DISTINCT floor((age_days - 1) / 30.4375))
            FILTER (WHERE NOT is_cancellation AND age_days <= 365)      AS active_months_365d,
        arg_max(country, invoice_ts)                                    AS country
    FROM history
    GROUP BY snapshot_date, customer_id
),

purchase_days AS (
    SELECT DISTINCT
        snapshot_date,
        customer_id,
        CAST(invoice_ts AS DATE) AS purchase_date
    FROM history
    WHERE NOT is_cancellation
),

gaps AS (
    SELECT
        snapshot_date,
        customer_id,
        date_diff('day', lag(purchase_date) OVER w, purchase_date) AS gap_days
    FROM purchase_days
    WINDOW w AS (PARTITION BY snapshot_date, customer_id ORDER BY purchase_date)
),

gap_stats AS (
    SELECT
        snapshot_date,
        customer_id,
        count(*)                AS n_purchase_days,
        avg(gap_days)           AS mean_gap_days,
        stddev_samp(gap_days)   AS std_gap_days
    FROM gaps
    GROUP BY snapshot_date, customer_id
),

line_stats AS (
    SELECT
        s.snapshot_date,
        l.customer_id,
        count(DISTINCT l.stock_code)    AS n_distinct_products_365d,
        avg(l.unit_price)               AS avg_unit_price_365d,
        median(l.quantity)              AS median_line_quantity_365d
    FROM snapshots AS s
    JOIN stg.order_lines AS l
      ON l.invoice_ts < s.snapshot_date
     AND l.invoice_ts >= s.snapshot_date - INTERVAL {{ active_window_days }} DAY
     AND NOT l.is_cancellation
     {% if single_customer %}AND l.customer_id = $customer_id{% endif %}
    GROUP BY s.snapshot_date, l.customer_id
)

SELECT
    o.snapshot_date,
    o.customer_id,
    o.country,
    CAST(month(o.snapshot_date) AS INTEGER)                     AS snapshot_month,
    o.recency_days,
    o.tenure_days,
    o.n_orders,
    {%- for w in windows %}
    o.n_orders_{{ w }}d,
    o.revenue_{{ w }}d,
    {%- endfor %}
    o.n_orders_same_period_ly,
    o.revenue_same_period_ly,
    o.gross_revenue,
    o.cancelled_amount / nullif(o.gross_revenue, 0)             AS cancel_rate,
    o.n_cancellations,
    o.avg_order_value,
    o.max_order_value,
    o.avg_products_per_order,
    o.avg_units_per_order,
    o.active_months_365d,
    g.n_purchase_days,
    g.mean_gap_days,
    g.std_gap_days,
    o.recency_days / nullif(g.mean_gap_days, 0)                 AS overdue_ratio,
    o.revenue_90d / nullif(o.revenue_365d, 0)                   AS revenue_share_90d,
    l.n_distinct_products_365d,
    l.avg_unit_price_365d,
    l.median_line_quantity_365d
FROM order_stats AS o
LEFT JOIN gap_stats AS g USING (snapshot_date, customer_id)
LEFT JOIN line_stats AS l USING (snapshot_date, customer_id)
WHERE o.recency_days <= {{ active_window_days }}
ORDER BY o.snapshot_date, o.customer_id
