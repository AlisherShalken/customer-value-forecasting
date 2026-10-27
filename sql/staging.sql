-- Staging layer: clean order lines and invoice-level orders.
--
-- Data quality rules (see README "Data"):
--   * The two source sheets overlap on 2010-12-01..2010-12-09 with identical
--     rows; the overlap is taken from the later sheet only.
--   * Exact duplicate lines are collapsed.
--   * Lines without a customer id cannot be attributed and are dropped.
--   * Only merchandise lines are kept. Postage, fees, manual adjustments,
--     bad-debt write-offs, samples and gift vouchers are not product revenue.
--   * Zero/negative prices are stock adjustments, not sales.

CREATE SCHEMA IF NOT EXISTS stg;

CREATE OR REPLACE TABLE stg.order_lines AS
WITH deduplicated AS (
    SELECT DISTINCT
        invoice,
        stock_code,
        quantity,
        invoice_ts,
        unit_price,
        customer_id,
        country
    FROM raw.transactions
    WHERE NOT (source_sheet = 'Year 2009-2010' AND invoice_ts >= TIMESTAMP '2010-12-01')
)
SELECT
    invoice,
    starts_with(invoice, 'C')               AS is_cancellation,
    upper(trim(stock_code))                 AS stock_code,
    quantity,
    invoice_ts,
    unit_price,
    quantity * unit_price                   AS line_amount,
    CAST(customer_id AS BIGINT)             AS customer_id,
    trim(country)                           AS country
FROM deduplicated
WHERE customer_id IS NOT NULL
  AND unit_price > 0
  AND quantity <> 0
  AND (
        regexp_matches(stock_code, '^[0-9]{5}')
     OR starts_with(upper(stock_code), 'DCGS')
  );

CREATE OR REPLACE TABLE stg.orders AS
SELECT
    invoice,
    customer_id,
    min(invoice_ts)                 AS invoice_ts,
    bool_or(is_cancellation)        AS is_cancellation,
    sum(line_amount)                AS amount,
    sum(quantity)                   AS units,
    count(*)                        AS n_lines,
    count(DISTINCT stock_code)      AS n_products,
    any_value(country)              AS country
FROM stg.order_lines
GROUP BY invoice, customer_id;

