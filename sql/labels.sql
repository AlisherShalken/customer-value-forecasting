-- Targets for each scored (snapshot_date, customer_id):
--   purchased : at least one non-cancelled order in [snapshot, snapshot + horizon)
--   revenue   : net merchandise revenue in that window, floored at zero
--               (cancellations of earlier orders can make the net negative).

SELECT
    p.snapshot_date,
    p.customer_id,
    count(o.invoice) FILTER (WHERE NOT o.is_cancellation) > 0  AS purchased,
    greatest(coalesce(sum(o.amount), 0), 0)                     AS revenue
FROM {{ population }} AS p
LEFT JOIN stg.orders AS o
  ON o.customer_id = p.customer_id
 AND o.invoice_ts >= p.snapshot_date
 AND o.invoice_ts <  p.snapshot_date + INTERVAL {{ horizon_days }} DAY
GROUP BY p.snapshot_date, p.customer_id
ORDER BY p.snapshot_date, p.customer_id
