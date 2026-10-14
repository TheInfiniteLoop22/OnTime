# FEATURE_SPEC.md — OnTime feature definitions

All features are computed **as of t₀ = `order_approved_at`**. Every feature has: name, family,
definition, source columns, leak status, and an explanation template for reason codes.
Implement each family in its own module under `src/ontime/features/`.

Legend for leak status: ✅ known at t₀ · ⚠️ historical, needs time-guard · ❌ forbidden

## Forbidden columns (never features)
`order_delivered_carrier_date`, `order_delivered_customer_date`, any `review_*` column,
`order_status` (filtering only), and any aggregate computed over future orders.

## 1. Logistics geometry — `features/geo.py`
| Feature | Definition | Leak | Reason template |
|---|---|---|---|
| `distance_km` | Haversine between primary seller zip centroid and customer zip centroid | ✅ | "Seller ships far: {v:,.0f} km from customer" |
| `log_distance_km` | log1p(distance_km) | ✅ | (same as above) |
| `same_state` | seller_state == customer_state | ✅ | "Seller and customer in the same state" |
| `customer_region` / `seller_region` | Map state → Brazilian macro-region (N, NE, CO, SE, S) | ✅ | "Delivery to the {v} region" |
| `customer_is_capital_metro` | customer zip prefix within capital-city prefix ranges (optional; or city == state capital) | ✅ | "Delivery outside a major metro" |
| `geo_imputed` | 1 if either centroid fell back to state centroid | ✅ | — |
| `n_distinct_sellers` | distinct sellers in order (multi-seller orders split shipments) | ✅ | "Order ships from {v} different sellers" |

## 2. Promise & handling windows — `features/calendar.py`
| Feature | Definition | Leak | Reason template |
|---|---|---|---|
| `promised_days` | estimated_delivery_date − approved_at (days) | ✅ | "Tight promise: {v:.0f} days to deliver" / "Generous promise: {v:.0f} days" |
| `approval_lag_hours` | approved_at − purchase_timestamp | ✅ | "Payment approval took {v:.0f} hours" |
| `handling_window_days` | max(shipping_limit_date) − approved_at | ✅ | "Seller has only {v:.1f} days to hand over to carrier" |
| `promise_per_1000km` | promised_days / (distance_km/1000 + 1) | ✅ | "Promise is tight for the distance" |

## 3. Calendar — `features/calendar.py`
| Feature | Definition | Leak | Reason template |
|---|---|---|---|
| `approve_dow`, `approve_hour`, `approve_month` | from approved_at | ✅ | "Approved on a {v}" |
| `is_weekend_approval` | Sat/Sun | ✅ | "Approved on a weekend" |
| `days_to_next_holiday` | using `holidays.Brazil()` (national) | ✅ | "Holiday in {v} days" |
| `holidays_in_promise_window` | count of national holidays between approval and estimated date | ✅ | "{v} public holidays fall within the delivery window" |
| `is_black_friday_window` | approved within configurable window around Black Friday | ✅ | "Ordered during the Black Friday rush" |

## 4. Order composition & product — `features/product.py`
| Feature | Definition | Leak | Reason template |
|---|---|---|---|
| `n_items` | count of order_items rows | ✅ | "Order has {v} items" |
| `total_price`, `total_freight` | sums | ✅ | — |
| `freight_ratio` | total_freight / total_price | ✅ | "High shipping cost relative to price" |
| `total_weight_kg` | sum(weight_g)/1000 | ✅ | "Heavy shipment: {v:.1f} kg" |
| `total_volume_l` | sum(L×H×W)/1000 | ✅ | "Bulky shipment: {v:.0f} litres" |
| `max_item_weight_kg` | max over items | ✅ | — |
| `category_group` | English category collapsed to ~10–15 groups (mapping in config); rare → "other" | ✅ | "Category: {v}" |
| `product_photos_qty_mean` | mean photos (listing quality proxy) | ✅ | — |
| `payment_type` | dominant payment type (by value) | ✅ | "Paid by {v}" |
| `payment_installments_max` | max installments | ✅ | — |

## 5. Leak-safe history features — `features/history.py`  ⚠️
These are the most important and most error-prone features. **Rule:** for order *o* approved at
t₀(o), a past order *p* may contribute only if `delivered_customer_date(p) < t₀(o)` — i.e.,
its outcome was *observable* before *o*'s decision time. Using `approved_at(p) < t₀(o)` alone is
**not enough** (p might still be in transit, its lateness unknown) — this subtle point is a
great interview talking point.

Implementation sketch (vectorized, no per-row Python loops over 100k orders):
- Build an "outcome events" table: (seller_id, event_time = delivered_customer_date, is_late).
- Build a "queries" table: (seller_id, query_time = approved_at, order_id).
- Use `pd.merge_asof` on time per seller (or DuckDB ASOF JOIN / window functions) over cumulative
  counts of outcome events to get, for each query, cumulative n_past and n_past_late strictly
  before query_time.

| Feature | Definition | Reason template |
|---|---|---|
| `seller_n_past` | # of seller's orders with outcome known before t₀ | "New seller: only {v} completed orders" |
| `seller_late_rate_shrunk` | (n_past_late + α·m)/(n_past + α), with m = global late rate known before t₀ (computed the same leak-safe way) and α from config (e.g., 20). Bayesian/beta-binomial shrinkage. | "Seller's past on-time record is weak: {v:.0%} late" |
| `seller_late_rate_90d_shrunk` | same, restricted to outcomes in the 90 days before t₀ | "Seller has been slipping recently" |
| `route_late_rate_shrunk` | same logic keyed on (seller_state, customer_state) | "This route ({v}) is often late" |
| `customer_state_late_rate_shrunk` | same keyed on customer_state | — |
| `seller_load_7d` | # of the seller's orders **approved** in the 7 days before t₀ (known at t₀, no outcome needed) | "Seller is busy: {v} orders in the last week" |
| `seller_days_since_first_order` | t₀ − seller's first approved_at | "Seller joined {v:.0f} days ago" |
| `global_late_rate_30d` | late rate of all outcomes known in the 30 days before t₀ (system-wide stress signal) | "Network-wide delays are elevated" |

### Required unit tests (tests/test_history_leakage.py)
1. Construct a tiny DataFrame where a past order is approved before t₀ but delivered **after**
   t₀ → it must NOT count toward history.
2. An order must never count itself.
3. Shuffling row order does not change any feature value.
4. A seller with zero history gets exactly the prior m (shrinkage edge case).
5. A property test: for random synthetic data, recompute features with a naive O(n²) loop and
   assert equality with the vectorized implementation.

## 6. Preprocessing per model
- **Logistic regression:** median impute + missing indicators, standard scale numerics,
  one-hot categoricals (min_frequency), log-transform skewed numerics (distance, weight, volume,
  price). Built as a sklearn `ColumnTransformer` + `Pipeline`.
- **EBM:** raw numerics + categoricals (EBM bins internally); limit interactions (e.g., 10).
- **LightGBM:** raw numerics, categoricals as pandas `category` dtype.
- All transformers are fit on **train only**.

## 7. Reason-code generation — `explain/reason_codes.py`
- Input: per-order contribution vector (SHAP for LightGBM or EBM local scores) + raw feature values.
- Group derived/related features into one reason (e.g., `distance_km` + `log_distance_km` +
  `promise_per_1000km` → "distance" group) by summing contributions within a group, so users
  don't see the same reason twice.
- Output top 3 positive (risk-increasing) and top 2 negative groups above a minimum magnitude,
  rendered from templates in `configs/reason_templates.yaml`.
- Test: every feature used by the model maps to a group and a template (no orphan features).
