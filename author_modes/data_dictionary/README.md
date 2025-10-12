# Data Dictionary Template (Creator Operating Modes)

Use `data_dictionary_template.yaml` to define the core source datasets required by the feature builder and clustering pipeline.

How to use:
1. Copy the template and fill in the actual `path` (filesystem, Hive table, or warehouse view) and column availability.
2. Keep the `grain`, `primary_key`, and `partitioning` sections accurate; they are used for sanity checks.
3. Ensure time columns are in UTC or specify the timezone to avoid weekly window drift.
4. If a dataset is unavailable (e.g., `traffic_events`), leave it out; the feature builder degrades gracefully.

Key entities:
- `authors`: creator master data
- `short_videos`: posts with/without shopping cart metadata and funnel metrics
- `live_sessions`: livestream sessions
- `shop_window`: storefront exposure/click/add-to-cart/orders
- `orders`: attributed downstream orders/GMV
- `products`: product attributes for category/price/margin features
- `traffic_events`: optional raw events for CTR computation

Tip: Maintain this template in source control and version it alongside the pipeline configs in `author_modes/configs`.
