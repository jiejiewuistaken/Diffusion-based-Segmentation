-- =============================================================================
-- Step2: 每 metric 全局按 score 取 Top10000（跨垂类一起排，便于看自然分布）
-- 依赖: dm_growth_da.tmp_ht_author_tgi_score_7d
-- 产出: dm_growth_da.tmp_ht_author_tgi_top10000_7d
-- =============================================================================

SET spark.sql.adaptive.enabled = TRUE;

CREATE TABLE IF NOT EXISTS dm_growth_da.tmp_ht_author_tgi_top10000_7d (
    metric                  STRING,
    rn                      INT,
    category                STRING,
    author_id               BIGINT,
    vv                      BIGINT,
    uv                      BIGINT,
    act                     DOUBLE,
    rate                    DOUBLE,
    bench_rate              DOUBLE,
    tgi_raw                 DOUBLE,
    tgi_shrink              DOUBLE,
    score                   DOUBLE,
    top1_user_vv_ratio      DOUBLE,
    author_quality_level    INT,
    is_quality              INT
) PARTITIONED BY (date STRING)
STORED AS PARQUET;

INSERT OVERWRITE TABLE dm_growth_da.tmp_ht_author_tgi_top10000_7d
PARTITION (date = '${date}')
SELECT  metric,
        rn,
        category,
        author_id,
        vv,
        uv,
        act,
        rate,
        bench_rate,
        tgi_raw,
        tgi_shrink,
        score,
        top1_user_vv_ratio,
        author_quality_level,
        is_quality
FROM    (
            SELECT  metric,
                    category,
                    author_id,
                    vv,
                    uv,
                    act,
                    rate,
                    bench_rate,
                    tgi_raw,
                    tgi_shrink,
                    score,
                    top1_user_vv_ratio,
                    author_quality_level,
                    is_quality,
                    ROW_NUMBER() OVER (
                        PARTITION BY metric
                        ORDER BY score DESC, tgi_shrink DESC, uv DESC, author_id
                    ) AS rn
            FROM    dm_growth_da.tmp_ht_author_tgi_score_7d
            WHERE   date = '${date}'
        ) t
WHERE   rn <= 10000
;
