-- =============================================================================
-- Step7【消费】: 每 metric × 每垂类 Top10000
-- 依赖: dm_growth_da.tmp_ht_author_tgi_score_7d
-- 产出: dm_growth_da.tmp_ht_author_tgi_top10000_bycat_7d
-- 用途: 看各垂类内的优质率
-- =============================================================================

SET spark.sql.adaptive.enabled = TRUE;

CREATE TABLE IF NOT EXISTS dm_growth_da.tmp_ht_author_tgi_top10000_bycat_7d (
    metric                  STRING,
    category                STRING,
    rn                      INT,
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

INSERT OVERWRITE TABLE dm_growth_da.tmp_ht_author_tgi_top10000_bycat_7d
PARTITION (date = '${date}')
SELECT  metric,
        category,
        rn,
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
                        PARTITION BY metric, category
                        ORDER BY score DESC, tgi_shrink DESC, uv DESC, author_id
                    ) AS rn
            FROM    dm_growth_da.tmp_ht_author_tgi_score_7d
            WHERE   date = '${date}'
        ) t
WHERE   rn <= 10000
;
