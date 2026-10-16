-- =============================================================================
-- Step6【评论】: 优质作者 Top10000 的垂类分布（主看分布）
-- 依赖: dm_growth_da.tmp_ht_author_tgi_cmt_top10000_quality_7d
-- =============================================================================

SELECT  date,
        metric,
        COUNT(*) AS author_cnt
FROM    dm_growth_da.tmp_ht_author_tgi_cmt_top10000_quality_7d
WHERE   date = '${date}'
GROUP BY
        date,
        metric
ORDER BY
        metric
;

SELECT  date,
        metric,
        category,
        COUNT(*) AS author_cnt,
        COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY date, metric) AS category_share
FROM    dm_growth_da.tmp_ht_author_tgi_cmt_top10000_quality_7d
WHERE   date = '${date}'
GROUP BY
        date,
        metric,
        category
ORDER BY
        metric,
        author_cnt DESC
;
