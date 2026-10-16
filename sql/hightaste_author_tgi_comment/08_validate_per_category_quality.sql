-- =============================================================================
-- Step8【评论】: 每垂类 Top10000 的优质率（主看优质率）
-- 依赖: dm_growth_da.tmp_ht_author_tgi_cmt_top10000_bycat_7d
-- =============================================================================

SELECT  date,
        metric,
        category,
        COUNT(*) AS author_cnt,
        SUM(is_quality) AS quality_cnt,
        SUM(is_quality) / COUNT(*) AS quality_rate
FROM    dm_growth_da.tmp_ht_author_tgi_cmt_top10000_bycat_7d
WHERE   date = '${date}'
GROUP BY
        date,
        metric,
        category
ORDER BY
        metric,
        category
;

SELECT  date,
        metric,
        COUNT(*) AS author_cnt,
        SUM(is_quality) AS quality_cnt,
        SUM(is_quality) / COUNT(*) AS quality_rate
FROM    dm_growth_da.tmp_ht_author_tgi_cmt_top10000_bycat_7d
WHERE   date = '${date}'
GROUP BY
        date,
        metric
ORDER BY
        metric
;
