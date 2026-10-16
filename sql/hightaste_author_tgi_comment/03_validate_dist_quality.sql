-- =============================================================================
-- Step3【评论6指标】: Top10000 垂类分布 + 优质率
-- 依赖: dm_growth_da.tmp_ht_author_tgi_cmt_top10000_7d
-- =============================================================================

-- 3a) 每指标：整体优质率 + Top10000 规模
SELECT  date,
        metric,
        COUNT(*) AS author_cnt,
        SUM(is_quality) AS quality_cnt,
        SUM(is_quality) / COUNT(*) AS quality_rate
FROM    dm_growth_da.tmp_ht_author_tgi_cmt_top10000_7d
WHERE   date = '${date}'
GROUP BY
        date,
        metric
ORDER BY
        metric
;

-- 3b) 每指标 × 垂类：分布人数 / 占 Top10000 比例 / 该垂类内优质率
SELECT  date,
        metric,
        category,
        COUNT(*) AS author_cnt,
        COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY date, metric) AS category_share,
        SUM(is_quality) AS quality_cnt,
        SUM(is_quality) / COUNT(*) AS quality_rate_in_category
FROM    dm_growth_da.tmp_ht_author_tgi_cmt_top10000_7d
WHERE   date = '${date}'
GROUP BY
        date,
        metric,
        category
ORDER BY
        metric,
        author_cnt DESC
;
