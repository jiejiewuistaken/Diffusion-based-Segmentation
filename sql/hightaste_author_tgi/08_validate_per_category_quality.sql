-- =============================================================================
-- Step8【消费】: 每垂类 Top10000 的优质率（主看优质率）
-- 依赖: dm_growth_da.tmp_ht_author_tgi_top10000_bycat_7d
-- =============================================================================

-- 8a) 每指标 × 垂类：池子规模 + 优质率
SELECT  date,
        metric,
        category,
        COUNT(*) AS author_cnt,
        SUM(is_quality) AS quality_cnt,
        SUM(is_quality) / COUNT(*) AS quality_rate
FROM    dm_growth_da.tmp_ht_author_tgi_top10000_bycat_7d
WHERE   date = '${date}'
GROUP BY
        date,
        metric,
        category
ORDER BY
        metric,
        category
;

-- 8b) 每指标：各垂类 Top10000 合并后的整体优质率（便于和全局大盘 Top10000 对比）
SELECT  date,
        metric,
        COUNT(*) AS author_cnt,
        SUM(is_quality) AS quality_cnt,
        SUM(is_quality) / COUNT(*) AS quality_rate
FROM    dm_growth_da.tmp_ht_author_tgi_top10000_bycat_7d
WHERE   date = '${date}'
GROUP BY
        date,
        metric
ORDER BY
        metric
;
