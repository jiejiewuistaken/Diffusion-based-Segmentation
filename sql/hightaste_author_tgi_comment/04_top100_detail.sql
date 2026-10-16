-- =============================================================================
-- Step4【评论6指标】: 每指标 Top100 作者明细
-- 依赖: dm_growth_da.tmp_ht_author_tgi_cmt_top10000_7d
-- =============================================================================

SELECT  date,
        metric,
        rn,
        category,
        author_id,
        vv,
        uv,
        rate,
        bench_rate,
        tgi_raw,
        tgi_shrink,
        score,
        top1_user_vv_ratio,
        author_quality_level,
        is_quality
FROM    dm_growth_da.tmp_ht_author_tgi_cmt_top10000_7d
WHERE   date = '${date}'
AND     rn <= 100
ORDER BY
        metric,
        rn
;
