-- 垂类高鉴赏力人群 → 作者相对偏好 TGI（7 日窗，外溢找相似好作者）
-- ${date}：窗口结束日；消费取 [date-6, date]；鉴赏人群取 date 分区快照。
--
-- 口径摘要：
--   1) 只保留「鉴赏用户 × 其合格垂类」上的消费；作者挂主垂类后与人群垂类对齐
--   2) UA 窗内 vv>=2；作者门槛占位 vv>=100 AND uv>=20
--   3) 率 = sum(act)/sum(vv)；分母 = 同人群同垂类 leave-one-out 的 sum/sum
--   4) 收缩 α=100（固定，快试）：tgi = 100 * (act+α*r_b)/( (vv+α)*r_b )
--   5) score = (tgi-100) * ln(1+uv)；每 category×metric Top 10000
--   6) 指标不含关注；合成后置，本任务只出分指标榜
--
-- 可选：将 seed_author 换成标杆 3000 表，用于诊断重叠（不参与过滤）

SET spark.sql.adaptive.enabled = TRUE;
SET spark.sql.shuffle.partitions = 5000;

WITH
params AS (
    SELECT  '${date}' AS end_date,
            date_format(date_sub(to_date('${date}', 'yyyyMMdd'), 6), 'yyyyMMdd') AS start_date,
            2 AS ua_vv_min,
            100 AS author_vv_min,
            20 AS author_uv_min,
            100.0 AS shrink_alpha,
            10000 AS top_k
),

-- 若有标杆作者表可替换此 CTE；无则 in_seed_3000 恒为 0
seed_author AS (
    SELECT  CAST(NULL AS BIGINT) AS author_id
    WHERE   1 = 0
),

hightaste_user AS (
    SELECT  DISTINCT
            h.user_id,
            CASE
                WHEN h.tag_name IS NULL OR h.tag_name IN ('', ',', ',,') OR h.tag_name IN ('未知', '未分类') THEN '未知'
                WHEN h.tag_name LIKE '%游戏%' THEN '游戏'
                WHEN h.tag_name LIKE '%医疗健康%' THEN '医疗健康'
                WHEN h.tag_name LIKE '%音乐%' THEN '音乐'
                WHEN h.tag_name LIKE '%文化%' THEN '文化'
                WHEN h.tag_name LIKE '%时政%' THEN '社会时政'
                WHEN h.tag_name LIKE '%汽车%' THEN '汽车'
                WHEN h.tag_name LIKE '%美食%' THEN '美食'
                WHEN h.tag_name LIKE '%随拍%' THEN '随拍'
                WHEN h.tag_name = 'AI原声影像' THEN 'AI原生影像'
                WHEN h.tag_name LIKE '%,%' THEN split(h.tag_name, ',')[0]
                ELSE h.tag_name
            END AS category
    FROM    dm_growth_da.hightaste_user_vote_weight_wjj h
    CROSS   JOIN params p
    WHERE   h.date = p.end_date
    AND     h.user_id > 0
    AND     h.tag_name IS NOT NULL
),

consume_item AS (
    SELECT  c.date,
            c.user_id,
            c.item_author_id AS author_id,
            CASE
                WHEN c.v2_first_tag_name IS NULL
                     OR c.v2_first_tag_name IN ('', ',', ',,')
                     OR c.v2_first_tag_name IN ('未知', '未分类') THEN '未知'
                WHEN c.v2_first_tag_name LIKE '%游戏%' THEN '游戏'
                WHEN c.v2_first_tag_name LIKE '%医疗健康%' THEN '医疗健康'
                WHEN c.v2_first_tag_name LIKE '%音乐%' THEN '音乐'
                WHEN c.v2_first_tag_name LIKE '%文化%' THEN '文化'
                WHEN c.v2_first_tag_name LIKE '%时政%' THEN '社会时政'
                WHEN c.v2_first_tag_name LIKE '%汽车%' THEN '汽车'
                WHEN c.v2_first_tag_name LIKE '%美食%' THEN '美食'
                WHEN c.v2_first_tag_name LIKE '%随拍%' THEN '随拍'
                WHEN c.v2_first_tag_name = 'AI原声影像' THEN 'AI原生影像'
                ELSE c.v2_first_tag_name
            END AS category,
            GREATEST(COALESCE(c.feed_page_vv_1d, 0), 0) AS vv,
            GREATEST(COALESCE(c.feed_page_comment_cnt_1d, 0), 0) AS comment_cnt,
            GREATEST(COALESCE(c.feed_page_download_cnt_1d, 0), 0) AS download_cnt,
            GREATEST(COALESCE(c.feed_page_finish_vv_1d, 0), 0) AS finish_vv,
            GREATEST(COALESCE(c.feed_page_outer_share_cnt_1d, 0), 0) AS outer_share_cnt,
            GREATEST(COALESCE(c.feed_page_like_cnt_1d, 0), 0) AS like_cnt,
            GREATEST(COALESCE(c.feed_page_favourite_video_cnt_1d, 0), 0) AS favourite_cnt,
            GREATEST(COALESCE(c.feed_page_enter_personal_detail_cnt_1d, 0), 0) AS enter_profile_cnt,
            GREATEST(COALESCE(c.feed_page_share_cnt_1d, 0), 0) AS share_cnt,
            GREATEST(COALESCE(c.feed_page_inner_share_cnt_1d, 0), 0) AS inner_share_cnt
    FROM    dm_growth_da.tmp_feed_consume_base_di_new c
    CROSS   JOIN params p
    WHERE   c.date >= p.start_date
    AND     c.date <= p.end_date
    AND     c.user_id > 0
    AND     c.item_author_id > 0
    AND     COALESCE(c.feed_page_vv_1d, 0) > 0
),

-- 仅保留：鉴赏用户在其合格垂类上的消费
ht_consume AS (
    SELECT  c.*
    FROM    consume_item c
    INNER   JOIN hightaste_user h
    ON      c.user_id = h.user_id
    AND     c.category = h.category
),

ua_raw AS (
    SELECT  user_id,
            author_id,
            category,
            SUM(vv) AS vv,
            SUM(comment_cnt) AS comment_cnt,
            SUM(download_cnt) AS download_cnt,
            SUM(finish_vv) AS finish_vv,
            SUM(outer_share_cnt) AS outer_share_cnt,
            SUM(like_cnt) AS like_cnt,
            SUM(favourite_cnt) AS favourite_cnt,
            SUM(enter_profile_cnt) AS enter_profile_cnt,
            SUM(share_cnt) AS share_cnt,
            SUM(inner_share_cnt) AS inner_share_cnt
    FROM    ht_consume
    GROUP BY
            user_id,
            author_id,
            category
),

ua_valid AS (
    SELECT  u.*
    FROM    ua_raw u
    CROSS   JOIN params p
    WHERE   u.vv >= p.ua_vv_min
),

-- 作者主垂类：鉴赏人群窗内消费 vv 最大的 category
author_primary AS (
    SELECT  author_id,
            category
    FROM    (
                SELECT  author_id,
                        category,
                        ROW_NUMBER() OVER (
                            PARTITION BY author_id
                            ORDER BY category_vv DESC, category
                        ) AS rn
                FROM    (
                            SELECT  author_id,
                                    category,
                                    SUM(vv) AS category_vv
                            FROM    ua_raw
                            GROUP BY
                                    author_id,
                                    category
                        ) s
            ) t
    WHERE   rn = 1
),

ua_primary AS (
    SELECT  u.*
    FROM    ua_valid u
    INNER   JOIN author_primary ap
    ON      u.author_id = ap.author_id
    AND     u.category = ap.category
),

author_agg AS (
    SELECT  category,
            author_id,
            SUM(vv) AS vv,
            COUNT(DISTINCT user_id) AS uv,
            SUM(comment_cnt) AS comment_cnt,
            SUM(download_cnt) AS download_cnt,
            SUM(finish_vv) AS finish_vv,
            SUM(outer_share_cnt) AS outer_share_cnt,
            SUM(like_cnt) AS like_cnt,
            SUM(favourite_cnt) AS favourite_cnt,
            SUM(enter_profile_cnt) AS enter_profile_cnt,
            SUM(share_cnt) AS share_cnt,
            SUM(inner_share_cnt) AS inner_share_cnt,
            -- 单用户 vv 浓度（诊断/可选过滤）
            MAX(vv) AS top1_user_vv
    FROM    ua_primary
    GROUP BY
            category,
            author_id
),

-- 宽表转 metric 长表（不含关注）
author_metric AS (
    SELECT  category,
            author_id,
            vv,
            uv,
            top1_user_vv,
            metric,
            act
    FROM    author_agg
    LATERAL VIEW stack(
        9,
        'feed_page_comment_cnt_1d', CAST(comment_cnt AS DOUBLE),
        'feed_page_download_cnt_1d', CAST(download_cnt AS DOUBLE),
        'feed_page_finish_vv_1d', CAST(finish_vv AS DOUBLE),
        'feed_page_outer_share_cnt_1d', CAST(outer_share_cnt AS DOUBLE),
        'feed_page_like_cnt_1d', CAST(like_cnt AS DOUBLE),
        'feed_page_favourite_video_cnt_1d', CAST(favourite_cnt AS DOUBLE),
        'feed_page_enter_personal_detail_cnt_1d', CAST(enter_profile_cnt AS DOUBLE),
        'feed_page_share_cnt_1d', CAST(share_cnt AS DOUBLE),
        'feed_page_inner_share_cnt_1d', CAST(inner_share_cnt AS DOUBLE)
    ) s AS metric, act
),

category_metric_total AS (
    SELECT  category,
            metric,
            SUM(vv) AS cat_vv,
            SUM(act) AS cat_act
    FROM    author_metric
    GROUP BY
            category,
            metric
),

scored AS (
    SELECT  p.end_date AS date,
            a.category,
            a.metric,
            a.author_id,
            a.vv,
            a.uv,
            a.act,
            a.act / a.vv AS rate,
            (t.cat_act - a.act) / (t.cat_vv - a.vv) AS bench_rate,
            100.0 * (a.act / a.vv)
                / ((t.cat_act - a.act) / (t.cat_vv - a.vv)) AS tgi_raw,
            100.0 * (a.act + p.shrink_alpha * ((t.cat_act - a.act) / (t.cat_vv - a.vv)))
                / ((a.vv + p.shrink_alpha) * ((t.cat_act - a.act) / (t.cat_vv - a.vv))) AS tgi_shrink,
            (
                100.0 * (a.act + p.shrink_alpha * ((t.cat_act - a.act) / (t.cat_vv - a.vv)))
                    / ((a.vv + p.shrink_alpha) * ((t.cat_act - a.act) / (t.cat_vv - a.vv)))
                - 100.0
            ) * ln(1.0 + a.uv) AS score,
            a.top1_user_vv / a.vv AS top1_user_vv_ratio,
            CASE WHEN s.author_id IS NOT NULL THEN 1 ELSE 0 END AS in_seed_3000,
            p.top_k,
            p.author_vv_min,
            p.author_uv_min
    FROM    author_metric a
    INNER   JOIN category_metric_total t
    ON      a.category = t.category
    AND     a.metric = t.metric
    CROSS   JOIN params p
    LEFT    JOIN seed_author s
    ON      a.author_id = s.author_id
    WHERE   a.vv >= p.author_vv_min
    AND     a.uv >= p.author_uv_min
    AND     t.cat_vv > a.vv
    AND     (t.cat_act - a.act) > 0
),

filtered AS (
    SELECT  *
    FROM    scored
    WHERE   tgi_shrink > 100
),

ranked AS (
    SELECT  date,
            category,
            metric,
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
            in_seed_3000,
            ROW_NUMBER() OVER (
                PARTITION BY date, category, metric
                ORDER BY score DESC, tgi_shrink DESC, uv DESC, author_id
            ) AS rn
    FROM    filtered
)

SELECT  date,
        category,
        metric,
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
        in_seed_3000,
        rn
FROM    ranked
WHERE   rn <= 10000
ORDER BY
        date,
        category,
        metric,
        rn
;
