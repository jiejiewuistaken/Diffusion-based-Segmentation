-- =============================================================================
-- Step1【评论6指标】: 鉴赏人群下作者×指标 score，落中间表（不截 Top）
-- 表: dm_growth_da.tmp_ht_author_tgi_cmt_score_7d PARTITION(date='${date}')
--
-- 与消费版相同的 TGI/收缩/score 逻辑；动作来自评论供给表，vv 仍来自消费表。
--   消费/评论窗 [${date-6}, ${date}]（调度宏）；人群/作者维表分区 ${date}
--   评论缺失按 0；item 级 LEFT JOIN 后汇总到 user×author×category
--   人群不限 user_tag_ranking；作者垂类用 dm_author_stats_info_df.v2_first_tag_name
--   leave-one-out 分母在「同垂类」内；rate=sum/sum；α=100
--   score=(tgi_shrink-100)*ln(1+uv)；仅保留 tgi_shrink>100 且过作者门槛
-- =============================================================================

SET spark.sql.adaptive.enabled = TRUE;
SET spark.sql.shuffle.partitions = 5000;

CREATE TABLE IF NOT EXISTS dm_growth_da.tmp_ht_author_tgi_cmt_score_7d (
    category                STRING,
    metric                  STRING,
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

WITH
params AS (
    SELECT  '${date}' AS end_date,
            '${date-6}' AS start_date,
            2 AS ua_vv_min,
            100 AS author_vv_min,
            20 AS author_uv_min,
            100.0 AS shrink_alpha
),

author_dim AS (
    SELECT  a.author_id,
            CASE
                WHEN a.v2_first_tag_name IS NULL
                     OR a.v2_first_tag_name IN ('', ',', ',,')
                     OR a.v2_first_tag_name IN ('未知', '未分类') THEN '未知'
                WHEN a.v2_first_tag_name LIKE '%游戏%' THEN '游戏'
                WHEN a.v2_first_tag_name LIKE '%医疗健康%' THEN '医疗健康'
                WHEN a.v2_first_tag_name LIKE '%音乐%' THEN '音乐'
                WHEN a.v2_first_tag_name LIKE '%文化%' THEN '文化'
                WHEN a.v2_first_tag_name LIKE '%时政%' THEN '社会时政'
                WHEN a.v2_first_tag_name LIKE '%汽车%' THEN '汽车'
                WHEN a.v2_first_tag_name LIKE '%美食%' THEN '美食'
                WHEN a.v2_first_tag_name LIKE '%随拍%' THEN '随拍'
                WHEN a.v2_first_tag_name = 'AI原声影像' THEN 'AI原生影像'
                WHEN a.v2_first_tag_name LIKE '%,%' THEN split(a.v2_first_tag_name, ',')[0]
                ELSE a.v2_first_tag_name
            END AS category,
            CAST(a.author_quality_level AS INT) AS author_quality_level,
            CASE WHEN CAST(a.author_quality_level AS INT) = 1 THEN 1 ELSE 0 END AS is_quality
    FROM    aweme_medium.dm_author_stats_info_df a
    CROSS   JOIN params p
    WHERE   a.date = p.end_date
    AND     a.author_id > 0
),

hightaste_user AS (
    SELECT  DISTINCT
            h.user_id,
            CASE
                WHEN h.tag_name IS NULL
                     OR h.tag_name IN ('', ',', ',,')
                     OR h.tag_name IN ('未知', '未分类') THEN '未知'
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

-- vv 来自消费；保留 item_id 以便挂评论
consume_item AS (
    SELECT  c.date,
            c.user_id,
            c.item_id,
            c.item_author_id AS author_id,
            GREATEST(COALESCE(c.feed_page_vv_1d, 0), 0) AS vv
    FROM    dm_growth_da.tmp_feed_consume_base_di_new c
    CROSS   JOIN params p
    WHERE   c.date >= p.start_date
    AND     c.date <= p.end_date
    AND     c.user_id > 0
    AND     c.item_author_id > 0
    AND     COALESCE(c.feed_page_vv_1d, 0) > 0
),

comment_item AS (
    SELECT  cm.date,
            cm.user_id,
            cm.item_id,
            SUM(GREATEST(COALESCE(cm.reply_cmt_cnt, 0), 0)) AS reply_cmt_cnt,
            SUM(GREATEST(COALESCE(cm.send_cmt_cnt, 0), 0)) AS send_cmt_cnt,
            SUM(GREATEST(COALESCE(cm.send_target_cmt_cnt, 0), 0)) AS send_target_cmt_cnt,
            SUM(
                GREATEST(COALESCE(cm.send_high_pre_v4_cmt_cnt, 0), 0)
            ) AS send_high_pre_v4_cmt_cnt,
            SUM(GREATEST(COALESCE(cm.reply_target_cmt_cnt, 0), 0)) AS reply_target_cmt_cnt,
            SUM(
                GREATEST(COALESCE(cm.reply_high_pre_v4_cmt_cnt, 0), 0)
            ) AS reply_high_pre_v4_cmt_cnt
    FROM    dm_growth_da.tmp_feed_comment_supply_label_di_new cm
    CROSS   JOIN params p
    WHERE   cm.date >= p.start_date
    AND     cm.date <= p.end_date
    AND     cm.user_id > 0
    AND     (
                COALESCE(cm.reply_cmt_cnt, 0) > 0
                OR COALESCE(cm.send_cmt_cnt, 0) > 0
                OR COALESCE(cm.send_target_cmt_cnt, 0) > 0
                OR COALESCE(cm.send_high_pre_v4_cmt_cnt, 0) > 0
                OR COALESCE(cm.reply_target_cmt_cnt, 0) > 0
                OR COALESCE(cm.reply_high_pre_v4_cmt_cnt, 0) > 0
            )
    GROUP BY
            cm.date,
            cm.user_id,
            cm.item_id
),

-- 消费 LEFT JOIN 评论（缺失评论按 0）；再与作者垂类、鉴赏人群对齐
ht_consume AS (
    SELECT  b.user_id,
            b.author_id,
            d.category,
            d.author_quality_level,
            d.is_quality,
            b.vv,
            COALESCE(cm.reply_cmt_cnt, 0) AS reply_cmt_cnt,
            COALESCE(cm.send_cmt_cnt, 0) AS send_cmt_cnt,
            COALESCE(cm.send_target_cmt_cnt, 0) AS send_target_cmt_cnt,
            COALESCE(cm.send_high_pre_v4_cmt_cnt, 0) AS send_high_pre_v4_cmt_cnt,
            COALESCE(cm.reply_target_cmt_cnt, 0) AS reply_target_cmt_cnt,
            COALESCE(cm.reply_high_pre_v4_cmt_cnt, 0) AS reply_high_pre_v4_cmt_cnt
    FROM    consume_item b
    LEFT    JOIN comment_item cm
    ON      b.date = cm.date
    AND     b.user_id = cm.user_id
    AND     b.item_id = cm.item_id
    INNER   JOIN author_dim d
    ON      b.author_id = d.author_id
    INNER   JOIN hightaste_user h
    ON      b.user_id = h.user_id
    AND     d.category = h.category
),

ua_raw AS (
    SELECT  user_id,
            author_id,
            category,
            MAX(author_quality_level) AS author_quality_level,
            MAX(is_quality) AS is_quality,
            SUM(vv) AS vv,
            SUM(reply_cmt_cnt) AS reply_cmt_cnt,
            SUM(send_cmt_cnt) AS send_cmt_cnt,
            SUM(send_target_cmt_cnt) AS send_target_cmt_cnt,
            SUM(send_high_pre_v4_cmt_cnt) AS send_high_pre_v4_cmt_cnt,
            SUM(reply_target_cmt_cnt) AS reply_target_cmt_cnt,
            SUM(reply_high_pre_v4_cmt_cnt) AS reply_high_pre_v4_cmt_cnt
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

author_agg AS (
    SELECT  category,
            author_id,
            MAX(author_quality_level) AS author_quality_level,
            MAX(is_quality) AS is_quality,
            SUM(vv) AS vv,
            COUNT(DISTINCT user_id) AS uv,
            SUM(reply_cmt_cnt) AS reply_cmt_cnt,
            SUM(send_cmt_cnt) AS send_cmt_cnt,
            SUM(send_target_cmt_cnt) AS send_target_cmt_cnt,
            SUM(send_high_pre_v4_cmt_cnt) AS send_high_pre_v4_cmt_cnt,
            SUM(reply_target_cmt_cnt) AS reply_target_cmt_cnt,
            SUM(reply_high_pre_v4_cmt_cnt) AS reply_high_pre_v4_cmt_cnt,
            MAX(vv) AS top1_user_vv
    FROM    ua_valid
    GROUP BY
            category,
            author_id
),

author_metric AS (
    SELECT  category,
            author_id,
            author_quality_level,
            is_quality,
            vv,
            uv,
            top1_user_vv,
            metric,
            act
    FROM    author_agg
    LATERAL VIEW stack(
        6,
        'reply_cmt_cnt', CAST(reply_cmt_cnt AS DOUBLE),
        'send_cmt_cnt', CAST(send_cmt_cnt AS DOUBLE),
        'send_target_cmt_cnt', CAST(send_target_cmt_cnt AS DOUBLE),
        'send_high_pre_v4_cmt_cnt', CAST(send_high_pre_v4_cmt_cnt AS DOUBLE),
        'reply_target_cmt_cnt', CAST(reply_target_cmt_cnt AS DOUBLE),
        'reply_high_pre_v4_cmt_cnt', CAST(reply_high_pre_v4_cmt_cnt AS DOUBLE)
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
            a.author_quality_level,
            a.is_quality
    FROM    author_metric a
    INNER   JOIN category_metric_total t
    ON      a.category = t.category
    AND     a.metric = t.metric
    CROSS   JOIN params p
    WHERE   a.vv >= p.author_vv_min
    AND     a.uv >= p.author_uv_min
    AND     t.cat_vv > a.vv
    AND     (t.cat_act - a.act) > 0
)

INSERT OVERWRITE TABLE dm_growth_da.tmp_ht_author_tgi_cmt_score_7d
PARTITION (date = '${date}')
SELECT  category,
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
        author_quality_level,
        is_quality
FROM    scored
WHERE   tgi_shrink > 100
;
