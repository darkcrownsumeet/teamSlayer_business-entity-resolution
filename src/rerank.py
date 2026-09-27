"""Stage-2 blocking: LightGBM reranker over stage-1 candidates, then adaptive cut."""

import lightgbm as lgb
from src.blocking import SRC

FEATS = ["src", "rk", "score", "n_keys", "score_rel", "name_jw", "cmp_jw", "cmp_contains",
         "name_jacc", "addr_jacc", "num_shared", "s_nums", "t_nums", "t_addr_null", "pc_match"]


def feat_sql(cand, s1_src, tgt_src):
    return f"""
    WITH c AS (SELECT * FROM {cand}),
    s AS (SELECT eid(entity_id) AS id, name_core, addr_v2, postcode FROM {s1_src}
          WHERE eid(entity_id) IN (SELECT DISTINCT s1_id FROM c)),
    t AS (SELECT eid(entity_id) AS id, name_core, addr_v2, postcode FROM {tgt_src}
          WHERE eid(entity_id) IN (SELECT DISTINCT tgt_id FROM c)),
    j AS (
        SELECT c.s1_id, c.tgt_id, c.score, c.n_keys, c.rk, c.tgt_id // {SRC} AS src,
               coalesce(s.name_core, '') AS sn, coalesce(t.name_core, '') AS tn,
               t.addr_v2 AS ta, s.postcode AS sp, t.postcode AS tp,
               string_split(coalesce(s.name_core, ''), ' ') AS snt, string_split(coalesce(t.name_core, ''), ' ') AS tnt,
               string_split(coalesce(s.addr_v2, ''), ' ')   AS sat, string_split(coalesce(t.addr_v2, ''), ' ')   AS tat,
               regexp_extract_all(coalesce(s.addr_v2, ''), '\\d+') AS snum,
               regexp_extract_all(coalesce(t.addr_v2, ''), '\\d+') AS tnum
        FROM c JOIN s ON s.id = c.s1_id JOIN t ON t.id = c.tgt_id
    )
    SELECT s1_id, tgt_id, src, rk, score, n_keys,
           score / max(score) OVER (PARTITION BY s1_id, src)                        AS score_rel,
           jaro_winkler_similarity(sn, tn)                                          AS name_jw,
           jaro_winkler_similarity(replace(sn, ' ', ''), replace(tn, ' ', ''))      AS cmp_jw,
           CASE WHEN length(sn) > 2 AND length(tn) > 2
                     AND (contains(replace(tn, ' ', ''), replace(sn, ' ', ''))
                          OR contains(replace(sn, ' ', ''), replace(tn, ' ', ''))) THEN 1 ELSE 0 END AS cmp_contains,
           len(list_intersect(snt, tnt)) / greatest(len(list_distinct(list_concat(snt, tnt))), 1) AS name_jacc,
           len(list_intersect(sat, tat)) / greatest(len(list_distinct(list_concat(sat, tat))), 1) AS addr_jacc,
           len(list_intersect(snum, tnum))                                          AS num_shared,
           len(snum) AS s_nums, len(tnum) AS t_nums,
           (ta IS NULL)::INT                                                        AS t_addr_null,
           CASE WHEN sp IS NULL OR tp IS NULL THEN -1 WHEN sp = tp THEN 1 ELSE 0 END AS pc_match
    FROM j"""


def load(path):
    return lgb.Booster(model_file=str(path))


def rerank_and_cut(con, cand_tbl, s1_src, tgt_src, booster, out_tbl, tau=0.01, cap=5):
    """Score cand_tbl, keep top-`cap` per (S1, source) with p >= tau, append to out_tbl."""
    df = con.execute(feat_sql(cand_tbl, s1_src, tgt_src)).fetchdf()
    if df.empty:
        return 0
    df["p"] = booster.predict(df[FEATS])
    df = df.sort_values(["s1_id", "src", "p"], ascending=[True, True, False])
    df["rk2"] = df.groupby(["s1_id", "src"]).cumcount() + 1
    cut = df.loc[(df["rk2"] <= cap) & (df["p"] >= tau), ["s1_id", "tgt_id", "src", "p", "rk2"]]
    con.register("_cut_df", cut)
    con.execute(f"INSERT INTO {out_tbl} SELECT * FROM _cut_df")
    con.unregister("_cut_df")
    return len(cut)