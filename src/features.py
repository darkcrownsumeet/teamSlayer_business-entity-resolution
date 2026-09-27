"""Matcher features for (S1, target) candidate pairs from the reranked cut.

Required tables (build once per dataset): name_idf, addr_idf, tok_odds, raw_names.
"""

import re
import time

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cpdist
from rapidfuzz.distance import JaroWinkler, Levenshtein

from src import normalize as N

# v1 features (matcher_lgb.txt)
FEATS = [
    "src", "p", "rk2", "own_max_p", "own_sum_p", "own_n", "own_n_hi", "p_gap_own",
    "rev_rk", "n_claim", "tgt_max_p", "tgt_second_p", "other_best_p", "p_margin",
    "xs_name_jw", "xs_addr_jacc",
    "name_jw", "cmp_jw", "cmp_contains", "name_jacc", "first_tok_eq", "s_len", "t_len", "len_ratio",
    "tsort", "tset", "cmp_partial", "skel_ratio", "legal_cmp", "name_num_cmp",
    "addr_jacc", "addr_tset", "num_shared", "s_nums", "t_nums", "s_addr_null", "t_addr_null", "pc_match",
]
# v2 additions (matcher_xgb_v2.json)
FEATS_V2 = FEATS + [
    "name_idf_jacc", "idf_shared", "idf_unshared_max", "name_n_unshared",
    "wratio", "cmp_lev", "cmp_prefix",
    "first_num_cmp", "addr_last_eq", "addr_jw", "addr_alpha_jacc",
]
# v3 additions: number-difference shape, twins / distractors, duplicated-token artifacts
FEATS_V3 = FEATS_V2 + [
    "num_absdiff_log", "num_lev", "num_trunc",
    "xs_num_eq", "n_twins", "addr_rank", "name_rank",
    "t_dup_toks", "s_dup_toks",
]
# v4 additions: distractor signatures (token log-odds, legal-form transition), IDF address overlap, raw format
# v4 features are implemented but NOT used by the submitted model (future work). Submitted model uses FEATS_V3.
FEATS_V4 = FEATS_V3 + [
    "add_lo_max", "add_lo_min", "add_lo_sum", "n_added",
    "drop_lo_max", "drop_lo_min", "drop_lo_sum", "n_dropped",
    "lf_same", "lf_dropped", "lf_added", "lf_subset", "lf_changed",
    "addr_idf_shared", "addr_idf_jacc", "addr_idf_unshared_max",
    "t_all_caps", "t_mixed_case", "t_lower", "t_symbols", "s_all_caps",
    "raw_name_eq_ci", "raw_len_diff",
    "t_has_alias", "s_has_alias", "alias_best_tset", "alias_best_jw",
]

LEGAL = N.LEGAL_FORMS - {"and"}
UNSEEN_IDF = 15.0
ALIAS_RE = re.compile(r"\b(?:d/b/a|dba|doing business as|a/k/a|aka|also known as|f/k/a|fka|"
                      r"formerly known as|formerly|t/a|trading as|nee)\b")
HELPER_COLS = ["s_name", "t_name", "sn", "tn", "sa", "ta", "sp", "tp", "cty", "sn1", "tn1", "s_raw", "t_raw"]


# ============================================================
# Macros and one-off tables
# ============================================================
def setup_macros(con):
    con.execute("""CREATE OR REPLACE MACRO tl(x) AS
        list_distinct(list_filter(string_split(coalesce(x, ''), ' '), y -> y <> ''))""")
    con.execute("""CREATE OR REPLACE MACRO jac(a, b) AS
        CASE WHEN len(tl(a)) = 0 OR len(tl(b)) = 0 THEN NULL
             ELSE len(list_intersect(tl(a), tl(b))) / len(list_distinct(list_concat(tl(a), tl(b)))) END""")
    con.execute(r"CREATE OR REPLACE MACRO nums(x) AS list_distinct(regexp_extract_all(coalesce(x, ''), '\d+'))")


def build_rev(con, cut_tbl, out_tbl):
    """Competition context per target across ALL S1 in cut_tbl."""
    con.execute(f"""
    CREATE OR REPLACE TABLE {out_tbl} AS
    SELECT *,
           row_number() OVER w_ord                   AS rev_rk,
           count(*)     OVER w_all                   AS n_claim,
           max(p)       OVER w_all                   AS tgt_max_p,
           coalesce(nth_value(p, 2) OVER w_full, 0)  AS tgt_second_p
    FROM {cut_tbl}
    WINDOW w_ord  AS (PARTITION BY tgt_id ORDER BY p DESC, s1_id),
           w_all  AS (PARTITION BY tgt_id),
           w_full AS (PARTITION BY tgt_id ORDER BY p DESC, s1_id
                      ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)
    """)


def build_name_idf(con, tgt_src, out_tbl="name_idf"):
    """IDF of name_core tokens over the target pool, per country."""
    con.execute(f"""
    CREATE OR REPLACE TABLE {out_tbl} AS
    WITH t AS (SELECT country_key AS c, unnest(tl(name_core)) AS tok FROM {tgt_src}),
         n AS (SELECT country_key AS c, count(*) AS n FROM {tgt_src} GROUP BY 1)
    SELECT t.c, t.tok, ln(n.n / count(*)) AS idf
    FROM t JOIN n USING (c) GROUP BY t.c, t.tok, n.n
    """)


def build_addr_idf(con, tgt_src, out_tbl="addr_idf"):
    """IDF of addr_v2 tokens over the target pool, per country."""
    con.execute(f"""
    CREATE OR REPLACE TABLE {out_tbl} AS
    WITH t AS (SELECT country_key AS c, unnest(tl(addr_v2)) AS tok FROM {tgt_src} WHERE addr_v2 IS NOT NULL),
         n AS (SELECT country_key AS c, count(*) AS n FROM {tgt_src} GROUP BY 1)
    SELECT t.c, t.tok, ln(n.n / count(*)) AS idf
    FROM t JOIN n USING (c) GROUP BY t.c, t.tok, n.n
    """)


def build_raw_names(con, raw_s1_src, raw_tgt_src, out_tbl="raw_names"):
    """Raw (un-normalized) business names keyed by integer id, for format features."""
    con.execute(f"""
    CREATE OR REPLACE TABLE {out_tbl} AS
    SELECT eid(entity_id) AS id, business_name AS raw FROM {raw_s1_src}
    UNION ALL
    SELECT eid(entity_id) AS id, business_name AS raw FROM {raw_tgt_src}
    """)


def build_token_odds(con, rows_tbl, s1_src, tgt_src, out_tbl="tok_odds", alpha=1.0, min_n=5, hard_p=0.3):
    """Log-odds (match vs hard non-match) of name tokens ADDED in the target / DROPPED from S1.
    rows_tbl: candidate rows with label y and reranker p, from S1 NOT used for matcher training."""
    con.execute(f"""
    CREATE OR REPLACE TABLE {out_tbl} AS
    WITH r AS (SELECT r.s1_id, r.tgt_id, r.y, s.name_v2 AS sn, t.name_v2 AS tn
               FROM (SELECT * FROM {rows_tbl} WHERE y = 1 OR p >= {hard_p}) r
               JOIN (SELECT eid(entity_id) AS id, name_v2 FROM {s1_src}) s ON s.id = r.s1_id
               JOIN (SELECT eid(entity_id) AS id, name_v2 FROM {tgt_src}) t ON t.id = r.tgt_id),
    tt AS (SELECT s1_id, tgt_id, y, unnest(tl(tn)) AS tok FROM r),
    st AS (SELECT s1_id, tgt_id, y, unnest(tl(sn)) AS tok FROM r),
    ev AS (SELECT 'add' AS kind, tok, y FROM (SELECT * FROM tt ANTI JOIN st USING (s1_id, tgt_id, tok))
           UNION ALL
           SELECT 'drop', tok, y FROM (SELECT * FROM st ANTI JOIN tt USING (s1_id, tgt_id, tok))),
    tot AS (SELECT sum(y) AS n1, sum(1 - y) AS n0 FROM r)
    SELECT kind, tok,
           ln((sum(y) + {alpha}) / (tot.n1 + {alpha})) - ln((sum(1 - y) + {alpha}) / (tot.n0 + {alpha})) AS lo,
           count(*) AS n
    FROM ev, tot
    GROUP BY kind, tok, tot.n1, tot.n0
    HAVING count(*) >= {min_n}
    """)


# ============================================================
# Pair features
# ============================================================
def sql_features(cand, s1_src, tgt_src, idf_tbl="name_idf", addr_idf_tbl="addr_idf",
                 odds_tbl="tok_odds", raw_tbl="raw_names"):
    """cand must contain ALL candidate rows of its S1s (columns of the rev table)."""
    return rf"""
    WITH c AS (SELECT * FROM {cand}),
    s AS (SELECT eid(entity_id) AS id, country_key, name_v2, name_core, addr_v2, postcode FROM {s1_src}
          WHERE eid(entity_id) IN (SELECT DISTINCT s1_id FROM c)),
    t AS (SELECT eid(entity_id) AS id, name_v2, name_core, addr_v2, postcode FROM {tgt_src}
          WHERE eid(entity_id) IN (SELECT DISTINCT tgt_id FROM c)),
    rw AS (SELECT id, raw FROM {raw_tbl}
           WHERE id IN (SELECT DISTINCT s1_id FROM c) OR id IN (SELECT DISTINCT tgt_id FROM c)),
    own AS (SELECT s1_id, max(p) AS own_max_p, sum(p) AS own_sum_p, count(*) AS own_n,
                   count(*) FILTER (WHERE p >= 0.5) AS own_n_hi
            FROM c GROUP BY 1),
    xs AS (SELECT a.s1_id, a.tgt_id,
                  max(jaro_winkler_similarity(coalesce(ta.name_core, ''), coalesce(tb.name_core, ''))) AS xs_name_jw,
                  max(jac(ta.addr_v2, tb.addr_v2)) AS xs_addr_jacc,
                  max((regexp_extract(coalesce(ta.addr_v2, ''), '\d+') <> ''
                       AND regexp_extract(coalesce(ta.addr_v2, ''), '\d+')
                         = regexp_extract(coalesce(tb.addr_v2, ''), '\d+'))::INT) AS xs_num_eq
           FROM c a
           JOIN c b ON a.s1_id = b.s1_id AND a.tgt_id <> b.tgt_id AND b.p >= 0.5
           JOIN t ta ON ta.id = a.tgt_id
           JOIN t tb ON tb.id = b.tgt_id
           GROUP BY 1, 2),
    tw AS (SELECT a.s1_id, a.tgt_id,
                  count(*) FILTER (WHERE jaro_winkler_similarity(coalesce(ta.name_core, ''),
                                                                 coalesce(tb.name_core, '')) >= 0.95) AS n_twins
           FROM c a
           JOIN c b ON a.s1_id = b.s1_id AND a.tgt_id <> b.tgt_id
           JOIN t ta ON ta.id = a.tgt_id
           JOIN t tb ON tb.id = b.tgt_id
           GROUP BY 1, 2),
    j AS (SELECT c.s1_id, c.tgt_id, c.src, c.p, c.rk2, c.rev_rk, c.n_claim, c.tgt_max_p, c.tgt_second_p,
                 o.own_max_p, o.own_sum_p, o.own_n, o.own_n_hi,
                 x.xs_name_jw, x.xs_addr_jacc, coalesce(x.xs_num_eq, -1) AS xs_num_eq,
                 coalesce(w.n_twins, 0) AS n_twins,
                 s.country_key AS cty,
                 coalesce(s.name_v2, '')   AS s_name, coalesce(t.name_v2, '')   AS t_name,
                 coalesce(s.name_core, '') AS sn,     coalesce(t.name_core, '') AS tn,
                 coalesce(s.addr_v2, '')   AS sa,     coalesce(t.addr_v2, '')   AS ta,
                 coalesce(rs.raw, '')      AS s_raw,  coalesce(rt.raw, '')      AS t_raw,
                 regexp_extract(coalesce(s.addr_v2, ''), '\d+') AS sn1,
                 regexp_extract(coalesce(t.addr_v2, ''), '\d+') AS tn1,
                 (s.addr_v2 IS NULL)::INT  AS s_addr_null, (t.addr_v2 IS NULL)::INT AS t_addr_null,
                 s.postcode AS sp, t.postcode AS tp
          FROM c
          JOIN own o USING (s1_id)
          LEFT JOIN xs x USING (s1_id, tgt_id)
          LEFT JOIN tw w USING (s1_id, tgt_id)
          JOIN s ON s.id = c.s1_id
          JOIN t ON t.id = c.tgt_id
          LEFT JOIN rw rs ON rs.id = c.s1_id
          LEFT JOIN rw rt ON rt.id = c.tgt_id),
    -- name-token IDF overlap
    tk AS (SELECT s1_id, tgt_id, cty, tok,
                  list_contains(tl(sn), tok) AND list_contains(tl(tn), tok) AS shared
           FROM (SELECT s1_id, tgt_id, cty, sn, tn,
                        unnest(list_distinct(list_concat(tl(sn), tl(tn)))) AS tok FROM j)),
    idf AS (SELECT tk.s1_id, tk.tgt_id,
                   sum(coalesce(i.idf, {UNSEEN_IDF})) FILTER (WHERE shared)      AS idf_shared,
                   sum(coalesce(i.idf, {UNSEEN_IDF}))                            AS idf_total,
                   max(coalesce(i.idf, {UNSEEN_IDF})) FILTER (WHERE NOT shared)  AS idf_unshared_max,
                   count(*) FILTER (WHERE NOT shared)                            AS name_n_unshared
            FROM tk LEFT JOIN {idf_tbl} i ON i.c = tk.cty AND i.tok = tk.tok
            GROUP BY 1, 2),
    -- address-token IDF overlap
    atk AS (SELECT s1_id, tgt_id, cty, tok,
                   list_contains(tl(sa), tok) AND list_contains(tl(ta), tok) AS shared
            FROM (SELECT s1_id, tgt_id, cty, sa, ta,
                         unnest(list_distinct(list_concat(tl(sa), tl(ta)))) AS tok FROM j)),
    aidf AS (SELECT atk.s1_id, atk.tgt_id,
                    sum(coalesce(i.idf, {UNSEEN_IDF})) FILTER (WHERE shared)     AS addr_idf_shared,
                    sum(coalesce(i.idf, {UNSEEN_IDF}))                           AS addr_idf_total,
                    max(coalesce(i.idf, {UNSEEN_IDF})) FILTER (WHERE NOT shared) AS addr_idf_unshared_max
             FROM atk LEFT JOIN {addr_idf_tbl} i ON i.c = atk.cty AND i.tok = atk.tok
             GROUP BY 1, 2),
    -- added / dropped name tokens (full name_v2) with learned distractor log-odds
    ttk AS (SELECT s1_id, tgt_id, unnest(tl(t_name)) AS tok FROM j),
    stk AS (SELECT s1_id, tgt_id, unnest(tl(s_name)) AS tok FROM j),
    addt AS (SELECT a.s1_id, a.tgt_id, coalesce(o.lo, 0) AS lo
             FROM (SELECT * FROM ttk ANTI JOIN stk USING (s1_id, tgt_id, tok)) a
             LEFT JOIN {odds_tbl} o ON o.kind = 'add' AND o.tok = a.tok),
    dropt AS (SELECT d.s1_id, d.tgt_id, coalesce(o.lo, 0) AS lo
              FROM (SELECT * FROM stk ANTI JOIN ttk USING (s1_id, tgt_id, tok)) d
              LEFT JOIN {odds_tbl} o ON o.kind = 'drop' AND o.tok = d.tok),
    ao AS (SELECT s1_id, tgt_id, max(lo) AS add_lo_max, min(lo) AS add_lo_min,
                  sum(lo) AS add_lo_sum, count(*) AS n_added FROM addt GROUP BY 1, 2),
    do_ AS (SELECT s1_id, tgt_id, max(lo) AS drop_lo_max, min(lo) AS drop_lo_min,
                   sum(lo) AS drop_lo_sum, count(*) AS n_dropped FROM dropt GROUP BY 1, 2)
    SELECT j.*,
           coalesce(idf.idf_shared, 0)                                     AS idf_shared,
           coalesce(idf.idf_shared, 0) / nullif(idf.idf_total, 0)          AS name_idf_jacc,
           coalesce(idf.idf_unshared_max, 0)                               AS idf_unshared_max,
           coalesce(idf.name_n_unshared, 0)                                AS name_n_unshared,
           coalesce(aidf.addr_idf_shared, 0)                               AS addr_idf_shared,
           coalesce(aidf.addr_idf_shared, 0) / nullif(aidf.addr_idf_total, 0) AS addr_idf_jacc,
           coalesce(aidf.addr_idf_unshared_max, 0)                         AS addr_idf_unshared_max,
           coalesce(ao.add_lo_max, 0)  AS add_lo_max,  coalesce(ao.add_lo_min, 0)  AS add_lo_min,
           coalesce(ao.add_lo_sum, 0)  AS add_lo_sum,  coalesce(ao.n_added, 0)     AS n_added,
           coalesce(do_.drop_lo_max, 0) AS drop_lo_max, coalesce(do_.drop_lo_min, 0) AS drop_lo_min,
           coalesce(do_.drop_lo_sum, 0) AS drop_lo_sum, coalesce(do_.n_dropped, 0)   AS n_dropped,
           CASE WHEN rev_rk = 1 THEN tgt_second_p ELSE tgt_max_p END      AS other_best_p,
           p - CASE WHEN rev_rk = 1 THEN tgt_second_p ELSE tgt_max_p END  AS p_margin,
           p - own_max_p                                                  AS p_gap_own,
           jaro_winkler_similarity(sn, tn)                                AS name_jw,
           jaro_winkler_similarity(replace(sn, ' ', ''), replace(tn, ' ', '')) AS cmp_jw,
           CASE WHEN length(sn) > 2 AND length(tn) > 2
                     AND (contains(replace(tn, ' ', ''), replace(sn, ' ', ''))
                          OR contains(replace(sn, ' ', ''), replace(tn, ' ', ''))) THEN 1 ELSE 0 END AS cmp_contains,
           jac(sn, tn)                                                    AS name_jacc,
           jac(sa, ta)                                                    AS addr_jacc,
           jac(regexp_replace(sa, '\d+', '', 'g'), regexp_replace(ta, '\d+', '', 'g')) AS addr_alpha_jacc,
           jaro_winkler_similarity(sa, ta)                                AS addr_jw,
           len(list_intersect(nums(sa), nums(ta)))                        AS num_shared,
           len(nums(sa)) AS s_nums, len(nums(ta)) AS t_nums,
           CASE WHEN sn1 = '' OR tn1 = '' THEN -1 WHEN sn1 = tn1 THEN 1 ELSE 0 END AS first_num_cmp,
           CASE WHEN sn1 = '' OR tn1 = '' THEN NULL
                ELSE ln(1 + abs(TRY_CAST(sn1 AS DOUBLE) - TRY_CAST(tn1 AS DOUBLE))) END AS num_absdiff_log,
           CASE WHEN sn1 = '' OR tn1 = '' THEN NULL ELSE levenshtein(sn1, tn1) END AS num_lev,
           CASE WHEN sn1 = '' OR tn1 = '' THEN -1
                WHEN sn1 <> tn1 AND (starts_with(sn1, tn1) OR starts_with(tn1, sn1)
                                     OR ends_with(sn1, tn1) OR ends_with(tn1, sn1)) THEN 1 ELSE 0 END AS num_trunc,
           CASE WHEN sa = '' OR ta = '' THEN -1
                WHEN string_split(sa, ' ')[-1] = string_split(ta, ' ')[-1] THEN 1 ELSE 0 END AS addr_last_eq,
           rank() OVER (PARTITION BY j.s1_id ORDER BY jaro_winkler_similarity(sa, ta) DESC) AS addr_rank,
           rank() OVER (PARTITION BY j.s1_id ORDER BY jaro_winkler_similarity(sn, tn) DESC) AS name_rank,
           CASE WHEN t_name = '' THEN 0 ELSE len(string_split(t_name, ' ')) - len(tl(t_name)) END AS t_dup_toks,
           CASE WHEN s_name = '' THEN 0 ELSE len(string_split(s_name, ' ')) - len(tl(s_name)) END AS s_dup_toks,
           -- raw formatting (A1: distractors carry less formatting noise)
           (t_raw <> '' AND t_raw = upper(t_raw) AND t_raw <> lower(t_raw))::INT  AS t_all_caps,
           (t_raw <> upper(t_raw) AND t_raw <> lower(t_raw))::INT                 AS t_mixed_case,
           (t_raw <> '' AND t_raw = lower(t_raw) AND t_raw <> upper(t_raw))::INT  AS t_lower,
           regexp_matches(t_raw, '[()\[\]#*]')::INT                               AS t_symbols,
           (s_raw <> '' AND s_raw = upper(s_raw) AND s_raw <> lower(s_raw))::INT  AS s_all_caps,
           (lower(s_raw) = lower(t_raw))::INT                                     AS raw_name_eq_ci,
           length(t_raw) - length(s_raw)                                          AS raw_len_diff,
           CASE WHEN sp IS NULL OR tp IS NULL THEN -1 WHEN sp = tp THEN 1 ELSE 0 END AS pc_match,
           (split_part(sn, ' ', 1) = split_part(tn, ' ', 1))::INT         AS first_tok_eq,
           length(sn) AS s_len, length(tn) AS t_len
    FROM j
    LEFT JOIN idf  USING (s1_id, tgt_id)
    LEFT JOIN aidf USING (s1_id, tgt_id)
    LEFT JOIN ao   USING (s1_id, tgt_id)
    LEFT JOIN do_  USING (s1_id, tgt_id)"""


def _cmp3(a, b):
    """-1: one side empty, 0: both present and disjoint (conflict), 1: overlap."""
    if not a or not b:
        return -1
    return 1 if a & b else 0


def _legal_transition(a, b):
    """Exact legal-form transition S1 -> target: none / same / dropped / added / subset / changed."""
    sa = tuple(t for t in a.split() if t in LEGAL)
    tb = tuple(t for t in b.split() if t in LEGAL)
    if not sa and not tb:
        return "none"
    if sa == tb:
        return "same"
    if sa and not tb:
        return "dropped"
    if tb and not sa:
        return "added"
    if set(sa) < set(tb) or set(tb) < set(sa):
        return "subset"
    return "changed"


def _prefix_share(a, b):
    if not a or not b:
        return 0.0
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n / min(len(a), len(b))

def _alias_parts(name):
    parts = [p.strip() for p in ALIAS_RE.split(name) if p and p.strip()]
    return parts or [name]


def py_features(df):
    sn, tn = df["sn"].tolist(), df["tn"].tolist()
    sa, ta = df["sa"].tolist(), df["ta"].tolist()
    sc, tc = [x.replace(" ", "") for x in sn], [x.replace(" ", "") for x in tn]

    df["tsort"] = cpdist(sn, tn, scorer=fuzz.token_sort_ratio, workers=-1)
    df["tset"] = cpdist(sn, tn, scorer=fuzz.token_set_ratio, workers=-1)
    df["wratio"] = cpdist(sn, tn, scorer=fuzz.WRatio, workers=-1)
    df["cmp_partial"] = cpdist(sc, tc, scorer=fuzz.partial_ratio, workers=-1)
    df["cmp_lev"] = cpdist(sc, tc, scorer=Levenshtein.normalized_similarity, workers=-1)
    df["addr_tset"] = cpdist(sa, ta, scorer=fuzz.token_set_ratio, workers=-1)
    df["cmp_prefix"] = [_prefix_share(a, b) for a, b in zip(sc, tc)]

    skel = {u: (N.skeleton(u) or "") for u in set(sn) | set(tn)}
    df["skel_ratio"] = cpdist([skel[x] for x in sn], [skel[x] for x in tn], scorer=fuzz.ratio, workers=-1)

    sl = [set(x.split()) & LEGAL for x in df["s_name"]]
    tl = [set(x.split()) & LEGAL for x in df["t_name"]]
    df["legal_cmp"] = [_cmp3(a, b) for a, b in zip(sl, tl)]

    lt = [_legal_transition(a, b) for a, b in zip(df["s_name"], df["t_name"])]
    for k in ("same", "dropped", "added", "subset", "changed"):
        df[f"lf_{k}"] = [int(x == k) for x in lt]

    num_re = re.compile(r"\d+")
    df["name_num_cmp"] = [_cmp3(set(num_re.findall(a)), set(num_re.findall(b)))
                          for a, b in zip(df["s_name"], df["t_name"])]

    ls, lt_ = df["s_len"].to_numpy(float), df["t_len"].to_numpy(float)
    df["len_ratio"] = np.where(np.maximum(ls, lt_) > 0, np.minimum(ls, lt_) / np.maximum(ls, lt_), 0.0)

        # alias-aware similarity: 'x dba y' -> best-matching part vs the other side
    s_alias = [bool(ALIAS_RE.search(x)) for x in df["s_name"]]
    t_alias = [bool(ALIAS_RE.search(x)) for x in df["t_name"]]
    df["s_has_alias"] = np.array(s_alias, dtype=int)
    df["t_has_alias"] = np.array(t_alias, dtype=int)
    best_tset = df["tset"].to_numpy(float).copy()
    best_jw = cpdist(df["s_name"].tolist(), df["t_name"].tolist(),
                     scorer=JaroWinkler.normalized_similarity, workers=-1).astype(float)
    for i in np.flatnonzero(np.array(s_alias) | np.array(t_alias)):
        sp, tp = _alias_parts(df["s_name"].iat[i]), _alias_parts(df["t_name"].iat[i])
        best_tset[i] = max(fuzz.token_set_ratio(a, b) for a in sp for b in tp)
        best_jw[i] = max(JaroWinkler.normalized_similarity(a, b) for a in sp for b in tp)
    df["alias_best_tset"] = best_tset
    df["alias_best_jw"] = best_jw
    
    return df.drop(columns=[c for c in HELPER_COLS if c in df.columns])


def build_features(con, cand_tbl, s1_src, tgt_src, **tbls):
    df = con.execute(sql_features(cand_tbl, s1_src, tgt_src, **tbls)).fetchdf()
    return py_features(df) if len(df) else df


def build_features_chunked(con, rows_tbl, s1_src, tgt_src, n_chunks=16, feats=None,
                           extra=("y",), log=print, **tbls):
    """Features for rows_tbl in S1-hash chunks; keeps ids + feats (+ extra cols if present) as float32."""
    feats = feats or FEATS_V4
    parts, t0 = [], time.time()
    for j in range(n_chunks):
        con.execute(f"CREATE OR REPLACE TABLE _fc AS SELECT * FROM {rows_tbl} WHERE hash(s1_id) % {n_chunks} = {j}")
        df = build_features(con, "_fc", s1_src, tgt_src, **tbls)
        if len(df):
            keep = ["s1_id", "tgt_id"] + feats + [e for e in extra if e in df.columns]
            df = df[keep].copy()
            df[feats] = df[feats].astype("float32")
            parts.append(df)
        if (j + 1) % 4 == 0:
            log(f"   features {j + 1}/{n_chunks}  {time.time() - t0:6.0f}s")
    con.execute("DROP TABLE IF EXISTS _fc")
    return pd.concat(parts, ignore_index=True)