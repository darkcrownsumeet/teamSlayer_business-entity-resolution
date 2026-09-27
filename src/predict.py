"""Full-pool scoring with the matcher, decision rule, submission writing."""

import csv
import gc
import time

import lightgbm as lgb

from src import features as F


def load(path):
    return lgb.Booster(model_file=str(path))


def score_all(con, rev_tbl, out_tbl, s1_src, tgt_src, predict_fn, n_chunks=32, log=print):
    """Matcher probability q for every row of rev_tbl, in S1-hash chunks.
    predict_fn(features_df) -> array of probabilities."""
    con.execute(f"CREATE OR REPLACE TABLE {out_tbl} (s1_id BIGINT, tgt_id BIGINT, q DOUBLE)")
    t0 = time.time()
    for j in range(n_chunks):
        con.execute(f"CREATE OR REPLACE TABLE _sc AS SELECT * FROM {rev_tbl} WHERE hash(s1_id) % {n_chunks} = {j}")
        df = F.build_features(con, "_sc", s1_src, tgt_src)
        if len(df):
            df["q"] = predict_fn(df)
            con.register("_q", df[["s1_id", "tgt_id", "q"]])
            con.execute(f"INSERT INTO {out_tbl} SELECT * FROM _q")
            con.unregister("_q")
        del df
        gc.collect()
        if (j + 1) % 8 == 0:
            log(f"   scored {j + 1}/{n_chunks}  {time.time() - t0:6.0f}s")
    con.execute("DROP TABLE IF EXISTS _sc")


def decide(con, q_tbl, out_tbl, tau, one_to_many=True):
    """Keep pairs with q >= tau; if one_to_many, each target goes only to its highest-q S1."""
    keep = "r = 1" if one_to_many else "TRUE"
    con.execute(f"""
    CREATE OR REPLACE TABLE {out_tbl} AS
    SELECT s1_id, tgt_id, q FROM (
        SELECT *, row_number() OVER (PARTITION BY tgt_id ORDER BY q DESC, s1_id) AS r
        FROM {q_tbl} WHERE q >= {tau}
    ) WHERE {keep}""")


def write_submission(con, pairs_tbl, s1_src, tgt_src, col, path):
    """One row per S1 in s1_src; comma-joined target entity_ids (empty if none)."""
    df = con.execute(f"""
    WITH p AS (SELECT DISTINCT s1_id, tgt_id FROM {pairs_tbl}),
         tid AS (SELECT eid(entity_id) AS id, entity_id FROM {tgt_src}
                 WHERE eid(entity_id) IN (SELECT tgt_id FROM p)),
         agg AS (SELECT p.s1_id, string_agg(tid.entity_id, ',' ORDER BY tid.entity_id) AS ids
                 FROM p JOIN tid ON tid.id = p.tgt_id GROUP BY 1)
    SELECT s.entity_id AS source1_entity_id, coalesce(a.ids, '') AS {col}
    FROM {s1_src} s LEFT JOIN agg a ON a.s1_id = eid(s.entity_id)
    ORDER BY s.entity_id""").fetchdf()
    df.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_NONE)
    return df