"""Official challenge metric: per-S1 F0.5, macro-averaged, singletons included."""


def eval_f05(con, pred, label, s1_tbl="eval_s1", pairs_tbl="eval_pairs"):
    return con.execute(f"""
    WITH pr AS (SELECT DISTINCT s1_id, tgt_id FROM {pred}),
         nt AS (SELECT s1_id, count(*) AS nt FROM {pairs_tbl} GROUP BY 1),
         np AS (SELECT s1_id, count(*) AS np FROM pr GROUP BY 1),
         tp AS (SELECT p.s1_id, count(*) AS tp FROM pr p JOIN {pairs_tbl} g USING (s1_id, tgt_id) GROUP BY 1),
         per AS (SELECT e.s1_id, coalesce(nt, 0) nt, coalesce(np, 0) np, coalesce(tp, 0) tp
                 FROM {s1_tbl} e LEFT JOIN nt USING (s1_id) LEFT JOIN np USING (s1_id) LEFT JOIN tp USING (s1_id)),
         f AS (SELECT *, CASE WHEN nt = 0 AND np = 0 THEN 1.0
                              WHEN tp = 0 THEN 0.0
                              ELSE 1.25 * (tp / np) * (tp / nt) / (0.25 * (tp / np) + (tp / nt)) END AS f05
               FROM per)
    SELECT '{label}' AS method,
           avg(f05)                           AS macro_f05,
           avg(CASE WHEN nt = 0 THEN f05 END) AS f05_singletons,
           avg(CASE WHEN nt > 0 THEN f05 END) AS f05_matched,
           sum(tp) / greatest(sum(np), 1)     AS micro_precision,
           sum(tp) / sum(nt)                  AS micro_recall,
           avg(np)                            AS avg_pred
    FROM f""").fetchdf()