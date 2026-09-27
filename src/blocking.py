"""
Combined-key blocking in DuckDB.

Keys per record (within country):
  name x addr, name x name, addr x addr pairs built from each record's rarest tokens
  (NAME_TOP name tokens, ADDR_TOP address tokens; rarity = df over targets, restricted
  to tokens that also occur in the S1 vocabulary), plus exact name_core and compact-name keys.
Keys with target block size > KEY_CAP are dropped. Candidates are ranked per (S1, source)
by the sum of key IDF weights and cut at K_MAX.

Memory is bounded at any scale:
  - targets processed per source in id-hash chunks (n_chunks)
  - S1 processed in id-hash chunks (s1_chunks) for key building and scoring
  - large DISTINCT / GROUP BY on keys done in key-hash partitions (KEY_PARTS)
"""

import time

SRC = 100_000_000_000  # eid(): 'S2-123' -> 2*SRC + 123 ; source = id // SRC
KEY_PARTS = 8          # key-hash partitions for large key aggregates


def setup(con, threads=4, memory_limit="4GB", temp_dir=None):
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute("SET preserve_insertion_order=false")
    if temp_dir:
        con.execute(f"SET temp_directory='{temp_dir}'")
    con.execute(f"""CREATE OR REPLACE MACRO eid(x) AS
        (CASE left(x, 2) WHEN 'S1' THEN 1 WHEN 'S2' THEN 2 ELSE 3 END) * {SRC}
        + CAST(substr(x, 4) AS BIGINT)""")
    con.execute("""CREATE OR REPLACE MACRO toks(txt) AS
        list_distinct(list_filter(string_split(coalesce(txt, ''), ' '),
                                  x -> length(x) >= 2 OR regexp_full_match(x, '[0-9]+')))""")


def tok_stream(src, c):
    return f"""
    SELECT eid(entity_id) AS id, 'n' AS f, hash(tok) AS h
    FROM (SELECT entity_id, unnest(toks(name_core)) AS tok FROM {src} WHERE country_key = '{c}')
    UNION ALL
    SELECT eid(entity_id), 'a', hash(tok)
    FROM (SELECT entity_id, unnest(toks(addr_v2)) AS tok FROM {src} WHERE country_key = '{c}')"""


def rec_sql(src, c, name_top, addr_top):
    return f"""
    SELECT t.id,
           arg_min(t.h, (d.df::HUGEINT << 64) + t.h::HUGEINT, {name_top}) FILTER (WHERE t.f = 'n') AS ns,
           arg_min(t.h, (d.df::HUGEINT << 64) + t.h::HUGEINT, {addr_top}) FILTER (WHERE t.f = 'a') AS ad
    FROM ({tok_stream(src, c)}) t
    JOIN df_c d ON t.f = d.f AND t.h = d.h
    GROUP BY t.id"""


def pair_sql(rec_tbl, tag, l1, l2, cond):
    return f"""
    SELECT id, hash(concat_ws('|', '{tag}', x::VARCHAR, y::VARCHAR)) AS k
    FROM (SELECT id, x, unnest(l2) AS y
          FROM (SELECT id, unnest({l1}) AS x, {l2} AS l2 FROM {rec_tbl}
                WHERE {l1} IS NOT NULL AND {l2} IS NOT NULL))
    WHERE {cond}"""


def keys_sql(src, rec_tbl, c):
    cmp_ = "replace(name_core, ' ', '')"
    base = f"FROM {src} WHERE country_key = '{c}' AND name_core IS NOT NULL"
    return " UNION ALL ".join([
        # combined token keys from each record's rarest name / address tokens
        pair_sql(rec_tbl, "na", "ns", "ad", "TRUE"),
        pair_sql(rec_tbl, "nn", "ns", "ns", "x < y"),
        pair_sql(rec_tbl, "aa", "ad", "ad", "x < y"),
        # exact core name / compact name / sorted-token core (word order)
        f"SELECT eid(entity_id) AS id, hash(concat_ws('|', 'core', name_core)) AS k {base}",
        f"SELECT eid(entity_id) AS id, hash(concat_ws('|', 'cmp', {cmp_})) AS k {base}",
        f"""SELECT eid(entity_id) AS id,
                   hash(concat_ws('|', 'srt', array_to_string(list_sort(tl(name_core)), ' '))) AS k {base}""",
        # typo / truncation channel WITH first address number (small blocks)
        f"""SELECT eid(entity_id) AS id,
                   hash(concat_ws('|', 'p6', left({cmp_}, 6), coalesce(regexp_extract(addr_v2, '\\d+'), ''))) AS k
            {base} AND length({cmp_}) >= 6""",
        f"""SELECT eid(entity_id) AS id,
                   hash(concat_ws('|', 's6', right({cmp_}, 6), coalesce(regexp_extract(addr_v2, '\\d+'), ''))) AS k
            {base} AND length({cmp_}) >= 6""",
        # typo / truncation channel WITHOUT number (name-only targets; longer affix keeps blocks small)
        f"""SELECT eid(entity_id) AS id, hash(concat_ws('|', 'p8', left({cmp_}, 8))) AS k
            {base} AND length({cmp_}) >= 10""",
        f"""SELECT eid(entity_id) AS id, hash(concat_ws('|', 's8', right({cmp_}, 8))) AS k
            {base} AND length({cmp_}) >= 10""",
    ])

def run_blocking(con, out_tbl, s1_all, tgt_parts, s1_ids_tbl=None,
                 name_top=3, addr_top=4, key_cap=300, k_max=50, n_chunks=16,
                 s1_chunks=1, on_chunk=None, log=print):
    """Candidates for S1 records (all of s1_all, or only ids in s1_ids_tbl.s1_id).

    s1_chunks > 1 : S1 side processed in id-hash chunks (needed for millions of S1).
    on_chunk      : if given, each chunk's top-k_max candidates are written to table
                    'cand_chunk' and on_chunk('cand_chunk') is called (e.g. rerank + cut);
                    otherwise they are appended to out_tbl.
    """
    tgt = "(" + " UNION ALL ".join(f"SELECT * FROM {p}" for p in tgt_parts) + ")"
    s1 = s1_all if s1_ids_tbl is None else \
        f"(SELECT k.* FROM {s1_all} k SEMI JOIN {s1_ids_tbl} e ON eid(k.entity_id) = e.s1_id)"
    cand_cols = "(s1_id BIGINT, tgt_id BIGINT, score DOUBLE, n_keys BIGINT, rk BIGINT)"
    if on_chunk is None:
        con.execute(f"CREATE OR REPLACE TABLE {out_tbl} {cand_cols}")

    t0 = time.time()
    countries = [r[0] for r in con.execute(f"SELECT DISTINCT country_key FROM {s1}").fetchall()]
    for c in countries:
        # 1. token df over targets, restricted to S1 vocabulary
        con.execute(f"CREATE OR REPLACE TABLE s1_vocab AS SELECT DISTINCT f, h FROM ({tok_stream(s1_all, c)})")
        con.execute(f"""CREATE OR REPLACE TABLE df_c AS
            SELECT t.f, t.h, count(*) AS df FROM ({tok_stream(tgt, c)}) t
            SEMI JOIN s1_vocab v ON t.f = v.f AND t.h = v.h GROUP BY ALL""")
        log(f"   {out_tbl} {c:8s} token df     {time.time() - t0:7.0f}s")

        # 2. S1 keys (chunked over S1), distinct key set (partitioned over key hash)
        con.execute("CREATE OR REPLACE TABLE s_keys (id BIGINT, k UBIGINT)")
        for j in range(s1_chunks):
            s1c = s1 if s1_chunks == 1 else f"(SELECT * FROM {s1} WHERE hash(entity_id) % {s1_chunks} = {j})"
            con.execute(f"CREATE OR REPLACE TABLE s_rec AS {rec_sql(s1c, c, name_top, addr_top)}")
            con.execute(f"INSERT INTO s_keys SELECT DISTINCT * FROM ({keys_sql(s1c, 's_rec', c)})")
        con.execute("CREATE OR REPLACE TABLE s_kset (k UBIGINT)")
        for i in range(KEY_PARTS):
            con.execute(f"INSERT INTO s_kset SELECT DISTINCT k FROM s_keys WHERE k % {KEY_PARTS} = {i}")
        log(f"   {out_tbl} {c:8s} S1 keys      {time.time() - t0:7.0f}s")

        # 3. target keys (chunked), only keys S1 also has
        con.execute("CREATE OR REPLACE TABLE t_keys (id BIGINT, k UBIGINT)")
        for part in tgt_parts:
            for i in range(n_chunks):
                chunk = f"(SELECT * FROM {part} WHERE hash(entity_id) % {n_chunks} = {i})"
                con.execute(f"CREATE OR REPLACE TABLE t_rec AS {rec_sql(chunk, c, name_top, addr_top)}")
                con.execute(f"""INSERT INTO t_keys
                    SELECT t.id, t.k FROM ({keys_sql(chunk, 't_rec', c)}) t
                    SEMI JOIN s_kset s ON t.k = s.k""")
        log(f"   {out_tbl} {c:8s} target keys  {time.time() - t0:7.0f}s")

        # 3b. key block sizes (partitioned over key hash), drop blocks larger than key_cap
        con.execute("CREATE OR REPLACE TABLE key_w (k UBIGINT, df BIGINT)")
        for i in range(KEY_PARTS):
            con.execute(f"""INSERT INTO key_w
                SELECT k, count(*) AS df FROM t_keys WHERE k % {KEY_PARTS} = {i}
                GROUP BY 1 HAVING count(*) <= {key_cap}""")
        log(f"   {out_tbl} {c:8s} key weights  {time.time() - t0:7.0f}s")

        # 4. scoring (chunked over S1)
        for j in range(s1_chunks):
            dest = "cand_chunk" if on_chunk else out_tbl
            if on_chunk:
                con.execute(f"CREATE OR REPLACE TABLE cand_chunk {cand_cols}")
            s_part = "s_keys" if s1_chunks == 1 else f"(SELECT * FROM s_keys WHERE hash(id) % {s1_chunks} = {j})"
            con.execute(f"""
            INSERT INTO {dest}
            SELECT * FROM (
                SELECT s1_id, tgt_id, score, n_keys,
                       row_number() OVER (PARTITION BY s1_id, tgt_id // {SRC} ORDER BY score DESC, tgt_id) AS rk
                FROM (SELECT s.id AS s1_id, t.id AS tgt_id, sum(ln(1e7 / w.df)) AS score, count(*) AS n_keys
                      FROM {s_part} s JOIN key_w w USING (k) JOIN t_keys t USING (k) GROUP BY ALL)
            ) WHERE rk <= {k_max}""")
            if on_chunk:
                on_chunk("cand_chunk")
            if s1_chunks > 1 and (j + 1) % 8 == 0:
                log(f"   {out_tbl} {c:8s} scored {j + 1}/{s1_chunks} {time.time() - t0:7.0f}s")

    for t in ("s1_vocab", "df_c", "s_rec", "s_keys", "s_kset", "t_rec", "t_keys", "key_w", "cand_chunk"):
        con.execute(f"DROP TABLE IF EXISTS {t}")
    con.execute("CHECKPOINT")


def eval_blocking(con, cand, label, s1_tbl="eval_s1", pairs_tbl="eval_pairs"):
    """Pair recall, S1 coverage, ceiling macro-F0.5 (perfect matcher on these candidates), candidate sizes."""
    return con.execute(f"""
    WITH c     AS (SELECT DISTINCT s1_id, tgt_id FROM {cand}),
         ntrue AS (SELECT s1_id, count(*) AS nt FROM {pairs_tbl} GROUP BY 1),
         hits  AS (SELECT p.s1_id, count(*) AS h FROM {pairs_tbl} p JOIN c USING (s1_id, tgt_id) GROUP BY 1),
         ncand AS (SELECT s1_id, count(*) AS nc FROM c GROUP BY 1),
         per   AS (SELECT e.s1_id, coalesce(nt, 0) nt, coalesce(h, 0) h, coalesce(nc, 0) nc
                   FROM {s1_tbl} e LEFT JOIN ntrue USING (s1_id) LEFT JOIN hits USING (s1_id)
                   LEFT JOIN ncand USING (s1_id))
    SELECT '{label}' AS method,
           sum(h) / sum(nt)                                              AS pair_recall,
           avg(CASE WHEN nt > 0 THEN (h > 0)::INT END)                   AS s1_coverage,
           avg(CASE WHEN nt = 0 THEN 1.0 WHEN h = 0 THEN 0.0
                    ELSE 1.25 * (h / nt) / (0.25 + h / nt) END)          AS ceiling_f05,
           avg(nc) AS avg_cands, median(nc) AS med_cands,
           quantile_cont(nc, 0.99) AS p99_cands, max(nc) AS max_cands
    FROM per""").fetchdf()