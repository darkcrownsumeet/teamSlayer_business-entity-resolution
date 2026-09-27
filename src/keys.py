"""Build normalized key tables (artifacts/keys_v2/<source>.parquet) from raw parquet copies."""

import re
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as papq

from src import normalize as N

K2_COLS = ["entity_id", "country_key", "name_v2", "name_core", "addr_v2", "postcode"]
K2_SCHEMA = pa.schema([(c, pa.string()) for c in K2_COLS])
POST_RE = re.compile(r"(?<!\d)(\d{3}\s?\d{3}|\d{5})(?!\d)")


def extract_postcode(addr):
    if not isinstance(addr, str):
        return None
    m = POST_RE.findall(addr)
    return m[-1].replace(" ", "") if m else None


def build_keys(con, raw_parquet: Path, out_parquet: Path, batch=250_000, overwrite=False):
    """Stream raw records in batches, normalize, write key parquet atomically (.tmp -> rename)."""
    if out_parquet.exists() and not overwrite:
        return
    tmp = out_parquet.with_suffix(".tmp")
    reader = con.execute(f"""
        SELECT entity_id, business_name, business_address, country
        FROM read_parquet('{raw_parquet.as_posix()}')
    """).to_arrow_reader(batch)
    writer = papq.ParquetWriter(tmp.as_posix(), K2_SCHEMA, compression="zstd")
    try:
        for rb in reader:
            ids, names, addrs, ctys = (rb.column(i).to_pylist() for i in range(4))
            nm = [N.normalize_name_v2(x) for x in names]
            ad = [N.normalize_address_v2(x) for x in addrs]
            writer.write_table(pa.Table.from_pydict({
                "entity_id":   ids,
                "country_key": [N.normalize_country_v2(x) for x in ctys],
                "name_v2":     nm,
                "name_core":   [N.name_core(x) for x in nm],
                "addr_v2":     ad,
                "postcode":    [extract_postcode(x) for x in ad],
            }, schema=K2_SCHEMA))
    finally:
        writer.close()
    tmp.replace(out_parquet)


def build_all(con, parq_dir: Path, keys_dir: Path, names, overwrite=False, log=print):
    keys_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        t0 = time.time()
        build_keys(con, parq_dir / f"{name}.parquet", keys_dir / f"{name}.parquet", overwrite=overwrite)
        log(f"   keys {name:16s} {time.time() - t0:7.1f}s")