"""
Normalization for the Business Entity Resolution challenge.

v1: kept only for comparison with artifacts/keys (do not use in the final pipeline).
v2: mojibake repair, learned transliteration (artifacts/translit_dict.json), anyascii fallback,
    legal-form / address canonicalization, mined variant maps (artifacts/canon_mined.json),
    digit-for-letter noise repair, leading-zero stripping, web-token removal, single-letter
    merging (l l c -> llc), honorific / legal-form stripping at both ends of name_core,
    adjacent duplicate collapse, light plural stemming, French rules, phonetic skeleton.

All resources are offline and derived from the provided training data or generic language rules.
"""

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
from anyascii import anyascii

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


# ============================================================
# Shared
# ============================================================
MISSING_NAME_TOKENS = {"na", "nan", "null"}


def is_missing_value(value):
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() == ""


# ============================================================
# v1 (comparison only)
# ============================================================
def normalize_text(value):
    if is_missing_value(value):
        return None
    value = unicodedata.normalize("NFKC", str(value))
    value = value.casefold()
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def normalize_name(value):
    value = normalize_text(value)
    if value is None or value in MISSING_NAME_TOKENS:
        return None
    return value


def normalize_address(value):
    return normalize_text(value)


def normalize_address_compact(value):
    value = normalize_address(value)
    if value is None:
        return None
    value = re.sub(r"[^\w\s]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def normalize_country(value):
    return normalize_text(value)


# ============================================================
# v2 hand dictionaries (generic language normalization)
# ============================================================
LEGAL_CANON = {
    "pvt": "private", "ltd": "limited", "co": "company", "corp": "corporation",
    "inc": "incorporated", "intl": "international", "mfg": "manufacturing",
    "bros": "brothers", "&": "and",
    # French
    "et": "and", "cie": "company", "ste": "societe", "ets": "etablissements",
}

LEGAL_FORMS = {
    "private", "limited", "incorporated", "corporation", "company", "llc", "llp",
    "plc", "lp", "pllc", "pc", "and",
    # French
    "sarl", "sas", "sasu", "sa", "eurl", "snc", "sci", "scop", "fils", "freres",
}

ADDR_CANON = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
    "bd": "boulevard", "cir": "circle", "ln": "lane", "dr": "drive", "ct": "court",
    "hwy": "highway", "pkwy": "parkway", "fl": "floor", "flr": "floor", "bldg": "building",
    "opp": "opposite", "nr": "near", "grd": "ground",
    # French (no labelled FR data -> rules, not mining)
    "r": "rue", "chem": "chemin", "imp": "impasse", "pl": "place", "rte": "route",
    "fbg": "faubourg", "sq": "square", "qu": "quai",
}

WEB_TOKENS = {"www", "com", "net", "org", "http", "https"}

# stripped from the FRONT of name_core (in addition to LEGAL_FORMS)
NAME_PREFIXES = {"the", "sri", "shri", "shree", "mr", "ms", "mrs", "m/s",
                 "societe", "etablissements"}

# stripped from the END of name_core (country suffix noise: 'union de gitane france')
TRAILING_NOISE = {"france"}


# ============================================================
# Mined variant maps (hand rules win; targets composed through hand map; chains resolved)
# ============================================================
MINED_BLOCKLIST = {"saint"}  # ambiguous: 'st' = saint vs street


def _compose(mined, hand):
    """Merge mined variant->canonical into hand rules so the final map is idempotent."""
    hand_vals = set(hand.values())
    out = {}
    for v, c in mined.items():
        if v in hand or v in hand_vals or v in MINED_BLOCKLIST:
            continue  # hand rules own these tokens (prevents pvt<->private swaps)
        out[v] = hand.get(c, c)
    for _ in range(5):  # resolve chains v->c->d
        out = {v: out.get(c, c) for v, c in out.items()}
    merged = {v: c for v, c in out.items() if v != c}  # drops 2-cycles
    merged.update(hand)
    return merged


_MINED = ARTIFACTS / "canon_mined.json"
if _MINED.exists():
    _m = json.loads(_MINED.read_text(encoding="utf-8"))
    LEGAL_CANON = _compose(_m.get("name", {}), LEGAL_CANON)
    ADDR_CANON = _compose(_m.get("addr", {}), ADDR_CANON)


# ============================================================
# Learned transliteration dictionary (script token -> English token)
# ============================================================
def _is_latin_word(w):
    """True if every letter is ASCII or Latin-script (accented Latin -> handled by anyascii)."""
    return all(ord(c) < 128 or not c.isalpha() or unicodedata.name(c, "").startswith("LATIN")
               for c in w)


_TRANSLIT = ARTIFACTS / "translit_dict.json"
TRANSLIT = {}
if _TRANSLIT.exists():
    TRANSLIT = {k: v for k, v in json.loads(_TRANSLIT.read_text(encoding="utf-8")).items()
                if len(k) >= 2 and not _is_latin_word(k)}


# ============================================================
# v2 regexes
# ============================================================
REPLACEMENT_RE = re.compile(r"ï¿½|\ufffd")
MOJIBAKE_RE = re.compile(r"[\u0080-\u009f]")
MOJIBAKE_STRIP_RE = re.compile(r"[\u00c2\u00c3\u00e2\u0101]?[\u0080-\u009f]+")
ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200d\u2060\ufeff]")
PLACEHOLDER_RE = re.compile(r"<\s*(null|none|nan|na)\s*>")
TOKEN_RE = re.compile(r"[^\s.,;:()\[\]/\\\-|'\"]+")  # identical to raw_tokens() used in mining
NONALNUM_RE = re.compile(r"[^a-z0-9&/ ]+")
NUM_RE = re.compile(r"\b\d+\b")
NUM_PREFIX_RE = re.compile(r"\b(?:ndeg|no|nr|num)\s+(?=\d)")  # 'n° 14' -> 'ndeg 14' -> '14'
SINGLE_LETTERS_RE = re.compile(r"\b([a-z]) (?=[a-z]\b)")      # 'l l c' -> 'llc', 'j w l' -> 'jwl'
LEET_TABLE = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "6": "g", "8": "b"})


# ============================================================
# v2 helpers
# ============================================================
def fix_mojibake(text):
    """Repair UTF-8 read as Latin-1 ('â\\x80\\x99' -> '’'); strip replacement-char junk."""
    text = REPLACEMENT_RE.sub(" ", text)
    if not MOJIBAKE_RE.search(text):
        return text
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return MOJIBAKE_STRIP_RE.sub(" ", text)


def _apply_translit(text):
    """Replace whole non-Latin tokens with their learned English form; everything else untouched."""
    if not TRANSLIT:
        return text
    return TOKEN_RE.sub(lambda m: TRANSLIT.get(m.group(0), m.group(0)), text)


def to_latin(value):
    """fix mojibake -> NFKC -> drop zero-width -> learned translit -> anyascii fallback -> casefold."""
    if is_missing_value(value):
        return None
    text = fix_mojibake(str(value))  # must run before NFKC (NFKC rewrites ½ -> 1⁄2)
    text = unicodedata.normalize("NFKC", text)
    text = ZERO_WIDTH_RE.sub("", text).casefold()
    text = _apply_translit(text)
    text = anyascii(text).casefold()
    text = PLACEHOLDER_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip() or None


def _fix_leet(tok):
    """hea1th->health, 6reat->great. Only tokens with >=3 letters whose digits are all in 0134568,
    so house numbers (21b, 101a), ordinals (1st, 3rd) and short codes (a1, 3m) are untouched."""
    letters = sum(c.isalpha() for c in tok)
    digits = {c for c in tok if c.isdigit()}
    if letters >= 3 and digits and digits <= set("0134568"):
        return tok.translate(LEET_TABLE)
    return tok


def _canon_tokens(text, canon):
    text = NONALNUM_RE.sub(" ", text)
    toks = [canon.get(t, t) for t in (_fix_leet(t) for t in text.split())]
    return " ".join(" ".join(toks).split())  # canon values may contain spaces


def _strip_zeros(m):
    """03211 -> 3211 on both sides (consistency matters more than ZIP fidelity)."""
    return m.group(0).lstrip("0") or "0"


def _stem(tok):
    """Light plural stem: hotels->hotel, partners->partner; keeps -ss/-us/-is words (lotus, genesis)."""
    if (len(tok) >= 5 and tok.endswith("s") and not tok.endswith(("ss", "us", "is"))
            and not tok.isdigit()):
        return tok[:-1]
    return tok


# ============================================================
# v2 public API
# ============================================================
def normalize_name_v2(value):
    text = to_latin(value)
    if text is None or text in MISSING_NAME_TOKENS:
        return None
    text = _canon_tokens(text.replace("&", " & "), LEGAL_CANON)
    text = SINGLE_LETTERS_RE.sub(r"\1", text)
    toks = [t for t in text.split() if t not in WEB_TOKENS] or text.split()
    return " ".join(toks) or None


def name_core(name_v2):
    """Distinctive part of the name, used for keys and most name features:
      - collapse adjacent duplicates      'lotus care care'              -> 'lotus care'
      - strip trailing legal forms/noise  'maison comice and fils'       -> 'maison comice'
      - strip leading honorifics/legal    'private smart exports'        -> 'smart exports'
      - light plural stem                 'zephon hotels'                -> 'zephon hotel'
    Always keeps at least one token."""
    if not name_v2:
        return None
    toks = name_v2.split()
    toks = [t for i, t in enumerate(toks) if i == 0 or t != toks[i - 1]]
    while len(toks) > 1 and (toks[-1] in LEGAL_FORMS or toks[-1] in TRAILING_NOISE):
        toks.pop()
    while len(toks) > 1 and (toks[0] in NAME_PREFIXES or toks[0] in LEGAL_FORMS):
        toks.pop(0)
    return " ".join(_stem(t) for t in toks)


def normalize_address_v2(value):
    text = to_latin(value)
    if text is None:
        return None
    text = text.replace("g/f", " ground floor ").replace("/", " ")
    text = _canon_tokens(text, ADDR_CANON)
    text = NUM_PREFIX_RE.sub("", text)
    text = NUM_RE.sub(_strip_zeros, text)
    return " ".join(text.split()) or None


def normalize_country_v2(value):
    return to_latin(value)


# ============================================================
# Phonetic skeleton (matching feature; cross-script drift: praivet/praibhet/piraivet -> prbt)
# ============================================================
_SKEL_DIGRAPHS = [("rr", "t"), ("ph", "f"), ("bh", "b"), ("kh", "k"), ("gh", "g"),
                  ("th", "t"), ("dh", "t"), ("sh", "s"), ("ch", "c"), ("jh", "j")]
_SKEL_MAP = str.maketrans({"v": "b", "w": "b", "d": "t", "q": "k", "c": "k", "z": "j", "x": "k"})
_ANUSVARA_RE = re.compile(r"m(?=[^aeiouy])")  # vemcrs -> vencrs (Indic nasal before consonant)
_VOWELS_RE = re.compile(r"[aeiouy]")
_REPEAT_RE = re.compile(r"(.)\1+")


def skeleton(text):
    """Coarse phonetic key per token. Use as a matching feature, never as a match decision alone."""
    if not text:
        return None
    out = []
    for tok in text.split():
        if tok.isdigit():
            out.append(tok)
            continue
        t = tok
        for a, b in _SKEL_DIGRAPHS:
            t = t.replace(a, b)
        t = _ANUSVARA_RE.sub("n", t).translate(_SKEL_MAP)
        t = t[0] + _VOWELS_RE.sub("", t[1:])
        out.append(_REPEAT_RE.sub(r"\1", t))
    return " ".join(out)