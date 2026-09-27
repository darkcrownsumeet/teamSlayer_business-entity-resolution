# Business Entity Resolution — Methodology (Team Slayer)

## 1. Summary

We resolve Source 1 (S1) business entities against Source 2 / Source 3 (S2/S3) records with a four-stage pipeline:

1. **Normalization** — script transliteration learned from training pairs, noise repair, canonicalization of legal forms, abbreviations and numbers.
2. **Candidate generation, stage 1** — combined-key blocking (rare name × address token pairs, exact / compact / sorted names, typo prefix-suffix keys); scalable and memory-bounded.
3. **Candidate generation, stage 2** — a lightweight LightGBM reranker and an adaptive cut. Its output is `candidate_pairs.tsv`, the exact input of the matcher.
4. **Matching and decision** — an XGBoost matcher (GPU) with competition, cluster-consistency and number-shape features, followed by a one-to-many assignment and a threshold rule.

| Result | Value |
|---|---|
| Validation macro-F0.5 (US + India, held-out S1) | **0.9802** |
| Public leaderboard | **0.966** (first submission with an earlier pipeline: 0.9492) |
| Candidate ceiling macro-F0.5 (perfect matcher on our candidates, validation) | 0.993 |
| Candidate pair recall (validation) | 0.976 |
| Candidates per S1 — validation / test (US 7.0, India 7.3, France 12.3) | 6.65 / 7.7 |

## 2. Data and validation protocol

- Train: 2.21M S1, 5.03M S2, 5.29M S3 (US, India). Test: 1.73M S1, 4.89M S2, 5.08M S3 (US, India and **France, unseen in training**).
- 5.6% of S1 are singletons; matched S1 have 1–11 matches (mean ≈ 3.5). No S2/S3 record belongs to more than one S1, and matched pairs always share the country label (both verified on the full ground truth).
- **Validation split**: 441k held-out S1 (`notebooks/phase2_valid_gt.parquet`, created by `00_dataset_audit`) are never used to mine dictionaries or train any model. Metrics are reported on a fixed 20k-entity sample (69,252 true pairs). During blocking all S1 compete for targets, so competition effects are realistic.
- **Metric**: official per-S1 F0.5, macro-averaged, singletons included (`src/metrics.py`). Standard error on the 20k sample ≈ ±0.0007; differences below ~0.001 are treated as noise.
- Disjoint training-fold S1 sets: reranker (30k), matcher (600k). No leakage: row order and ID numbers are uncorrelated with matches (correlation ≈ 0.001) and are never used.

## 3. Normalization (`src/normalize.py`)

Applied identically to all sources, train and test.

| Step | Purpose | Example |
|---|---|---|
| Mojibake repair (before NFKC) | UTF-8 read as Latin-1, replacement characters | `Youngā\x80\x99s` → `young s` |
| NFKC, zero-width removal, casefold | Unicode consistency | |
| **Learned transliteration dictionary** | 10 Indic scripts (~25% of India targets) mapped token-by-token to the English word used in S1 | `प्राइवेट` → `private` |
| anyascii fallback | Unseen tokens, accents (French) | `Société` → `societe` |
| Legal-form / abbreviation canonicalization | Hand rules + mined variant maps, merged idempotently | `pvt ltd` → `private limited`, `rd` → `road` |
| Digit-for-letter repair (tokens with ≥3 letters) | OCR-style noise | `hea1th` → `health` |
| Number canonicalization | Leading zeros, `no`/`n°` prefixes | `03211` → `3211`, `n° 14` → `14` |
| Web tokens, spaced acronyms | | `clyialsystems.com` → `clyialsystems`, `l l c` → `llc` |
| `name_core` | Strip legal forms/honorifics at both ends and trailing country noise; collapse adjacent duplicates; light plural stem | `sarl maison comice et fils` → `maison comice` |
| French rules (small, generic) | No labelled French data | `r` → `rue`, `sarl`, `ets` |

**Learned resources (training pairs only; validation excluded):**
- *Transliteration dictionary* — word-for-word alignment of Indic-script targets with their S1 names. Held-out token accuracy 97.5–99.2% across all 10 scripts; 94.2% token coverage on test.
- *Variant maps* — aligned differing tokens of true pairs (e.g. `praivet` → `private`), merged with hand rules so the final map is idempotent (hand rules win, chains resolved, ambiguous `saint`/`st` blocked).

All resources are offline. `anyascii` is a static character table (ISC license). No external lookup, API, geocoder, registry or pretrained model is used anywhere.

## 4. Candidate generation / blocking (`src/blocking.py`, `src/rerank.py`)

### 4.1 Stage 1 — combined-key blocking
Within each country (never across countries):
- For every record, the 3 rarest name tokens and 4 rarest address tokens are selected (rarity = document frequency over the target pool, restricted to tokens that also occur in the S1 vocabulary, so typo tokens are not chosen).
- Keys: name×address, name×name and address×address token pairs; exact core name; compact name (spaces removed); sorted-token core; compact-name prefix/suffix with and without the first house number (typo/truncation channel).
- Keys whose target block exceeds 300 records are dropped (generic names).
- Candidates are scored by the sum of key IDF weights; top 50 per source (S2, S3) are kept.

**Scalability**: every step is a hash join or aggregate in DuckDB. Targets are processed per source in hash chunks, S1 in hash chunks, and large key aggregates in key-hash partitions, so memory stays bounded (4–9 GB) regardless of data size. No all-pairs comparison is performed at any point. Full train run (2.2M S1 × 10.3M targets) ≈ 3 h; test ≈ 2.5 h on a 16-thread CPU.

### 4.2 Stage 2 — reranker and adaptive cut
- LightGBM with 15 cheap features (key score/rank, Jaro-Winkler on core and compact names, containment, name/address token Jaccard, shared numbers, missing-address and postcode flags), trained on a disjoint 30k training-fold sample.
- Cut: keep candidates with p ≥ 0.003, at most 8 per source. **This set is `candidate_pairs.tsv`** and is exactly what the matcher scores.

| Blocking quality (validation) | Value |
|---|---|
| Stage-1 pair recall @ top-50/source | 0.979 |
| Final candidate pair recall | 0.976 |
| Candidates per S1 (mean / median / p99) | 6.65 / 6 / 16 |
| Reduction ratio vs. all same-country pairs | > 0.99999 |
| Ceiling macro-F0.5 | 0.993 |

## 5. Matcher (`src/features.py`)

**Model**: XGBoost (hist, CUDA), max depth 10, learning rate 0.04, early stopping on 10% of training S1. Trained on ~4.0M candidate pairs from 600k training-fold S1. Model file: `artifacts/matcher_xgb_v3_c3_big.json`.

**Features (58, `FEATS_V3`)**:

| Group | Examples | Motivation |
|---|---|---|
| Reranker / list context | p, rank, S1's max and sum of p, gap to the S1's best candidate | List-level context |
| **Competition across S1** | number of S1 claiming the target, best competing claim, margin | Each target belongs to at most one S1; resolves singletons whose best candidate belongs to someone else |
| **Cluster consistency** | similarity (name, address, house number) to the S1's other strong candidates; near-duplicate "twin" count; address/name rank within the S1's list | Matches of one entity corroborate each other; distractors are the "second copy" |
| Name similarity | Jaro-Winkler, token set/sort, WRatio, partial ratio and Levenshtein on the compact name, phonetic skeleton, IDF-weighted token overlap, legal-form agreement | Typos, transliteration drift, truncation |
| Address similarity | token Jaccard (with/without numbers), Jaro-Winkler, last-token agreement, postcode | |
| **Number-difference shape** | equal / log absolute difference / digit edit distance / truncation of the house number | Distractors shift the number by 1–99; true-match noise truncates or mistypes it |
| Artifacts | duplicated tokens in the raw name | Generated noise pattern |

## 6. Decision rule (`src/predict.py`)
1. Keep pairs with matcher probability q ≥ 0.7.
2. **One-to-many**: each target is assigned only to its highest-q S1.
3. An S1's best match needs q ≥ 0.7; each additional match of the same S1 needs q ≥ 0.8 (distractor siblings concentrate there).
4. Empty prediction when nothing passes (full credit on true singletons).

The rule was chosen on validation; alternatives (separate top-1 thresholds, relative-to-best thresholds, per-entity expected-F0.5) were within ±0.0003. Per-country checks on test show consistent behaviour (avg matches per S1: US 3.39, India 3.32, France 3.39; empty rate 5–6%).

## 7. Development history (validation macro-F0.5)

| Version | Change | Macro-F0.5 |
|---|---|---|
| Reranker used as matcher | baseline | 0.929 |
| + one-to-many | | 0.933 |
| Matcher v1 | competition, list context, cluster features | 0.968 |
| Matcher v2 | + IDF name overlap, address/number features, 3× data, GPU | 0.977 |
| Matcher v3 | + number-difference shape, twins, sibling number agreement | 0.981 |
| **Final: v3 features, cycle-3 normalization, 600k training S1, final rule** | submitted | **0.9802** |

Variants that did not help (within noise): deeper trees (depth 12), different seeds, 2–3 model ensembles, alternative decision rules, per-country thresholds.

**Data findings that shaped the design:**
- 26% of targets belong to no S1. About 9.5% of these are near-copies of a real entity ("distractors"): same name with a house number shifted by 1–99 (60% of distractors vs 4.6% of true matches), or a changed/added legal form or business word (e.g. `+ Holdings`, `+ Exports`, `Ltd → Pvt Ltd`).
- About 57% of the remaining candidate misses are name-only records with generic names shared by several S1 entities; about 17% of contested cases carry no distinguishing signal at all.

## 8. Future work (built or validated, not in the submitted model)
- **Stage-2 matcher with competitor-relative features**: for contested targets, the true owner is distinguishable from the wrong claimant by name, legal form or address in ~80% of cases.
- **Feature set v4** (implemented as `FEATS_V4`, not used): exact legal-form transitions, learned log-odds of added/dropped name tokens, alias-aware similarity (`dba`, `aka`, `fka`), IDF-weighted address overlap, raw formatting flags — each supported by audits on validation data.
- Training the matcher on all ~1.7M training-fold S1 (300k → 600k gave +0.0008).

## 9. Reproducibility
See `code/business_entity_resolution/README.md`. Notebooks run in order `00` → `04`: dataset audit and validation split → dictionary mining → Parquet conversion → normalized keys, reranker, train blocking and matcher → test blocking, scoring, decision, output files and validator. Seeds are fixed; dependencies are pinned in `requirements.txt`. The files in `artifacts/` are regenerable from the training data.

## 10. Models and licenses
No pretrained language model is used; all models are gradient-boosted trees trained from scratch (far below 8B parameters).

| Component | License |
|---|---|
| XGBoost | Apache-2.0 |
| LightGBM | MIT |
| DuckDB | MIT |
| anyascii | ISC |
| rapidfuzz | MIT |
| pandas, NumPy, scikit-learn | BSD-3 |
| PyArrow | Apache-2.0 |

## 11. Limitations
- France has no labelled data; its handling relies on language-neutral features and generic normalization rules and cannot be validated offline.
- Name-only targets with generic names shared by several S1 entities are often ambiguous by construction.
- Test has ~20% more targets per S1 than train (more unmatched records), so offline precision is somewhat optimistic (validation 0.9802 vs leaderboard 0.966).
