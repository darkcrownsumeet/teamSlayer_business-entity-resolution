# TeamSlayer Submission

## Environment
- **Python**: 3.14
- **Dependencies**: Install using `pip install -r requirements.txt`
- **Data**: Data should be placed in `dataset/train` and `dataset/test` (in TSV format).

## Run Order

| Notebook | Outputs |
|----------|---------|
| `00_dataset_audit` | `notebooks/phase2_valid_gt.parquet` (validation split) |
| `01_translit_mining` | `artifacts/translit_dict.json`, `canon_mined.json` |
| `02_parquet_and_setup` | `artifacts/parquet` |
| `03_train_pipeline` | `keys_v2`, `rerank_lgb_c3.txt`, train blocking, `matcher_xgb_v3_c3_big.json` |
| `04_test_pipeline` | test blocking, scoring, `output/matching_results.tsv`, `output/candidate_pairs.tsv` |

## Validation
To validate the submission outputs, run:
```bash
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

## Source Modules (`src/`)
- `normalize.py`: Text normalization and standardization utilities.
- `keys.py`: Key generation for blocking.
- `blocking.py`: Logic to pair entities and generate candidate pairs.
- `rerank.py`: Re-ranking mechanisms for the matched candidates.
- `features.py`: Feature engineering used by the models.
- `predict.py`: Model inference and scoring pipeline.
- `metrics.py`: Evaluation metrics calculations.
- `__init__.py`: Module initializer.

## Hardware Specs & Timing
- **Hardware**: 16-thread CPU, 16 GB RAM, NVIDIA RTX 5050 8 GB
- **Execution Time**: Train blocking takes ~3 hours, Test pipeline takes ~2.5 hours.

## Note
No external data, APIs, geocoders or pretrained models are used. anyascii is an offline character table. The files in artifacts/ are regenerable from the training data by running the notebooks.
