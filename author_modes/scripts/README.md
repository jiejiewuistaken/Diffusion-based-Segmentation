# Runbook

Examples

1) Build features from data dictionary
```bash
python -m author_modes.features.build_features \
  --data-dict author_modes/data_dictionary/data_dictionary_template.yaml \
  --config author_modes/configs/feature_config.yaml \
  --output author_modes/artifacts/features.csv
```

2) Train clustering and save assignments + embeddings
```bash
python -m author_modes.models.train_cluster \
  --features author_modes/artifacts/features.csv \
  --config author_modes/configs/model_config.yaml \
  --output author_modes/artifacts/cluster_assignments.csv
```

3) Evaluate clusters
```bash
python -m author_modes.evaluation.cluster_eval \
  --features author_modes/artifacts/features.csv \
  --assignments author_modes/artifacts/cluster_assignments.csv \
  --embeddings author_modes/artifacts/cluster_assignments.umap.csv \
  --output author_modes/artifacts/cluster_report
```
