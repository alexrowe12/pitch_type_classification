# pitch_type_classification
Pitch type classification project for Intro AI

## Preprocessing

Run preprocessing from the repo root:

```bash
python -m preprocess.download_clips --limit 100
python -m preprocess.process_clips --preview --debug --limit 20
python -m stage_a.export_frames --limit 25
python -m stage_a.build_weak_labels
python -m stage_a.make_review_queue --target-size 250
streamlit run stage_a/app.py
```

Directory layout:

- `data/clips/`: downloaded raw pitch clips
- `data/processed/`: processed numpy arrays
- `data/debug/`: debug visualizations
- `data/stage_a/`: Stage A shot-classification artifacts
- `research/mlb-youtube-repo/`: source dataset repo and metadata
