# pitch_type_classification
Pitch type classification project for Intro AI

## Workflow

Run commands from the repo root.

Download raw clips:

```bash
python -m preprocess.download_clips --limit 100
```

Stage A shot classification:

```bash
python -m stage_a.export_frames --limit 25
python -m stage_a.build_weak_labels
python -m stage_a.make_review_queue --target-size 250
streamlit run stage_a/app.py
python -m stage_a.prepare_train_labels
python -m stage_a.train_stage_a --epochs 3
python -m stage_a.infer_stage_a
python -m stage_a.export_debug_contacts --limit 100
```

Stage B release/catch localization:

```bash
python -m stage_b.export_candidates --min-stage-a-prob 0.98
python -m stage_b.build_weak_events
streamlit run stage_b/app.py
python -m stage_b.prepare_events
python -m stage_b.export_sequences --num-frames 12
python -m stage_b.export_debug_contacts --limit 100
```

Legacy baseline preprocessing:

```bash
python -m preprocess.process_clips --preview --debug --limit 20
```

Directory layout:

- `data/clips/`: downloaded raw pitch clips
- `data/stage_a/labels/manual_labels.csv`: tracked manual Stage A labels
- `data/stage_a/`: generated Stage A frames, models, predictions, and debug contact sheets
- `data/stage_b/`: generated Stage B candidate frames, event labels, sequences, and debug outputs
- `preprocess/`: raw clip download and legacy frame-processing scripts
- `stage_a/`: Stage A shot-classification scripts
- `stage_b/`: Stage B release/catch localization scripts
- `research/mlb-youtube-repo/`: source dataset repo and metadata

Generated artifacts under `data/` are ignored unless explicitly tracked.
