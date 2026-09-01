# ScaleTraj
SIGSPTIAL 2026
## Setup

```bash
conda create -n scaletraj python=3.11 -y
conda activate scaletraj

pip install -r requirements.txt
```

## Data

Datasets are not bundled. Download the raw data and run the preprocessing:

| Dataset | Where to get it                                                                                                       |
|---|-----------------------------------------------------------------------------------------------------------------------|
| GeoLife (Beijing) | [Microsoft Research](https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/) |
| Tencent Beijing | Released with [CoPB (Shao et al.)](https://github.com/tsinghua-fib-lab/CoPB) — contact the authors |

```bash
# GeoLife (~7 min)
python -m src.data_preprocess --dataset geolife_beijing \
    --geolife_dir "/path/to/Geolife Trajectories 1.3/Data" --relabel dbscan

# Tencent (~30 s)
python -m src.data_preprocess --dataset tencent_beijing \
    --stay_input /path/to/user_stay_points.txt \
    --hw_input   /path/to/user_hw.txt
```

Outputs land in `data/{dataset}/preprocessed/` (`train.jsonl`, `test.jsonl`,
`daily_stays.jsonl`, plus per-user anchors: `anchors.json` for GeoLife or
`anchors.pkl` for Tencent). Splits are chronological 80/20 per user.
Expected sizes: GeoLife 121 users / 4,457 train + 1,171 test days;
Tencent 500 users (`seed=42`) / 28,957 train + 7,507 test days.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
