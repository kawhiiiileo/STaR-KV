# Repository instructions

The supported compressed-KV method is `starkv`: native GQA group-mean scores,
online group MI, exponential TEM, exact SFR, and a protected recent window.
`examples/repro_ssp_kv.sh` is the ScreenSpot-Pro entry point.
Keep benchmark drivers in `eval/` and compression modules in `starkv/`.
Do not change algorithm parameters when reorganizing the repository.
Do not include datasets, model weights, local paths, or generated results.
Never edit measured scores or describe reference scores as verified reproductions.

Core components are `starkv_mi.py`, `starkv_temporal.py`, and `starkv_sfr.py`.
The compressor is `starkv_group_cluster.py`; shared grid/cache helpers are
`starkv_utils.py`. Model integration and measurements belong in `eval/`.
