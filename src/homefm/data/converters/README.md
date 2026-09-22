# Dataset converters

Every source is converted into a `HomeStream` of Home Tokens (docs/DESIGN.md §6.1).

| Dataset | Status | Notes |
|---|---|---|
| Synthetic (`homefm.data.synthetic`) | ✅ implemented | Episodes, captions, injected faults |
| CASAS (Milan, Aruba, …) | ✅ `casas.py` | Needs a hand-written `sensor_map.json` per home |
| Kasteren A / C | ⬜ todo | `.mat` / annotation files; item/room semantics in the paper appendix |
| UCI ADL Binary (Home B) | ⬜ todo | Sensor table includes location + type |
| Orange4Home | ⬜ todo | 236 sensors incl. continuous; keep scalars as scalar tokens (no binarisation) |
| MuRAL | ⬜ todo | Multi-resident, natural-language annotations → captions |
| ARAS | ⬜ todo | Multi-resident |
| REDD / UK-DALE / REFIT | ⬜ todo | Appliance power → scalar tokens for device-health pretraining |

Place raw files under `data/raw/<dataset>/` (git-ignored).
