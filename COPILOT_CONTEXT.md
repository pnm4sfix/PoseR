# PoseR — Copilot Project Context

## Environment
- **Python env**: `D:\CondaEnvs\PoseR6\python.exe`
- **napari version**: 0.6.6
- **Key packages**: torch 2.10 (CPU), ultralytics, imageio 2.37, imageio-ffmpeg 0.6.0, av (pyav), dask 2026.1, napari-plot, tables (pytables)

## Project Structure
```
src/poser/
  _panels/
    data_panel.py        ← session/video/pose management UI
    inference_panel.py   ← pose estimation + behaviour decoding UI
    annotation_panel.py  ← behaviour annotation UI
    analysis_panel.py    ← analysis UI
    train_panel.py       ← training UI
    _factories.py        ← zero-arg factory functions for napari plugin
  core/
    session.py           ← SessionManager, SessionEntry, get_session()
  models/                ← ST-GCN and graph conv model definitions
  _widget.py             ← original monolithic widget (legacy, keep for reference)
  napari.yaml            ← 5 widget contributions
setup.cfg                ← install_requires (add new deps here)
weights/zeb.pt           ← pretrained zebrafish pose model
```

## Architecture
- **5 panels** registered in `napari.yaml`, each created by a zero-arg factory in `_factories.py`
- **Shared session**: all panels share one `SessionManager` via `get_session(viewer)` singleton (keyed by `id(viewer)`) in `core/session.py`
- **SessionEntry fields**: `video_path`, `pose_path` (optional), `status` (`"pending"` / `"active"`), `layer_names`
- `activate(idx)` sets `entry.status = "active"` and loads layers

## Video Loading (data_panel.py `_load_entry_layers`)
- Uses **pyav** (via `imageio.v3`, plugin="pyav") — ships bundled FFmpeg, no system dep
- **Lazy dask array**: metadata read upfront (`iio.improps`), frames decoded in 64-frame chunks on demand via `dask.delayed` + `av` directly
- `props.shape` is `(N, H, W, C)` — use `props.shape[-3]` for H, `props.shape[-2]` for W
- Grayscale detection: peek first frame, check if R==G channels
- napari `add_image(vid_arr, rgb=is_rgb)` — no `VideoReaderNP`, no cv2

## Pose File Loading (data_panel.py `_load_pose_as_points`)
4-format cascade:
1. PoseR pytables (`tables.open_file`) — `ind.coords[:]` shape `(3, n_nodes, n_frames)`
2. DLC h5 (`pd.read_hdf`) → `_parse_dlc_df()`
3. DLC CSV (`pd.read_csv(header=[0,1,2])`) → `_parse_dlc_df()`
4. SLEAP h5py groups with x/y/ci datasets

## Known Issues / Decisions
- `VideoReaderNP` (napari-video) broken on napari ≥ 0.5 — `ndim` returns `len(shape)+1`, causes `np.transpose` error in napari's slice machinery. **Do not use.**
- `napari-video` removed from `setup.cfg`, replaced by `av` + `dask`
- `imageio.v3` plugin name for bundled ffmpeg is `"pyav"` (not `"ffmpeg"` — that's v2 only)

## Next Tasks (suggested)
- [ ] Test inference panel: run pose estimation on active video entry with `zeb.pt`
- [ ] Annotation panel: wire up behaviour labelling to session
- [ ] Analysis panel: behaviour extraction using session pose data
- [ ] Training panel: dataset preparation + lightning training loop
- [ ] Progress bar / threading for video load (currently blocks UI)
