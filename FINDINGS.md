# PoseR: why behaviour decoding does not reproduce on the test data

Investigation run against `v0.0.1b4` release assets on macOS (Apple Silicon),
Python 3.10, torch 2.14, napari 0.6.6.

Everything below is measured, not inferred. Commands are included so each
claim can be re-run.

---

## 1. Headline: the decoder works; the earlier failure was an evaluation error

Two published checkpoints, scored against their own datasets' test splits
and compared with the accuracy in each filename:

| dataset | species | nodes | reported | measured | balanced |
|---|---|---|---|---|---|
| ZebLR | zebrafish larvae | 19 | 0.91 | **0.904** | 0.906 |
| OFT full | mouse | 18 | 0.86 | **0.839** | 0.833 |

Both reproduce to within 0.02, across two species and two skeletons, so the
load -> preprocess -> predict -> score chain is sound end to end. These used
the preprocessed `.npy` arrays the preprint datasets ship, which bypass
preprocessing entirely and therefore test the model and loader in isolation.

`zeblr.ckpt` scored against the 61 hand-labelled bouts in
`test_classification_file.h5`, which does exercise the full preprocessing
path:

```
accuracy          0.951
balanced accuracy 0.973        chance = 0.50 (two classes occur)
right recall      0.95
```

An earlier draft of this report claimed the opposite — 0.323 balanced,
"worse than chance" — and built four rejected hypotheses on top of it
(coordinate conventions, class-index permutation, skeleton topology,
checkpoint/data mismatch). **All of that was wrong.** The cause was a single
missing argument in the evaluation script, not in PoseR.

### 1.1 What went wrong

`classification_data_to_bouts` declares:

```python
align_data: bool = False
```

The training path always overrides it — `training/data_prep.py:149` passes
`align_data=dcfg.align_data`, and `DataConfig.align_data` defaults to `True`.
The evaluation script did not pass it, so it defaulted to `False`.

Alignment rotates each bout so the animal faces a canonical direction.
Without it, a left turn and a right turn differ only by the animal's arbitrary
heading in the arena — the information distinguishing them is destroyed before
the model sees the data. Hence `right` bouts being read as `left`, and hence
no amount of sign-flipping or label-permuting helping.

Same checkpoint, same data, alignment the only variable:

| | accuracy | balanced | `right` recall |
|---|---|---|---|
| `align_data=False` | 0.426 | 0.323 | 0.45 |
| `align_data=True` | **0.951** | **0.973** | **0.95** |

### 1.2 The lesson for anyone evaluating a decoder

Preprocessing at scoring time must match preprocessing at training time
exactly. `classification_data_to_bouts` defaulting `align_data` to `False`
while every real caller passes `True` is a trap: the mismatch produces no
error, no warning, and a plausible-looking confusion matrix. Mirror
`training/data_prep.py:139-150` rather than passing arguments by hand.

### 1.3 Checkpoint compatibility, for reference

Architecture read from the weights (`poser model inspect`), since stored
metadata is unreliable:

| checkpoint | nodes | in_ch | classes | loads? |
|---|---|---|---|---|
| `zeblr.ckpt` | 19 | 3 | 3 | yes — **the one matching the test data** |
| `zebtensor.ckpt` | 19 | 3 | 30 | yes |
| `oftfull.ckpt` | 18 | 3 | 3 | yes (mouse2) |
| `oftbody.ckpt` | 13 | 3 | 3 | yes (mouse1) |
| `zebrep.ckpt` | 8 | 1 | 13 | no — no 8-node layout in `models/graph.py` |
| `calms21.ckpt` | — | — | — | no — pickled `WindowsPath` |
| `flyvfly.ckpt` | — | — | — | no — same |

---

## 2. Bugs found and fixed en route

Each of these blocked the pipeline before the question in §1 could even be
asked. All are committed.

### 2.1 Checkpoints could not be loaded at all

**torch 2.6 flipped `weights_only` to `True`.** These checkpoints pickle a
`poser._loader.HyperParams`, which is then rejected:

```
UnpicklingError: Unsupported global: GLOBAL poser._loader.HyperParams
```

The raw `torch.load` calls already passed `weights_only=False`; the three
`ST_GCN_18.load_from_checkpoint` calls did not. (`5ebe94b`)

**Constructor arguments were never supplied.** Lightning only replays
hyper-parameters that were saved, and `zebrep.ckpt` stores only `hparams`:

```
TypeError: ST_GCN_18.__init__() missing 4 required positional arguments:
'in_channels', 'num_class', 'graph_cfg', and 'data_cfg'
```

Now rebuilt from the weights, which cannot drift from the architecture they
describe. (`25c8fc2`, `04479f2`)

### 2.2 `transform: None` silently disabled all preprocessing

`zeblr.ckpt` stores `data_cfg["transform"] = None`. The panel read it as a
value rather than as "unset", and `PoseDataset.__getitem__` skips centring,
alignment and padding entirely when transform is None. Measured on the same
checkpoint and data, with transform as the only variable:

```
transform=None               -> {1: 2000}          every frame one class
transform=center,align,pad   -> {0: 1716, 1: 284}
```

This is why the ethogram rendered as a single solid block. (`2ff5477`)

### 2.3 Negative frame indices silently discarded bouts

The detectors pad bouts outwards — `(peak_start - 20, peak_end + 20)` — so
`start` is routinely negative near the beginning of a recording. Two places
used it directly as a slice index, where numpy reads from the far end and
returns an empty window:

- `preprocess_bouts` raised `IndexError: index 0 is out of bounds for axis 1
  with size 0`, but only once the window was large enough to go negative.
- `check_behaviour_confidence` returned `False`, **silently dropping every
  bout in the first 20 frames regardless of its actual confidence**:

```
interior bout (100,140):     True
bout at the start (-15,25):  False      # with confidence 0.95 everywhere
ci[..., -15:25].shape = (5, 0)          # empty
```

Both now clamp. (`9730727`, `75d6823`)

### 2.4 The points array disagreed with itself

`behaviour_decode` built the `(V*T, 3)` array bout detection consumes with the
frame column laid out node-major and the coordinates laid out frame-major:

```
row |  frame_idx says  |  y value actually is  | agree?
  0 |        0         |   node 0, frame 0     |  yes
  1 |        1         |   node 1, frame 0     |  NO
  2 |        2         |   node 2, frame 0     |  NO
  3 |        3         |   node 0, frame 1     |  NO
```

4 of 12 rows aligned by coincidence. `orthogonal_variance` then does
`points.reshape(n_nodes, -1, 3)`, so what it treated as one node's trajectory
was a round-robin across body parts — every frame-to-frame difference was a
distance between *different* body parts. No error; just wrong bouts.

The GUI path (`_panels/analysis_panel.py:317`) had it right all along; only
the batch path transposed. (`898fe99`)

### 2.5 Both detectors crashed below 10 fps

`int(fps / 10)` reaches 0, and `gaussian_filter1d` divides by sigma squared:

```
fps=25 -> 19 bouts
fps=9  -> ZeroDivisionError: float division by zero
fps=1  -> ZeroDivisionError
```

(`1791763`)

### 2.6 `egocentric_variance`'s `amd_threshold` did nothing

Accepted as a parameter, never used — the prominence was hardcoded to
`amd * 7`:

```
amd_threshold=0.5   -> 19 bouts
amd_threshold=50.0  -> 19 bouts      (identical)
```

Parameter removed. (`1791763`)

### 2.7 Inference ran on CPU on Apple Silicon

Five copies of `cuda if torch.cuda.is_available() else cpu`, with no `mps`
branch, and `_resolve_model` never called `.to(device)` for PoseR-pretrained
names at all:

```
cpu  200 frames in 11.0s  ->  54.8 ms/frame
mps  200 frames in  4.7s  ->  23.6 ms/frame
```

For the 38,065-frame test video that is ~15 minutes rather than hours.
(`9a2d1fc`)

### 2.8 Panels only talked to each other in one open order

Cross-panel wiring was done at construction and only looked backwards, so
predictions reached the ethogram only if Ethogram was opened *before*
Inference:

```
open order                    before   after
ethogram -> inference         True     True
inference -> ethogram         False    True
```

(`061b861`)

### 2.9 Scoring crashed on a partial class set

`benchmark_model_performance` named every class in the project's `label_dict`
but did not tell sklearn which labels those were:

```
ValueError: Number of classes, 3, does not match size of target_names, 4
```

That is the normal case — a recording need not contain every defined
behaviour. (`db5622a`)

---

## 3. Still open

**A napari segfault during inference.** Crash report is unambiguous about the
site but not the trigger:

```
libGL.dylib        glGetIntegerv        ← SIGSEGV, invalid address 0x348
_ctypes…so         PyCFuncPtr_call
python3.10         bounded_lru_cache_wrapper
python3.10         slot_tp_init / type_call
```

An OpenGL call with no current context, from vispy, during visual
construction. Inference runs on raw `threading.Thread(daemon=True)` workers
(5 sites) with no shutdown handling, so closing napari mid-run leaves a
worker that then emits into a destroyed canvas. Unconfirmed; needs a stack
sample taken while it is wedged.

**A UI freeze when loading the full test video.** Every component measures
fast in isolation — pose read 0.06 s, 723k-point layer build 0.12 s, video
metadata 0.03 s, `Image` layer over a 38,065-frame dask array 0.12 s with one
chunk decoded — so the cause is not any of them. GL-side cost cannot be
measured headlessly. Needs `sample <pid>` during the freeze.

**`poser._loader.HyperParams` is baked into every released checkpoint** by
import path. Any refactor that moves or deletes `_loader.py` breaks all seven
unless that path stays resolvable.

**`cli/main.py` has the same missing-constructor-args bug** the GUI had
(`main.py:360`); it will fail identically.

**`training/data_prep.py:147` passes a bool where a node index is expected.**
`classification_data_to_bouts(center: Optional[int])` wants a node index, but
`data_prep` passes `center=dcfg.center_data`, which is `DataConfig.center_data:
bool = True`. `True` is used as index 1, so bouts are centred on node 1 rather
than on `center_node`. Unverified against training outcomes, but the types do
not line up.

**Seven of the nine published datasets cannot be used as benchmarks.**
Only ZebLR and OFT full are complete. The others:

- `ZebRep` ships `Zebtest.npy` but no `Zebtest_labels.npy`.
- `OFT/bodyonly` has a 13-node checkpoint, but the arrays in its folder are
  byte-identical to `OFT/full`'s 18-node ones, and its own config says
  `dataset: "OFT_full"`. The bodyonly arrays were never published.
- `CALMS21` task1 and task2, `FlyVFly` and `PAIRR24M` pickle a
  `pathlib.WindowsPath` and cannot be unpickled off Windows.
- `ZebTensor` downloaded empty.

Also note both OFT configs declare `V: 18` while their own `keypoints` list
has 13 entries, so the config is not a reliable source for node count. Read
it from the weights.

**`models/graph.py` shifts every species skeleton by one node.** The edge lists
are written 0-based but converted as if 1-based:

```python
neighbor_1base = [[0, 1], [0, 5], ..., [17, 18]]
neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
```

For `zebrafishlarvae` this leaves `Tailend` with degree 0, `Tail8` with degree
1, and three edges pointing at index -1 (which numpy resolves to the tail tip),
so the graph claims the snout is adjacent to the tail.

**This is not currently breaking inference**, because `A` is saved in the
state dict and `load_state_dict` overwrites the constructed graph. Verified:
`zeblr.ckpt`'s `A` matches the off-by-one graph exactly, so training and
inference are self-consistent.

It does matter for *new* training: correcting it produces models whose `A` is
incompatible with every released checkpoint. Fix deliberately and re-baseline,
or leave it and document it — but do not fix it silently.

---

## 4. How to reproduce

```bash
conda activate PoseR
PY=/opt/anaconda3/envs/PoseR/bin/python
ZEB=data/preprint_datasets/ZebLR

# what a checkpoint actually contains, read from the weights
$PY -m poser.cli.main model inspect data/BehaviourModels/preprint/zeblr.ckpt

# reproduce the published 0.91 on ZebLR
$PY ../scripts/score_decoder.py "$ZEB"/lightning_logs/version_0/epoch=5-*.ckpt \
    --data "$ZEB/Zebtest.npy" --config "$ZEB/decoder_config.yml"

# the same against the GUI's hand-labelled bouts, which also exercises
# preprocessing; add --no-align to reproduce the 0.323 failure
$PY ../scripts/score_decoder.py data/BehaviourModels/preprint/zeblr.repaired.ckpt \
    --data data/PoseRTestData/test_classification_file.h5 \
    --config data/PoseRTestData/decoder_config.yml
```

A checkpoint that fails to load with a missing-arguments `TypeError` stores no
architecture; embed one first, then score it:

```bash
$PY -m poser.cli.main model repair <ckpt> --config <decoder_config.yml>
```

`score_decoder.py` lives in `../scripts/`, outside the repo, with its own
README. It takes a checkpoint, labelled data (`.npy` pair or GUI `.h5`) and
the dataset's `decoder_config.yml`, and reports accuracy, balanced accuracy
and a confusion matrix.
