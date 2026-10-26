# LERN: Local Evidence Rank Normalization

LERN adapts a frozen vision-language model by retrieving visually similar past images, averaging their complete class-margin vectors, and comparing class-specific historical ranks. The image encoder and text prototypes remain fixed.

## Installation

Use Python 3.10 or newer. Install a PyTorch build suitable for your device,
then install the lightweight package:

```bash
git clone https://github.com/mendicant04/LERN.git
cd LERN
python -m pip install -e .
python examples/synthetic_stream.py
```

Raw-image evaluation also needs the image/model dependencies:

```bash
python -m pip install -e ".[images]"
```

The image frontend was checked with PyTorch 2.8, torchvision 0.23,
OpenCLIP 3.3, timm 1.0, and transformers 4.44. Cached-feature evaluation
needs only NumPy and PyTorch. All paths are supplied through command-line
options; pretrained weights and datasets stay outside the repository.

## Evaluate MedMNIST images

Download the official 224-pixel datasets:

```bash
python scripts/download_medmnist.py --root data/medmnist --tasks all
```

Run the complete test sets with fresh state for stream orders 0, 1, and 2:

```bash
python -m lern run --data-dir data/medmnist --tasks all \
  --seeds 0,1,2 --device cuda --output outputs/biomedclip
python -m lern summarize --input outputs/biomedclip
```

The default model is
`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` on Hugging Face.
The first run downloads it. A local checkpoint can be selected with
`--checkpoint-dir checkpoints/biomedclip`; include the OpenCLIP model
configuration, weights, BERT configuration, and tokenizer files in that
directory. Standard OpenCLIP backbones can use, for example,
`--model ViT-B-16 --pretrained openai`.

Use `--tasks breastmnist --seeds 0 --limit 16` for a small smoke run.
`--limit 0`, the default, evaluates every test image. The output records
the actual sample count. Existing results are preserved unless
`--overwrite` is supplied.

The bundled class names follow the evaluated expanded MedMNIST vocabulary.
The default prompt is `this is a photo of {}`. Supply `--template`
repeatedly to use a fixed template ensemble. The raw-image frontend
supports timm-based OpenCLIP vision towers and standard OpenCLIP ViTs.
For a custom tower, supply features through the interface below.

## Evaluate frozen feature caches

This interface also supports PubMedCLIP, DermLIP, and other backbones
when their projected and preprojection features are available.

```bash
python -m lern cache --input features/breastmnist.npz \
  --task breastmnist --seeds 0,1,2 --device cpu \
  --output outputs/cached
python -m lern summarize --input outputs/cached
```

Each NPZ contains aligned arrays in the original dataset order:

| Array | Shape | Meaning |
|---|---|---|
| `h` | N × D | Preprojection image features |
| `v` | N × E | Projected image features |
| `text` | C × E | Fixed class text prototypes, in label order |
| `indices` | N | Unique sample IDs; optional, defaults to row positions |
| `y` | N | Integer targets for evaluation only; optional |


`preprojection` and `images` are accepted aliases for `h` and `v`.
`targets` is an alias for `y`. Labels never enter the adapter.
The evaluator saves predictions in stream order together with their
original IDs. Cached execution measures classifier behavior; raw-image
execution additionally performs actual image encoding.

## Online API and protocol

`examples/synthetic_stream.py` shows the one-image-at-a-time interface
without a model download. The core adapter is exported as `LERN`;
its existing implementation name remains available for integration.
Internal statistics use float64. Image encoding uses float32 in the
raw-image command, with TF32 disabled.

LERN maintains disjoint reference and calibration banks. Odd arrivals enter
the reference bank, and even arrivals enter the calibration bank, after
prediction. The scorer refreshes at powers of two, with
`k = ceil(sqrt(reference_count))`. The first eight predictions use the
frozen classifier. The default stores the complete history; its memory
therefore grows with stream length. Integer sample IDs must be unique
and stable across repeated evaluations.

Reset adaptation by constructing a new adapter for each dataset and
stream order. Cached calls may use `--chunk` to process several arrivals
efficiently while preserving the causal prediction order.

## Outputs and tests

Each run writes `TASK_seedN.npz` with `indices`, `predictions`, and
`frozen`, plus a JSON file with sample count, configuration, ACC, and BA.
ACC and BA are percentages. `summarize` writes `per_dataset.csv` and
`summary.json`, averaging equally over datasets and stream orders.

```bash
python -m unittest discover -s tests -v
```

Tests cover causality, streaming behavior, invalid inputs, and the
evaluation interface. They run on CPU without datasets or checkpoints.

## Acknowledgments

The model frontend uses [OpenCLIP](https://github.com/mlfoundations/open_clip).
The benchmark uses [MedMNIST](https://github.com/MedMNIST/MedMNIST).
Pretrained models and datasets retain their respective licenses.
