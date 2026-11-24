# Python examples

Use Python 3.12 on Linux with a compatible NVIDIA CUDA GPU. Training and serving need separate Python environments. Run the commands below from the package root.

## Training

```bash
python3.12 -m venv .venv-training
source .venv-training/bin/activate
python -m pip install --upgrade pip
```

Install a PyTorch CUDA build compatible with your driver using the [official selector](https://pytorch.org/get-started/locally/), then run:

```bash
python -m pip install -r scripts/training/requirements.txt
python -m pip check
python scripts/training/train_qlora.py --num-examples 16 --max-steps 1 --output-dir scripts/training/adapter-smoke
python scripts/training/train_qlora.py
```

The full training run saves the adapter and tokenizer to `scripts/training/adapter/`. Model and dataset downloads require internet access.

## Serving

Open another terminal and create a separate serving environment:

```bash
python3.12 -m venv .venv-serving
source .venv-serving/bin/activate
python -m pip install --upgrade pip
python -m pip install -r scripts/serving/requirements.txt
python -m pip check
python scripts/serving/serve_vllm_rag.py --adapter-dir scripts/training/adapter
```

For a different machine, copy the saved adapter directory to that machine and pass its location with `--adapter-dir`. Use `--base-only` to deliberately run without an adapter.

The serving script performs offline inference. Inspect the retrieved passages and raw answers: valid citation IDs do not prove that a passage supports the answer. Check the pinned vLLM release's [GPU requirements](https://docs.vllm.ai/en/v0.27.1/getting_started/installation/gpu/) against the target machine.
