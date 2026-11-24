"""Run a small retrieval-augmented generation example with vLLM.

Use a dedicated Linux/NVIDIA environment. Install requirements.txt,
copy the trained adapter directory alongside this script, then run:

    python serve_vllm_rag.py

Use --adapter-dir for another saved adapter or --base-only to use the base model.
The embedding model runs on CPU; vLLM uses the GPU. See ../README.md for setup.
"""
import argparse
import json
import re
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from sentence_transformers import SentenceTransformer, util

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = Path(__file__).resolve().parent / "adapter"

CORPUS = [
    "LoRA freezes W and learns a low-rank BA update; r*(d_in+d_out) parameters train per matrix.",
    "PagedAttention stores the KV-cache in non-contiguous blocks to cut waste.",
    "GRPO standardises rewards within a group so different reward scales contribute comparably.",
    "Dividing by the group standard deviation puts every prompt group on the same scale, so a "
    "high-variance group cannot dominate the update and a low-variance group still teaches the model.",
    "Nucleus sampling draws from the smallest set of tokens whose probability mass exceeds p.",
    "Nicosia is the capital of Cyprus.",
]

# Illustrative retrieval threshold, used by the retrieval example. Inspect the retrieved
# passages: a high similarity score does not prove that a passage answers a question.
MIN_SIM = 0.35


def build_retriever():
    embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    C = embed.encode(CORPUS, convert_to_tensor=True)

    def retrieve(query, k=2, min_sim=None):
        if k < 1:
            raise ValueError("k must be positive")
        if min_sim is None:
            min_sim = MIN_SIM
        q = embed.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(q, C)[0]
        top = scores.topk(min(k, len(CORPUS)))
        return [CORPUS[int(i)] for s, i in zip(top.values, top.indices) if float(s) >= min_sim]

    return retrieve


def build_prompt(query, docs):
    """Use numbered passages, preserve their meaning, and abstain without evidence."""
    context = "\n".join(f"[{i}] {d}" for i, d in enumerate(docs)) or "(no relevant passages found)"
    return (
        "Answer using only the numbered context passages below. Preserve their meaning.\n"
        "Give one or two short sentences and finish with the citation of the passage used, such as [0].\n"
        "If the context does not answer the question, reply with exactly: unknown\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )



def citation_is_valid(raw, docs):
    """Check citation IDs only; this does not check whether the text supports the answer."""
    cited = [int(i) for i in re.findall(r"\[(\d+)\]", raw)]
    return bool(cited) and all(i < len(docs) for i in cited) and not raw.lower().startswith("unknown")


def validate_adapter(path):
    config_path = path / "adapter_config.json"
    if not config_path.is_file() or not (path / "adapter_model.safetensors").is_file():
        raise ValueError(f"Incomplete adapter at {path}. Copy the saved adapter directory or use --base-only.")
    config = json.loads(config_path.read_text())
    if config.get("base_model_name_or_path") != BASE or config.get("r") != 16:
        raise ValueError(f"Expected an r=16 adapter trained on {BASE}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--adapter-dir", type=Path, default=ADAPTER_DIR)
    group.add_argument("--base-only", action="store_true", help="Explicitly run without a trained adapter")
    options = parser.parse_args()
    has_adapter = not options.base_only
    adapter_dir = options.adapter_dir.expanduser().resolve()
    if has_adapter:
        validate_adapter(adapter_dir)

    llm = LLM(
        model=BASE,
        dtype="float16",                 # "bfloat16" on Ampere or newer
        gpu_memory_utilization=0.80,
        enable_prefix_caching=True,      # reuse the KV-cache across shared prefixes
        enable_lora=has_adapter,         # hot-swap adapters per request
        max_lora_rank=16,               # matches the training lab
        max_model_len=2048,
        # tensor_parallel_size=2,        # uncomment for 2-GPU tensor parallelism
    )

    retrieve = build_retriever()
    # Greedy (temperature 0) and no repetition penalty: a grounded answer has to copy prompt tokens.
    sp = SamplingParams(temperature=0.0, max_tokens=128, repetition_penalty=1.0)
    lora_request = LoRARequest("session1-adapter", 1, str(adapter_dir)) if has_adapter else None
    tokenizer = llm.get_tokenizer()

    for query in ["Why does GRPO divide by the group standard deviation?",   # answered by the course passages
                  "What is the capital of Cyprus?",                            # answered by one passage
                  "Who painted the Mona Lisa?"]:                               # not answerable -> unknown
        docs = retrieve(query)
        prompt = build_prompt(query, docs)
        # Use the same chat format as SFT, with no second addition of special tokens.
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True,
        )
        raw = llm.generate([{"prompt_token_ids": prompt_ids}], sp,
                           lora_request=lora_request)[0].outputs[0].text.strip()
        # Citation ID check: only an answer that cites a passage really present
        # in the context is shown; "unknown" or a citation of a non-existent passage -> "unknown".
        valid_citation = citation_is_valid(raw, docs)
        print("Q:", query)
        print("   context:", docs or "(no relevant passages found)")
        print("   raw model output:", repr(raw))
        print("   displayed answer:", raw if valid_citation else "unknown")


if __name__ == "__main__":
    main()
