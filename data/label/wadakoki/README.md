# data/label/wadakoki/

How the Qwen and Gemma report scores in label `L` (see `docs/index.en.md`, section 2.1) were generated: the prompt, the generation settings, and the code. The score files themselves are source labels and are not part of this repository (see `data/README.md`).

| File | Content |
|---|---|
| `prompt_en_v0_4.md` | Severity prompt v0.4 (English), shared by both models |
| `gemma_v04.json`, `qwen_v04.json` | Model (repository, revision, GGUF file), llama.cpp release, sampling, seed |
| `make_reports.py` | Writes the input JSONL: `StudyInstanceUID` and the original `Report` of the 4,349 train studies with no official labels |
| `launch_server.py` | Starts `llama-server` on 127.0.0.1 with the config's settings |
| `generate.py` | Sends each report and stores the responses (resumable) |
| `harvest.py` | Checks the stored responses and exports the accepted scores to `labels_accepted.csv` |

## Settings

| | Gemma v0.4 | Qwen v0.4 |
|---|---|---|
| Model | `unsloth/gemma-4-26B-A4B-it-GGUF`, `gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf` (revision `c099eb48…`) | `unsloth/Qwen3.8-27B-GGUF`, `Qwen3.8-27B-UD-Q4_K_XL.gguf` (revision `4ca72078…`) |
| Engine | llama.cpp `b11026`, 8 parallel slots, 8,192 tokens of context per slot | llama.cpp `b11026`, 16 parallel slots, 8,192 tokens of context per slot |
| Sampling | temperature 0.4, top_p 0.95, top_k 64 | temperature 0.6, top_p 0.95, top_k 20 |
| Common | up to 2,304 tokens, thinking on (budget 1,024), output fixed by a JSON schema, up to 3 attempts | same |
| Seed | first 4 bytes of `sha256(StudyInstanceUID)` + 340918 + attempt index | same, with 390918 |

- Each request has the prompt as the system message and `Report:` followed by the original report as the user message. No other cases, existing labels, or image-model predictions are sent.
- The output gives, per finding, a `score` (0.05–0.95, two decimals), a `source` (direct / uncertain / conflicting / inferred / default) and a short `evidence` (≤ 120 characters). `default` is accepted only with the fixed default score and evidence; an unreadable report returns `{"error": "unreadable_report"}`.
- The accepted `score` is used as the label as is. Studies that were never accepted are left out, not filled: 4 for Gemma (refused) and 4 for Qwen (1 refused, 3 that returned the default for every finding and were removed in the quality check), so both label files have 4,345 rows.
- Gold studies are not used. Both runs used a Colab A100.

## Commands

```bash
uv run python data/label/wadakoki/make_reports.py --train-csv input/rsna-knee-abnormality-detection/train.csv --out output/llm_labels/reports.jsonl
uv run python data/label/wadakoki/launch_server.py --binary <llama-server> --model <gguf> --config data/label/wadakoki/gemma_v04.json --output output/llm_labels/gemma_v04/server
uv run python data/label/wadakoki/generate.py --config data/label/wadakoki/gemma_v04.json --reports output/llm_labels/reports.jsonl --root output/llm_labels/gemma_v04 --attempt full
uv run python data/label/wadakoki/harvest.py --root output/llm_labels/gemma_v04 --output output/llm_labels/gemma_v04/harvest
```

LLM generation is not fully deterministic on GPUs with parallel slots even with fixed seeds, so regenerated scores can differ slightly from the ones we used.
