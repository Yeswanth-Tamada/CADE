# Running the CADE experiments on a GPU laptop

This guide is for running the long experiments (LinUCB training with real playbooks, playbook quality, library
growth) on a Windows laptop with an NVIDIA GPU. On a GPU they take hours instead of days. Nothing here needs the
Google Cloud VMs, and nothing costs money.

## 0. Check the GPU (2 minutes)

Open **Task Manager → Performance → GPU** and note the GPU name and **Dedicated GPU memory** (VRAM). Then pick the
models:

| VRAM | Generator / judge | Matches the paper? |
|---|---|---|
| 6–8 GB | `llama3.2:3b` / `qwen2.5:3b` (or `gemma3:4b` as judge) | different, smaller models |
| 12 GB | `qwen3:14b` / `gemma3:12b` | **yes (Table 3)** — Ollama swaps the two models in and out, a little slower |
| 16 GB+ | `qwen3:14b` / `gemma3:12b` | **yes (Table 3)**, comfortably |

Results are only comparable between runs that used the same two models.

## 1. Install (once, ~20 minutes)

1. **NVIDIA driver**: up to date (GeForce Experience or nvidia.com).
2. **Python 3.11**: python.org → Windows installer → tick "Add python.exe to PATH".
3. **Git**: git-scm.com.
4. **Ollama**: ollama.com/download. It uses the NVIDIA GPU automatically.

Open **PowerShell** and download the models (pick the pair from step 0):

```powershell
ollama pull llama3.2:3b          # always needed (intent model)
ollama pull qwen2.5:3b           # small judge   -- or: ollama pull gemma3:4b
# for the paper's models instead:
ollama pull qwen3:14b
ollama pull gemma3:12b
```

## 2. Get CADE and set it up (once, ~15 minutes)

```powershell
cd $HOME\Documents
git clone https://github.com/Yeswanth-Tamada/CADE.git
cd CADE
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-tools.txt
Copy-Item .env.example .env
```

(If `Activate.ps1` is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then try again.)

Open `.env` in Notepad and set these lines (paper models shown; use the small pair for 6–8 GB):

```
CADE_LLM_PROVIDER=ollama
CADE_OLLAMA_MODEL=llama3.2:3b
CADE_GENERATOR_MODEL=qwen3:14b
CADE_JUDGE_MODEL=gemma3:12b
CADE_LLM_REQUEST_TIMEOUT_SECONDS=300
CADE_EMBEDDER=sentence_transformers
CADE_VECTOR_INDEX=faiss
```

Qwen3's "thinking" mode is switched off automatically (it would slow every call down and disturb the JSON answers).
CADE's prompts fit Ollama's default 4,096-token context window (about 3,200 tokens are used); only if Ollama's log
(`%LOCALAPPDATA%\Ollama\server.log`) shows `truncated = 1` add `CADE_OLLAMA_NUM_CTX=8192` to `.env`.

## 3. Check the setup (~5 minutes)

```powershell
python -m pytest -q                                          # all tests should pass (offline, no models used)
python scripts/eval_routing_generated.py --time-test         # ONE real synthesis; prints how long it took
```

While the time test runs, `ollama ps` in another PowerShell window should show the model with **100% GPU**. If it
says CPU, the GPU is not being used (update the NVIDIA driver, restart Ollama).

The time test's number × about 100–130 is roughly how long step 4 takes.

## 4. Run the experiments (hours; can be stopped and restarted)

Keep the laptop **plugged in, lid open, on a hard surface**. The scripts stop Windows from idle-sleeping while they
run (closing the lid still sleeps it). If anything stops them, run the same command again: finished playbooks are
saved and are not redone.

```powershell
$d = "reports\$(Get-Date -Format yyyy-MM-dd)"
python scripts/eval_routing_generated.py --out-dir "$d\routing_generated"
python scripts/eval_quality_paper.py --out-dir "$d\quality" --routing-results "$d\routing_generated\results.json"
python scripts/eval_library_growth.py --out-dir "$d\growth"
```

The first command writes most of the playbooks; the other two reuse them and add the rest.

## 5. Send the results back

Zip and send these two folders:

- `reports\<date>\` — the three reports (`REPORT.md` and `results.json` each)
- `reports\quality_cache\<generator>__<judge>\` — every synthesized playbook (lets the other side rebuild reports
  without re-running anything)

Nothing in them is secret: they contain playbooks, scores and timings for public Splunk Attack Data scenarios.
