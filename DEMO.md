# CADE demo — one command

A launcher that sets up the whole project on a fresh laptop and opens the demo wall in Chrome.
It works even if nothing but Python is installed.

## Run it

**Windows** — double-click **`run_demo.bat`** (or in PowerShell: `powershell -ExecutionPolicy Bypass -File run_demo.ps1`).

**macOS / Linux**
```bash
chmod +x run_demo.sh
./run_demo.sh
```

That is all. The script:
1. finds Python 3.10+ (and stops with a download link if it is missing);
2. creates a virtual environment and installs the dependencies;
3. downloads the MITRE ATT&CK data if it is missing (~50 MB, once);
4. uses your local **Ollama** models if Ollama is installed, otherwise a built-in **mock** model, so the pipeline always runs;
5. starts CADE and opens `demo.html` in Chrome;
6. prints an API key (also copied to your clipboard) and a sample incident.

Then, in the page: **Live → paste the key → Connect → + New incident → paste the incident → Run**, and watch
*understanding → routing → synthesis → execution* fill in.

## Two modes

| | What it needs | What is real |
|---|---|---|
| **Portable (default)** | just Python (Ollama optional) | the full intent / routing / evolution / fitness pipeline is real computation. **Tool results are SIMULATED** and the wall labels them SYNTHETIC. LLM reasoning is real if Ollama is installed, mock otherwise. |
| **`-Live` / `--live`** | your own `.env` with real Elasticsearch, VirusTotal, Sigma, plus Ollama | real tools answer against your indexed data. This is **not portable** — it needs your private keys and services, so don't expect it to work on someone else's laptop. |

To get real local LLM reasoning in portable mode, install Ollama from <https://ollama.com/download> before running;
the script pulls `llama3.2:3b` and `qwen2.5:3b` the first time.

## Options

| Flag | Effect |
|---|---|
| `-Full` / `--full` | install the paper's embeddings (sentence-transformers + FAISS). Large and slow; not needed for the demo. |
| `-Live` / `--live` | use the real tools from your `.env` instead of the simulator. |
| `-NoModels` / `--no-models` | do not download Ollama models. |
| `-Port 8030` / `--port 8030` | run on a different port (default 8020). |

## Stop it
- **Windows:** close the minimized `python … uvicorn` window.
- **macOS / Linux:** `kill` the PID the script printed (server log: `cade_demo_server.log`).

## Sending this to someone
Zip the project folder but leave out the big, regenerated parts:
- `.venv/`, any `*.db`, `lab/`, and `app/data/attack/*.json`.

The script rebuilds the virtual environment and re-downloads the ATT&CK data on their machine.

## Notes
- The launcher never edits your `.env` and never touches your other databases. The demo uses its own `cade_demo.db`.
- The "demo testbed" diagram on the wall shows the reference VM topology (labelled *not part of paper 968*).
  In portable mode everything actually runs locally in simulate mode; the wall's status text and the SYNTHETIC
  badges always reflect what truly happened.
- This is a research/reference implementation, not a production SOAR platform.
