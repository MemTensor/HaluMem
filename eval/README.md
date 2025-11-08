# Evaluation — Memory Systems with HaluMem

This folder contains the evaluation toolkit for measuring hallucination and memory performance of different memory systems on the **HaluMem** benchmark.

---

## 🚀 Quickstart

```bash
# 1. enter eval folder
cd eval

# 2. install dependencies (Poetry)
poetry install --with eval
```

> If you prefer `pip` or `conda`, install the packages listed in `pyproject.toml` accordingly.

---

## ⚙️ Configuration

1. Copy the example environment file and fill in your own keys and service endpoints:

```bash
cp .env-example .env
```

2. Edit `.env` to include the required configurations for your setup.
   The key parameters are grouped as follows:

#### 🔑 Model Configuration

| Variable             | Description                                     | Example                     |
| -------------------- | ----------------------------------------------- | --------------------------- |
| `OPENAI_API_KEY`     | API key for OpenAI models                       | `sk-xxxx`                   |
| `OPENAI_BASE_URL`    | Base URL for custom OpenAI-compatible endpoints | `https://api.openai.com/v1` |
| `OPENAI_MODEL`       | Model name used for evaluation                  | `gpt-4o`                    |
| `OPENAI_MAX_TOKENS`  | Maximum token limit per request                 | `16384`                     |
| `OPENAI_TEMPERATURE` | Sampling temperature for generation             | `0.0`                       |
| `OPENAI_TIMEOUT`     | Timeout (seconds) for API calls                 | `300`                       |

#### 🔁 Retry Policy

| Variable                              | Description                                 | Example     |
| ------------------------------------- | ------------------------------------------- | ----------- |
| `RETRY_TIMES`                         | Maximum retry attempts                      | `3`         |
| `WAIT_TIME_LOWER` / `WAIT_TIME_UPPER` | Random wait range (seconds) between retries | `10` / `30` |

#### 🧠 Memory Systems

| Variable                                                     | Description                      | Example                                                      |
| ------------------------------------------------------------ | -------------------------------- | ------------------------------------------------------------ |
| `MEM0_API_KEY`                                               | API key for Mem0                 | `xxx`                                                        |
| `ZEP_API_KEY`                                                | API key for Zep Memory System    | `xxx`                                                        |
| `MEMOS_KEY`                                                  | Auth token for local MemOS       | `"Token mpg-xxxxx"`                                          |
| `SUPERMEMORY_API_KEY`                                        | API key for Supermemory          | `xxx`                                                        |
| `MEMOS_URL`                                                  | Local MemOS endpoint             | `"http://127.0.0.1:8001"`                                    |
| `MEMOS_ONLINE_URL`                                           | Online MemOS endpoint (optional) | `"https://memos.memtensor.cn/api/openmem/v1"`                |
| `MEMOBASE_PROJECT_URL`                                       | Memobase project endpoint        | `"http://127.0.0.1:8001"`                                    |
| `MEMOBASE_PROJECT_TOKEN`                                     | Auth token for Memobase          | `"secret"`                                                   |
| `MEMOBASE_DB_HOST` / `PORT` / `USER` / `PASSWORD` / `DB_NAME` | Local DB config for Memobase     | `"127.0.0.1"`, `8002`, `"user_name"`, `"password"`, `"database_name"` |

> 💡 Tip:
>
> * If you're evaluating only one specific memory system (e.g., Mem0), you can omit the configuration settings for all other systems.
> * For local deployments, ensure corresponding services are running before evaluation.

---

## 🔧 Local deployments (MemOS / Memobase)

For MemOS and Memobase we rely on local/service deployments. See these projects for install & runtime instructions:

* MemOS: [https://github.com/MemTensor/MemOS](https://github.com/MemTensor/MemOS)
* Memobase: [https://github.com/memodb-io/memobase](https://github.com/memodb-io/memobase)

Make sure the services are running and the corresponding endpoints are reachable before running the evaluation scripts.

---

## 📁 Supported Memory System Wrappers

The repo includes evaluation adapters for multiple memory systems. Filenames map to system wrappers:

* `eval_memzero.py` — Mem0 (default)
* `eval_memzero_graph.py` — Mem0 (Graph variant)
* `eval_memos.py` — MemOS
* `eval_memobase.py` — Memobase
* `eval_zep.py` — Zep
* `eval_supermemory.py` — SuperMemory

Each adapter follows the same input / output contract so the downstream scorer can compare systems fairly.

---

## 🧪 Running the Evaluation

To evaluate memory systems using **HaluMem**, execute the following commands step by step:

1. Generate run artifacts (extract memories & generate predictions): using **Mem0** as an example, first run the following command to process dialogue input, extract memory points, and perform QA retrieval.

   ```bash
   python eval_memzero.py
   ```

   Please set data_path and version in eval_memzero.py as needed:
   * `data_path` : path to dataset
   * `version` : evaluation version identifier (used to tag outputs)

   Specifically, for Zep, due to its asynchronous design, we provide an additional `run_task` argument in `eval_zep.py` to specify either the dialogue addition task (`run_task="add"`) or the memory retrieval task (`run_task="search"`, which includes both memory update and QA retrieval). The dialogue addition task must be executed first, and once the dialogue processing is complete (progress can be monitored via threads and episodes on the official platform), the memory retrieval task can then be performed.

2. Score & aggregate results: run the following command to evaluate **Mem0** on memory extraction, memory update, and memory QA tasks, and aggregate the results.

   ```bash
   python evaluation.py --frame memzero --version default
   ```

   * `--frame` : which adapter/system frame to evaluate (e.g., `memzero`, `memos`, `memobase`)
   * `--version` : same identifier used during run generation

3. Outputs: the final results will be saved in the `results` folder.
   

These artifacts enable reproducibility and further analysis (error inspection, evidence linking, per-memory-point breakdowns).

---