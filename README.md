# GenAI-Perf on Triton Inference Server (vLLM backend)

Quick notes + a repeatable workflow to run **GenAI-Perf** (Perf Analyzer wrapper) against a **Triton Inference Server** deployment hosting a **vLLM** model, plus a short guide to interpreting the results.

> **Reference (may be outdated):** https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/docs/tutorial.html

---

## What this repo/doc is for

- Launch the NVIDIA Triton SDK container that includes `genai-perf` / `perf_analyzer`
- Verify your Triton endpoint is reachable
- Run a **warm-up** and a **real measurement** with streaming enabled
- Interpret the **LLM Metrics** table
- Sanity-check the numbers with a few quick calculations

---

## Prerequisites

- A running Triton server reachable from your host (examples assume `localhost`)
- A deployed model named `vllm_model` (HTTP endpoint used below: `:8000`, gRPC endpoint: `:8001`)
- Docker with NVIDIA Container Toolkit (`--gpus=all`)

---

- Model on test:
```
sudo docker run --rm -d --gpus=all --runtime=nvidia   --name triton-vllm-serve   --entrypoint /opt/tritonserver/bin/tritonserver   -p 8000:8000 -p 8001:8001 -p 8002:8002   -v "$PWD/Documents/triton/vllm_backend/samples/model_repository:/models"   triton-vllm-gptoss:25.08-hotfix5   --model-repository=/models
```


---
---
  
## Run the Triton SDK container (includes GenAI-Perf)

```bash
export RELEASE="24.06"
docker run -it --net=host --gpus=all nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk
```

> `--net=host` is used so the container can reach your Triton server on `localhost:*`.

---

## Quick connectivity test (HTTP generate)

From your host (or from the container if networking matches), call Triton’s HTTP endpoint:

```bash
curl -X POST localhost:8000/v2/models/vllm_model/generate \
  -d '{"text_input": "What is Triton Inference Server?", "parameters": {"stream": false, "temperature": 0, "max_tokens": 50}}'
```

Example response (truncated):

```json
{"model_name":"vllm_model","model_version":"1","text_output":"What is Triton Inference Server?..."}
```

---

## Warm-up run

Warm-up primes caches and stabilizes early measurements.

```bash
genai-perf -m vllm_model --service-kind triton --backend vllm -u localhost:8001 \
  --streaming --synthetic-input-tokens-mean 200 --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 100 --output-tokens-stddev 0 --output-tokens-mean-deterministic \
  --num-prompts 10 --concurrency 1
```

Example output (excerpt):

```text
LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃                Statistic ┃           avg ┃           min ┃           max ┃           p99 ┃           p90 ┃           p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ Time to first token (ns) │    46,993,496 │    29,975,042 │    94,178,902 │    89,480,725 │    55,784,297 │    47,668,953 │
│ Inter token latency (ns) │    18,594,863 │    15,580,181 │    21,211,021 │    21,188,443 │    20,724,169 │    19,857,881 │
│     Request latency (ns) │ 2,468,158,071 │ 2,447,858,527 │ 2,556,108,930 │ 2,545,300,491 │ 2,471,626,202 │ 2,466,068,090 │
│         Num output token │           132 │           115 │           156 │           155 │           149 │           141 │
│          Num input token │           200 │           200 │           200 │           200 │           200 │           200 │
└──────────────────────────┴───────────────┴───────────────┴───────────────┴───────────────┴───────────────┴───────────────┘
Output token throughput (per sec): 53.58
Request throughput (per sec): 0.41
```

---

## Real measurement run

```bash
genai-perf -m vllm_model --service-kind triton --backend vllm -u localhost:8001 \
  --streaming --synthetic-input-tokens-mean 200 --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 100 --output-tokens-stddev 0 --output-tokens-mean-deterministic \
  --num-prompts 50 --concurrency 1
```

Example output (excerpt):

```text
LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃                Statistic ┃           avg ┃           min ┃           max ┃           p99 ┃           p90 ┃           p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ Time to first token (ns) │    44,263,597 │    28,438,223 │    87,805,434 │    83,723,554 │    54,422,435 │    46,879,044 │
│ Inter token latency (ns) │    19,283,304 │    15,700,461 │    20,665,405 │    20,599,649 │    20,157,857 │    20,000,876 │
│     Request latency (ns) │ 2,465,556,222 │ 2,446,314,775 │ 2,547,081,921 │ 2,537,578,239 │ 2,472,466,746 │ 2,464,234,681 │
│         Num output token │           127 │           118 │           155 │           152 │           135 │           128 │
│          Num input token │           200 │           200 │           200 │           200 │           200 │           200 │
└──────────────────────────┴───────────────┴───────────────┴───────────────┴───────────────┴───────────────┴───────────────┘
Output token throughput (per sec): 51.56
Request throughput (per sec): 0.41
```

---

## How to read the table

### Statistic columns (one-liners)

- **avg**: arithmetic mean across requests in the run
- **min / max**: best / worst single observed value
- **p75 / p90 / p99**: percentiles (e.g. **p99** means 99% of requests were ≤ that value)

### Metric rows (one-liners)

- **Time to first token (TTFT)**: time from sending the request until the first output token arrives (streaming UX “snappiness”)
- **Inter token latency**: average time between consecutive output tokens once generation has started (inverse ≈ tokens/sec)
- **Request latency**: end-to-end time for the full response (TTFT + generation + overhead)
- **Num output token**: how many tokens the model actually generated per request
- **Num input token**: how many tokens were in the prompt per request
- **Output token throughput**: aggregate generated tokens per second across the run
- **Request throughput**: requests completed per second across the run

---

## Sanity-checking the averages (example math)

Using example “avg” numbers:

### 1) Convert to human units

- **TTFT avg** = 60,089,038 ns = **60.09 ms**
- **Inter-token latency avg** = 20,034,708 ns = **20.03 ms/token**
- **Request latency avg** = 2,480,921,379 ns = **2.4809 s**
- **Avg output tokens** ≈ **123 tokens**
- **Avg input tokens** = **200 tokens**

### 2) Derive generation speed from inter-token latency

- **Tokens/sec** ≈ 1 / (20.034708 ms)  
  = 1 / 0.020034708 s  
  = **49.91 tokens/sec**

(Usually close to the tool’s **Output token throughput**; small differences come from overhead and rounding.)

### 3) Check if request latency matches “TTFT + generation time”

Approx:

- Total time ≈ TTFT + (output_tokens − 1) × inter_token_latency  
  = 60.09 ms + 122 × 20.03 ms  
  = 60.09 ms + 2444.23 ms  
  = **2504.32 ms** ≈ **2.504 s**

Measured request latency avg = **2.481 s**, so you’re within ~**23 ms** (normal variance: measurement method, overlap/batching, and averaging).

### 4) Cross-check throughput

If request throughput is ~0.40 req/s and each request outputs ~123 tokens:

- Output tokens/sec ≈ 0.40 × 123 = **49.2 tokens/sec**

That should line up with reported output token throughput (with minor rounding differences).

---

## One-line interpretation (concurrency=1, streaming)

- **TTFT ~60 ms**: first token arrives fast (good interactive feel)
- **~20 ms/token → ~50 tok/s**: steady-state generation speed
- **Total latency ~2.48 s**: dominated by generating ~123 tokens at ~20 ms/token
- **Throughput ~0.40 req/s**: expected when each request takes ~2.48 s at concurrency 1

---

## Notes / Tips

- Keep warm-up + measurement settings consistent (tokens, streaming, concurrency) when comparing runs.
- Increase `--concurrency` to explore saturation behavior and throughput scaling.
- If results are noisy, increase `--num-prompts` and/or use a longer `--measurement-interval` in the underlying Perf Analyzer settings (GenAI-Perf will pass these through).
