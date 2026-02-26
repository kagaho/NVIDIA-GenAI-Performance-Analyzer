GenAI-Perf Tool (on top of Triton Inference Server)
==================================================

Reference (outdated): https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/docs/tutorial.html

rteixeira@shockwave:~/Documents/triton/GenAI-Perf$ export RELEASE="24.06"
rteixeira@shockwave:~/Documents/triton/GenAI-Perf$ docker run -it --net=host --gpus=all  nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk

# Testing access:
 127 rteixeira@shockwave:~/Documents/triton/GenAI-Perf$ curl -X POST localhost:8000/v2/models/vllm_model/generate -d '{"text_input": "What is Triton Inference Server?", "parameters": {"stream": false, "temperature": 0, "max_tokens": 50}}'
{"model_name":"vllm_model","model_version":"1","text_output":"What is Triton Inference Server?**  \n   - Triton Inference Server is a tool that helps run machine learning models on servers. It supports multiple frameworks like TensorFlow, PyTorch, and ONNX, and can handle many requests at once.\n\n2. **Why use Trit"}


# WARM UP
root@shockwave:/workspace# genai-perf -m vllm_model --service-kind triton --backend vllm -u localhost:8001 \
  --streaming --synthetic-input-tokens-mean 200 --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 100 --output-tokens-stddev 0 --output-tokens-mean-deterministic \
  --num-prompts 10 --concurrency 1
2026-02-03 10:52 [INFO] genai_perf.wrapper:137 - Running Perf Analyzer : 'perf_analyzer -m vllm_model --async --input-data artifacts/vllm_model-triton-vllm-concurrency1/llm_inputs.json --service-kind triton -u localhost:8001 --measurement-interval 10000 --stability-percentage 999 --profile-export-file artifacts/vllm_model-triton-vllm-concurrency1/profile_export.json -i grpc --streaming --concurrency-range 1'
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




# REAL MEASUREMENT
root@shockwave:/workspace# genai-perf -m vllm_model --service-kind triton --backend vllm -u localhost:8001 \
  --streaming --synthetic-input-tokens-mean 200 --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 100 --output-tokens-stddev 0 --output-tokens-mean-deterministic \
  --num-prompts 50 --concurrency 1
2026-02-03 10:37 [INFO] genai_perf.wrapper:137 - Running Perf Analyzer : 'perf_analyzer -m vllm_model --async --input-data artifacts/vllm_model-triton-vllm-concurrency1/llm_inputs.json --service-kind triton -u localhost:8001 --measurement-interval 10000 --stability-percentage 999 --profile-export-file artifacts/vllm_model-triton-vllm-concurrency1/profile_export.json -i grpc --streaming --concurrency-range 1'

2026-02-03 10:53 [INFO] genai_perf.wrapper:137 - Running Perf Analyzer : 'perf_analyzer -m vllm_model --async --input-data artifacts/vllm_model-triton-vllm-concurrency1/llm_inputs.json --service-kind triton -u localhost:8001 --measurement-interval 10000 --stability-percentage 999 --profile-export-file artifacts/vllm_model-triton-vllm-concurrency1/profile_export.json -i grpc --streaming --concurrency-range 1'
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

# ANALYSIS
==========

Statistics columns (1 line each)

- avg: arithmetic mean across the 50 requests.
- min / max: best / worst single observed value.
- p75 / p90 / p99: 75th / 90th / 99th percentile (e.g., p99 = 99% of requests were ≤ that value).

Metrics rows (1 line each)

- Time to first token (TTFT): time from sending the request until the first output token arrives (streaming UX “snappiness”).
- Inter token latency: average time between consecutive output tokens once generation has started (inverse ≈ tokens/sec).
- Request latency: total end-to-end time for the full response (TTFT + generating all tokens + overhead).
- Num output token: how many tokens the model actually generated per request.
- Num input token: how many tokens were in the prompt per request.
- Output token throughput: aggregate generated tokens per second across the run.
- Request throughput: requests completed per second across the run.

Redo the calculations from your “avg” numbers

1) Convert the key averages to human units
- TTFT avg = 60,089,038 ns = 60.09 ms

- Inter-token latency avg = 20,034,708 ns = 20.03 ms/token

- Request latency avg = 2,480,921,379 ns = 2.4809 s

- Avg output tokens = 123 tokens

- Avg input tokens = 200 tokens

2) Derive generation speed from inter-token latency

- Tokens/sec ≈ 1 / (20.034708 ms)
  = 1 / 0.020034708 s
  = 49.91 tokens/sec (close to reported 49.42 tokens/sec; small differences come from overhead + rounding).

3) Check if request latency matches “TTFT + generation time”

- Approx total time ≈ TTFT + (output_tokens − 1) × inter_token_latency
  = 60.09 ms + 122 × 20.03 ms
  = 60.09 ms + 2444.23 ms
  = 2504.32 ms = 2.504 s

Measured request latency avg = 2.481 s, so you’re within ~23 ms (that gap is normal: measurement method, batching/overlap, and how “inter-token latency” is averaged).

4) Cross-check throughput using request throughput

- If request throughput is ~0.40 req/s and each request outputs ~123 tokens:
     - Output tokens/sec ≈ 0.40 × 123 = 49.2 tokens/sec
       Reported output token throughput is 49.42 tokens/sec → consistent (0.40 is rounded; actual is ~1/2.4809 = 0.403 req/s).

1-line interpretation of what this run says (concurrency=1, streaming)

- TTFT ~60 ms: first token comes fast (good interactive feel).
- ~20 ms/token → ~50 tok/s: steady-state generation speed is ~50 tokens/sec.
- Total latency ~2.48 s mainly comes from generating ~123 tokens at ~20 ms/token (generation dominates).
- Throughput ~0.40 req/s is exactly what you’d expect when each request takes ~2.48 s at concurrency 1.



