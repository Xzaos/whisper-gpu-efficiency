# Efficiency report

GPUs: Tesla T4, Tesla T4. torch 2.11.0+cu130 (CUDA 13.0), transformers 4.57.6, peft 0.21.2.

## Configs (10 warmup + 40 timed steps)

| config | batch | samples/s | ms/step | peak GB (GiB) | loss at last step |
|---|---|---|---|---|---|
| 1 full fp32 | 8 | 2.0 | 4043 | 12.79 | 0.560 |
| 2 full fp16 | 8 | 7.4 | 1075 | 10.21 | 0.569 |
| 3 full fp16 + ckpt | 8 | 6.0 | 1325 | 4.76 | 0.568 |
| 4 lora fp16 | 8 | 7.0 | 1136 | 8.78 | 0.522 |

## Max batch size that fits

Each batch size ran 2 warmup + 4 timed steps, stopping at the first OOM.

| config | max batch | tried | samples/s at max | peak GB at max |
|---|---|---|---|---|
| 2 full fp16 | 8 | 4, 8, 16 (OOM) | 7.5 | 10.21 |
| 3 full fp16 + ckpt | 32 | 4, 8, 16, 32, 64 (OOM) | 6.4 | 9.54 |
| 4 lora fp16 | 8 | 4, 8, 16 (OOM) | 7.0 | 8.78 |

## DDP scaling (LoRA fp16)

| run | batch x accum x gpus = global | samples/s | ms/step | peak GB |
|---|---|---|---|---|
| 1 GPU | 8 x 2 x 1 = 16 | 7.1 | 2267 | 8.83 |
| 2 GPU DDP | 8 x 1 x 2 = 16 | 13.8 | 1159 | 8.82 |

Single-GPU baseline fell back to 8 x accum 2 (batch 16 OOM).

Scaling efficiency = throughput_2gpu / (2 x throughput_1gpu): 0.978

## Equivalence check (LoRA fp16, dropout off, same data order, 30 steps)

| run | loss step 1 | loss step 30 | fp16 skipped steps |
|---|---|---|---|
| a: 1 GPU, batch 16 | OOM | _ | _ |
| b: 1 GPU, batch 8 x accum 2 | 3.7591 | 0.5692 | 3 (steps 5, 8, 9) |
| b rerun (noise floor) | 3.7591 | 0.4737 | 0 |
| c: DDP 2 x 8 | 3.7591 | 0.4987 | 1 (steps 5) |

| pair | max per-step abs loss diff | at step |
|---|---|---|
| b_vs_b_repeat | 9.77e+00 | 9 |
| b_vs_c | 9.68e+00 | 9 |
| b_repeat_vs_c | 3.36e-01 | 6 |
