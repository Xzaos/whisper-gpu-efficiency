# Efficiency report

GPUs: Tesla T4, Tesla T4. torch 2.11.0+cu130 (CUDA 13.0), transformers 4.57.6, peft 0.21.2.

## Configs (10 warmup + 40 timed steps)

| config | batch | samples/s | ms/step | peak GB (GiB) | loss at last step |
|---|---|---|---|---|---|
| 1 full fp32 | 8 | 2.0 | 4046 | 12.79 | 0.560 |
| 2 full fp16 | 8 | 6.8 | 1170 | 10.21 | 0.569 |
| 3 full fp16 + ckpt | 8 | 5.5 | 1453 | 4.76 | 0.569 |
| 4 lora fp16 | 8 | 6.4 | 1251 | 8.78 | 0.520 |

## Max batch size that fits

Each batch size ran 2 warmup + 4 timed steps, stopping at the first OOM.

| config | max batch | tried | samples/s at max | peak GB at max |
|---|---|---|---|---|
| 2 full fp16 | 8 | 4, 8, 16 (OOM) | 6.8 | 10.21 |
| 3 full fp16 + ckpt | 32 | 4, 8, 16, 32, 64 (OOM) | 5.6 | 9.54 |
| 4 lora fp16 | 8 | 4, 8, 16 (OOM) | 6.5 | 8.78 |

## DDP scaling (LoRA fp16)

| run | batch x accum x gpus = global | samples/s | ms/step | peak GB |
|---|---|---|---|---|
| 1 GPU | 8 x 2 x 1 = 16 | 6.4 | 2496 | 8.83 |
| 2 GPU DDP | 8 x 1 x 2 = 16 | 12.7 | 1262 | 8.82 |

Single-GPU baseline fell back to 8 x accum 2 (batch 16 OOM).

Scaling efficiency = throughput_2gpu / (2 x throughput_1gpu): 0.989

## Equivalence check (LoRA fp16, dropout off, same data order, 30 steps)

| run | loss step 1 | loss step 30 | fp16 skipped steps |
|---|---|---|---|
| a: 1 GPU, batch 16 | OOM | _ | _ |
| b: 1 GPU, batch 8 x accum 2 | 3.7591 | 0.4782 | 0 |
| b rerun (noise floor) | 3.7591 | 0.4774 | 0 |
| c: DDP 2 x 8 | 3.7591 | 0.5504 | 2 |

| pair | max per-step abs loss diff | at step |
|---|---|---|
| b_vs_b_repeat | 1.10e-02 | 16 |
| b_vs_c | 3.74e-01 | 7 |
| b_repeat_vs_c | 3.72e-01 | 7 |
