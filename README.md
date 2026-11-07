# whisper-gpu-efficiency

What do mixed precision, gradient checkpointing, LoRA and 2-GPU data parallelism actually buy when fine-tuning whisper-small on a Kaggle "GPU T4 x2" machine? I measured each one with the same data, the same step count and a fixed sample order, and checked whether the 2-GPU and gradient-accumulation setups compute the same thing as a plain big batch.

Short version: fp16 is the big win for speed, checkpointing buys batch size and not speed, LoRA saved little here, and DDP scaled almost perfectly. The equivalence check did not pass: DDP drifted away from batch 8 x accum 2, and I have a likely cause but have not shown it.

Everything below comes from `results/report.md` and `results/raw/*.json`, from one run on two Tesla T4s (torch 2.11.0+cu130, transformers 4.57.6, peft 0.21.2).

## Setup

- Data: 400 FLEURS hi_in train clips (seed 0). Whisper log-mel features and label ids are computed once and saved to one tensor file, so a timed step is only the GPU.
- Each config runs 10 untimed warmup steps, then 40 timed steps. I record samples/s, ms/step, peak memory (`torch.cuda.max_memory_allocated`, in GiB) and the loss at the last step. A CUDA OOM is caught, the cache is emptied and the cell says OOM.
- Full fine-tune uses AdamW at 1e-5, LoRA (r=32 on q, k, v, out_proj, fc1, fc2) uses 5e-4. Both clip gradients at 1.0 and use no schedule. Last-step losses are not comparable across configs, they only show training did not blow up.
- fp16 means autocast plus GradScaler, with the weights kept in fp32 (the T4 has no fast bf16). With LoRA the frozen base weights also stay fp32.
- The loss is a token sum divided by the token count of the whole global batch. Averaging per-micro-batch means gives a different gradient when micro-batches hold different numbers of tokens, so batch 8 x accum 2 would not match batch 16.

## Results

| config | batch | samples/s | ms/step | peak GB | loss |
|---|---|---|---|---|---|
| 1 full fp32 | 8 | 2.0 | 4046 | 12.79 | 0.560 |
| 2 full fp16 | 8 | 6.8 | 1170 | 10.21 | 0.569 |
| 3 full fp16 + checkpointing | 8 | 5.5 | 1453 | 4.76 | 0.569 |
| 4 LoRA fp16 | 8 | 6.4 | 1251 | 8.78 | 0.520 |

Largest batch that fits, trying 4, 8, 16, 32, 64 and stopping at the first OOM. These sweep runs used 2 warmup and 4 timed steps, to keep the whole run under 25 minutes.

| config | max batch | samples/s at max | peak GB at max |
|---|---|---|---|
| 2 full fp16 | 8 | 6.8 | 10.21 |
| 3 full fp16 + checkpointing | 32 | 5.6 | 9.54 |
| 4 LoRA fp16 | 8 | 6.5 | 8.78 |

### Mixed precision

fp16 autocast took throughput from 2.0 to 6.8 samples/s, 3.4 times faster. Peak memory went from 12.79 to 10.21 GiB, so the memory saving is much smaller than the speed gain: the weights, gradients and optimizer state stay in fp32.

### Gradient checkpointing

At batch 8 it cut peak memory from 10.21 to 4.76 GiB and cost 1170 to 1453 ms/step (24% slower). The largest batch went from 8 to 32. But throughput at batch 32 is 5.6 samples/s, still below plain fp16 at batch 8 (6.8). So checkpointing buys batch size and not speed. I did not test whether a larger batch helps final quality, which is the only reason I would want it here.

### LoRA

LoRA only dropped peak memory from 10.21 to 8.78 GiB and was not faster (6.4 vs 6.8 samples/s). It also did not raise the largest batch, which stays at 8. My reading is that activations dominate: Whisper pads every input to 30 s, and LoRA still backpropagates through the whole network, so it saves optimizer state and little else. I did not profile activations separately, so this is the explanation that fits the numbers and not something I measured.

This also explains why [whisper-hindi-telephony](https://github.com/Xzaos/whisper-hindi-telephony) ran out of memory at batch 16 on a T4 even with LoRA: here batch 16 is also OOM for LoRA fp16.

### DDP scaling

LoRA fp16, global batch 16. Batch 16 on one GPU ran out of memory, so the single-GPU baseline is 8 x accum 2.

| run | batch x accum x gpus = global | samples/s | ms/step | peak GB |
|---|---|---|---|---|
| 1 GPU | 8 x 2 x 1 = 16 | 6.4 | 2496 | 8.83 |
| 2 GPU DDP | 8 x 1 x 2 = 16 | 12.7 | 1262 | 8.82 |

Scaling efficiency = throughput_2gpu / (2 x throughput_1gpu) = 0.989. Only the LoRA gradients are all-reduced, which is small, so near-linear scaling is not surprising, but it is one machine and one run.

### Equivalence check

30 steps of LoRA fp16 with dropout off and the same data order. Run (a), one GPU at batch 16, ran out of memory, so there is no direct large-batch reference. I compared (b) one GPU at batch 8 x accum 2, a rerun of (b) as a noise floor, and (c) DDP 2 x 8.

| run | loss step 1 | loss step 30 | fp16 skipped steps |
|---|---|---|---|
| b: 1 GPU, batch 8 x accum 2 | 3.7591 | 0.4782 | 0 |
| b rerun | 3.7591 | 0.4774 | 0 |
| c: DDP 2 x 8 | 3.7591 | 0.5504 | 2 |

| pair | max per-step abs loss diff | at step |
|---|---|---|
| b vs b rerun | 1.10e-02 | 16 |
| b vs c | 3.74e-01 | 7 |
| b rerun vs c | 3.72e-01 | 7 |

The two identical runs differ by up to 1.1e-2, and DDP differs from accumulation by 3.7e-1, about 34 times the noise floor. The three runs agree through step 4 and DDP separates at step 5 to 6. The check did not pass.

The one difference I can see in the data is that the DDP run skipped 2 optimizer steps because of fp16 gradient overflow, and the accumulation runs skipped none. A skipped step means that run made two fewer updates, which would leave its loss higher at the end (0.550 vs 0.478). I think this is the leading explanation, but I have not shown it. Why DDP overflows more is also open. One guess: each rank scales its loss by the world size before DDP averages, so per-rank scaled gradients are twice as large as in the accumulation run at the same GradScaler scale.

On loss normalisation, which I checked in the code: the DDP path divides by the global token count, not each rank's local count. Every rank computes the token count from the full global batch slice (same seed, same order), so no all-reduce is needed, and the loss is scaled by world size so that DDP's gradient averaging gives the sum over the global batch. Local normalisation is therefore not a second cause. I have not tested the code path on a GPU beyond this run, though the CPU gloo test shows DDP matching a single process in fp32.

## Running it

```bash
pip install -e ".[dev]"
pytest -q                      # CPU only, tiny random Whisper config, gloo for the DDP test
bash scripts/run_kaggle.sh     # on a Kaggle GPU T4 x2 machine, HF_TOKEN set
```

The script prints a timestamp per stage. Before each stage it runs `pip uninstall -y torchao` and sets `USE_TF=0` and `TRANSFORMERS_NO_TF=1`.

## Limitations

- 40 timed steps per config, and the max-batch sweep used only 4 timed steps per batch size.
- One machine type (Kaggle T4 x2, PCIe, no NVLink), so the scaling number says nothing about other interconnects.
- whisper-small only, one seed, one run per config.
- No FSDP or DeepSpeed, so nothing here says what sharding would buy.
- fp16 runs keep fp32 master weights under autocast; I did not try casting the frozen LoRA base to fp16.
- Features are copied from CPU to GPU every step, which is inside the timed region.
- The equivalence check failed and its cause is not established.

## Next steps

Rerun the equivalence check with a lower GradScaler `init_scale`, or in fp32, to see whether the DDP gap falls to the noise floor. If it does, the skipped steps are the cause.
