# Diagnosis of the DDP skipped steps

Run status: c_fp16 ok, b_fp16 ok, b_fp32 ok, c_fp32 ok

## Which steps and ranks found inf

c_fp16: 4 (step, rank) events
- step 5, rank 0, scale 65536.0, loss finite True, before all-reduce on this rank: 162 params; after all-reduce: 162 params (e.g. base_model.model.model.decoder.layers.0.encoder_attn.k_proj.lora_A.default.weight, base_model.model.model.decoder.layers.0.encoder_attn.k_proj.lora_B.default.weight, base_model.model.model.decoder.layers.0.encoder_attn.out_proj.lora_A.default.weight, base_model.model.model.decoder.layers.0.encoder_attn.out_proj.lora_B.default.weight)
- step 5, rank 1, scale 65536.0, loss finite True, before all-reduce on this rank: 1 params; after all-reduce: 162 params (e.g. base_model.model.model.decoder.layers.0.fc1.lora_B.default.weight)
- step 6, rank 0, scale 32768.0, loss finite True, before all-reduce on this rank: 10 params; after all-reduce: 10 params (e.g. base_model.model.model.decoder.layers.0.encoder_attn.q_proj.lora_A.default.weight, base_model.model.model.decoder.layers.0.encoder_attn.q_proj.lora_B.default.weight, base_model.model.model.decoder.layers.0.self_attn.k_proj.lora_A.default.weight, base_model.model.model.decoder.layers.0.self_attn.k_proj.lora_B.default.weight)
- step 6, rank 1, scale 32768.0, loss finite True, after all-reduce: 10 params (e.g. base_model.model.model.decoder.layers.0.self_attn.k_proj.lora_A.default.weight, base_model.model.model.decoder.layers.0.self_attn.k_proj.lora_B.default.weight, base_model.model.model.decoder.layers.0.self_attn.v_proj.lora_A.default.weight, base_model.model.model.decoder.layers.0.self_attn.v_proj.lora_B.default.weight)

b_fp16: 3 (step, rank) events
- step 5, rank 0, scale 65536.0, loss finite True, before all-reduce on this rank: 2 params; after all-reduce: 2 params (e.g. base_model.model.model.decoder.layers.0.fc1.lora_B.default.weight, base_model.model.model.decoder.layers.0.self_attn.v_proj.lora_B.default.weight)
- step 8, rank 0, scale 32768.0, loss finite True, before all-reduce on this rank: 12 params; after all-reduce: 12 params (e.g. base_model.model.model.decoder.layers.0.encoder_attn.q_proj.lora_A.default.weight, base_model.model.model.decoder.layers.0.encoder_attn.q_proj.lora_B.default.weight, base_model.model.model.decoder.layers.0.fc1.lora_B.default.weight, base_model.model.model.decoder.layers.0.self_attn.k_proj.lora_A.default.weight)
- step 9, rank 0, scale 16384.0, loss finite True, before all-reduce on this rank: 10 params; after all-reduce: 10 params (e.g. base_model.model.model.decoder.layers.0.encoder_attn.q_proj.lora_A.default.weight, base_model.model.model.decoder.layers.0.encoder_attn.q_proj.lora_B.default.weight, base_model.model.model.decoder.layers.0.self_attn.k_proj.lora_A.default.weight, base_model.model.model.decoder.layers.0.self_attn.k_proj.lora_B.default.weight)

b_fp32: no non-finite gradients

c_fp32: no non-finite gradients

## Do the DDP replicas diverge

- c_fp16: largest weight checksum difference between ranks 0.000e+00, never above 1e-9
- c_fp32: largest weight checksum difference between ranks 0.000e+00, never above 1e-9

Both DDP ranks together saw the same samples as b at every step (fp16): True

## Loss gaps (max per-step absolute difference)

| comparison | max abs diff | at step |
|---|---|---|
| b_vs_c_fp16 | 4.85e+00 | 9 |
| b_vs_c_fp32 | 4.65e-04 | 18 |
| fp16_noise_floor_b_vs_b_rerun | 9.77e+00 | 9 |
