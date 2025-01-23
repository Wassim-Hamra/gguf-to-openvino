# gguf-to-openvino
An example that reads GGUF and creates OpenVINO on the fly.

## Usage
1. Download GGUF file:
```sh
huggingface-cli download mradermacher/SmolLM2-135M-GGUF SmolLM2-135M.f16.gguf --local-dir models
```

2. Convert the model:
```sh
python convert_gguf.py --org_model_path models/SmolLM2-135M.f16.gguf --ov_model_path models/smollm_ov
```
