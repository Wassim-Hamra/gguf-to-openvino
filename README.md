# gguf-to-openvino
An example that reads GGUF and creates OpenVINO on the fly.

## Usage
1. Download GGUF file:
```sh
huggingface-cli download "atwine/Llama-2-7b-chat-f16-gguf" "Llama-2-7b-chat-f16.gguf"
```

2. Convert the model:
```sh
python convert_gguf.py --org_model_path /home/<user_name>/.cache/huggingface/hub/models--atwine--Llama-2-7b-chat-f16-gguf/snapshots/140c851f3a1f5b9503a2c5d231122d02a0ae6c3c/Llama-2-7b-chat-f16.gguf --ov_model_path models/
```
