```shell
nohup /root/miniconda3/envs/sgl/bin/python -m sglang.launch_server --model-path /root/models/Qwen3-8B --served-model-name Qwen3-8B --host 127.0.0.1 --port 30000 --attention-backend fa3 --reasoning-parser qwen3 > sglang.log 2>&1 &
```
