To run the image built as l2o, given no entrypoint, use:

```
docker run --gpus all -itd l2o -- bash -c 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate main && git clone https://github.com/FadyRezkGhattas/Cyrus.git && cd Cyrus/ppo && python ppo_atari_envpool_xla_jax_scan.py'
```

with the entrypoint, only do:
```
docker run --gpus all -itd l2o 'git clone https://github.com/FadyRezkGhattas/Cyrus.git && cd Cyrus/ppo && python ppo_atari_envpool_xla_jax_scan.py' 
```

Building the image and running a container or without entrypoint as above, the container tears itself down once the task running is finished. Make sure to persist necessary files in mounted volumes or to some cloud storage such as huggingface hub.