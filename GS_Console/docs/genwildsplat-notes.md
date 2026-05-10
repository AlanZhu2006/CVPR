# GenWildSplat Notes For LingBot Live

Repository cloned at:

```text
/home/nvidia/twork/lingbot-map/CVPR/third_party_research/GenWildSplat
```

Useful takeaways from the released implementation:

- GenWildSplat is feed-forward sparse-view reconstruction, not an online SLAM
  system. The README reports roughly 3 seconds on a single A6000 for 2-6
  unposed images.
- The core output is a Gaussian set plus predicted context poses. See
  `demo_gradio.py:get_reconstructed_scene()` and `src/eval_nvs_video.py`.
- The practical design pattern to borrow is short-window Gaussian refresh:
  keep LingBot/cuVSLAM responsible for realtime RGB, pose, depth, and colored
  clouds, then periodically refresh a bounded Gaussian seed for rendering.
- On Jetson, do not put GenWildSplat itself on the critical realtime path
  without benchmarking. Its pinned environment is Python 3.10, PyTorch 2.4,
  CUDA 12.4, `gsplat`, and `torch_scatter`; the advertised 3s timing is for
  A6000, not Orin.

For the current GS Console stack this maps to:

- realtime path: HikRobot RGB -> cuVSLAM pose -> LingBot depth -> ROS2 topics
- monitor path: worker windows -> TSDF/Gaussian seed -> latest camera render
- optional cadence: `START_GAUSSIAN_MONITOR_RENDER=1`,
  `GAUSSIAN_RENDER_INTERVAL_SEC=3.0`
