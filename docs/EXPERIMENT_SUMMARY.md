# Experiment Summary

This file condenses the final report's experimental story into repo-friendly
notes. Use it for slides, README updates, and oral presentation.

## Experiment 1: Live Edge Pipeline

Question:

```text
Can RGB, pose, and dense geometry stream without blocking each other?
```

Result:

- RGB and ROS image topics stayed around 4-5 fps.
- LingBot dense updates arrived asynchronously at seconds-level latency.
- The run recorded 7 dense updates, 103 trajectory poses, 73 RGB keyframes,
  zero worker failures, and zero queue drops.

Solved issue:

```text
Avoids forcing LingBot dense reconstruction to run at RGB frame rate.
```

## Experiment 2: Full-Video LingBot Geometry

Question:

```text
Does LingBot provide usable pose/depth/point geometry for the 301-frame video?
```

Result:

- LingBot produced depth, confidence, intrinsics, extrinsics, and colored
  point outputs for the video.
- Real LingBot depth/pose fixed early WebUI failure modes caused by synthetic
  or low-resolution fallback depth.

Solved issue:

```text
Provides the geometric scaffold required by LG-GS.
```

## Experiment 3: Raw Local GenWildSplat Chunks

Question:

```text
Can GenWildSplat produce useful Gaussian visual assets from sparse views?
```

Result:

- Local sofa/table chunks looked sharp.
- Full playback failed because local Gaussian assets did not cover the whole
  trajectory.
- Raw GenWildSplat gauge did not match the LingBot world frame.

Solved issue:

```text
Confirms GenWildSplat is a useful local visual prior, not a complete long-video global mapper.
```

## Experiment 4: Overlap Chunk Atlas

Question:

```text
Can overlapping temporal chunks preserve local quality across a longer video?
```

Recorded variants:

| Variant | Splats | Alignment mean | Observation |
| --- | ---: | ---: | --- |
| `chunk000_120` | 1,039,068 | 0.0505 | best sofa/table local quality |
| `chunk060_180` | 878,408 | 0.0563 | acceptable transition |
| `chunk120_240` | 589,799 | 0.0889 | weaker hallway/transition |
| `chunk180_300` | 836,546 | 0.2229 | visibly worse late alignment |

Conclusion:

```text
Chunk atlas helps inspection but does not create a single coherent global Gaussian scene.
```

## Experiment 5: Single Global Keyframe Count

Question:

```text
Does adding more global keyframes solve long-video reconstruction?
```

Variants:

- `global_32ctx`
- `global_40ctx`
- `global_44ctx_sharp`
- `global_44ctx_postopt_s05_300`
- `48ctx` attempted but exceeded memory budget

Conclusion:

```text
More frames increased coverage but also increased blur, memory cost, and alignment difficulty.
```

Main lesson:

```text
More frames are not the same as better geometry for a feed-forward sparse-view Gaussian model.
```

## Experiment 6: LG-GS Variants

Question:

```text
Does LingBot-guided initialization improve pose synchronization and coordinate consistency?
```

| Variant | Splats | Preprocessing | Outcome |
| --- | ---: | --- | --- |
| `LG1 stride2` | 602,112 | early LG-GS, stride 2 | too sparse; grid artifacts |
| `LG2 wide stride2` | 456,876 | wide 518x294, stride 2 | better FOV; still sparse |
| `LG3 letterbox` | 1,365,504 | letterbox to square | denser; dark splats and grid artifacts |
| `LG4 clean` | 1,135,118 kept | letterbox + cleanup | fewer black gaps; display-only cleanup |
| `LG5 wide stride1` | 1,827,504 | wide 518x294, stride 1 | active report variant; best FOV consistency |

Active keyframes:

```text
0, 27, 55, 82, 109, 136, 164, 191, 218, 245, 273, 300
```

Conclusion:

```text
LG-GS improves pose synchronization and coordinate-frame consistency, but final visual quality still needs local photometric Gaussian optimization.
```

## Experiment 7: Field of View

Question:

```text
Is the RGB-vs-Gaussian mismatch caused by pose, crop, or both?
```

Result:

- Default square crop removed left/right content.
- Letterbox preserved FOV but introduced padded input distribution.
- Wide 518x294 best matched LingBot RGB/depth geometry.

Solved issue:

```text
Prevents crop-induced view mismatch from being mistaken for pose failure.
```

## Experiment 8: Display Artifacts

Question:

```text
Why do some LG-GS views show black speckles, grids, or blur?
```

Causes:

- many low-opacity splats
- tiny scale axes
- black background rendering
- scale inflation hiding holes but blurring edges

Conclusion:

```text
Display cleanup improves readability but does not replace photometric Gaussian optimization.
```

## Final Takeaway

Mono2Sim-GS demonstrates a plausible reconstruction layer:

```text
fast edge tracking
+ learned LingBot geometry
+ pose-guided Gaussian visual prior
+ WebUI inspection
```

The next research step is not another viewer. It is local/windowed photometric
Gaussian optimization initialized by LG-GS, followed by the future physical
TSDF/mesh branch.
