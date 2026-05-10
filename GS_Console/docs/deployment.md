# GS-SDF Docker Deployment

## 当前机器已确认可用的条件

- Docker 可用
- 已存在镜像 `gs_sdf_img:latest`
- 容器内已编译好 `neural_mapping_node`
- `--gpus all` 在容器内可见 RTX A6000

## 已准备好的脚本

- 通用训练脚本：
  `/home/chatsign/gs-sdf/scripts/train_gssdf.sh`
- tiny FAST-LIVO smoke test：
  `/home/chatsign/gs-sdf/scripts/smoke_fastlivo_tiny.sh`
- Host 侧 campus 配置：
  `/home/chatsign/gs-sdf/config/fastlivo_campus_host.yaml`
- Host 侧低迭代 smoke 配置：
  `/home/chatsign/gs-sdf/config/fastlivo_tiny_smoke.yaml`

## 先跑 smoke test

```bash
/home/chatsign/gs-sdf/scripts/smoke_fastlivo_tiny.sh
```

这个包已经检查过：

- 话题：
  `/aft_mapped_to_init`
  `/cloud_registered_body`
  `/origin_img`
- 图像分辨率：
  `640x512`

## 跑正式 FAST-LIVO bag

示例：

```bash
/home/chatsign/gs-sdf/scripts/train_gssdf.sh \
  /home/chatsign/data/data-002/3DGS_data/hku_campus_seq_00.bag \
  /home/chatsign/gs-sdf/config/fastlivo_campus_host.yaml \
  hku-campus-seq00
```

## 运行时行为

- 输入 bag 所在目录会被原样挂载进容器
- 如果该目录里没有：
  `images/`
  `depths/`
  `color_poses.txt`
  `depth_poses.txt`
  程序会自动从 bag 解析生成
- 输出结果会写到：
  `/home/chatsign/gs-sdf/runtime/output`
- 训练脚本会在容器内自动启动 `roscore`，不需要你先在宿主机起 ROS Master
- 日志会写到：
  `/home/chatsign/gs-sdf/runtime/logs`

## 注意

- 现在默认镜像是 `gs_sdf_img:latest`
- 如果你的 bag 不是 campus 相机参数，不要直接用 `fastlivo_campus_host.yaml`
- 这套脚本复用的是容器内的 `/root/gs_sdf_ws/devel/lib/neural_mapping/neural_mapping_node`
