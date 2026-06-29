# PPT分类实验记录

更新时间：2026-06-29

## 当前结论

当前推荐使用 `ppt_classify_4`，也就是训练实验 `logs/ppt_classify/4` 的结果。

选择 run4 的核心原因不是验证集 top1 最高，而是它在业务实际使用的高置信度条件下更稳定：长时间播放视频预警只把 `video` 且 `score > 0.99` 的片段计入连续播放视频，run4 在公平测试集上 `video score >= 0.99` 的误报为 0，召回还能保留 80/95。run8 的验证集指标接近 run4，但在同一个高置信度阈值下几乎不输出高置信度 video，不适合当前预警逻辑。

当前接入状态：

- 业务模型名：`ppt_classify_4`
- Triton 模型目录：`/home/yujiannan/Projects/AlgorithmUtils/models/triton/ppt_classify_4`
- 业务代码入口：`/home/yujiannan/Projects/AlgorithmUtils/app/model_zoo/ppt_classify.py`
- 开发 Triton 配置：`/home/yujiannan/Projects/AlgorithmUtils/deploy/开发机Triton.yaml`
- 双卡 5090 线上/开发配置本轮没有修改，重启服务前要确认实际加载了 `ppt_classify_4`

## 业务目标

PPT 画面变化后做一次分类，类别用于后续报警逻辑和调试分析。

| 类别 | 目录名 | 说明 |
| --- | --- | --- |
| PPT | `ppt` | 普通课件、板书式课件、文档式课件 |
| 网页 | `web` | 浏览器网页、网页应用、在线平台页面 |
| 视频 | `video` | PPT 区域或屏幕正在播放视频内容 |
| 其他 | `other` | 空白、桌面、文件管理器、无有效课件内容、难以归入前三类 |

长时间播放视频预警的实际使用条件：

- 当前分类必须是 `video`
- 分类置信度必须 `> 0.99`
- 连续播放视频时长超过配置项 `longPlayVideoTimeGreaterThan`
- 连续区间内变化次数也要达到阈值，避免单帧误判造成长时间误报

因此评估时除了看验证集 top1，还必须额外看高置信度 `video` 的 precision/recall，以及长时间播放视频单测表现。

## 数据集

### 分组划分数据集

路径：`/DATA/yujiannan/Datasets/20260625_ppt_classify_group_split`

| split | other | ppt | video | web | total |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 629 | 4638 | 651 | 446 | 6364 |
| val | 84 | 556 | 90 | 100 | 830 |
| test | 104 | 545 | 95 | 88 | 832 |
| total | 817 | 5739 | 836 | 634 | 8026 |

这个数据集是当前主要训练/公平评估数据集。划分时按来源视频做 group split，避免同一个视频里的相似截图同时进入 train 和 val/test。

临时划分脚本已经删除，保留可复现逻辑如下：

- Label Studio 导出文件：`/home/yujiannan/Projects/AlgorithmUtils/scripts/classify/project-49-at-2026-06-25-06-41-d3b4cb77.json`
- Label Studio 数据目录：`/DATA/yujiannan/label_studio/data/PPT分类`
- 只读取任务名 `PPT分类`
- 排除路径中包含 `Codex公平测试集` 的图片
- 读取每个任务最新且未取消的 annotation
- 标签映射：`PPT -> ppt`，`网页 -> web`，`视频 -> video`，`其他 -> other`
- `跳过` 标签不进入数据集
- 用图片 SHA1 去重；同一图片重复出现且标签冲突时跳过后出现的样本
- source key 优先从相对路径里提取 `v\d+`，提取不到时用父目录或文件名
- split 比例为 train/val/test = 0.8/0.1/0.1
- 分配 split 时按 source 作为不可拆分单元，同时尽量平衡类别比例和总量
- 输出文件复制到 `split/class_name/` 下，文件名冲突时追加后缀

### 少数类过采样数据集

路径：`/DATA/yujiannan/Datasets/20260625_ppt_classify_group_split_minority_oversample`

| split | other | ppt | video | web | total |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 2500 | 4638 | 2500 | 2500 | 12138 |
| val | 84 | 556 | 90 | 100 | 830 |
| test | 104 | 545 | 95 | 88 | 832 |
| total | 2688 | 5739 | 2685 | 2688 | 13800 |

临时过采样脚本已经删除，保留可复现逻辑如下：

- source dataset 为 `20260625_ppt_classify_group_split`
- target dataset 为 `20260625_ppt_classify_group_split_minority_oversample`
- 只对 train 过采样，val/test 原样复制
- train 中每个少数类最少补到 2500 张
- 补样方式是重复复制原图，重复文件名格式为 `{stem}_dup{i:04d}{suffix}`

这个数据集只用于 run8。它改善了验证集表面指标，但破坏了高置信度 `video` 输出能力，不作为当前推荐路线。

## 训练脚本

当前训练入口：`/home/yujiannan/Projects/ultralytics/scripts/train_ppt_classify.py`

当前脚本最后一次用于 run8，后续继续实验时要先明确是否要沿用 run8 的过采样数据和 `yolo11s`。如果目标是复现当前线上推荐模型，应把配置改回 run4：

```python
model = YOLO("yolo11n-cls.pt")
data = "/DATA/yujiannan/Datasets/20260625_ppt_classify_group_split"
name = "4"  # 新实验不要复用这个 name
auto_augment = "randaugment"
erasing = 0.05
hsv_h = hsv_s = hsv_v = 0.0
```

固定策略：

- `imgsz=224`，这个任务是简单整屏分类，224 够用
- 从 ImageNet 预训练分类模型开始训练，不从旧 PPT 实验继续微调，避免数据泄露和实验不可比
- 使用 `LetterBoxClassificationTrainer`，避免中心裁剪裁掉浏览器栏、播放器控件、PPT 边界等全局结构
- `fliplr=0.0`，屏幕截图左右翻转没有业务意义

## 训练实验记录

| run | 数据集 | 模型 | 关键变化 | best val top1 | best val loss | 结论 |
| --- | --- | --- | --- | ---: | ---: | --- |
| 1 | `20260623_updating` | `yolo11n-cls.pt` | SGD，弱 HSV | 0.995190 @15 | 0.015420 @62 | 旧随机/更新数据集，疑似同源泄露，只能作早期 baseline |
| 2 | `20260623_updating` | `yolo11n-cls.pt` | AdamW，弱 HSV | 0.997600 @11 | 0.013810 @39 | 同旧数据集，不作为公平结论 |
| 3 | group split | `yolo11n-cls.pt` | SGD，弱 HSV | 0.903610 @16 | 0.525700 @3 | 分组后难度明显升高，说明旧数据集指标虚高 |
| 4 | group split | `yolo11n-cls.pt` | `randaugment` + `erasing=0.05` | 0.961450 @40 | 0.273630 @10 | 当前推荐模型，验证和业务高置信度 video 表现最均衡 |
| 5 | group split | `yolo11n-cls.pt` | 强 HSV，不用 randaugment | 0.896390 @31 | 0.509000 @3 | 颜色扰动单独加强效果不好 |
| 6 | group split | `yolo11s-cls.pt` | `randaugment` + `erasing=0.05` | 0.954220 @46 | 0.237440 @5 | val loss 较好，但高置信度 video 能力差 |
| 7 | group split | `yolo11s-cls.pt` | `randaugment`，无 erasing | 0.959040 @26 | 0.225520 @5 | val loss 较好，但公平测试和阈值表现差于 run4 |
| 8 | oversample | `yolo11s-cls.pt` | 少数类 train 过采样 + `randaugment` | 0.961450 @4 | 0.228640 @4 | val top1 接近 run4，但 `score > 0.99` 下 video 基本不可用 |

## 公平测试集结果

测试集：`/DATA/yujiannan/Datasets/20260625_ppt_classify_group_split/test`

### run4 混淆矩阵

按 `true -> pred` 记录：

| true | pred other | pred ppt | pred video | pred web | total |
| --- | ---: | ---: | ---: | ---: | ---: |
| other | 104 | 0 | 0 | 0 | 104 |
| ppt | 14 | 525 | 6 | 0 | 545 |
| video | 0 | 11 | 84 | 0 | 95 |
| web | 20 | 0 | 0 | 68 | 88 |

run4 test top1：`0.938702`

run4 的主要错误：

- `video -> ppt`：11 张，影响长时间播放视频召回
- `ppt -> video`：6 张，但在 `score >= 0.99` 下没有形成 video 误报
- `web -> other`：20 张，对当前长时间播放视频预警影响不大

### 高置信度 video 指标

| run | test top1 | video 阈值 | TP | FP | FN | precision | recall | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4 | 0.938702 | 0.50 | 84 | 6 | 11 | 0.9333 | 0.8842 | 普通分类阈值下召回较好 |
| 4 | 0.938702 | 0.90 | 80 | 2 | 15 | 0.9756 | 0.8421 | 误报已经很低 |
| 4 | 0.938702 | 0.99 | 80 | 0 | 15 | 1.0000 | 0.8421 | 符合当前预警策略 |
| 7 | 0.914663 | 0.99 | 1 | 5 | 94 | 0.1667 | 0.0105 | 高置信度 video 不可用 |
| 8 | 0.933894 | 0.99 | 0 | 3 | 95 | 0.0000 | 0.0000 | 高置信度 video 不可用 |

## 长时间播放视频单测记录

单测数据：`/home/yujiannan/Projects/AlgorithmUtils/test_data/case/ppt_long_play_video`

对比方式：

- 用独立临时 Triton 容器加载 OCR 和 PPT 分类模型
- 旧模型日志：`/home/yujiannan/Projects/AlgorithmUtils/logs/ppt_long_play_video_old_model.log`
- run4 新模型日志：`/home/yujiannan/Projects/AlgorithmUtils/logs/ppt_long_play_video_ppt_classify_4.log`

| 模型 | 通过 | 失败 | 误报 | 漏报 | 变化 |
| --- | ---: | ---: | ---: | ---: | --- |
| 旧 `ppt_classify` | 53 | 47 | 0 | 47 | baseline |
| 新 `ppt_classify_4` | 57 | 43 | 0 | 43 | 修复 case 3、51、88、98，无回退 |

当前失败仍然全部是漏报，说明策略偏保守。考虑到长时间播放视频预警是强业务报警，目前优先避免误报是合理的。

## 后续实验建议

优先补数据，而不是继续堆参数：

- 重点补 `video -> ppt` 的困难样本，尤其是播放器画面像普通 PPT、视频静止帧、视频里出现课件/文字的情况
- 补充 `web -> other` 和 `ppt -> other` 的边界样本，降低非视频类别之间的混淆
- 新实验必须继续使用 group split，不能按图片随机划分
- 每次实验除了记录 val top1，还要固定记录 test top1、`video score >= 0.99` precision/recall、长时间播放视频单测结果

可以尝试的下一组实验：

- 在 group split 数据集上继续用 `yolo11n-cls.pt`，只增加真实困难样本，不做重复图片过采样
- 如果继续尝试类均衡，优先使用采样权重或 loss 权重，而不是复制图片过采样
- 保持 `randaugment`，谨慎使用大幅 HSV；历史 run5 说明纯颜色扰动会伤害结构分类

