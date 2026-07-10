# 可视化脚本入口

本目录放置正式维护的可视化脚本。原则：

- 核心算法优先复用 `src/cognitivemap` 与 `script/*_pipeline`。
- `experiments/新数据` 只作为历史原型和迁移参考，不再作为默认输入；需要复查旧目录时显式传 `--input-dir`。
- 正式图表统一输出到 `result/figure/`。

当前脚本：

- `visualize_mix_jspsych_data.py`：从 jsPsych JSON 生成混合任务行为、固定 B/D/E 与地图还原图。

示例：

```bash
python script/visualization/visualize_mix_jspsych_data.py
python script/visualization/visualize_mix_jspsych_data.py --input-dir data/raw_data --output-dir result/figure/mix_jspsych
```
