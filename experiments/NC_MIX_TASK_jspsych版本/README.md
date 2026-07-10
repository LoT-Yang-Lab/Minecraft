# NC_MIX_TASK2（jsPsych 网页范式）

基于 **jsPsych 7** 的两个独立范式包，与 [`NC_MIX_TASK`](../NC_MIX_TASK) 桌面实验共用**固定地图与试次表**。目标：本地 HTTP 可跑，并兼容上传 **脑岛**（参考 [`AE_Part3_Cp_012/ae-paradigm/`](AE_Part3_Cp_012/ae-paradigm/) 的 `naodao-2021-12.js` 集成方式）。

## 目录

| 路径 | 说明 |
|------|------|
| [`navigation_jspsych/ae-paradigm/`](navigation_jspsych/ae-paradigm/) | 导航主实验（非地图编辑器）：练习 → 正式 |
| [`crafting_jspsych/ae-paradigm/`](crafting_jspsych/ae-paradigm/) | 九石阵主实验：练习 → 正式 |
| [`scripts/`](scripts/) | 从桌面项目**导出**网页用 JSON 的 Python 脚本 |

## 本地运行

**可直接双击 `ae-paradigm/index.html`**（`file://`）：实验数据通过 `materials/*_embed.js` 注入，**不再使用 `fetch` 加载 JSON**（浏览器会拦截本地 `fetch`）。

可选：在 `ae-paradigm` 下用 `python -m http.server 8000` 通过 `http://localhost:8000/` 打开（与脑岛部署更接近）。

**依赖文件**：各范式需包含 `dist/`（jsPsych）、`materials/nav_embed.js` 或 `materials/crafting_embed.js`（由 `scripts/export_*.py` 生成）、以及本地的 `dist/axios.min.js` 与 `dist/naodao-2021-12.js`（已与 AE 同源下载到仓库，**无需联网**即可加载脚本；脑岛线上仍可按平台要求使用 CDN）。

## 练习与正式

- 练习、正式试次均来自**同一张试次表**的前缀与后缀：由 [`js/config.js`](navigation_jspsych/ae-paradigm/js/config.js) / [`crafting_jspsych/.../config.js`](crafting_jspsych/ae-paradigm/js/config.js) 中的 **`practiceTrialCount`**（默认 **3**）控制：前 N 条为练习，其余为正式。
- 与桌面 `practice_main.py`（时长、访问覆盖等）**不等价**；论文方法中请勿混写。

## 重新导出 materials（地图或试次变更后）

在 `experiments/NC_MIX_TASK2/scripts` 下（需已安装 `NC_MIX_TASK` 的 `pygame` 等依赖）：

```bash
python export_nav_web.py
python export_crafting_web.py
```

- `export_nav_web.py` 写入 `navigation_jspsych/ae-paradigm/materials/nav_graph.json` 与 `trial_sequence.json`（与 `main2` 图论一致）。
- `export_crafting_web.py` 写入 `crafting_rules.json`（转化图 + 逆向边）与 `crafting_trials.json`（含与 `GameCrafting` 一致的订单目标展开）。

## 数据

- 实验结束会尝试**下载 JSON**（`navigation_nc_mix_*.json` / `crafting_nc_mix_*.json`）。
- `call-function` 试次的数据在字段 **`value`** 内（含 `phase`、`steps` 等）。脑岛部署时保留 `initJsPsych` 的 `extensions: [{ type: Naodao }]`（脚本已引入；离线时无 `Naodao` 则自动跳过扩展）。

## 设计说明

- **导航**：第二个任务为 **导航主实验**，不包含 Pygame 地图编辑器。
- **九石阵**：网页版未实现桌面 **R 重置槽位** 键；若需完全对齐可后续增加。
