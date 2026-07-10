/**
 * 练习 = 试次表前 N 条。
 * 图片优先从 NC_MIX_TASK 桌面素材目录加载（与 practice_ui / stone_images 同名规则），
 * 失败时回退 materials/img/ 下 SVG（generate_web_assets.py）。
 * 可覆盖：navAssetRoots, craftingStoneRoots, craftingBottleRoots（URL 数组，靠前优先）。
 */
window.NC_MIX_CONFIG = {
  /** 网页范式不再划分试次表练习；保留字段以免旧脚本报错，当前无引用。 */
  practiceTrialCount: 0,
  /** 与 navigation/practice_main 一致：探索阶段至少累计秒数、每站至少抵达次数 */
  explorationMinSeconds: 300,
  explorationMinVisitsPerNode: 2,
  assetsBase: "materials/",
  imgBase: "materials/img/",
};
