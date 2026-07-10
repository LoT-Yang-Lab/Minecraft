/**
 * 宝石图：materials/img/stone/ 下中文文件名（见 stoneImageFiles）。
 * 药水：materials/img/bottle/，默认使用 魔法药水1.png … 魔法药水3.png（见 bottleStems）。
 */
window.NC_MIX_CONFIG = {
  /** 网页范式不再划分试次表练习；保留字段以免旧脚本报错，当前无引用。 */
  practiceTrialCount: 0,
  /** 与 crafting/practice_main 一致：探索阶段至少累计秒数、每块石头至少抵达次数 */
  explorationMinSeconds: 300,
  explorationMinVisitsPerNode: 2,
  assetsBase: "materials/",
  imgBase: "materials/img/",
  /**
   * 九张图片文件名，第 1 项 = stone_01 … 第 9 项 = stone_09（与 crafting_embed 规则一致）。
   * 当前为目录下文件名字母序；若与地图不符，请调整顺序或设置 stoneImageMap。
   */
  stoneImageFiles: [
    "白钻.png",
    "粉钻.png",
    "紫宝石.png",
    "红钻.png",
    "绿宝石.png",
    "蓝宝石.png",
    "藏青宝石.png",
    "黄琥珀.png",
    "黑钻.png",
  ],
  /**
   * 界面显示名；与 stoneImageFiles 同序。也可用对象 { "stone_01": "红钻", ... }。
   * 可选 stoneImageMap：按 id 指定文件名（优先级高于 stoneImageFiles 顺序）。
   * 若未配 stoneDisplayNames，将从 stoneImageFiles 文件名（去扩展名）自动推导。
   */
  stoneDisplayNames: [
    "白钻",
    "粉钻",
    "紫宝石",
    "红钻",
    "绿宝石",
    "蓝宝石",
    "藏青宝石",
    "黄琥珀",
    "黑钻",
  ],
  /**
   * 三瓶药水图主文件名（无扩展名），顺序对应药水 1·2·3（Q/E、A/D、W）。
   * 与 asset_resolve 中候选扩展名组合；优先于 bottle_01 等旧占位名。
   */
  bottleStems: ["魔法药水1", "魔法药水2", "魔法药水3"],
};
