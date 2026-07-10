/**
 * 与 NC_MIX_TASK2/js/asset_resolve.js 同步；供 ae-paradigm 单独拷贝运行（不依赖 ../../js）。
 */
(function (global) {
  var IMAGE_EXTS = [".png", ".webp", ".jpg", ".jpeg", ".svg", ".PNG", ".WEBP", ".JPG"];

  /** 与 NC_MIX_TASK navigation/app/common/station_names.py 站名展示一致 */
  var STATION_SHAPE_LABELS = [
    "红色三角形站",
    "蓝色正方形站",
    "绿色圆形站",
    "橙色菱形站",
    "紫色五角星站",
    "粉色六边形站",
    "黄色十字站",
    "青色倒三角站",
    "棕色五边形站",
  ];
  var STATION_NAMES_ZH = [
    "Ⅰ站",
    "Ⅱ站",
    "Ⅲ站",
    "Ⅳ站",
    "Ⅴ站",
    "Ⅵ站",
    "Ⅶ站",
    "Ⅷ站",
    "Ⅸ站",
  ];
  var STATION_ICON_EN = [
    "I",
    "II",
    "III",
    "IV",
    "V",
    "VI",
    "VII",
    "VIII",
    "IX",
  ];

  function cfgArray(key, fallback) {
    var c = global.NC_MIX_CONFIG || {};
    if (Array.isArray(c[key]) && c[key].length) return c[key];
    return fallback.slice();
  }

  function joinUrl(root, name, ext) {
    var r = root.replace(/\/?$/, "/");
    return r + name + ext;
  }

  function navStationCandidates(code) {
    var roots = cfgArray("navAssetRoots", [
      "materials/img/nav/",
      "../../../NC_MIX_TASK/navigation/app/assets/stations/",
      "../../../NC_MIX_TASK/navigation/app/assets/",
    ]);
    var stub = (global.NC_MIX_CONFIG && global.NC_MIX_CONFIG.imgBase) || "materials/img/";
    /** 与 station_names / main2 一致：优先彩色几何形状 SVG，再回退罗马数字等旧素材名 */
    var shapeFirst =
      stub.replace(/\/?$/, "/") +
      "nav/station_" +
      String(code).padStart(2, "0") +
      ".svg";
    var names = [];
    if (code >= 1 && code <= 9) {
      names.push(STATION_NAMES_ZH[code - 1].replace("站", ""));
      var en = STATION_ICON_EN[code - 1];
      if (names.indexOf(en) < 0) names.push(en);
    }
    names.push(String(code));

    var out = [shapeFirst];
    for (var ri = 0; ri < roots.length; ri++) {
      for (var ni = 0; ni < names.length; ni++) {
        for (var ei = 0; ei < IMAGE_EXTS.length; ei++) {
          var u = joinUrl(roots[ri], names[ni], IMAGE_EXTS[ei]);
          if (u !== shapeFirst) out.push(u);
        }
      }
    }
    return out;
  }

  function stoneIndexFromStateId(stateId) {
    var m = /^stone_(\d+)$/.exec(stateId);
    return m ? parseInt(m[1], 10) : 0;
  }

  /** NC_MIX_CONFIG.stoneImageFiles（按 stone_01…09）或 stoneImageMap（按 id）优先 */
  function stonePrimaryUrls(stateId) {
    var cfg = global.NC_MIX_CONFIG || {};
    var roots = cfgArray("craftingStoneRoots", [
      "materials/img/stone/",
      "../../../NC_MIX_TASK/crafting/assets/stone/",
    ]);
    var filename = null;
    if (cfg.stoneImageMap && typeof cfg.stoneImageMap === "object" && cfg.stoneImageMap[stateId]) {
      filename = String(cfg.stoneImageMap[stateId]);
    } else if (Array.isArray(cfg.stoneImageFiles) && cfg.stoneImageFiles.length) {
      var idx = stoneIndexFromStateId(stateId);
      if (idx >= 1 && idx <= cfg.stoneImageFiles.length) {
        filename = String(cfg.stoneImageFiles[idx - 1]);
      }
    }
    if (!filename) return [];
    var out = [];
    for (var ri = 0; ri < roots.length; ri++) {
      out.push(roots[ri].replace(/\/?$/, "/") + filename);
    }
    return out;
  }

  function stoneStemCandidates(stateId) {
    var m = /^stone_(\d+)$/.exec(stateId);
    var idx = m ? parseInt(m[1], 10) : 0;
    var stems = [];
    if (idx >= 1 && idx <= 9) {
      stems.push(
        "stone_" + String(idx).padStart(2, "0"),
        "stone_" + idx,
        "gem_" + String(idx).padStart(2, "0"),
        "gem_" + idx,
        String(idx).padStart(2, "0"),
        String(idx),
        "宝石" + String(idx).padStart(2, "0"),
        "宝石" + idx
      );
    }
    if (stems.indexOf(stateId) < 0) stems.push(stateId);
    return stems;
  }

  function stoneUrlCandidates(stateId) {
    var roots = cfgArray("craftingStoneRoots", [
      "materials/img/stone/",
      "../../../NC_MIX_TASK/crafting/assets/stone/",
    ]);
    var primary = stonePrimaryUrls(stateId);
    var stems = stoneStemCandidates(stateId);
    var out = [];
    for (var pi = 0; pi < primary.length; pi++) {
      out.push(primary[pi]);
    }
    for (var ri = 0; ri < roots.length; ri++) {
      for (var si = 0; si < stems.length; si++) {
        for (var ei = 0; ei < IMAGE_EXTS.length; ei++) {
          out.push(joinUrl(roots[ri], stems[si], IMAGE_EXTS[ei]));
        }
      }
    }
    var m = /^stone_(\d+)$/.exec(stateId);
    var n = m ? m[1] : "01";
    var stub =
      ((global.NC_MIX_CONFIG && global.NC_MIX_CONFIG.imgBase) || "materials/img/").replace(
        /\/?$/,
        "/"
      ) + "stone/stone_" + String(n).padStart(2, "0") + ".svg";
    out.push(stub);
    return out;
  }

  function bottleStemCandidates(index) {
    if (index < 1 || index > 3) return [];
    var i = index;
    return [
      "bottle_" + String(i).padStart(2, "0"),
      "bottle_" + i,
      "魔法药水" + i,
      "药水" + i,
      "potion_" + String(i).padStart(2, "0"),
      "potion_" + i,
      "magic_bottle_" + String(i).padStart(2, "0"),
      "magic_bottle_" + i,
      String(i),
    ];
  }

  function bottleUrlCandidates(index) {
    var roots = cfgArray("craftingBottleRoots", [
      "materials/img/bottle/",
      "../../../NC_MIX_TASK/crafting/assets/bottle/",
    ]);
    var stems = bottleStemCandidates(index);
    var out = [];
    for (var ri = 0; ri < roots.length; ri++) {
      for (var si = 0; si < stems.length; si++) {
        for (var ei = 0; ei < IMAGE_EXTS.length; ei++) {
          out.push(joinUrl(roots[ri], stems[si], IMAGE_EXTS[ei]));
        }
      }
    }
    var stub =
      ((global.NC_MIX_CONFIG && global.NC_MIX_CONFIG.imgBase) || "materials/img/").replace(
        /\/?$/,
        "/"
      ) +
      "bottle/bottle_" +
      String(index).padStart(2, "0") +
      ".svg";
    out.push(stub);
    return out;
  }

  function applyImageChain(img, urls, index) {
    if (!img || !urls || !urls.length) return;
    index = index || 0;
    if (index >= urls.length) {
      img.style.display = "none";
      img.removeAttribute("src");
      return;
    }
    img.onerror = function () {
      applyImageChain(img, urls, index + 1);
    };
    img.onload = function () {
      img.style.display = "";
    };
    img.src = urls[index];
  }

  global.NCMixAssets = {
    navStationCandidates: navStationCandidates,
    stoneUrlCandidates: stoneUrlCandidates,
    bottleUrlCandidates: bottleUrlCandidates,
    applyImageChain: applyImageChain,
    stationNameZh: function (code) {
      if (code >= 1 && code <= 9) return STATION_SHAPE_LABELS[code - 1];
      return "站" + code;
    },
  };
})(window);
