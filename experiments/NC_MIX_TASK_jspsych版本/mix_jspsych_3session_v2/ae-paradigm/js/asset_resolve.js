/**
 * 与 NC_MIX_TASK 桌面版一致的素材解析：
 * - 导航：practice_ui._iter_station_asset_paths
 * - 九石：stone_images / bottle_images 的 stem 与扩展名顺序
 *
 * 在 <img> 上对候选 URL 依次 onerror 回退；可在 NC_MIX_CONFIG 中覆盖各 root 列表。
 */
(function (global) {
  var IMAGE_EXTS = [".png", ".webp", ".jpg", ".jpeg", ".svg", ".PNG", ".WEBP", ".JPG"];

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
    var names = [];
    if (code >= 1 && code <= 9) {
      names.push(STATION_NAMES_ZH[code - 1].replace("站", ""));
      var en = STATION_ICON_EN[code - 1];
      if (names.indexOf(en) < 0) names.push(en);
    }
    names.push(String(code));

    var out = [];
    for (var ri = 0; ri < roots.length; ri++) {
      for (var ni = 0; ni < names.length; ni++) {
        for (var ei = 0; ei < IMAGE_EXTS.length; ei++) {
          out.push(joinUrl(roots[ri], names[ni], IMAGE_EXTS[ei]));
        }
      }
    }
    var stub = (global.NC_MIX_CONFIG && global.NC_MIX_CONFIG.imgBase) || "materials/img/";
    out.push(
      stub.replace(/\/?$/, "/") +
        "nav/station_" +
        String(code).padStart(2, "0") +
        ".svg"
    );
    return out;
  }

  function stoneIndexFromStateId(stateId) {
    var m = /^stone_(\d+)$/.exec(stateId);
    return m ? parseInt(m[1], 10) : 0;
  }

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
    var cfg = global.NC_MIX_CONFIG || {};
    var explicit =
      Array.isArray(cfg.bottleStems) && cfg.bottleStems[i - 1]
        ? String(cfg.bottleStems[i - 1]).trim()
        : "";
    var out = [];
    function add(stem) {
      if (stem && out.indexOf(stem) < 0) out.push(stem);
    }
    if (explicit) add(explicit);
    add("魔法药水" + i);
    add("魔法药水 " + i);
    add("药水" + i);
    add("potion_" + String(i).padStart(2, "0"));
    add("potion_" + i);
    add("magic_bottle_" + String(i).padStart(2, "0"));
    add("magic_bottle_" + i);
    add("bottle_" + String(i).padStart(2, "0"));
    add("bottle_" + i);
    add(String(i));
    return out;
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
    var imgBase = ((global.NC_MIX_CONFIG && global.NC_MIX_CONFIG.imgBase) || "materials/img/").replace(
      /\/?$/,
      "/"
    );
    var stemsForStub = bottleStemCandidates(index);
    var stubStem = stemsForStub.length ? stemsForStub[0] : "魔法药水" + index;
    for (var ei = 0; ei < IMAGE_EXTS.length; ei++) {
      out.push(imgBase + "bottle/" + stubStem + IMAGE_EXTS[ei]);
    }
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
      if (code >= 1 && code <= 9) return STATION_NAMES_ZH[code - 1];
      return "站" + code;
    },
  };
})(window);
