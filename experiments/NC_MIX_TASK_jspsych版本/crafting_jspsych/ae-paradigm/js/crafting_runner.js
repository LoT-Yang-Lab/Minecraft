/**
 * 九石阵：与 GameCrafting 订单逻辑一致。
 * 试次表与 crafting/src/proposal5_trial_schedule.py 一致：每块 20 试次、grid:loop:tie=8:8:4；
 * 网页单独任务使用 Session2 序列（base_seed 20260408 → session_seed 20260410），配方与桌面 _make_crafting_trial 同为 rng 42。
 * UI：操作区 + 药水；仅 phase === "main" 时渲染「目标区」面板并显示目标石块图（与桌面正式任务一致）。
 * 练习试次（practice）不显示目标区，订单信息仅在 nc-pr 一行文字中给出；自由探索阶段无目标区。
 */
(function (global) {
  var applyChain =
    global.NCMixAssets && global.NCMixAssets.applyImageChain
      ? global.NCMixAssets.applyImageChain.bind(global.NCMixAssets)
      : null;
  var stoneCands =
    global.NCMixAssets && global.NCMixAssets.stoneUrlCandidates
      ? global.NCMixAssets.stoneUrlCandidates.bind(global.NCMixAssets)
      : null;
  var bottleCands =
    global.NCMixAssets && global.NCMixAssets.bottleUrlCandidates
      ? global.NCMixAssets.bottleUrlCandidates.bind(global.NCMixAssets)
      : null;

  var KEY_TO_EDGE = {
    q: { ring: 1, dir: 1 },
    e: { ring: 1, dir: -1 },
    a: { ring: 2, dir: 1 },
    d: { ring: 2, dir: -1 },
    w: { ring: 3, dir: 0 },
  };

  /** 与 NC_MIX_TASK crafting main_crafting：ACTION_BTN_HIGHLIGHT_S / ACTION_BTN_ERROR_S */
  var POTION_FLASH_OK_MS = 220;
  var POTION_FLASH_ERR_MS = 280;

  function normalizeStoneId(sid) {
    var m = /^stone_(\d+)$/i.exec(String(sid || ""));
    if (!m) return String(sid || "");
    return "stone_" + String(parseInt(m[1], 10)).padStart(2, "0");
  }

  function setImgSrcIfChanged(img, src) {
    if (!img || !src) return;
    if (img.dataset && img.dataset.ncSrc === src) {
      img.style.display = "";
      return;
    }
    img.src = src;
    if (img.dataset) img.dataset.ncSrc = src;
    img.style.display = "";
  }

  function keyToPotionSlot(keyLower) {
    if (keyLower === "q" || keyLower === "e") return 1;
    if (keyLower === "a" || keyLower === "d") return 2;
    if (keyLower === "w") return 3;
    return 0;
  }

  function flashPotionSlot(slot, ok) {
    if (slot < 1 || slot > 3) return;
    var el = document.getElementById("nc-pot-wrap-" + slot);
    if (!el) return;
    if (el._ncFlashTimer) clearTimeout(el._ncFlashTimer);
    el.classList.remove("nc-pot--flash-ok", "nc-pot--flash-err");
    el.offsetWidth;
    el.classList.add(ok ? "nc-pot--flash-ok" : "nc-pot--flash-err");
    var ms = ok ? POTION_FLASH_OK_MS : POTION_FLASH_ERR_MS;
    el._ncFlashTimer = setTimeout(function () {
      el.classList.remove("nc-pot--flash-ok", "nc-pot--flash-err");
      el._ncFlashTimer = null;
    }, ms);
  }

  function bindStoneImg(img, sid) {
    if (!img) return;
    sid = normalizeStoneId(sid);
    var pref = global.NC_MIX_ASSET_PREFETCH;
    if (pref && pref.stone && pref.stone[sid]) {
      setImgSrcIfChanged(img, pref.stone[sid]);
      return;
    }
    if (applyChain && stoneCands) applyChain(img, stoneCands(sid));
    else {
      var m = /^stone_(\d+)$/.exec(sid);
      var n = m ? m[1] : "01";
      img.src =
        ((global.NC_MIX_CONFIG && global.NC_MIX_CONFIG.imgBase) || "materials/img/").replace(
          /\/?$/,
          "/"
        ) +
        "stone/stone_" +
        String(n).padStart(2, "0") +
        ".svg";
    }
  }

  function nextState(rules, state, keyLower) {
    var spec = KEY_TO_EDGE[keyLower];
    if (!spec) return { ok: false, next: null, reason: "bad_key" };
    var p1 = rules.potion1,
      p1r = rules.potion1_rev,
      p2 = rules.potion2,
      p2r = rules.potion2_rev,
      p3 = rules.potion3;
    if (spec.ring === 1) {
      if (spec.dir === 1)
        return { ok: p1[state] != null, next: p1[state] || null, reason: "no_edge" };
      return { ok: p1r[state] != null, next: p1r[state] || null, reason: "no_edge" };
    }
    if (spec.ring === 2) {
      if (spec.dir === 1)
        return { ok: p2[state] != null, next: p2[state] || null, reason: "no_edge" };
      return { ok: p2r[state] != null, next: p2r[state] || null, reason: "no_edge" };
    }
    var n3 = p3[state];
    return { ok: n3 != null, next: n3 || null, reason: "no_edge" };
  }

  function displayName(sid) {
    var n = stoneIndex(sid);
    if (n) return "石块 " + n;
    return sid;
  }

  /** NC_MIX_CONFIG.stoneDisplayNames：数组 9 项或 { stone_01: "…" } */
  function getStoneDisplayName(sid) {
    var cfg = global.NC_MIX_CONFIG || {};
    var map = cfg.stoneDisplayNames;
    if (map && typeof map === "object" && !Array.isArray(map) && map[sid]) {
      return String(map[sid]);
    }
    if (Array.isArray(map) && map.length) {
      var n = stoneIndex(sid);
      if (n >= 1 && n <= map.length) {
        var s = map[n - 1];
        if (s != null && String(s).length) return String(s);
      }
    }
    var files = cfg.stoneImageFiles;
    if (Array.isArray(files) && files.length) {
      var ni = stoneIndex(sid);
      if (ni >= 1 && ni <= files.length) {
        var fn = files[ni - 1];
        if (fn != null && String(fn).length) {
          return String(fn).replace(/\.[^/.]+$/, "");
        }
      }
    }
    return displayName(sid);
  }

  function stoneIndex(sid) {
    var m = /^stone_(\d+)$/.exec(sid);
    if (!m) return 0;
    return parseInt(m[1], 10);
  }

  /**
   * @param {object|null} progressOpt 正式试次进度，如 { current: 3, total: 12 }；仅 phase===main 时传入并展示
   */
  function runCraftingTrial(display_element, rules, trial, phase, trialIndex, done, progressOpt) {
    var state = trial.start_state;
    var targets = trial.order_targets || [];
    var startState = trial.start_state;
    var orderIndex = 0;
    var steps = [];
    var t0 = performance.now();
    var lastT = t0;
    var stepCount = 0;
    var waitingOrder = false;
    var phaseNorm = String(phase || "").toLowerCase();
    /** 仅正式实验（main）展示右侧目标区与目标石块可视化；练习（practice）不展示 */
    var showTargetZone = phaseNorm === "main";

    var wrap = document.createElement("div");
    wrap.className =
      "nc-craft-trial nc-craft-trial--" +
      (phaseNorm || "unknown") +
      (showTargetZone ? "" : " nc-craft-trial--no-target");
    var targetBlock = showTargetZone
      ? '<div class="nc-craft-panel nc-craft-panel--target">' +
        '<div class="nc-craft-cap">目标区</div>' +
        '<div class="nc-craft-gem-slot nc-craft-gem-slot--target">' +
        '<img id="nc-order-img" class="nc-craft-order-img" alt="" />' +
        "</div>" +
        '<div id="nc-order-txt" class="nc-craft-sub"></div>' +
        "</div>"
      : "";
    var trialProgressBar = "";
    if (
      phaseNorm === "main" &&
      progressOpt &&
      typeof progressOpt.current === "number" &&
      typeof progressOpt.total === "number" &&
      progressOpt.total >= 1 &&
      progressOpt.current >= 1 &&
      progressOpt.current <= progressOpt.total
    ) {
      trialProgressBar =
        '<div class="nc-trial-progress" aria-live="polite">试次 ' +
        progressOpt.current +
        " / " +
        progressOpt.total +
        "</div>";
    }
    wrap.innerHTML =
      trialProgressBar +
      '<div class="nc-craft-panel nc-craft-panel--op">' +
      '<div class="nc-craft-cap">操作区</div>' +
      '<div class="nc-craft-gem-slot">' +
      '<img id="nc-current-img" class="nc-craft-current-img" alt="" />' +
      "</div>" +
      '<div id="nc-current-txt" class="nc-craft-sub"></div>' +
      "</div>" +
      '<div class="nc-potions">' +
      '<div class="nc-pot" id="nc-pot-wrap-1">' +
      '<img id="nc-bottle-1" alt=""/>' +
      "<div>魔法药水1</div>" +
      '<span class="nc-pot-key">Q · E</span>' +
      "</div>" +
      '<div class="nc-pot" id="nc-pot-wrap-2">' +
      '<img id="nc-bottle-2" alt=""/>' +
      "<div>魔法药水2</div>" +
      '<span class="nc-pot-key">A · D</span>' +
      "</div>" +
      '<div class="nc-pot" id="nc-pot-wrap-3">' +
      '<img id="nc-bottle-3" alt=""/>' +
      "<div>魔法药水3</div>" +
      '<span class="nc-pot-key">W</span>' +
      "</div>" +
      "</div>" +
      targetBlock +
      '<div class="nc-craft-meta"><span id="nc-pr"></span></div>' +
      '<div id="nc-cmsg" class="nc-msg"></div>';

    display_element.innerHTML = "";
    display_element.appendChild(wrap);

    function initBottles() {
      for (var bi = 1; bi <= 3; bi++) {
        var b = document.getElementById("nc-bottle-" + bi);
        var pref = global.NC_MIX_ASSET_PREFETCH;
        var hit = pref && pref.bottle && pref.bottle[String(bi)];
        if (b && hit) {
          setImgSrcIfChanged(b, hit);
        } else if (b && applyChain && bottleCands) {
          applyChain(b, bottleCands(bi));
        }
      }
    }
    initBottles();

    function refreshUI() {
      var oImg = showTargetZone ? document.getElementById("nc-order-img") : null;
      var oTx = showTargetZone ? document.getElementById("nc-order-txt") : null;
      var cImg = document.getElementById("nc-current-img");
      var cTx = document.getElementById("nc-current-txt");
      var pr = document.getElementById("nc-pr");

      bindStoneImg(cImg, state);
      cTx.textContent = getStoneDisplayName(state);

      if (orderIndex >= targets.length) {
        if (oImg) oImg.style.opacity = "0.4";
        if (oTx) oTx.textContent = "—";
        pr.textContent = "试次完成 · 按空格继续";
        return;
      }
      var tg = targets[orderIndex];
      if (showTargetZone && oImg && oTx) {
        bindStoneImg(oImg, tg);
        oImg.style.opacity = "1";
        oTx.textContent = getStoneDisplayName(tg);
        pr.textContent =
          "订单 " + (orderIndex + 1) + " / " + targets.length + " · 订单数 " + targets.length;
      } else {
        pr.textContent =
          "练习试次 · 子任务 " +
          (orderIndex + 1) +
          " / " +
          targets.length +
          " · 订单宝石：" +
          getStoneDisplayName(tg);
      }
    }
    refreshUI();

    function finishTrial() {
      window.removeEventListener("keydown", onKey);
      done({
        phase: phase,
        domain: "crafting",
        trial_index: trialIndex,
        trial_progress:
          progressOpt &&
          typeof progressOpt.current === "number" &&
          typeof progressOpt.total === "number"
            ? { current: progressOpt.current, total: progressOpt.total }
            : null,
        trial_id: trial.trial_id,
        steps: steps,
        total_steps: stepCount,
        outcome: "completed",
        total_time_ms: Math.round(performance.now() - t0),
      });
    }

    function onKey(ev) {
      if (waitingOrder) {
        if (ev.key === " " || ev.key === "Enter") {
          ev.preventDefault();
          waitingOrder = false;
          document.getElementById("nc-cmsg").textContent = "";
          if (orderIndex >= targets.length) finishTrial();
          else refreshUI();
        }
        return;
      }
      var k = ev.key.toLowerCase();
      if (!KEY_TO_EDGE[k]) return;
      ev.preventDefault();
      var now = performance.now();
      var rtMs = Math.round(now - lastT);
      lastT = now;
      stepCount += 1;

      var target = targets[orderIndex];
      var mv = nextState(rules, state, k);
      var slot = keyToPotionSlot(k);
      if (!mv.ok || !mv.next) {
        flashPotionSlot(slot, false);
        steps.push({
          step: stepCount,
          key: k,
          valid: false,
          prev: state,
          next: state,
          order_target: target,
          rt_ms: rtMs,
        });
        document.getElementById("nc-cmsg").textContent = "无效操作";
        return;
      }
      flashPotionSlot(slot, true);
      var prev = state;
      state = mv.next;
      var hit = state === target;
      steps.push({
        step: stepCount,
        key: k,
        valid: true,
        prev: prev,
        next: state,
        order_target: target,
        order_completed: hit,
        rt_ms: rtMs,
      });
      document.getElementById("nc-cmsg").textContent = "";

      if (hit) {
        orderIndex += 1;
        if (orderIndex >= targets.length) {
          waitingOrder = true;
          document.getElementById("nc-cmsg").textContent = "本试次完成（按空格继续）";
          refreshUI();
          return;
        }
        state = startState;
        lastT = performance.now();
        waitingOrder = false;
        refreshUI();
      } else {
        refreshUI();
      }
    }

    window.addEventListener("keydown", onKey);
  }

  /**
   * 自由探索：对齐 crafting/practice_main — 九块石头各至少 minVisits 次且累计时长 >= minSeconds；
   * 达标自动结束。
   */
  function runCraftExploration(display_element, rules, done) {
    var cfg = global.NC_MIX_CONFIG || {};
    var minMs = (cfg.explorationMinSeconds || 300) * 1000;
    var minVisits = cfg.explorationMinVisitsPerNode || 2;
    var STONE_IDS = [];
    var si;
    for (si = 1; si <= 9; si++) {
      STONE_IDS.push("stone_" + String(si).padStart(2, "0"));
    }

    var startStone = "stone_01";
    var state = startStone;
    var visitCounts = {};
    STONE_IDS.forEach(function (s) {
      visitCounts[s] = 0;
    });
    visitCounts[state] += 1;

    var roundStartedAt = performance.now();
    var steps = [];
    var stepCount = 0;
    var lastT = roundStartedAt;
    var finished = false;
    var tickTimer = null;

    var wrap = document.createElement("div");
    wrap.className = "nc-craft-trial nc-craft-trial--explore nc-craft-trial--no-target";
    wrap.innerHTML =
      '<div id="nc-craft-exp-panel" class="nc-craft-exp-panel"></div>' +
      '<div class="nc-craft-panel nc-craft-panel--op">' +
      '<div class="nc-craft-cap">操作区</div>' +
      '<div class="nc-craft-gem-slot">' +
      '<img id="nc-current-img" class="nc-craft-current-img" alt="" />' +
      "</div>" +
      '<div id="nc-current-txt" class="nc-craft-sub"></div>' +
      "</div>" +
      '<div class="nc-potions">' +
      '<div class="nc-pot" id="nc-pot-wrap-1">' +
      '<img id="nc-bottle-1" alt=""/>' +
      "<div>魔法药水1</div>" +
      '<span class="nc-pot-key">Q · E</span>' +
      "</div>" +
      '<div class="nc-pot" id="nc-pot-wrap-2">' +
      '<img id="nc-bottle-2" alt=""/>' +
      "<div>魔法药水2</div>" +
      '<span class="nc-pot-key">A · D</span>' +
      "</div>" +
      '<div class="nc-pot" id="nc-pot-wrap-3">' +
      '<img id="nc-bottle-3" alt=""/>' +
      "<div>魔法药水3</div>" +
      '<span class="nc-pot-key">W</span>' +
      "</div>" +
      "</div>" +
      '<div id="nc-cmsg" class="nc-msg"></div>';

    display_element.innerHTML = "";
    display_element.appendChild(wrap);

    function initBottles() {
      for (var bi = 1; bi <= 3; bi++) {
        var b = document.getElementById("nc-bottle-" + bi);
        var pref = global.NC_MIX_ASSET_PREFETCH;
        var hit = pref && pref.bottle && pref.bottle[String(bi)];
        if (b && hit) {
          setImgSrcIfChanged(b, hit);
        } else if (b && applyChain && bottleCands) {
          applyChain(b, bottleCands(bi));
        }
      }
    }
    initBottles();

    function refreshExploreUI() {
      var cImg = document.getElementById("nc-current-img");
      var cTx = document.getElementById("nc-current-txt");
      bindStoneImg(cImg, state);
      cTx.textContent = getStoneDisplayName(state);
    }
    refreshExploreUI();

    function metrics() {
      var mastered = 0;
      STONE_IDS.forEach(function (s) {
        if ((visitCounts[s] || 0) >= minVisits) mastered += 1;
      });
      var masteryOk = mastered === STONE_IDS.length;
      var elapsedMs = performance.now() - roundStartedAt;
      var timeOk = elapsedMs >= minMs;
      return { masteryOk: masteryOk, timeOk: timeOk, elapsedMs: elapsedMs, mastered: mastered };
    }

    function updateExpPanel() {
      var el = document.getElementById("nc-craft-exp-panel");
      if (!el) return;
      var m = metrics();
      var elapsedSec = m.elapsedMs / 1000;
      var em = Math.floor(elapsedSec / 60);
      var es = Math.floor(elapsedSec % 60);
      var reqM = Math.floor(minMs / 60000);
      var reqS = Math.floor((minMs % 60000) / 1000);
      el.innerHTML =
        "<strong>练习一 · 自由探索</strong> · 本轮时长 " +
        String(em).padStart(2, "0") +
        ":" +
        String(es).padStart(2, "0") +
        " / " +
        String(reqM).padStart(2, "0") +
        ":" +
        String(reqS).padStart(2, "0") +
        " · 覆盖达标 " +
        m.mastered +
        "/" +
        STONE_IDS.length +
        "（每宝石≥" +
        minVisits +
        "次）· 时间" +
        (m.timeOk ? "✓" : "未达标") +
        " · 覆盖" +
        (m.masteryOk ? "✓" : "未达标");
      if (!finished && m.timeOk && m.masteryOk) {
        finishExploration(true);
      }
    }

    function finishExploration(completed) {
      if (finished) return;
      finished = true;
      if (tickTimer) clearInterval(tickTimer);
      window.removeEventListener("keydown", onKey);
      done({
        phase: "exploration",
        domain: "crafting",
        completed: completed,
        exploration_skipped: !completed,
        visit_counts: visitCounts,
        steps: steps,
        min_seconds: minMs / 1000,
        min_visits_per_node: minVisits,
        total_time_ms: Math.round(performance.now() - roundStartedAt),
      });
    }

    function onKey(ev) {
      if (finished) return;
      var k = ev.key.toLowerCase();
      if (!KEY_TO_EDGE[k]) return;
      ev.preventDefault();
      var now = performance.now();
      var rtMs = Math.round(now - lastT);
      lastT = now;
      stepCount += 1;

      var mv = nextState(rules, state, k);
      var slot = keyToPotionSlot(k);
      if (!mv.ok || !mv.next) {
        flashPotionSlot(slot, false);
        steps.push({
          step: stepCount,
          key: k,
          valid: false,
          prev: state,
          next: state,
          rt_ms: rtMs,
        });
        document.getElementById("nc-cmsg").textContent = "无效操作";
        return;
      }
      flashPotionSlot(slot, true);
      var prev = state;
      state = mv.next;
      steps.push({
        step: stepCount,
        key: k,
        valid: true,
        prev: prev,
        next: state,
        rt_ms: rtMs,
      });
      visitCounts[state] = (visitCounts[state] || 0) + 1;
      document.getElementById("nc-cmsg").textContent = "";
      refreshExploreUI();
      updateExpPanel();
    }

    window.addEventListener("keydown", onKey);
    tickTimer = setInterval(updateExpPanel, 400);
    updateExpPanel();
  }

  global.NCCraftingRunner = {
    runCraftingTrial: runCraftingTrial,
    runCraftExploration: runCraftExploration,
  };
})(window);
