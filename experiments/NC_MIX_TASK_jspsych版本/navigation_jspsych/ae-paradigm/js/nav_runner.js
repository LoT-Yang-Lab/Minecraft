/**
 * 导航试次：与 main2 一致；(targetA,targetB)=(起点,终点)；Q/E/A/D/W；
 * 试次表与 crafting/src/proposal5_trial_schedule.py 一致：每块 20 试次、grid:loop:tie=8:8:4、30 对素材库；
 * 网页单独任务使用 Session1 序列（base_seed 20260408 → session_seed 20260409）。
 * 有效移动：与桌面相同的沿线滑动动画（0.3s、ease-out quad）；五向线无透明度区分。
 */
(function (global) {
  var applyChain =
    global.NCMixAssets && global.NCMixAssets.applyImageChain
      ? global.NCMixAssets.applyImageChain.bind(global.NCMixAssets)
      : null;
  var navCandidates =
    global.NCMixAssets && global.NCMixAssets.navStationCandidates
      ? global.NCMixAssets.navStationCandidates.bind(global.NCMixAssets)
      : null;

  var CX = 120;
  var CY = 120;
  var R = 88;
  /** 与下方 #nc-vis-svg width/height、viewBox 一致，用于目标站图例与中心白圆同屏比例 */
  var NAV_VIS_SVG_CSS = 300;
  var NAV_VIS_VB = 240;
  /** 中心白圆直径 2×26（viewBox 单位）映射到 CSS 像素 = 正式试次「目标」图例整体边长（与地图中心 token 同比例） */
  var GOAL_STATION_IMG_CSS = Math.round((2 * 26 * NAV_VIS_SVG_CSS) / NAV_VIS_VB);
  /** 与地图中心 token 局部几何一致：白圆直径 2×26（viewBox 单位） */
  var TOKEN_VB_DIAM = 100;
  var TOKEN_VB_CENTER = 26;
  /** 与 main2._VIS_ANIM_DURATION 一致 */
  var ANIM_MS = 300;
  var LINE_W = 7;
  /** 外圆 r=26 不变；内嵌形状 54（36×1.5），超出圆域由 clipPath 裁切（与目标图例内联 SVG 一致） */
  var CENTER_SHAPE_SIZE = 100;
  var CENTER_SHAPE_HALF = CENTER_SHAPE_SIZE / 2;
  /** 裁剪圆半径（略小于白底内接区，避免画出描边圆） */
  var SHAPE_CLIP_R = 24;

  var KEY_MAP = {
    q: { mode: "bus", direction: "next" },
    e: { mode: "bus", direction: "prev" },
    a: { mode: "light_rail", direction: "next" },
    d: { mode: "light_rail", direction: "prev" },
    w: { mode: "metro", direction: "next" },
  };

  var MODE_ORDER = [
    ["bus", "next"],
    ["bus", "prev"],
    ["light_rail", "next"],
    ["light_rail", "prev"],
    ["metro", "next"],
  ];

  /** 与 main2._MODE_DIR_COLORS 一致：每条线 (mode,direction) 独立深/浅色 */
  var MODE_DIR_COLORS = ["#1e64d2", "#82c3ff", "#1e8c3c", "#8ce1a0", "#8237c8"];

  function modeDirIndex(mode, dir) {
    for (var i = 0; i < MODE_ORDER.length; i++) {
      if (MODE_ORDER[i][0] === mode && MODE_ORDER[i][1] === dir) return i;
    }
    return 0;
  }

  function codeName(graph, code) {
    var row = (graph.codebook || []).find(function (c) {
      return c.code === code;
    });
    return row ? row.name : "编码" + code;
  }

  function bindStationImg(img, code) {
    if (!img) return;
    if (applyChain && navCandidates) applyChain(img, navCandidates(code));
    else
      img.src =
        ((global.NC_MIX_CONFIG && global.NC_MIX_CONFIG.imgBase) || "materials/img/").replace(
          /\/?$/,
          "/"
        ) +
        "nav/station_" +
        String(code).padStart(2, "0") +
        ".svg";
  }

  function findTransition(graph, code, mode, direction) {
    var list = graph.transitions[String(code)] || [];
    return (
      list.find(function (t) {
        return t.mode === mode && t.direction === direction;
      }) || null
    );
  }

  function modeDirSet(graph, code) {
    var raw = graph.mode_dirs_at[String(code)] || [];
    var s = {};
    raw.forEach(function (pair) {
      s[pair[0] + "|" + pair[1]] = true;
    });
    return s;
  }

  function keyForModeDir(mode, dir) {
    for (var k in KEY_MAP) {
      var m = KEY_MAP[k];
      if (m.mode === mode && m.direction === dir) return k;
    }
    return "";
  }

  function escXml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function escAttr(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/"/g, "&quot;")
      .replace(/</g, "&lt;");
  }

  /** 站点形状 SVG（与 materials/img/nav/station_XX.svg 一致） */
  function stationShapeUrl(code) {
    var n = parseInt(code, 10);
    if (!n || n < 1 || n > 9) n = 1;
    var stub =
      (global.NC_MIX_CONFIG && global.NC_MIX_CONFIG.imgBase) || "materials/img/";
    return stub.replace(/\/?$/, "/") + "nav/station_" + String(n).padStart(2, "0") + ".svg";
  }

  /**
   * 正式试次「目标站点」旁图例：与 #nc-vis-svg 中心同一套几何（r=26 白圆 + #6488c8 描边 + 54 内图 + clip r=24），
   * 仅 viewBox 缩到 token 直径 52，再用 width/height 缩放到与大地图中心相同的 CSS 像素。
   */
  function goalLegendTokenSvg(code) {
    var url = stationShapeUrl(code);
    var clipId =
      "nc-goal-leg-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 9);
    var vc = TOKEN_VB_CENTER;
    return (
      '<svg class="nc-goal-legend-token" width="' +
      GOAL_STATION_IMG_CSS +
      '" height="' +
      GOAL_STATION_IMG_CSS +
      '" viewBox="0 0 ' +
      TOKEN_VB_DIAM +
      " " +
      TOKEN_VB_DIAM +
      '" overflow="visible" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" focusable="false" aria-hidden="true">' +
      "<defs><clipPath id=\"" +
      clipId +
      '"><circle cx="' +
      vc +
      '" cy="' +
      vc +
      '" r="' +
      SHAPE_CLIP_R +
      '"/></clipPath></defs>' +
      '<circle cx="' +
      vc +
      '" cy="' +
      vc +
      '" r="26" fill="#ffffff" stroke="#6488c8" stroke-width="3"/>' +
      '<image href="' +
      escAttr(url) +
      '" xlink:href="' +
      escAttr(url) +
      '" x="' +
      (vc - CENTER_SHAPE_HALF) +
      '" y="' +
      (vc - CENTER_SHAPE_HALF) +
      '" width="' +
      CENTER_SHAPE_SIZE +
      '" height="' +
      CENTER_SHAPE_SIZE +
      '" clip-path="url(#' +
      clipId +
      ')" preserveAspectRatio="xMidYMid meet"/>' +
      "</svg>"
    );
  }

  function svgStationImage(x, y, size, code, cls, clipPathId) {
    var url = stationShapeUrl(code);
    var c = cls ? ' class="' + cls + '"' : "";
    var clip =
      clipPathId && String(clipPathId).length
        ? ' clip-path="url(#' + String(clipPathId) + ')"'
        : "";
    return (
      "<image" +
      c +
      ' href="' +
      escAttr(url) +
      '" xlink:href="' +
      escAttr(url) +
      '" x="' +
      x +
      '" y="' +
      y +
      '" width="' +
      size +
      '" height="' +
      size +
      '"' +
      clip +
      ' preserveAspectRatio="xMidYMid meet"/>'
    );
  }

  /**
   * 静态五向图：与 main2 一致 — 每条线为 (mode,direction) 的深/浅色；
   * 端点仅为白底描边小圆（不展示「下一站」形状，避免预知移动结果）；
   * 中心为白底圆 + 当前站形状图（与移动动画中的包裹圆一致）。
   */
  function drawVis(svgEl, graph, code, flashKey) {
    var avail = modeDirSet(graph, code);
    var parts = [];
    parts.push(
      "<defs><clipPath id=\"nc-center-shape-clip\"><circle cx=\"" +
        CX +
        "\" cy=\"" +
        CY +
        "\" r=\"" +
        SHAPE_CLIP_R +
        '"/></clipPath></defs>'
    );

    MODE_ORDER.forEach(function (md, i) {
      var mode = md[0];
      var dir = md[1];
      var ang = ((-90 + (360 / 5) * i) * Math.PI) / 180;
      var x2 = CX + R * Math.cos(ang);
      var y2 = CY + R * Math.sin(ang);
      var col = MODE_DIR_COLORS[i] || "#999";
      var kk = keyForModeDir(mode, dir);
      var flash = flashKey && flashKey === kk;
      if (flash) {
        col = "#ff3737";
      }
      parts.push(
        '<line class="nc-vis-line" x1="' +
          CX +
          '" y1="' +
          CY +
          '" x2="' +
          x2 +
          '" y2="' +
          y2 +
          '" stroke="' +
          col +
          '" stroke-width="' +
          LINE_W +
          '" stroke-linecap="round" opacity="1"/>'
      );
      parts.push(
        '<g transform="translate(' +
          x2.toFixed(2) +
          "," +
          y2.toFixed(2) +
          ')">' +
          '<circle r="10" fill="#ffffff" stroke="' +
          col +
          '" stroke-width="2"/>' +
          "</g>"
      );
    });

    parts.push(
      '<circle cx="' +
        CX +
        '" cy="' +
        CY +
        '" r="26" fill="#ffffff" stroke="#6488c8" stroke-width="3"/>'
    );
    parts.push(
      svgStationImage(
        CX - CENTER_SHAPE_HALF,
        CY - CENTER_SHAPE_HALF,
        CENTER_SHAPE_SIZE,
        code,
        "nc-vis-center-shape",
        "nc-center-shape-clip"
      )
    );

    svgEl.innerHTML = parts.join("");
  }

  /**
   * 沿线滑动：整段彩线（该向深/浅色）+ 白底描边圆与圆内当前站形状图一并移动（ease-out quad），
   * 与静态中心「圆裹形状」视觉一致。
   */
  function drawVisAnimating(svgEl, graph, code, mode, direction, eased) {
    var i = modeDirIndex(mode, direction);
    var ang = ((-90 + (360 / 5) * i) * Math.PI) / 180;
    var x2 = CX + R * Math.cos(ang);
    var y2 = CY + R * Math.sin(ang);
    var col = MODE_DIR_COLORS[i] || "#999";
    var ax = CX + (x2 - CX) * eased;
    var ay = CY + (y2 - CY) * eased;
    var axs = ax.toFixed(2);
    var ays = ay.toFixed(2);
    var parts = [];
    parts.push(
      "<defs><clipPath id=\"nc-move-shape-clip\"><circle cx=\"" +
        axs +
        "\" cy=\"" +
        ays +
        "\" r=\"" +
        SHAPE_CLIP_R +
        '"/></clipPath></defs>'
    );
    parts.push(
      '<line x1="' +
        CX +
        '" y1="' +
        CY +
        '" x2="' +
        x2 +
        '" y2="' +
        y2 +
        '" stroke="' +
        col +
        '" stroke-width="' +
        LINE_W +
        '" stroke-linecap="round"/>'
    );
    parts.push(
      '<circle cx="' +
        axs +
        '" cy="' +
        ays +
        '" r="26" fill="#ffffff" stroke="' +
        col +
        '" stroke-width="3"/>'
    );
    parts.push(
      svgStationImage(
        ax - CENTER_SHAPE_HALF,
        ay - CENTER_SHAPE_HALF,
        CENTER_SHAPE_SIZE,
        code,
        "nc-vis-move-shape",
        "nc-move-shape-clip"
      )
    );
    svgEl.innerHTML = parts.join("");
  }

  function runMoveAnimation(svg, graph, fromCode, mode, direction, onComplete) {
    var t0 = performance.now();
    function frame(now) {
      var elapsed = (now - t0) / ANIM_MS;
      var t = Math.min(1, elapsed);
      var eased = 1 - (1 - t) * (1 - t);
      drawVisAnimating(svg, graph, fromCode, mode, direction, eased);
      if (t < 1) {
        requestAnimationFrame(frame);
      } else {
        onComplete();
      }
    }
    requestAnimationFrame(frame);
  }

  function render(displayEl, graph, start, goal, getCode, onKey, opts) {
    opts = opts || {};
    var exploration = !!opts.exploration;
    var tp = opts.trialProgress;
    var progressBar = "";
    if (
      !exploration &&
      tp &&
      typeof tp.current === "number" &&
      typeof tp.total === "number" &&
      tp.total >= 1 &&
      tp.current >= 1 &&
      tp.current <= tp.total
    ) {
      progressBar =
        '<div class="nc-trial-progress" aria-live="polite">试次 ' +
        tp.current +
        " / " +
        tp.total +
        "</div>";
    }
    var wrap = document.createElement("div");
    wrap.className = "nc-nav-trial";
    var metaTop = exploration
      ? '<div id="nc-exp-panel" class="nc-exp-panel"></div>'
      : '<div class="nc-nav-line nc-nav-line--goal" style="--nc-goal-token-px:' +
        GOAL_STATION_IMG_CSS +
        'px">' +
        '<span id="nc-goal-line" class="nc-nav-line-txt"></span>' +
        '<span id="nc-goal-legend-wrap" class="nc-goal-legend-wrap">' +
        goalLegendTokenSvg(goal) +
        "</span>" +
        "</div>";
    wrap.innerHTML =
      '<div class="nc-nav-meta">' +
      progressBar +
      metaTop +
      "</div>" +
      '<div class="nc-nav-vis-wrap">' +
      '<div class="nc-nav-vis-title">请用按键选择交通工具</div>' +
      '<div class="nc-nav-vis-frame">' +
      '<svg id="nc-vis-svg" class="nc-vis-svg" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 ' +
        NAV_VIS_VB +
        " " +
        NAV_VIS_VB +
        '" width="' +
        NAV_VIS_SVG_CSS +
        '" height="' +
        NAV_VIS_SVG_CSS +
        '" aria-label="交通示意图"></svg>' +
      "</div>" +
      '<div class="nc-nav-legend nc-nav-legend--detailed">' +
      '<span><i class="nc-leg-line" style="background:#1e64d2"></i><i class="nc-leg-line nc-leg-line--2" style="background:#82c3ff"></i>公交（前）Q · 公交（后）E</span>' +
      '<span><i class="nc-leg-line" style="background:#1e8c3c"></i><i class="nc-leg-line nc-leg-line--2" style="background:#8ce1a0"></i>地铁（前）A · 地铁（后）D</span>' +
      '<span><i class="nc-leg-line" style="background:#8237c8"></i>快速巴士 W</span>' +
      "</div>" +
      "</div>" +
      '<div id="nc-msg" class="nc-msg"></div>';

    displayEl.innerHTML = "";
    displayEl.appendChild(wrap);

    if (!exploration) {
      document.getElementById("nc-goal-line").textContent = "目标站点：";
    }

    var svg = document.getElementById("nc-vis-svg");
    drawVis(svg, graph, start, null);

    function handler(ev) {
      var k = ev.key.toLowerCase();
      if (!KEY_MAP[k]) return;
      ev.preventDefault();
      onKey(k, KEY_MAP[k]);
    }
    window.addEventListener("keydown", handler);
    return function cleanup() {
      window.removeEventListener("keydown", handler);
    };
  }

  /**
   * @param {object|null} progressOpt 正式试次进度，如 { current: 3, total: 12 }；仅用于界面展示
   */
  function runNavTrial(display_element, graph, trialRow, phase, trialIndex, done, progressOpt) {
    var start = parseInt(trialRow.targetA, 10);
    var goal = parseInt(trialRow.targetB, 10);
    var code = start;
    var steps = [];
    var t0 = performance.now();
    var lastT = t0;
    var stepIdx = 0;
    var trialAnimating = false;
    var waitingContinue = false;

    function getCode() {
      return code;
    }

    function finishTrial() {
      cleanupKb();
      window.removeEventListener("keydown", continueHandler);
      done({
        phase: phase,
        domain: "navigation",
        trial_index: trialIndex,
        trial_progress:
          progressOpt &&
          typeof progressOpt.current === "number" &&
          typeof progressOpt.total === "number"
            ? { current: progressOpt.current, total: progressOpt.total }
            : null,
        start_code: start,
        goal_code: goal,
        steps: steps,
        total_steps: stepIdx,
        outcome: "reached_goal",
        total_time_ms: Math.round(performance.now() - t0),
      });
    }

    function continueHandler(ev) {
      if (!waitingContinue) return;
      if (ev.key === " " || ev.key === "Enter") {
        ev.preventDefault();
        waitingContinue = false;
        finishTrial();
      }
    }

    function handleAction(keyLower, md) {
      if (trialAnimating || waitingContinue) return;

      var tr = findTransition(graph, code, md.mode, md.direction);
      var stepStart = performance.now();
      var rtMs = Math.round(stepStart - lastT);

      if (!tr) {
        lastT = stepStart;
        steps.push({
          step: stepIdx + 1,
          from_code: code,
          key: keyLower,
          valid: false,
          to_code: null,
          rt_ms: rtMs,
        });
        var msg = document.getElementById("nc-msg");
        if (msg) msg.textContent = "当前站不可使用该交通方式";
        var svgBad = document.getElementById("nc-vis-svg");
        if (svgBad) drawVis(svgBad, graph, code, keyLower);
        setTimeout(function () {
          var s = document.getElementById("nc-vis-svg");
          if (s) drawVis(s, graph, code, null);
        }, 360);
        return;
      }

      var prev = code;
      var nextCode = tr.next_code;
      var svg = document.getElementById("nc-vis-svg");

      stepIdx += 1;
      steps.push({
        step: stepIdx,
        from_code: prev,
        key: keyLower,
        valid: true,
        mode: tr.mode,
        direction: tr.direction,
        to_code: nextCode,
        rt_ms: rtMs,
      });

      trialAnimating = true;
      if (svg) {
        runMoveAnimation(svg, graph, prev, tr.mode, tr.direction, function () {
          trialAnimating = false;
          code = nextCode;
          lastT = performance.now();

          var svgEl = document.getElementById("nc-vis-svg");
          if (svgEl) drawVis(svgEl, graph, code, null);

          var msg2 = document.getElementById("nc-msg");
          if (msg2) msg2.textContent = "";

          if (code === goal) {
            waitingContinue = true;
            if (msg2) msg2.textContent = "本试次完成（按空格继续）";
          }
        });
      } else {
        trialAnimating = false;
        code = nextCode;
        lastT = performance.now();
        if (code === goal) {
          waitingContinue = true;
          var msg3 = document.getElementById("nc-msg");
          if (msg3) msg3.textContent = "本试次完成（按空格继续）";
        }
      }
    }

    var renderOpts = null;
    if (
      progressOpt &&
      typeof progressOpt.current === "number" &&
      typeof progressOpt.total === "number"
    ) {
      renderOpts = {
        trialProgress: { current: progressOpt.current, total: progressOpt.total },
      };
    }
    var cleanupKb = render(
      display_element,
      graph,
      start,
      goal,
      getCode,
      function (k, md) {
        if (trialAnimating || waitingContinue) return;
        handleAction(k, md);
      },
      renderOpts
    );
    window.addEventListener("keydown", continueHandler);
  }

  /**
   * 自由探索：对齐 navigation/practice_main — 每站至少 minVisits 次且累计时长 >= minSeconds；
   * 达标自动结束。
   */
  function runNavExploration(display_element, graph, done) {
    var cfg = global.NC_MIX_CONFIG || {};
    var minMs = (cfg.explorationMinSeconds || 300) * 1000;
    var minVisits = cfg.explorationMinVisitsPerNode || 2;
    var codes = (graph.codebook || [])
      .map(function (c) {
        return c.code;
      })
      .sort(function (a, b) {
        return a - b;
      });
    if (!codes.length) codes = [1, 2, 3, 4, 5, 6, 7, 8, 9];

    var visitCounts = {};
    codes.forEach(function (c) {
      visitCounts[c] = 0;
    });
    var startCode = codes[0];
    visitCounts[startCode] += 1;

    var code = startCode;
    var roundStartedAt = performance.now();
    var expSteps = [];
    var stepIdx = 0;
    var trialAnimating = false;
    var tickTimer = null;
    var finished = false;
    var lastT = performance.now();

    function metrics() {
      var mastered = 0;
      codes.forEach(function (c) {
        if ((visitCounts[c] || 0) >= minVisits) mastered += 1;
      });
      var masteryOk = mastered === codes.length;
      var elapsedMs = performance.now() - roundStartedAt;
      var timeOk = elapsedMs >= minMs;
      return { masteryOk: masteryOk, timeOk: timeOk, elapsedMs: elapsedMs, mastered: mastered };
    }

    function updateExpPanel() {
      var el = document.getElementById("nc-exp-panel");
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
        codes.length +
        "（每站≥" +
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
      if (cleanupKb) cleanupKb();
      done({
        phase: "exploration",
        domain: "navigation",
        completed: completed,
        exploration_skipped: !completed,
        visit_counts: visitCounts,
        steps: expSteps,
        min_seconds: minMs / 1000,
        min_visits_per_node: minVisits,
        total_time_ms: Math.round(performance.now() - roundStartedAt),
      });
    }

    function getCode() {
      return code;
    }

    function handleAction(keyLower, md) {
      if (trialAnimating || finished) return;

      var tr = findTransition(graph, code, md.mode, md.direction);
      var stepStart = performance.now();
      var rtMs = Math.round(stepStart - lastT);

      if (!tr) {
        lastT = stepStart;
        expSteps.push({
          step: stepIdx + 1,
          from_code: code,
          key: keyLower,
          valid: false,
          to_code: null,
          rt_ms: rtMs,
        });
        var msg = document.getElementById("nc-msg");
        if (msg) msg.textContent = "当前站不可使用该交通方式";
        var svgBad = document.getElementById("nc-vis-svg");
        if (svgBad) drawVis(svgBad, graph, code, keyLower);
        setTimeout(function () {
          var s = document.getElementById("nc-vis-svg");
          if (s) drawVis(s, graph, code, null);
        }, 360);
        return;
      }

      var prev = code;
      var nextCode = tr.next_code;
      var svg = document.getElementById("nc-vis-svg");

      stepIdx += 1;
      expSteps.push({
        step: stepIdx,
        from_code: prev,
        key: keyLower,
        valid: true,
        mode: tr.mode,
        direction: tr.direction,
        to_code: nextCode,
        rt_ms: rtMs,
      });

      trialAnimating = true;
      if (svg) {
        runMoveAnimation(svg, graph, prev, tr.mode, tr.direction, function () {
          trialAnimating = false;
          code = nextCode;
          lastT = performance.now();
          visitCounts[nextCode] = (visitCounts[nextCode] || 0) + 1;

          var svgEl = document.getElementById("nc-vis-svg");
          if (svgEl) drawVis(svgEl, graph, code, null);

          var msg2 = document.getElementById("nc-msg");
          if (msg2) msg2.textContent = "";
          updateExpPanel();
        });
      } else {
        trialAnimating = false;
        code = nextCode;
        lastT = performance.now();
        visitCounts[nextCode] = (visitCounts[nextCode] || 0) + 1;
        updateExpPanel();
      }
    }

    var cleanupKb = render(
      display_element,
      graph,
      startCode,
      startCode,
      getCode,
      function (k, md) {
        if (trialAnimating) return;
        handleAction(k, md);
      },
      { exploration: true }
    );

    tickTimer = setInterval(updateExpPanel, 400);
    updateExpPanel();
  }

  global.NCNavRunner = { runNavTrial: runNavTrial, runNavExploration: runNavExploration };
})(window);
