/**
 * 尾部答题 B/D/E 嵌入 jsPsych v3（主观评分版：无正确答案/自信评分）。
 * 依赖：NCGridGeometry（grid_geometry.js）、NCMixAssets + NC_MIX_CONFIG（石块图）。
 */
(function (global) {
  var gm = global.NCGridGeometry;
  if (!gm || !gm.neighbor) {
    throw new Error("tail_task_embed_v3.js 需在 grid_geometry.js 之后加载");
  }

  var ALL = [1, 2, 3, 4, 5, 6, 7, 8, 9];
  var RAPID_NEXT = { 1: 3, 3: 9, 9: 7, 7: 1 };

  var NAV_NODES = {
    1: { ch: "\u25b2", color: "#E63C3C", name: "\u7ea2\u8272\u4e09\u89d2\u5f62\u7ad9" },
    2: { ch: "\u25a0", color: "#3C78E6", name: "\u84dd\u8272\u6b63\u65b9\u5f62\u7ad9" },
    3: { ch: "\u25cf", color: "#32B450", name: "\u7eff\u8272\u5706\u5f62\u7ad9" },
    4: { ch: "\u25c6", color: "#E69628", name: "\u6a59\u8272\u83f1\u5f62\u7ad9" },
    5: { ch: "\u2605", color: "#A050DC", name: "\u7d2b\u8272\u4e94\u89d2\u661f\u7ad9" },
    6: { ch: "\u2b22", color: "#E66EAA", name: "\u7c89\u8272\u516d\u8fb9\u5f62\u7ad9" },
    7: { ch: "\u271a", color: "#DCC828", name: "\u9ec4\u8272\u5341\u5b57\u7ad9" },
    8: { ch: "\u25bc", color: "#28BEC8", name: "\u9752\u8272\u5012\u4e09\u89d2\u7ad9" },
    9: { ch: "\u2b1f", color: "#A06E3C", name: "\u68d5\u8272\u4e94\u8fb9\u5f62\u7ad9" },
  };

  var STONE_NAMES = [
    "\u77f3\u5757\u4e00",
    "\u77f3\u5757\u4e8c",
    "\u77f3\u5757\u4e09",
    "\u77f3\u5757\u56db",
    "\u77f3\u5757\u4e94",
    "\u77f3\u5757\u516d",
    "\u77f3\u5757\u4e03",
    "\u77f3\u5757\u516b",
    "\u77f3\u5757\u4e5d",
  ];

  function mulberry32(a) {
    return function () {
      var t = (a += 0x6d2b79f5);
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function hashSeed(str) {
    var h = 2166136261 >>> 0;
    var s = String(str || "");
    for (var i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return h >>> 0;
  }

  function busSet() {
    var s = {};
    function add(a, b) {
      var k = a < b ? a + "-" + b : b + "-" + a;
      s[k] = true;
    }
    [[1, 2], [2, 3], [4, 5], [5, 6], [7, 8], [8, 9]].forEach(function (p) {
      add(p[0], p[1]);
    });
    return s;
  }
  function subwaySet() {
    var s = {};
    function add(a, b) {
      var k = a < b ? a + "-" + b : b + "-" + a;
      s[k] = true;
    }
    [[1, 4], [4, 7], [2, 5], [5, 8], [3, 6], [6, 9]].forEach(function (p) {
      add(p[0], p[1]);
    });
    return s;
  }
  var _BUS = busSet();
  var _SUB = subwaySet();

  function directedAction(a, b) {
    var pairKey = a < b ? a + "-" + b : b + "-" + a;
    if (_BUS[pairKey]) return b > a ? "bus_e" : "bus_q";
    if (_SUB[pairKey]) return b > a ? "subway_d" : "subway_a";
    if (RAPID_NEXT[a] === b) return "rapidbus_w";
    return null;
  }

  function gridNeighbors() {
    var nb = {};
    for (var i = 1; i <= 9; i++) nb[i] = {};
    function addU(a, b) {
      nb[a][b] = true;
      nb[b][a] = true;
    }
    [[1, 2], [2, 3], [4, 5], [5, 6], [7, 8], [8, 9], [1, 4], [4, 7], [2, 5], [5, 8], [3, 6], [6, 9]].forEach(
      function (p) {
        addU(p[0], p[1]);
      }
    );
    return nb;
  }
  var _GRID_NB = gridNeighbors();

  function gridDistance(a, b) {
    if (a === b) return 0;
    var visited = {};
    visited[a] = true;
    var frontier = [a];
    var d = 0;
    while (frontier.length) {
      d++;
      var nxt = [];
      for (var fi = 0; fi < frontier.length; fi++) {
        var u = frontier[fi];
        var keys = Object.keys(_GRID_NB[u]);
        for (var ki = 0; ki < keys.length; ki++) {
          var v = parseInt(keys[ki], 10);
          if (v === b) return d;
          if (!visited[v]) {
            visited[v] = true;
            nxt.push(v);
          }
        }
      }
      frontier = nxt;
    }
    return -1;
  }

  /** 与 tail_task._direct_targets 一致：含 W 一站出边 */
  function directTargetsOneStep(node) {
    var out = {};
    var keys = Object.keys(_GRID_NB[node]);
    for (var i = 0; i < keys.length; i++) out[parseInt(keys[i], 10)] = true;
    if (RAPID_NEXT[node]) out[RAPID_NEXT[node]] = true;
    return out;
  }

  /** 用 NCGridGeometry 验证一站（五键）是否与 directTargetsOneStep 一致 */
  function assertGeomMatchesDirectTargets() {
    for (var n = 1; n <= 9; n++) {
      var py = directTargetsOneStep(n);
      var js = {};
      gm.KEYS.forEach(function (k) {
        var t = gm.neighbor(n, k);
        if (t) js[t] = true;
      });
      var pk = Object.keys(py).sort().join(",");
      var jk = Object.keys(js).sort().join(",");
      if (pk !== jk) {
        console.warn("tail_task_embed_v3: neighbor mismatch at", n, py, js);
        break;
      }
    }
  }
  assertGeomMatchesDirectTargets();

  function buildPhaseB() {
    var trials = [];
    for (var i = 1; i <= 9; i++) {
      for (var j = i + 1; j <= 9; j++) {
        var d = gridDistance(i, j);
        trials.push({
          id: "B-" + String(trials.length + 1).padStart(2, "0"),
          phase: "B",
          a: i,
          b: j,
          grid_distance: d,
          category: "d" + d,
        });
      }
    }
    return trials;
  }

  function seqActions(seq) {
    var out = [];
    for (var i = 0; i < seq.length - 1; i++) {
      var act = directedAction(seq[i], seq[i + 1]);
      if (!act) throw new Error("bad seq " + seq[i] + "->" + seq[i + 1]);
      out.push(act);
    }
    return out;
  }

  function buildPhaseD() {
    var raw = [
      ["D-G1", [1, 2, 5], "G_corner"],
      ["D-G2", [5, 6, 3], "G_corner"],
      ["D-G3", [7, 4, 5], "G_corner"],
      ["D-G4", [9, 6, 5], "G_corner"],
      ["D-G5", [4, 5, 8], "G_corner"],
      ["D-F1", [1, 2, 5, 4], "F_face"],
      ["D-F2", [2, 3, 6, 5], "F_face"],
      ["D-F3", [4, 5, 8, 7], "F_face"],
      ["D-F4", [5, 6, 9, 8], "F_face"],
      ["D-F5", [4, 1, 2, 5], "F_face"],
      ["D-L1", [1, 3, 9], "L_ring"],
      ["D-L2", [3, 9, 7], "L_ring"],
      ["D-L3", [9, 7, 1], "L_ring"],
      ["D-L4", [1, 3, 9, 7], "L_ring"],
      ["D-S1", [1, 2, 3], "S_straight"],
      ["D-S2", [1, 4, 7], "S_straight"],
      ["D-S3", [4, 5, 6], "S_straight"],
      ["D-M1", [1, 3, 6, 5], "M_mixed"],
      ["D-M2", [9, 7, 8, 5], "M_mixed"],
      ["D-M3", [7, 1, 2, 5], "M_mixed"],
    ];
    var trials = [];
    for (var r = 0; r < raw.length; r++) {
      trials.push({
        id: raw[r][0],
        phase: "D",
        sequence: raw[r][1],
        actions: seqActions(raw[r][1]),
        category: raw[r][2],
        length: raw[r][1].length,
      });
    }
    return trials;
  }

  function buildPhaseE() {
    var trials = [];
    for (var s = 1; s <= 9; s++) {
      for (var e = 1; e <= 9; e++) {
        if (s === e) continue;
        if (gridDistance(s, e) !== 3) continue;
        var dt = directTargetsOneStep(s);
        if (dt[e]) throw new Error("E pair one-step: " + s + "->" + e);
        trials.push({
          id: "E-" + String(trials.length + 1).padStart(2, "0"),
          phase: "E",
          start: s,
          end: e,
          category: "grid_d3",
          grid_distance: 3,
        });
      }
    }
    return trials;
  }

  function pseudoShuffle(items, keyFn, maxConsec, rng) {
    var arr = items.slice();
    for (var attempt = 0; attempt < 5000; attempt++) {
      for (var i = arr.length - 1; i > 0; i--) {
        var j = Math.floor(rng() * (i + 1));
        var t = arr[i];
        arr[i] = arr[j];
        arr[j] = t;
      }
      var ok = true;
      var run = 1;
      for (var k = 1; k < arr.length; k++) {
        if (keyFn(arr[k]) === keyFn(arr[k - 1])) {
          run++;
          if (run > maxConsec) {
            ok = false;
            break;
          }
        } else run = 1;
      }
      if (ok) return arr;
    }
    return arr;
  }

  function buildFullTrialList(seed) {
    var rng = mulberry32(seed >>> 0);
    var B = pseudoShuffle(
      buildPhaseB(),
      function (t) {
        return t.category;
      },
      3,
      rng
    );
    var D = pseudoShuffle(
      buildPhaseD(),
      function (t) {
        return t.category;
      },
      2,
      rng
    );
    var E = pseudoShuffle(
      buildPhaseE(),
      function (t) {
        return t.start;
      },
      2,
      rng
    );
    var out = [];
    out.push({ phase: "intro", id: "intro_B", phase_name: "B" });
    out = out.concat(B);
    out.push({ phase: "intro", id: "intro_D", phase_name: "D" });
    out = out.concat(D);
    out.push({ phase: "intro", id: "intro_E", phase_name: "E" });
    out = out.concat(E);
    out.push({ phase: "end", id: "end" });
    return out;
  }

  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function navNodeHtml(code) {
    var n = NAV_NODES[code];
    var num = parseInt(code, 10);
    var url = "materials/img/nav/station_" + String(num || 1).padStart(2, "0") + ".svg";
    if (!n) return '<img src="' + esc(url) + '" alt="" style="width:64px;height:64px;object-fit:contain;filter:drop-shadow(0 2px 4px rgba(0,0,0,.4))"/>';
    return (
      '<div style="text-align:center">' +
      '<img src="' +
      esc(url) +
      '" alt="" style="width:64px;height:64px;object-fit:contain;filter:drop-shadow(0 2px 4px rgba(0,0,0,.4))"/>' +
      "</div>"
    );
  }

  function craftNodeHtml(code) {
    var assets = global.NCMixAssets;
    var sid = "stone_" + String(code).padStart(2, "0");
    var url =
      assets && typeof assets.stoneUrlCandidates === "function"
        ? assets.stoneUrlCandidates(sid)[0]
        : "materials/img/stone/stone_" + String(code).padStart(2, "0") + ".svg";
    return (
      '<div style="text-align:center">' +
      '<img src="' +
      esc(url) +
      '" alt="" style="width:64px;height:64px;object-fit:contain;filter:drop-shadow(0 2px 4px rgba(0,0,0,.4))"/>' +
      "</div>"
    );
  }

  function nodeHtml(domain, code) {
    return domain === "crafting" ? craftNodeHtml(code) : navNodeHtml(code);
  }

  function dashedArrowHtml(i) {
    var mid = "nc-tail-v3-arrow-" + String(i || 0);
    return (
      '<span style="display:inline-flex;align-items:center;justify-content:center;margin:0 10px;vertical-align:middle" aria-hidden="true">' +
      '<svg width="44" height="18" viewBox="0 0 44 18" focusable="false" aria-hidden="true" xmlns="http://www.w3.org/2000/svg">' +
      '<defs><marker id="' +
      mid +
      '" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#8b95a8"/></marker></defs>' +
      '<line x1="2" y1="9" x2="38" y2="9" stroke="#8b95a8" stroke-width="2" stroke-dasharray="5 4" stroke-linecap="round" marker-end="url(#' +
      mid +
      ')"/>' +
      "</svg></span>"
    );
  }

  function tailWrap(inner, progressHtml) {
    return (
      '<div class="nc-tail" style="max-width:720px;margin:0 auto;padding:20px 18px 32px;color:#e8ecf6;">' +
      (progressHtml || "") +
      inner +
      "</div>"
    );
  }

  function btnStyle() {
    return (
      "margin:8px;padding:12px 16px;font-size:15px;border-radius:8px;border:1px solid #5a6a8a;" +
      "background:#3d4a60;color:#fff;cursor:pointer;"
    );
  }

  function phaseDQuestion(domain) {
    return domain === "crafting"
      ? "\u4ee5\u4e0b\u52a0\u5de5\u6d41\u7a0b\u5728\u4f60\u773c\u91cc\u591a\u5927\u7a0b\u5ea6\u4e0a\u53ef\u4ee5\u6784\u6210\u4e00\u6761\u52a0\u5de5\u5de5\u5e8f\u3002"
      : "\u4ee5\u4e0b\u7ad9\u70b9\u5728\u4f60\u773c\u91cc\u591a\u5927\u7a0b\u5ea6\u4e0a\u53ef\u4ee5\u6784\u6210\u4e00\u6761\u7ebf\u8def\u3002";
  }

  var INTROS = {
    B: {
      title: "\u9636\u6bb5 B \u00b7 \u8282\u70b9\u95f4\u53ef\u8fbe\u5bb9\u6613\u7a0b\u5ea6\uff08\u5171 36 \u9898\uff09",
      body:
        "\u4f60\u5c06\u770b\u5230 36 \u5bf9\u8282\u70b9\uff08\u5168\u679a\u4e3e\uff0c\u6bcf\u5bf9\u4e00\u6b21\uff09\u3002" +
        "\u8bf7\u51ed\u4f60\u7684\u76f4\u89c9\u5224\u65ad\uff1a<strong>\u8fd9\u4e24\u4e2a\u8282\u70b9\u4e4b\u95f4\u591a\u5bb9\u6613\u5230\u8fbe\uff1f</strong><br><br>" +
        "\u2460 \u5f88\u96be\u5230\u8fbe \u2192 \u2464 \u5f88\u5bb9\u6613\u5230\u8fbe\u3002",
    },
    D: {
      title: "\u9636\u6bb5 D \u00b7 \u52a8\u4f5c\u7ec4\u6574\u4f53\u611f\uff08\u5171 20 \u9898\uff09",
      body:
        "\u4f60\u5c06\u770b\u5230 20 \u7ec4\u56fe\u5f62\u5316\u7684\u5e8f\u5217\u3002" +
        "\u8bf7\u7528 1\u20135 \u8bc4\u4ef7\u5b83\u5728\u4f60\u773c\u91cc\u6784\u6210\u4e00\u6761\u7ed3\u6784\u5316\u5e8f\u5217\u7684\u7a0b\u5ea6\u3002<br><br>" +
        "\u2460 \u5b8c\u5168\u4e0d\u50cf\u6574\u4f53 \u2192 \u2464 \u975e\u5e38\u50cf\u4e00\u4e2a\u6574\u4f53",
    },
    E: {
      title: "\u9636\u6bb5 E \u00b7 \u4e2d\u8f6c\u8282\u70b9\uff08\u5171 16 \u9898\uff09",
      body:
        "\u6bcf\u8bd5\u7ed9\u5b9a\u8d77\u70b9\u4e0e\u7ec8\u70b9\uff08\u4e0d\u8003\u8651\u73af\u7ebf\u65f6\u6700\u5c0f\u8ddd\u79bb\u5747\u4e3a 3\uff09\u3002" +
        "\u8bf7\u5728\u5269\u4f59 7 \u4e2a\u8282\u70b9\u4e2d<strong>\u9009 1 \u4e2a</strong>\u4f5c\u4e3a\u4f60\u8111\u4e2d\u6700\u5148\u60f3\u5230\u7684\u4e2d\u8f6c\u3002<br><br>" +
        "\u70b9\u51fb\u8282\u70b9\u5373\u53ef\u63d0\u4ea4\u672c\u9898\u3002",
    },
  };

  function phaseProgress(list, idx, phaseLetter) {
    var total = 0;
    var done = 0;
    for (var i = 0; i < list.length; i++) {
      if (list[i].phase === phaseLetter) total++;
    }
    for (var j = 0; j < idx; j++) {
      if (list[j].phase === phaseLetter) done++;
    }
    return { current: done + 1, total: total };
  }

  global.NCTailTaskEmbedV3 = {
    buildFullTrialList: buildFullTrialList,
    appendTailPhaseToTimeline: function (timeline, domain, participantId, experimentOrder) {
      var list = buildFullTrialList(
        hashSeed(
          String(participantId || "anon") +
            "|" +
            domain +
            "|" +
            String(experimentOrder || "") +
            "|tailv3"
        )
      );
      for (var idx = 0; idx < list.length; idx++) {
        var t = list[idx];
        if (t.phase === "intro") {
          var intro = INTROS[t.phase_name];
          timeline.push({
            type: jsPsychHtmlKeyboardResponse,
            stimulus: tailWrap(
              "<h2 style='font-size:20px;margin:0 0 14px'>" + esc(intro.title) + "</h2>" +
                "<p style='line-height:1.7;font-size:15px;color:#c4ccd8'>" +
                intro.body +
                "</p>" +
                "<p class='nc-inst-hint' style='margin-top:20px'>\u6309 <strong>\u7a7a\u683c</strong> \u7ee7\u7eed</p>",
              ""
            ),
            choices: [" "],
            data: {
              phase: "tail_task",
              tail_domain: domain === "crafting" ? "crafting" : "navigation",
              tail_subphase: "intro",
              tail_intro_phase: t.phase_name,
              screen_id: "tail_intro_" + t.phase_name,
            },
          });
          continue;
        }
        if (t.phase === "end") {
          timeline.push({
            type: jsPsychHtmlKeyboardResponse,
            stimulus: tailWrap(
              "<h2 style='font-size:20px'>\u7b54\u9898\u5c0f\u8282\u7ed3\u675f</h2>" +
                "<p style='color:#bcc6d8'>\u6309\u7a7a\u683c\u8fdb\u5165\u4e0b\u4e00\u9636\u6bb5\u5b9e\u9a8c\u3002</p>",
              ""
            ),
            choices: [" "],
            data: {
              phase: "tail_task",
              tail_domain: domain === "crafting" ? "crafting" : "navigation",
              tail_subphase: "outro",
              screen_id: "tail_block_end",
            },
          });
          continue;
        }

        var prog = phaseProgress(list, idx, t.phase);
        var progHtml =
          '<div style="text-align:center;font-size:13px;color:#9eb4d8;margin-bottom:14px">' +
          "\u7b54\u9898 " +
          (domain === "crafting" ? "crafting" : "navigation") +
          " \u00b7 \u9636\u6bb5 " +
          t.phase +
          " \uff1a " +
          prog.current +
          " / " +
          prog.total +
          "</div>";

        (function (trialIdx, trial, phtml) {
          timeline.push({
            type: jsPsychCallFunction,
            async: true,
            data: {
              phase: "tail_task",
              tail_domain: domain === "crafting" ? "crafting" : "navigation",
              tail_subphase: trial.phase,
              tail_trial_id: trial.id,
              tail_category: trial.category || "",
              screen_id: "tail_trial_" + trial.phase + "_" + trial.id,
              tail_present_version: "v3",
            },
            func: function (done) {
              var root = window.jsPsych.getDisplayElement();
              var t0 = performance.now();
              var seedLocal = hashSeed(
                String(participantId || "anon") + "|" + domain + "|" + String(experimentOrder || "") + "|tailv3"
              );

              if (trial.phase === "B") {
                var rngB = mulberry32((seedLocal + trialIdx * 997) >>> 0);
                var left = trial.a;
                var right = trial.b;
                if (rngB() < 0.5) {
                  left = trial.b;
                  right = trial.a;
                }
                root.innerHTML = tailWrap(
                  "<p style='font-size:16px;margin-bottom:16px'>\u8fd9\u4e24\u4e2a\u8282\u70b9\u4e4b\u95f4\u591a\u5bb9\u6613\u5230\u8fbe\uff1f</p>" +
                    '<div style="display:flex;align-items:flex-end;justify-content:center;gap:28px;flex-wrap:wrap;margin-bottom:20px">' +
                    '<div style="flex:0 0 auto">' +
                    nodeHtml(domain, left) +
                    "</div>" +
                    '<div style="font-size:28px;color:#8b95a8;padding-bottom:20px">\u2194</div>' +
                    '<div style="flex:0 0 auto">' +
                    nodeHtml(domain, right) +
                    "</div></div>" +
                    "<div>" +
                    [1, 2, 3, 4, 5]
                      .map(function (s) {
                        var labels = [
                          "\u2460 \u5f88\u96be",
                          "\u2461 \u8f83\u96be",
                          "\u2462 \u4e00\u822c",
                          "\u2463 \u8f83\u5bb9\u6613",
                          "\u2464 \u5f88\u5bb9\u6613",
                        ];
                        return (
                          "<button type='button' class='nc-tail-b1' data-s='" +
                          s +
                          "' style='" +
                          btnStyle() +
                          "'>" +
                          labels[s - 1] +
                          "</button>"
                        );
                      })
                      .join(" ") +
                    "</div>",
                  phtml
                );
                root.querySelectorAll(".nc-tail-b1").forEach(function (btn) {
                  btn.onclick = function () {
                    var rating = parseInt(btn.getAttribute("data-s"), 10);
                    var rt = Math.round(performance.now() - t0);
                    done({
                      tail_response_reachability: rating,
                      tail_rt_ms: rt,
                      tail_stimulus_pair: trial.a + "-" + trial.b,
                      tail_grid_distance: trial.grid_distance,
                      value: {
                        tail_phase: "B",
                        trial_id: trial.id,
                        response: rating,
                        grid_distance: trial.grid_distance,
                        a: trial.a,
                        b: trial.b,
                        display_left: left,
                        display_right: right,
                      },
                    });
                  };
                });
              } else if (trial.phase === "D") {
                var seq = trial.sequence;
                var parts = [];
                for (var si = 0; si < seq.length; si++) {
                  parts.push(
                    '<span style="display:inline-block;vertical-align:middle">' +
                      nodeHtml(domain, seq[si]) +
                      "</span>"
                  );
                  if (si < seq.length - 1) {
                    parts.push(dashedArrowHtml(si));
                  }
                }
                root.innerHTML = tailWrap(
                  "<p style='font-size:15px;margin-bottom:14px'>" +
                    phaseDQuestion(domain) +
                    "</p>" +
                    '<div style="display:flex;flex-wrap:wrap;align-items:center;justify-content:center;gap:4px;margin-bottom:18px">' +
                    parts.join("") +
                    "</div>" +
                    "<div>" +
                    [1, 2, 3, 4, 5]
                      .map(function (lv) {
                        return (
                          "<button type='button' class='nc-tail-d' data-lv='" +
                          lv +
                          "' style='" +
                          btnStyle() +
                          "'>" +
                          lv +
                          "</button>"
                        );
                      })
                      .join("") +
                    "</div>",
                  phtml
                );
                root.querySelectorAll(".nc-tail-d").forEach(function (btn) {
                  btn.onclick = function () {
                    var lv = parseInt(btn.getAttribute("data-lv"), 10);
                    var rt = Math.round(performance.now() - t0);
                    var seqStr = seq.join("->");
                    done({
                      tail_response_fluency: lv,
                      tail_rt_ms: rt,
                      tail_stimulus_sequence: seqStr,
                      value: {
                        tail_phase: "D",
                        trial_id: trial.id,
                        response: lv,
                        category: trial.category,
                        sequence: seq.slice(),
                      },
                    });
                  };
                });
              } else if (trial.phase === "E") {
                var candE = ALL.filter(function (n) {
                  return n !== trial.start && n !== trial.end;
                });
                candE.sort(function (a, b) {
                  return a - b;
                });
                root.innerHTML = tailWrap(
                  "<p style='font-size:16px;margin-bottom:12px'>\u8bf7\u9009\u62e9\u4f60\u5fc3\u91cc\u8fd9\u6761\u8def\u7ebf\u4e0a\u6700\u7406\u60f3\u7684\u4e2d\u8f6c\u8282\u70b9\uff1a</p>" +
                    '<div style="display:flex;align-items:center;justify-content:center;gap:14px;flex-wrap:wrap;margin-bottom:18px">' +
                    '<span style="font-size:15px;color:#c4ccd8;font-weight:600">起点：</span>' +
                    '<span style="display:inline-block">' +
                    nodeHtml(domain, trial.start) +
                    "</span>" +
                    '<span style="font-size:26px;color:#8b95a8">\u2192</span>' +
                    '<span style="font-size:15px;color:#c4ccd8;font-weight:600">终点：</span>' +
                    '<span style="display:inline-block">' +
                    nodeHtml(domain, trial.end) +
                    "</span></div>" +
                    "<div style='display:flex;flex-wrap:wrap;gap:10px;justify-content:center'>" +
                    candE
                      .map(function (n) {
                        return (
                          "<button type='button' class='nc-tail-eh' data-n='" +
                          n +
                          "' style='" +
                          btnStyle() +
                          "min-width:100px'>" +
                          "<span style='display:block;transform:scale(0.85)'>" +
                          nodeHtml(domain, n) +
                          "</span></button>"
                        );
                      })
                      .join("") +
                    "</div>",
                  phtml
                );
                root.querySelectorAll(".nc-tail-eh").forEach(function (btn) {
                  btn.onclick = function () {
                    var chosen = parseInt(btn.getAttribute("data-n"), 10);
                    var rt = Math.round(performance.now() - t0);
                    done({
                      tail_response_hub: chosen,
                      tail_rt_ms: rt,
                      tail_stimulus_route: trial.start + "->" + trial.end,
                      tail_e_hub_display_order: "numeric_asc",
                      tail_e_hub_options: candE.slice(),
                      tail_grid_distance: trial.grid_distance,
                      value: {
                        tail_phase: "E",
                        trial_id: trial.id,
                        chosen: chosen,
                        start: trial.start,
                        end: trial.end,
                        grid_distance: trial.grid_distance,
                        hub_options_order: candE.slice(),
                        hub_display_order: "numeric_asc",
                      },
                    });
                  };
                });
              }
            },
          });
        })(idx, t, progHtml);
      }
    },
  };
})(window);
