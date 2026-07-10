/*
 * Adaptive B/D/E after Session 3.
 * Builds participant-specific probes from jsPsych main-task rows.
 */
(function (global) {
  var gm = global.NCGridGeometry;
  if (!gm || !gm.neighbor) {
    throw new Error("adaptive_tail_task_embed_v3.js 需在 grid_geometry.js 之后加载");
  }

  var ALL = [1, 2, 3, 4, 5, 6, 7, 8, 9];
  var KEYS = ["q", "e", "a", "d", "w"];
  var RAPID_NEXT = { 1: 3, 3: 9, 9: 7, 7: 1 };
  var POS = {
    1: [0, 0],
    2: [0, 1],
    3: [0, 2],
    4: [1, 0],
    5: [1, 1],
    6: [1, 2],
    7: [2, 0],
    8: [2, 1],
    9: [2, 2],
  };
  var STRUCTURAL_TOKEN = "S";
  var CHUNK_MAX_PRIMITIVE_LEN = 3;
  var CHUNK_MAX_ITERATIONS = 20;
  var CHUNK_JS_THRESHOLD = 0.05;
  var CHUNK_CONVERGENCE_WINDOW = 3;
  var CHUNK_MIN_JOINT_PROBABILITY = 0.03;
  var CHUNK_MIN_LOG_BF = Math.log(3);
  var CHUNK_SCORE_RATIO_THRESHOLD = 0.9;
  var CHUNK_BDEU_ESS = 1.0;
  var CHUNK_BOOTSTRAP_ITERATIONS = 80;
  var LANDMARK_BOOTSTRAP_ITERATIONS = 500;
  var LANDMARK_SAMPLE_RATIO = 0.8;
  var LANDMARK_SELECTION_THRESHOLD = 0.7;
  var LANDMARK_MAX_CANDIDATES = 4;
  var LANDMARK_EPS = 1e-12;
  var LANDMARK_MIN_EDGE_COST = 1e-9;

  function hashSeed(str) {
    var h = 2166136261 >>> 0;
    var s = String(str || "");
    for (var i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return h >>> 0;
  }

  function mulberry32(a) {
    return function () {
      var t = (a += 0x6d2b79f5);
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function shuffle(arr, rng) {
    var out = arr.slice();
    for (var i = out.length - 1; i > 0; i--) {
      var j = Math.floor(rng() * (i + 1));
      var tmp = out[i];
      out[i] = out[j];
      out[j] = tmp;
    }
    return out;
  }

  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function toNode(x) {
    var n = parseInt(x, 10);
    return n >= 1 && n <= 9 ? n : null;
  }

  function neighbor(code, key) {
    return gm.neighbor(code, key);
  }

  function neighbors(code) {
    var out = [];
    KEYS.forEach(function (k) {
      var n = neighbor(code, k);
      if (n) out.push([n, k]);
    });
    return out;
  }

  function actionBetween(a, b) {
    for (var i = 0; i < KEYS.length; i++) {
      if (neighbor(a, KEYS[i]) === b) return KEYS[i];
    }
    return null;
  }

  function gridDistanceNoLoop(a, b) {
    var pa = POS[a];
    var pb = POS[b];
    return Math.abs(pa[0] - pb[0]) + Math.abs(pa[1] - pb[1]);
  }

  function shortestPath(start, goal) {
    if (start === goal) return [start];
    var queue = [{ node: start, path: [start] }];
    var seen = {};
    seen[start] = true;
    while (queue.length) {
      var cur = queue.shift();
      var nbs = neighbors(cur.node);
      for (var i = 0; i < nbs.length; i++) {
        var nxt = nbs[i][0];
        if (nxt === goal) return cur.path.concat([nxt]);
        if (!seen[nxt]) {
          seen[nxt] = true;
          queue.push({ node: nxt, path: cur.path.concat([nxt]) });
        }
      }
    }
    return [];
  }

  function sequenceActions(seq) {
    var out = [];
    for (var i = 0; i < seq.length - 1; i++) {
      var act = actionBetween(seq[i], seq[i + 1]);
      if (act) out.push(act);
      else {
        var path = shortestPath(seq[i], seq[i + 1]);
        for (var j = 0; j < path.length - 1; j++) {
          out.push(actionBetween(path[j], path[j + 1]) || "?");
        }
      }
    }
    return out;
  }

  function normalizeSteps(steps) {
    if (Array.isArray(steps)) return steps;
    if (typeof steps === "string" && steps.length) {
      try {
        var parsed = JSON.parse(steps);
        return Array.isArray(parsed) ? parsed : [];
      } catch (_err) {
        return [];
      }
    }
    return [];
  }

  function isFalseLike(value) {
    if (value === false || value === 0) return true;
    var text = String(value == null ? "" : value).trim().toLowerCase();
    return text === "false" || text === "0" || text === "invalid" || text === "no";
  }

  function flattenRecord(row) {
    var out = Object.assign({}, row || {});
    if (row && row.value && typeof row.value === "object" && !Array.isArray(row.value)) {
      Object.keys(row.value).forEach(function (key) {
        if (out[key] == null || out[key] === "") out[key] = row.value[key];
      });
    }
    return out;
  }

  function extractTrajectories(rows, domain) {
    var out = [];
    rows.forEach(function (row, idx) {
      var flat = flattenRecord(row);
      if (!flat || flat.domain !== domain || flat.phase !== "main") return;
      var rawSteps = normalizeSteps(flat.steps);
      if (!rawSteps.length) return;
      var states = [];
      var actions = [];
      rawSteps.forEach(function (st) {
        if (!st) return;
        if (isFalseLike(st.valid) || isFalseLike(st.executed)) return;
        var from = toNode(st.from_code);
        var to = toNode(st.to_code);
        var key = String(st.key || "").toLowerCase();
        if (!from || !to || from === to || KEYS.indexOf(key) < 0) return;
        if (!states.length) states.push(from);
        else if (states[states.length - 1] !== from) states.push(from);
        states.push(to);
        actions.push(key);
      });
      if (states.length < 2 || !actions.length) return;
      out.push({
        domain: domain,
        session: parseInt(flat.mix_session || 0, 10) || 0,
        order_index: idx,
        trial_id: String(flat.trial_id || flat.task_id || idx),
        states: states,
        actions: actions,
      });
    });
    out.sort(function (a, b) {
      return a.session - b.session || a.order_index - b.order_index;
    });
    return out;
  }

  function last80(trajs) {
    if (!trajs.length) return [];
    var drop = Math.ceil(trajs.length * 0.2);
    var kept = trajs.slice(drop);
    return kept.length ? kept : trajs.slice(-1);
  }

  function countKey(arr) {
    return arr.join(",");
  }

  function tokenParts(token) {
    return String(token || "").split("-").filter(Boolean);
  }

  function tokenPrimitiveLength(token) {
    if (token === STRUCTURAL_TOKEN) return 0;
    return tokenParts(token).length;
  }

  function isStructuralToken(token) {
    return token === STRUCTURAL_TOKEN || tokenParts(token).indexOf(STRUCTURAL_TOKEN) >= 0;
  }

  function logGamma(z) {
    var p = [
      676.5203681218851,
      -1259.1392167224028,
      771.32342877765313,
      -176.61502916214059,
      12.507343278686905,
      -0.13857109526572012,
      9.9843695780195716e-6,
      1.5056327351493116e-7,
    ];
    if (z < 0.5) {
      return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * z)) - logGamma(1 - z);
    }
    z -= 1;
    var x = 0.99999999999980993;
    for (var i = 0; i < p.length; i++) x += p[i] / (z + i + 1);
    var t = z + p.length - 0.5;
    return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
  }

  function bdeuBinaryScore(childValues, parentRows, parentStateCounts, ess) {
    var q = 1;
    for (var i = 0; i < parentStateCounts.length; i++) q *= parentStateCounts[i];
    var counts = [];
    for (var j = 0; j < q; j++) counts.push([0, 0]);
    for (var c = 0; c < childValues.length; c++) {
      var idx = 0;
      var mult = 1;
      for (var r = 0; r < parentRows.length; r++) {
        idx += parentRows[r][c] * mult;
        mult *= parentStateCounts[r];
      }
      counts[idx][childValues[c] ? 1 : 0] += 1;
    }
    var alphaParent = ess / q;
    var alphaCell = ess / (q * 2);
    var score = 0;
    for (var pc = 0; pc < q; pc++) {
      var n0 = counts[pc][0];
      var n1 = counts[pc][1];
      var nj = n0 + n1;
      score += logGamma(alphaParent) - logGamma(alphaParent + nj);
      score += logGamma(alphaCell + n0) - logGamma(alphaCell);
      score += logGamma(alphaCell + n1) - logGamma(alphaCell);
    }
    return score;
  }

  function elementFrequency(sequence) {
    var counts = {};
    var total = 0;
    sequence.forEach(function (token) {
      if (isStructuralToken(token)) return;
      counts[token] = (counts[token] || 0) + 1;
      total += 1;
    });
    var out = {};
    Object.keys(counts).forEach(function (token) {
      out[token] = counts[token] / Math.max(total, 1);
    });
    return out;
  }

  function scoringTokenCount(sequence) {
    var total = 0;
    sequence.forEach(function (token) {
      if (!isStructuralToken(token)) total += 1;
    });
    return total;
  }

  function jsDistance(a, b) {
    var support = {};
    Object.keys(a).forEach(function (k) { support[k] = true; });
    Object.keys(b).forEach(function (k) { support[k] = true; });
    var js = 0;
    Object.keys(support).forEach(function (k) {
      var p = a[k] || 0;
      var q = b[k] || 0;
      var m = 0.5 * (p + q);
      if (p > 0) js += 0.5 * p * Math.log(p / m);
      if (q > 0) js += 0.5 * q * Math.log(q / m);
    });
    return Math.sqrt(Math.max(js, 0));
  }

  function buildMiningStream(trajs) {
    var tokens = [];
    var states = [];
    trajs.forEach(function (tr) {
      for (var i = 0; i < tr.actions.length; i++) {
        tokens.push(tr.actions[i]);
        states.push(toNode(tr.states[i]) || -1);
      }
      tokens.push(STRUCTURAL_TOKEN);
      states.push(-1);
    });
    return { tokens: tokens, states: states };
  }

  function parseWithChunks(rawTokens, rawStates, chunks) {
    var sorted = chunks.slice().sort(function (a, b) {
      return tokenPrimitiveLength(b) - tokenPrimitiveLength(a) || a.localeCompare(b);
    });
    var outTokens = [];
    var outStates = [];
    var starts = [];
    var i = 0;
    while (i < rawTokens.length) {
      var matched = null;
      if (rawTokens[i] !== STRUCTURAL_TOKEN) {
        for (var c = 0; c < sorted.length; c++) {
          var parts = tokenParts(sorted[c]);
          var ok = parts.length > 0 && i + parts.length <= rawTokens.length;
          for (var j = 0; ok && j < parts.length; j++) {
            if (rawTokens[i + j] !== parts[j]) ok = false;
          }
          if (ok) {
            matched = sorted[c];
            break;
          }
        }
      }
      if (matched) {
        outTokens.push(matched);
        outStates.push(rawStates[i]);
        starts.push(i);
        i += tokenPrimitiveLength(matched);
      } else {
        outTokens.push(rawTokens[i]);
        outStates.push(rawStates[i]);
        starts.push(i);
        i += 1;
      }
    }
    return { tokens: outTokens, states: outStates, starts: starts };
  }

  function validPairRows(sequence, states) {
    var rows = [];
    for (var i = 0; i < sequence.length - 1; i++) {
      if (isStructuralToken(sequence[i]) || isStructuralToken(sequence[i + 1])) continue;
      var state = toNode(states[i]);
      if (!state) continue;
      rows.push({ index: i, state: state - 1 });
    }
    return rows;
  }

  function collectBdeuCandidates(sequence, states, learned) {
    var elements = {};
    sequence.forEach(function (token) {
      if (!isStructuralToken(token)) elements[token] = true;
    });
    var elementList = Object.keys(elements);
    var rows = validPairRows(sequence, states);
    var freq = elementFrequency(sequence);
    var tokenCount = scoringTokenCount(sequence);
    var learnedMap = {};
    learned.forEach(function (chunk) { learnedMap[chunk] = true; });
    var candidates = [];
    elementList.forEach(function (parent) {
      var parentValues = [];
      var stateValues = [];
      rows.forEach(function (row) {
        parentValues.push(sequence[row.index] === parent ? 1 : 0);
        stateValues.push(row.state);
      });
      var parentSum = parentValues.reduce(function (a, b) { return a + b; }, 0);
      if (!parentSum) return;
      elementList.forEach(function (child) {
        var chunk = parent + "-" + child;
        if (learnedMap[chunk]) return;
        if (tokenPrimitiveLength(parent) + tokenPrimitiveLength(child) > CHUNK_MAX_PRIMITIVE_LEN) return;
        var childValues = [];
        var joint = 0;
        rows.forEach(function (row, idx) {
          var isChild = sequence[row.index + 1] === child ? 1 : 0;
          childValues.push(isChild);
          if (parentValues[idx] && isChild) joint += 1;
        });
        var childSum = childValues.reduce(function (a, b) { return a + b; }, 0);
        if (!childSum) return;
        var pJoint = joint / Math.max(tokenCount, 1);
        if (pJoint < (freq[parent] || 0) * (freq[child] || 0)) return;
        if (pJoint < CHUNK_MIN_JOINT_PROBABILITY) return;
        var score1 = bdeuBinaryScore(childValues, [parentValues, stateValues], [2, 9], CHUNK_BDEU_ESS);
        var score0 = bdeuBinaryScore(childValues, [stateValues], [9], CHUNK_BDEU_ESS);
        var logBf = score1 - score0;
        if (logBf <= CHUNK_MIN_LOG_BF) return;
        candidates.push({
          chunk: chunk,
          parent: parent,
          child: child,
          score: logBf / Math.max(childValues.length, 1),
          log_bayes_factor: logBf,
          sample_count: childValues.length,
          joint_count: joint,
        });
      });
    });
    return candidates;
  }

  function chooseChunkCandidates(candidates) {
    if (!candidates.length) return [];
    var maxScore = candidates.reduce(function (m, c) { return Math.max(m, c.score); }, -Infinity);
    var logRatio = Math.log(CHUNK_SCORE_RATIO_THRESHOLD);
    return candidates.filter(function (c) {
      return c.score - maxScore > logRatio;
    });
  }

  function runBdeuChunkMining(trajs) {
    var stream = buildMiningStream(trajs);
    var rawTokens = stream.tokens;
    var rawStates = stream.states;
    var learned = [];
    var learnedMeta = {};
    var parsed = parseWithChunks(rawTokens, rawStates, learned);
    var jsHistory = [];
    for (var iter = 0; iter < CHUNK_MAX_ITERATIONS; iter++) {
      var candidates = collectBdeuCandidates(parsed.tokens, parsed.states, learned);
      var selected = chooseChunkCandidates(candidates);
      if (!selected.length) break;
      selected.forEach(function (c) {
        if (learned.indexOf(c.chunk) < 0) learned.push(c.chunk);
        learnedMeta[c.chunk] = c;
      });
      var oldFreq = elementFrequency(parsed.tokens);
      parsed = parseWithChunks(rawTokens, rawStates, learned);
      var nextFreq = elementFrequency(parsed.tokens);
      var js = jsDistance(oldFreq, nextFreq);
      jsHistory.push(js);
      if (jsHistory.length >= CHUNK_CONVERGENCE_WINDOW) {
        var recent = jsHistory.slice(-CHUNK_CONVERGENCE_WINDOW);
        var mean = recent.reduce(function (a, b) { return a + b; }, 0) / recent.length;
        if (mean < CHUNK_JS_THRESHOLD) break;
      }
    }
    return { chunks: learned, meta: learnedMeta, parsed: parsed, js_history: jsHistory };
  }

  function sampleTrajectories(trajs, sampleSize, rng) {
    var idxs = trajs.map(function (_tr, idx) { return idx; });
    for (var i = idxs.length - 1; i > 0; i--) {
      var j = Math.floor(rng() * (i + 1));
      var tmp = idxs[i];
      idxs[i] = idxs[j];
      idxs[j] = tmp;
    }
    return idxs.slice(0, sampleSize).map(function (idx) { return trajs[idx]; });
  }

  function trajectorySeed(domain, trajs, suffix) {
    return hashSeed(
      domain +
        "|" +
        suffix +
        "|" +
        trajs.map(function (tr) { return tr.session + ":" + tr.trial_id + ":" + tr.actions.join(""); }).join("|")
    );
  }

  function chunkRealizationSummary(trajs, chunk) {
    var parts = tokenParts(chunk);
    var realization = {};
    var total = 0;
    var support = {};
    trajs.forEach(function (tr) {
      for (var i = 0; i <= tr.actions.length - parts.length; i++) {
        var ok = true;
        for (var j = 0; j < parts.length; j++) {
          if (tr.actions[i + j] !== parts[j]) ok = false;
        }
        if (!ok) continue;
        var states = tr.states.slice(i, i + parts.length + 1).map(toNode).filter(Boolean);
        if (states.length !== parts.length + 1) continue;
        var key = countKey(states);
        realization[key] = (realization[key] || 0) + 1;
        support[tr.session + "|" + tr.trial_id] = true;
        total += 1;
      }
    });
    var best = Object.keys(realization).sort(function (a, b) {
      return realization[b] - realization[a] || a.localeCompare(b);
    })[0];
    return {
      count: total,
      support_trials: Object.keys(support).length,
      states: best ? best.split(",").map(function (x) { return parseInt(x, 10); }) : [],
      state_realizations: Object.keys(realization).map(function (key) {
        return { states: key.split(",").map(function (x) { return parseInt(x, 10); }), count: realization[key] };
      }),
    };
  }

  function fallbackChunks(domain) {
    return [
      { id: domain + "_chunk_fallback_1", actions: ["w", "w"], states: [1, 3, 9], length: 2, count: 0, support_trials: 0, score: 0, source: "fallback" },
      { id: domain + "_chunk_fallback_2", actions: ["e", "e"], states: [1, 2, 3], length: 2, count: 0, support_trials: 0, score: 0, source: "fallback" },
      { id: domain + "_chunk_fallback_3", actions: ["d", "d"], states: [1, 4, 7], length: 2, count: 0, support_trials: 0, score: 0, source: "fallback" },
      { id: domain + "_chunk_fallback_4", actions: ["q", "q"], states: [9, 8, 7], length: 2, count: 0, support_trials: 0, score: 0, source: "fallback" },
    ];
  }

  function mineChunks(domain, trajs, topK) {
    var result = runBdeuChunkMining(trajs);
    var stability = {};
    result.chunks.forEach(function (chunk) { stability[chunk] = 0; });
    if (trajs.length > 1 && result.chunks.length) {
      var sampleSize = Math.max(1, Math.ceil(trajs.length * 0.8));
      var rng = mulberry32(trajectorySeed(domain, trajs, "chunk-bootstrap"));
      for (var b = 0; b < CHUNK_BOOTSTRAP_ITERATIONS; b++) {
        var sample = sampleTrajectories(trajs, sampleSize, rng);
        var boot = runBdeuChunkMining(sample);
        var seen = {};
        boot.chunks.forEach(function (chunk) { seen[chunk] = true; });
        result.chunks.forEach(function (chunk) {
          if (seen[chunk]) stability[chunk] += 1;
        });
      }
    }
    var chunks = result.chunks.map(function (chunk) {
      var meta = result.meta[chunk] || {};
      var summary = chunkRealizationSummary(trajs, chunk);
      return {
        chunk: chunk,
        actions: tokenParts(chunk),
        states: summary.states,
        length: tokenPrimitiveLength(chunk),
        count: summary.count,
        support_trials: summary.support_trials,
        score: meta.score || 0,
        log_bayes_factor: meta.log_bayes_factor || 0,
        sample_count: meta.sample_count || 0,
        joint_count: meta.joint_count || 0,
        bootstrap_stability: CHUNK_BOOTSTRAP_ITERATIONS
          ? stability[chunk] / CHUNK_BOOTSTRAP_ITERATIONS
          : 0,
        state_realizations: summary.state_realizations,
        js_history: result.js_history,
        source: "bdeu_state_conditioned",
      };
    });
    chunks.sort(function (a, b) {
      return (
        b.bootstrap_stability - a.bootstrap_stability ||
        b.score - a.score ||
        b.log_bayes_factor - a.log_bayes_factor ||
        b.count - a.count ||
        a.chunk.localeCompare(b.chunk)
      );
    });
    chunks = chunks.slice(0, topK).map(function (item, idx) {
      item.id = domain + "_chunk_" + (idx + 1);
      return item;
    });
    if (chunks.length < topK) {
      chunks = chunks.concat(fallbackChunks(domain).slice(0, topK - chunks.length));
    }
    return chunks.slice(0, topK);
  }

  function emptyMatrix(n, fill) {
    var m = [];
    for (var i = 0; i < n; i++) {
      var row = [];
      for (var j = 0; j < n; j++) row.push(fill || 0);
      m.push(row);
    }
    return m;
  }

  function scoreLandmarks(trajs) {
    var n = ALL.length;
    var transition = emptyMatrix(n, 0);
    var visits = ALL.map(function () { return 0; });
    var coverage = ALL.map(function () { return 0; });
    var pairStates = {};
    trajs.forEach(function (tr) {
      var seen = {};
      tr.states.map(toNode).filter(Boolean).forEach(function (node) {
        visits[node - 1] += 1;
        seen[node] = true;
      });
      Object.keys(seen).forEach(function (node) {
        coverage[parseInt(node, 10) - 1] += 1;
      });
      for (var i = 0; i < tr.states.length - 1; i++) {
        var a = toNode(tr.states[i]);
        var b = toNode(tr.states[i + 1]);
        if (a && b) transition[a - 1][b - 1] += 1;
      }
      if (tr.states.length > 1) {
        var key = tr.states[0] + "->" + tr.states[tr.states.length - 1];
        pairStates[key] = pairStates[key] || {};
        tr.states.slice(1, -1).map(toNode).filter(Boolean).forEach(function (node) {
          pairStates[key][node] = true;
        });
      }
    });
    var nTrials = Math.max(trajs.length, 1);
    var pairCount = Math.max(Object.keys(pairStates).length, 1);
    var commonality = ALL.map(function () { return 0; });
    Object.keys(pairStates).forEach(function (key) {
      Object.keys(pairStates[key]).forEach(function (node) {
        commonality[parseInt(node, 10) - 1] += 1;
      });
    });
    var coverageFeature = coverage.map(function (v) { return v / nTrials; });
    var commonalityFeature = commonality.map(function (v) { return v / pairCount; });
    var betweenness = weightedBetweenness(transition);
    var rankCoverage = percentileRank(coverageFeature);
    var rankCommonality = percentileRank(commonalityFeature);
    var rankBetweenness = percentileRank(betweenness);
    var score = ALL.map(function (_node, idx) {
      return (rankCoverage[idx] + rankCommonality[idx] + rankBetweenness[idx]) / 3;
    });
    return {
      score: score,
      features: {
        coverage: coverageFeature,
        path_commonality: commonalityFeature,
        betweenness: betweenness,
        visits: visits,
        interior_count: interiorCounts(trajs),
      },
      transition_counts: transition,
    };
  }

  function interiorCounts(trajs) {
    var counts = ALL.map(function () { return 0; });
    trajs.forEach(function (tr) {
      tr.states.slice(1, -1).map(toNode).filter(Boolean).forEach(function (node) {
        counts[node - 1] += 1;
      });
    });
    return counts;
  }

  function percentileRank(values) {
    var n = values.length;
    if (n <= 1) return n === 1 ? [1] : [];
    var order = values.map(function (value, idx) { return { value: value, idx: idx }; });
    order.sort(function (a, b) { return a.value - b.value || a.idx - b.idx; });
    var ranks = [];
    for (var r = 0; r < n; r++) ranks.push(0);
    var start = 0;
    while (start < n) {
      var end = start + 1;
      while (end < n && order[end].value === order[start].value) end += 1;
      var avg = (start + end - 1) / 2 / (n - 1);
      for (var i = start; i < end; i++) ranks[order[i].idx] = avg;
      start = end;
    }
    return ranks;
  }

  function topScoreIndices(values, k) {
    return values.map(function (value, idx) {
      return { value: value, idx: idx };
    }).sort(function (a, b) {
      return b.value - a.value || a.idx - b.idx;
    }).slice(0, k).map(function (row) {
      return row.idx;
    });
  }

  function weightedBetweenness(transition) {
    var n = transition.length;
    var adjacency = [];
    var weights = emptyMatrix(n, Infinity);
    for (var i = 0; i < n; i++) {
      adjacency[i] = [];
      var rowSum = transition[i].reduce(function (a, b) { return a + b; }, 0);
      for (var j = 0; j < n; j++) {
        if (transition[i][j] <= 0 || rowSum <= 0) continue;
        adjacency[i].push(j);
        var p = transition[i][j] / rowSum;
        weights[i][j] = Math.max(-Math.log(Math.max(p, LANDMARK_EPS)), LANDMARK_MIN_EDGE_COST);
      }
    }
    var bet = [];
    for (var z = 0; z < n; z++) bet.push(0);
    for (var source = 0; source < n; source++) {
      var stack = [];
      var pred = [];
      var sigma = [];
      var dist = [];
      var queue = [{ node: source, dist: 0 }];
      for (var a = 0; a < n; a++) {
        pred[a] = [];
        sigma[a] = 0;
        dist[a] = Infinity;
      }
      sigma[source] = 1;
      dist[source] = 0;
      while (queue.length) {
        queue.sort(function (x, y) { return x.dist - y.dist; });
        var item = queue.shift();
        var v = item.node;
        if (item.dist > dist[v] + LANDMARK_EPS) continue;
        stack.push(v);
        adjacency[v].forEach(function (w) {
          var candidate = dist[v] + weights[v][w];
          if (candidate < dist[w] - LANDMARK_EPS) {
            dist[w] = candidate;
            queue.push({ node: w, dist: candidate });
            sigma[w] = sigma[v];
            pred[w] = [v];
          } else if (Math.abs(candidate - dist[w]) <= LANDMARK_EPS) {
            sigma[w] += sigma[v];
            pred[w].push(v);
          }
        });
      }
      var dependency = [];
      for (var d = 0; d < n; d++) dependency[d] = 0;
      while (stack.length) {
        var wNode = stack.pop();
        if (sigma[wNode] > 0) {
          var coeff = (1 + dependency[wNode]) / sigma[wNode];
          pred[wNode].forEach(function (vNode) {
            dependency[vNode] += sigma[vNode] * coeff;
          });
        }
        if (wNode !== source) bet[wNode] += dependency[wNode];
      }
    }
    if (n > 2) {
      for (var b = 0; b < n; b++) bet[b] /= (n - 1) * (n - 2);
    }
    return bet;
  }

  function fallbackLandmarks(domain) {
    return [5, 3, 9, 1].map(function (node, idx) {
      return {
        id: domain + "_landmark_fallback_" + (idx + 1),
        node: node,
        visit_count: 0,
        interior_count: 0,
        support_trials: 0,
        score: 0,
        source: "fallback",
      };
    });
  }

  function mineLandmarks(domain, trajs, topK) {
    var full = scoreLandmarks(trajs);
    var selectionCounts = ALL.map(function () { return 0; });
    if (trajs.length > 1) {
      var rng = mulberry32(trajectorySeed(domain, trajs, "landmark-bootstrap"));
      var sampleSize = Math.max(1, Math.ceil(trajs.length * LANDMARK_SAMPLE_RATIO));
      for (var b = 0; b < LANDMARK_BOOTSTRAP_ITERATIONS; b++) {
        var sample = sampleTrajectories(trajs, sampleSize, rng);
        var sampleScore = scoreLandmarks(sample).score;
        topScoreIndices(sampleScore, LANDMARK_MAX_CANDIDATES).forEach(function (idx) {
          selectionCounts[idx] += 1;
        });
      }
    }
    var rates = selectionCounts.map(function (count) {
      return LANDMARK_BOOTSTRAP_ITERATIONS ? count / LANDMARK_BOOTSTRAP_ITERATIONS : 0;
    });
    var ranked = ALL.map(function (node, idx) {
      return {
        node: node,
        score: full.score[idx],
        selection_rate: rates[idx],
        coverage: full.features.coverage[idx],
        path_commonality: full.features.path_commonality[idx],
        betweenness: full.features.betweenness[idx],
        visit_count: full.features.visits[idx],
        interior_count: full.features.interior_count[idx],
      };
    }).sort(function (a, b) {
      return b.selection_rate - a.selection_rate || b.score - a.score || a.node - b.node;
    });
    var stable = ranked.filter(function (r) { return r.selection_rate >= LANDMARK_SELECTION_THRESHOLD; });
    var selected = stable.concat(ranked.filter(function (r) {
      return stable.indexOf(r) < 0 && r.score > 0;
    })).slice(0, topK);
    var landmarks = selected.map(function (r, idx) {
      return {
        id: domain + "_landmark_" + (idx + 1),
        node: r.node,
        visit_count: r.visit_count,
        interior_count: r.interior_count,
        support_trials: Math.round(r.coverage * Math.max(trajs.length, 1)),
        score: r.score,
        coverage: r.coverage,
        path_commonality: r.path_commonality,
        betweenness: r.betweenness,
        selection_rate: r.selection_rate,
        bootstrap_stability: r.selection_rate,
        source: r.selection_rate >= LANDMARK_SELECTION_THRESHOLD ? "bootstrap_landmark" : "bootstrap_landmark_low_stability",
      };
    });
    var seen = {};
    landmarks.forEach(function (lm) {
      seen[lm.node] = true;
    });
    fallbackLandmarks(domain).forEach(function (lm) {
      if (landmarks.length < topK && !seen[lm.node]) landmarks.push(lm);
    });
    return landmarks.slice(0, topK);
  }

  function controlSequence(seq) {
    var candidates = [
      [7, 8, 9],
      [3, 6, 9],
      [4, 5, 2],
      [9, 7, 1],
      [2, 5, 8],
      [6, 5, 4],
      [1, 2, 5, 8],
    ];
    for (var i = 0; i < candidates.length; i++) {
      if (candidates[i].length === seq.length) return candidates[i].slice();
    }
    return seq.length <= 3 ? [1, 2, 5].slice(0, seq.length) : [1, 2, 5, 8].slice(0, seq.length);
  }

  function landmarkRoute(node) {
    var cands = [];
    ALL.forEach(function (start) {
      if (start === node) return;
      var left = shortestPath(start, node);
      if (!left.length || left.length > 3) return;
      ALL.forEach(function (end) {
        if (end === node || end === start) return;
        var right = shortestPath(node, end);
        if (!right.length || right.length > 3) return;
        var route = left.concat(right.slice(1));
        if (route.length < 3 || route.length > 5) return;
        cands.push({ score: Math.abs(route.length - 4), route: route });
      });
    });
    cands.sort(function (a, b) {
      return a.score - b.score || countKey(a.route).localeCompare(countKey(b.route));
    });
    if (cands.length) return cands[0].route;
    return node === 1 || node === 9 ? [2, node, 8] : [1, node, 9];
  }

  function hubOptions(start, end) {
    return ALL.filter(function (n) {
      return n !== start && n !== end;
    });
  }

  function common(domain, construct, condition, sourceId, source) {
    return {
      domain: domain,
      target_construct: construct,
      condition_label: condition,
      source_candidate_id: String(sourceId || ""),
      candidate_source: String(source || ""),
    };
  }

  function buildDomainTrials(domain, chunks, landmarks, rng, itemsPerPhase) {
    var B = [];
    var D = [];
    var E = [];
    chunks.forEach(function (ch) {
      var seq = (ch.states || []).map(toNode).filter(Boolean);
      if (seq.length < 2) return;
      var start = seq[0];
      var end = seq[seq.length - 1];
      var mid = seq[Math.floor(seq.length / 2)];
      var c = common(domain, "chunk", "chunk_experimental", ch.id, ch.source);
      c.candidate_chunk_id = ch.id;
      c.chunk_confidence = ch.bootstrap_stability || 0;
      c.chunk_log_bayes_factor = ch.log_bayes_factor || 0;
      c.chunk_score = ch.score || 0;
      B.push(Object.assign({}, c, { phase: "B", a: start, b: end, nodes: [start, end], directional: true, grid_distance: gridDistanceNoLoop(start, end), chunk_actions: ch.actions || [] }));
      D.push(Object.assign({}, c, { phase: "D", sequence: seq, nodes: seq, actions: sequenceActions(seq), length: seq.length }));
      if (mid !== start && mid !== end) {
        E.push(Object.assign({}, c, { phase: "E", start: start, end: end, hub_options: hubOptions(start, end), chunk_predicted_hub: mid, landmark_predicted_hub: null }));
      }
      var ctl = controlSequence(seq);
      var cc = common(domain, "chunk", "chunk_matched_control", ch.id, ch.source);
      cc.candidate_chunk_id = ch.id;
      cc.chunk_confidence = ch.bootstrap_stability || 0;
      cc.chunk_log_bayes_factor = ch.log_bayes_factor || 0;
      cc.chunk_score = ch.score || 0;
      B.push(Object.assign({}, cc, { phase: "B", a: ctl[0], b: ctl[ctl.length - 1], nodes: [ctl[0], ctl[ctl.length - 1]], directional: true, grid_distance: gridDistanceNoLoop(ctl[0], ctl[ctl.length - 1]), chunk_actions: sequenceActions(ctl) }));
      D.push(Object.assign({}, cc, { phase: "D", sequence: ctl, nodes: ctl, actions: sequenceActions(ctl), length: ctl.length }));
    });
    landmarks.forEach(function (lm) {
      var node = toNode(lm.node) || 5;
      var route = landmarkRoute(node);
      var start = route[0];
      var end = route[route.length - 1];
      var c = common(domain, "landmark", "landmark_experimental", lm.id, lm.source);
      c.candidate_landmark_id = lm.id;
      c.candidate_landmark_node = node;
      c.landmark_confidence = lm.selection_rate || 0;
      c.landmark_score = lm.score || 0;
      B.push(Object.assign({}, c, { phase: "B", a: start, b: end, nodes: [start, end], directional: true, grid_distance: gridDistanceNoLoop(start, end), landmark_node: node }));
      D.push(Object.assign({}, c, { phase: "D", sequence: route, nodes: route, actions: sequenceActions(route), length: route.length, landmark_node: node }));
      E.push(Object.assign({}, c, { phase: "E", start: start, end: end, hub_options: hubOptions(start, end), chunk_predicted_hub: null, landmark_predicted_hub: node, landmark_node: node }));
      var other = [5, 3, 7, 9, 1, 2, 4, 6, 8].filter(function (n) {
        return n !== node;
      })[0];
      var ctl = landmarkRoute(other);
      var cc = common(domain, "landmark", "landmark_matched_control", lm.id, lm.source);
      cc.candidate_landmark_id = lm.id;
      cc.candidate_landmark_node = node;
      cc.landmark_confidence = lm.selection_rate || 0;
      cc.landmark_score = lm.score || 0;
      D.push(Object.assign({}, cc, { phase: "D", sequence: ctl, nodes: ctl, actions: sequenceActions(ctl), length: ctl.length, landmark_node: node, control_landmark_node: other }));
    });

    function choose(rows, phase) {
      var chunkRows = shuffle(rows.filter(function (r) { return r.target_construct === "chunk"; }), rng);
      var landmarkRows = shuffle(rows.filter(function (r) { return r.target_construct === "landmark"; }), rng);
      var half = Math.max(1, Math.floor(itemsPerPhase / 2));
      var chosen = chunkRows.slice(0, half).concat(landmarkRows.slice(0, half));
      var rest = shuffle(chunkRows.slice(half).concat(landmarkRows.slice(half)), rng);
      chosen = chosen.concat(rest.slice(0, Math.max(0, itemsPerPhase - chosen.length)));
      while (chosen.length < itemsPerPhase) {
        var a = ALL[Math.floor(rng() * ALL.length)];
        var b = ALL[Math.floor(rng() * ALL.length)];
        if (a === b) b = ((b + 1) % 9) + 1;
        chosen.push(Object.assign({}, common(domain, "baseline", "adaptive_fallback", "none", "fallback"), { phase: phase, a: a, b: b, nodes: [a, b], directional: true, grid_distance: gridDistanceNoLoop(a, b) }));
      }
      return chosen.slice(0, itemsPerPhase).map(function (row, idx) {
        row.id = domain.slice(0, 3).toUpperCase() + "-" + phase + "-ADAPT-" + String(idx + 1).padStart(2, "0");
        return row;
      });
    }

    return choose(B, "B").concat(choose(D, "D")).concat(choose(E, "E"));
  }

  function buildAdaptivePackFromRows(rows, participantId, experimentOrder, itemsPerPhaseDomain) {
    var items = itemsPerPhaseDomain || 8;
    var rng = mulberry32(hashSeed(String(participantId || "anon") + "|" + String(experimentOrder || "") + "|adaptive-bde-v1"));
    var domains = {};
    var allTrials = [];
    ["navigation", "crafting"].forEach(function (domain) {
      var trajs = extractTrajectories(rows || [], domain);
      var windowRows = last80(trajs);
      var chunks = mineChunks(domain, windowRows, 4);
      var landmarks = mineLandmarks(domain, windowRows, 4);
      domains[domain] = {
        trial_count_total: trajs.length,
        trial_count_used: windowRows.length,
        window_rule: "ceil_drop_first_20_percent_keep_last_80_percent",
        chunk_method: "state_conditioned_bdeu_recursive_chunk_mining",
        chunk_bootstrap_iterations: CHUNK_BOOTSTRAP_ITERATIONS,
        landmark_method: "rank_aggregation_coverage_commonality_betweenness_bootstrap",
        landmark_bootstrap_iterations: LANDMARK_BOOTSTRAP_ITERATIONS,
        landmark_selection_threshold: LANDMARK_SELECTION_THRESHOLD,
        chunks: chunks,
        landmarks: landmarks,
      };
      allTrials = allTrials.concat(buildDomainTrials(domain, chunks, landmarks, rng, items));
    });
    var trials = [];
    ["B", "D", "E"].forEach(function (phase) {
      trials = trials.concat(shuffle(allTrials.filter(function (t) { return t.phase === phase; }), rng));
    });
    return {
      schema: "adaptive_bde_pack",
      version: "1.1-js-bdeu-bootstrap",
      participant_id: participantId || "anon",
      experiment_order: experimentOrder || "",
      source_data_window: "last_80_percent_by_domain",
      window_rule: "ceil_drop_first_20_percent_keep_last_80_percent",
      items_per_phase_domain: items,
      domains: domains,
      trials: trials,
    };
  }

  function navNodeHtml(code) {
    var num = parseInt(code, 10) || 1;
    var url = "materials/img/nav/station_" + String(num).padStart(2, "0") + ".svg";
    return '<img src="' + esc(url) + '" alt="" style="width:64px;height:64px;object-fit:contain;filter:drop-shadow(0 2px 4px rgba(0,0,0,.4))"/>';
  }

  function craftNodeHtml(code) {
    var assets = global.NCMixAssets;
    var sid = "stone_" + String(code).padStart(2, "0");
    var url =
      assets && typeof assets.stoneUrlCandidates === "function"
        ? assets.stoneUrlCandidates(sid)[0]
        : "materials/img/stone/stone_" + String(code).padStart(2, "0") + ".svg";
    return '<img src="' + esc(url) + '" alt="" style="width:64px;height:64px;object-fit:contain;filter:drop-shadow(0 2px 4px rgba(0,0,0,.4))"/>';
  }

  function nodeHtml(domain, code) {
    return domain === "crafting" ? craftNodeHtml(code) : navNodeHtml(code);
  }

  function wrap(inner) {
    return '<div class="nc-tail nc-adaptive-tail" style="max-width:760px;margin:0 auto;padding:20px 18px 32px;color:#e8ecf6;">' + inner + "</div>";
  }

  function btnStyle() {
    return "margin:8px;padding:12px 16px;font-size:15px;border-radius:8px;border:1px solid #5a6a8a;background:#3d4a60;color:#fff;cursor:pointer;";
  }

  function waitKeyOrClick(root, selector, validKeys) {
    return new Promise(function (resolve) {
      var done = false;
      function finish(value) {
        if (done) return;
        done = true;
        window.removeEventListener("keydown", onKey, true);
        resolve(value);
      }
      function onKey(ev) {
        if (ev.key === "Escape") finish(null);
        if (validKeys && validKeys.indexOf(ev.key) >= 0) finish(ev.key);
      }
      window.addEventListener("keydown", onKey, true);
      Array.prototype.forEach.call(root.querySelectorAll(selector), function (btn) {
        btn.addEventListener("click", function () {
          finish(btn.getAttribute("data-value"));
        });
      });
    });
  }

  function phaseIntro(root, phase, total) {
    var text = {
      B: "请根据刚才任务中的直觉，判断从左侧节点到右侧节点多容易到达。",
      D: "请判断呈现的节点序列在你眼里多大程度上像一条熟悉的线路或加工工序。",
      E: "请在候选节点中选择你心里这条路线上最理想的中转节点。",
    }[phase];
    root.innerHTML = wrap(
      "<h2 style='font-size:20px;margin:0 0 14px'>Adaptive 阶段 " +
        phase +
        "</h2><p style='line-height:1.7;font-size:15px;color:#c4ccd8'>" +
        esc(text) +
        "</p><p style='color:#9eb4d8'>本阶段共 " +
        total +
        " 题。</p><p class='nc-inst-hint' style='margin-top:20px'>按 <strong>空格</strong> 继续</p>"
    );
    return waitKeyOrClick(root, "button", [" "]).then(function (v) {
      return v !== null;
    });
  }

  function ratingButtons() {
    return [1, 2, 3, 4, 5]
      .map(function (s) {
        return '<button type="button" data-value="' + s + '" style="' + btnStyle() + 'min-width:72px">' + s + "</button>";
      })
      .join("");
  }

  function dashedArrowHtml(i) {
    var mid = "nc-adaptive-arrow-" + String(i || 0) + "-" + Math.floor(Math.random() * 100000);
    return (
      '<span style="display:inline-flex;align-items:center;justify-content:center;margin:0 8px;vertical-align:middle" aria-hidden="true">' +
      '<svg width="42" height="18" viewBox="0 0 42 18" focusable="false" aria-hidden="true" xmlns="http://www.w3.org/2000/svg">' +
      '<defs><marker id="' +
      mid +
      '" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#8b95a8"/></marker></defs>' +
      '<line x1="2" y1="9" x2="36" y2="9" stroke="#8b95a8" stroke-width="2" stroke-dasharray="5 4" stroke-linecap="round" marker-end="url(#' +
      mid +
      ')"/></svg></span>'
    );
  }

  function runB(root, trial, idx, total) {
    var domain = trial.domain === "crafting" ? "crafting" : "navigation";
    var nodes = trial.nodes || [trial.a, trial.b];
    var t0 = performance.now();
    root.innerHTML = wrap(
      "<div style='text-align:center;font-size:13px;color:#9eb4d8;margin-bottom:14px'>Adaptive B · " +
        esc(domain) +
        " · " +
        idx +
        " / " +
        total +
        "</div><p style='font-size:16px;margin-bottom:18px;text-align:center'>" +
        (domain === "crafting" ? "从左侧加工状态到右侧加工状态，在你眼里多容易到达？" : "从左侧站点到右侧站点，在你眼里多容易到达？") +
        "</p><div style='display:flex;align-items:center;justify-content:center;gap:28px;flex-wrap:wrap;margin-bottom:20px'>" +
        nodeHtml(domain, nodes[0]) +
        "<div style='font-size:28px;color:#8b95a8'>→</div>" +
        nodeHtml(domain, nodes[1]) +
        "</div><div style='text-align:center'>" +
        ratingButtons() +
        "</div>"
    );
    return waitKeyOrClick(root, "button[data-value]", ["1", "2", "3", "4", "5"]).then(function (value) {
      if (value == null) return null;
      return Object.assign({}, trial, { rating: parseInt(value, 10), rt_ms: Math.round(performance.now() - t0), presented_nodes: nodes });
    });
  }

  function runD(root, trial, idx, total) {
    var domain = trial.domain === "crafting" ? "crafting" : "navigation";
    var seq = trial.sequence || trial.nodes || [1, 2, 3];
    var t0 = performance.now();
    var seqHtml = seq
      .map(function (code, i) {
        return '<span style="display:inline-flex;align-items:center">' + nodeHtml(domain, code) + (i < seq.length - 1 ? dashedArrowHtml(i) : "") + "</span>";
      })
      .join("");
    root.innerHTML = wrap(
      "<div style='text-align:center;font-size:13px;color:#9eb4d8;margin-bottom:14px'>Adaptive D · " +
        esc(domain) +
        " · " +
        idx +
        " / " +
        total +
        "</div><p style='font-size:16px;margin-bottom:18px;text-align:center'>" +
        (domain === "crafting" ? "以下加工流程在你眼里多大程度上可以构成一条熟悉的加工工序？" : "以下站点在你眼里多大程度上可以构成一条熟悉的线路？") +
        "</p><div style='display:flex;align-items:center;justify-content:center;gap:4px;flex-wrap:wrap;margin:14px 0 22px'>" +
        seqHtml +
        "</div><div style='text-align:center'>" +
        ratingButtons() +
        "</div>"
    );
    return waitKeyOrClick(root, "button[data-value]", ["1", "2", "3", "4", "5"]).then(function (value) {
      if (value == null) return null;
      return Object.assign({}, trial, { rating: parseInt(value, 10), rt_ms: Math.round(performance.now() - t0), presented_sequence: seq });
    });
  }

  function runE(root, trial, idx, total) {
    var domain = trial.domain === "crafting" ? "crafting" : "navigation";
    var start = parseInt(trial.start || 1, 10);
    var end = parseInt(trial.end || 9, 10);
    var options = {};
    (trial.hub_options || hubOptions(start, end)).forEach(function (n) {
      options[parseInt(n, 10)] = true;
    });
    var t0 = performance.now();
    var grid = ALL.map(function (code) {
      var enabled = !!options[code];
      return (
        '<button type="button" data-value="' +
        code +
        '" ' +
        (enabled ? "" : "disabled ") +
        'style="width:92px;height:92px;margin:8px;border-radius:8px;border:1px solid ' +
        (enabled ? "#5a6a8a" : "#343b4a") +
        ";background:" +
        (enabled ? "#2f3849" : "#222731") +
        ";opacity:" +
        (enabled ? "1" : ".42") +
        ';cursor:' +
        (enabled ? "pointer" : "default") +
        '">' +
        nodeHtml(domain, code) +
        "</button>"
      );
    }).join("");
    root.innerHTML = wrap(
      "<div style='text-align:center;font-size:13px;color:#9eb4d8;margin-bottom:14px'>Adaptive E · " +
        esc(domain) +
        " · " +
        idx +
        " / " +
        total +
        "</div><p style='font-size:16px;margin-bottom:16px;text-align:center'>请选择你心里这条路线上最理想的中转节点：</p>" +
        "<div style='display:flex;align-items:center;justify-content:center;gap:16px;margin-bottom:18px'><span>起点</span>" +
        nodeHtml(domain, start) +
        "<span style='font-size:24px;color:#8b95a8'>→</span><span>终点</span>" +
        nodeHtml(domain, end) +
        "</div><div style='max-width:340px;margin:0 auto;display:grid;grid-template-columns:repeat(3,108px);justify-content:center'>" +
        grid +
        "</div><p style='text-align:center;color:#9eb4d8;font-size:13px;margin-top:14px'>点击可选节点，或按对应数字键 1-9。</p>"
    );
    var keys = Object.keys(options);
    return waitKeyOrClick(root, "button[data-value]:not([disabled])", keys).then(function (value) {
      if (value == null) return null;
      return Object.assign({}, trial, { chosen_hub: parseInt(value, 10), rt_ms: Math.round(performance.now() - t0), presented_start: start, presented_end: end });
    });
  }

  async function runAdaptiveBlock(root, pack) {
    var results = [];
    var interrupted = false;
    for (var p = 0; p < ["B", "D", "E"].length; p++) {
      var phase = ["B", "D", "E"][p];
      var rows = (pack.trials || []).filter(function (t) {
        return t.phase === phase;
      });
      if (!rows.length) continue;
      var ok = await phaseIntro(root, phase, rows.length);
      if (!ok) {
        interrupted = true;
        break;
      }
      for (var i = 0; i < rows.length; i++) {
        var got = null;
        if (phase === "B") got = await runB(root, rows[i], i + 1, rows.length);
        else if (phase === "D") got = await runD(root, rows[i], i + 1, rows.length);
        else got = await runE(root, rows[i], i + 1, rows.length);
        if (!got) {
          interrupted = true;
          break;
        }
        results.push(got);
      }
      if (interrupted) break;
    }
    return { results: results, interrupted: interrupted };
  }

  function adaptivePackSummary(pack) {
    var nav = (pack.domains && pack.domains.navigation) || {};
    var craft = (pack.domains && pack.domains.crafting) || {};
    return {
      trial_count: (pack.trials || []).length,
      navigation_total_trials: nav.trial_count_total || 0,
      navigation_used_trials: nav.trial_count_used || 0,
      crafting_total_trials: craft.trial_count_total || 0,
      crafting_used_trials: craft.trial_count_used || 0,
    };
  }

  function runAdaptivePreparationRest(root, participantId, experimentOrder) {
    return new Promise(function (resolve) {
      var minMs = 5000;
      var t0 = performance.now();
      var done = false;
      var pack = null;
      var error = null;
      var packReady = false;

      function cleanup() {
        window.removeEventListener("keydown", onKey, true);
      }

      function finish(interrupted) {
        if (done) return;
        done = true;
        cleanup();
        resolve({
          interrupted: !!interrupted,
          min_duration_ms: minMs,
          actual_duration_ms: Math.round(performance.now() - t0),
          background_task_used: true,
          background_task_done: packReady,
          background_error: error ? String(error && (error.stack || error.message || error)) : null,
          adaptive_pack: pack,
        });
      }

      function render() {
        if (done) return;
        var elapsed = performance.now() - t0;
        var remaining = Math.max(0, minMs - elapsed);
        var minElapsed = remaining <= 0;
        var hint = "";
        if (!minElapsed) {
          hint = Math.ceil(remaining / 1000) + " 秒后可继续。";
        } else if (!packReady) {
          hint = "系统正在准备下一阶段，请稍候。";
        } else {
          hint = "准备好后，请按空格继续。";
        }
        root.innerHTML = wrap(
          "<h2 style='font-size:22px;margin:0 0 14px'>请稍作休息</h2>" +
            "<p style='line-height:1.7;font-size:15px;color:#c4ccd8'>正式任务已完成。请稍作休息，准备进入最后一段答题。</p>" +
            "<p style='margin-top:18px;color:#9eb4d8'>" +
            esc(hint) +
            "</p>"
        );
        window.requestAnimationFrame(render);
      }

      function onKey(ev) {
        var elapsed = performance.now() - t0;
        if ((ev.key === " " || ev.code === "Space" || ev.key === "Enter") && elapsed >= minMs && packReady) {
          ev.preventDefault();
          finish(false);
        }
      }

      window.addEventListener("keydown", onKey, true);
      render();
      window.setTimeout(function () {
        try {
          var jsp = global.jsPsych;
          pack = buildAdaptivePackFromRows(jsp.data.get().values(), participantId, experimentOrder, 8);
          global.__NC_ADAPTIVE_BDE_PACK = pack;
        } catch (err) {
          error = err;
        } finally {
          packReady = true;
        }
      }, 0);
    });
  }

  global.NCAdaptiveTailTaskV3 = {
    buildAdaptivePackFromRows: buildAdaptivePackFromRows,
    appendAdaptiveTailPhaseToTimeline: function (timeline, participantId, experimentOrder) {
      timeline.push({
        type: jsPsychCallFunction,
        async: true,
        data: {
          screen_id: "adaptive_bde_rest",
          phase: "transition_rest",
          rest_context: "before_adaptive",
          experiment_order: experimentOrder,
          adaptive_tail_task_version: "v3",
        },
        func: function (done) {
          var root = global.jsPsych.getDisplayElement();
          runAdaptivePreparationRest(root, participantId, experimentOrder).then(function (out) {
            var payload = {
              phase: "transition_rest",
              rest_context: "before_adaptive",
              participant_id: participantId || "anon",
              experiment_order: experimentOrder || "",
              rest_min_duration_ms: out.min_duration_ms,
              rest_actual_duration_ms: out.actual_duration_ms,
              background_task_used: out.background_task_used,
              background_task_done: out.background_task_done,
              background_error: out.background_error,
              interrupted: out.interrupted,
              adaptive_pack_summary: out.adaptive_pack ? adaptivePackSummary(out.adaptive_pack) : null,
            };
            done(payload);
          });
        },
      });
      timeline.push({
        type: jsPsychHtmlKeyboardResponse,
        stimulus:
          "<div class='nc-inst'>" +
          "<h2>Adaptive B/D/E</h2>" +
          "<div class='nc-inst-tag'>Session 3 后测</div>" +
          "<p>接下来会进入最后一组 B/D/E 题目。</p>" +
          "<p>题目会同时覆盖<strong>导航任务</strong>与<strong>合成任务</strong>中的 landmark 和 chunk 表征。</p>" +
          "<p class='nc-inst-hint'>按 <strong>空格</strong> 开始答题。</p></div>",
        choices: [" "],
        data: {
          screen_id: "adaptive_bde_preface",
          phase: "adaptive_tail_task",
          adaptive_subphase: "preface",
          experiment_order: experimentOrder,
        },
      });
      timeline.push({
        type: jsPsychCallFunction,
        async: true,
        data: {
          screen_id: "adaptive_bde_runner",
          phase: "adaptive_tail_task",
          adaptive_subphase: "runner",
          experiment_order: experimentOrder,
          adaptive_tail_task_version: "v3",
        },
        func: function (done) {
          var jsp = global.jsPsych;
          var root = jsp.getDisplayElement();
          var priorRows = jsp.data.get().values();
          var pack = global.__NC_ADAPTIVE_BDE_PACK || buildAdaptivePackFromRows(priorRows, participantId, experimentOrder, 8);
          global.__NC_ADAPTIVE_BDE_PACK = pack;
          runAdaptiveBlock(root, pack).then(function (out) {
            done({
              phase: "adaptive_tail_task",
              adaptive_subphase: "complete",
              participant_id: participantId || "anon",
              experiment_order: experimentOrder || "",
              adaptive_tail_task_version: "v3",
              adaptive_pack: pack,
              adaptive_results: out.results,
              interrupted: out.interrupted,
              trial_count: out.results.length,
            });
          });
        },
      });
    },
  };
})(window);
