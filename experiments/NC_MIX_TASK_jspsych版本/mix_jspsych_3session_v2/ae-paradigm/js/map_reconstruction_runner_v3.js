/**
 * 地图还原 v3：纯空间摆放版。被试只把认为接近的节点摆在一起，不还原连线。
 * 依赖：NCGridGeometry（可选，仅用于校验提示）、NCMixAssets（图片链，与 nav/craft runner 一致）。
 */
(function (global) {
  var gm = global.NCGridGeometry;
  var applyChain =
    global.NCMixAssets && global.NCMixAssets.applyImageChain
      ? global.NCMixAssets.applyImageChain.bind(global.NCMixAssets)
      : null;
  var navCandidates =
    global.NCMixAssets && global.NCMixAssets.navStationCandidates
      ? global.NCMixAssets.navStationCandidates.bind(global.NCMixAssets)
      : null;
  var stoneCands =
    global.NCMixAssets && global.NCMixAssets.stoneUrlCandidates
      ? global.NCMixAssets.stoneUrlCandidates.bind(global.NCMixAssets)
      : null;

  function esc(s) {
    return String(s || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/"/g, "&quot;");
  }

  function bindNavImg(img, code) {
    if (!img) return;
    var pref = global.NC_MIX_ASSET_PREFETCH;
    if (pref && pref.nav && pref.nav[String(code)]) {
      img.src = pref.nav[String(code)];
      return;
    }
    if (applyChain && navCandidates) applyChain(img, navCandidates(code));
    else {
      img.src =
        ((global.NC_MIX_CONFIG && global.NC_MIX_CONFIG.imgBase) || "materials/img/").replace(
          /\/?$/,
          "/"
        ) +
        "nav/station_" +
        String(code).padStart(2, "0") +
        ".svg";
    }
  }

  function bindStoneImg(img, sid) {
    if (!img) return;
    sid = String(sid || "");
    var pref = global.NC_MIX_ASSET_PREFETCH;
    if (pref && pref.stone && pref.stone[sid]) {
      img.src = pref.stone[sid];
      return;
    }
    if (applyChain && stoneCands) applyChain(img, stoneCands(sid));
    else {
      var m = /^stone_(\d+)$/i.exec(sid);
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

  function stoneIdFromCode(c) {
    return "stone_" + String(c).padStart(2, "0");
  }

  /**
   * @param {"navigation"|"crafting"} domain
   * @param {function(object)} done  payload 写入 jsPsych trial
   */
  function runMapReconstruction(displayEl, domain, opts, done) {
    opts = opts || {};
    var experimentOrder = opts.experiment_order || "";
    var t0 = performance.now();

    var title =
      domain === "crafting"
        ? "地图还原 · 合成任务（九石与药水）"
        : "地图还原 · 导航任务（站点与交通工具）";
    var hintBody =
      domain === "crafting"
        ? "请将左侧 <strong>9 个石块</strong>全部拖入中央空白区域，按您理解的关系摆放；您认为越接近、越容易联想到一起的石块，请摆得越近。完成后点击「提交本节」。"
        : "请将左侧 <strong>9 个站点</strong>全部拖入中央空白区域，按您理解的关系摆放；您认为越接近、越容易联想到一起的站点，请摆得越近。完成后点击「提交本节」。";

    /** @type {Record<number, {xp:number,yp:number}>} */
    var placed = {};

    var wrap = document.createElement("div");
    wrap.className = "nc-map-rec nc-map-rec--v2 nc-map-rec--v3";
    wrap.innerHTML =
      "<div class='nc-map-rec-inner'>" +
      "<h2 class='nc-map-rec-title'>" +
      esc(title) +
      "</h2>" +
      "<p class='nc-map-rec-hint'>" +
      hintBody +
      "</p>" +
      "<div class='nc-map-rec-main'>" +
      "<div class='nc-map-rec-side nc-map-rec-side--v2'>" +
      "<div class='nc-map-rec-cap'>\u7d20\u6750\uff081\u20139\uff0c\u4e24\u5217\uff1b\u62d6\u5230\u753b\u677f\uff09</div>" +
      "<div id='nc-map-rec-palette' class='nc-map-rec-palette'></div>" +
      "</div>" +
      "<div id='nc-map-rec-board-wrap' class='nc-map-rec-board-wrap nc-map-rec-board-wrap--v2'>" +
      "<div class='nc-map-rec-board-cap'>\u753b\u677f\uff08\u62d6\u52a8\u6446\u653e\u8282\u70b9\uff1b\u8ddd\u79bb\u8868\u793a\u4f60\u7406\u89e3\u4e2d\u7684\u63a5\u8fd1\u7a0b\u5ea6\uff09</div>" +
      "<div id='nc-map-rec-board' class='nc-map-rec-board' tabindex='0'>" +
      '<svg id="nc-map-rec-svg" class="nc-map-rec-svg" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none"></svg>' +
      "<div id='nc-map-rec-nodes' class='nc-map-rec-nodes'></div>" +
      "</div>" +
      "</div>" +
      "</div>" +
      "<div class='nc-map-rec-tools'>" +
      "<div class='nc-map-rec-actions'>" +
      "<button type='button' id='nc-map-rec-submit' class='nc-map-rec-btn'>提交本节</button>" +
      "</div></div></div>";

    displayEl.innerHTML = "";
    displayEl.appendChild(wrap);

    var board = document.getElementById("nc-map-rec-board");
    var svg = document.getElementById("nc-map-rec-svg");
    var nodesLayer = document.getElementById("nc-map-rec-nodes");
    var palette = document.getElementById("nc-map-rec-palette");

    for (var c = 1; c <= 9; c++) {
      var tile = document.createElement("div");
      tile.className = "nc-map-rec-palette-item";
      tile.dataset.code = String(c);
      var img = document.createElement("img");
      img.alt = domain === "crafting" ? stoneIdFromCode(c) : "站点 " + c;
      img.draggable = false;
      if (domain === "crafting") bindStoneImg(img, stoneIdFromCode(c));
      else bindNavImg(img, c);
      tile.appendChild(img);
      palette.appendChild(tile);
    }

    function boardRect() {
      return board.getBoundingClientRect();
    }

    function pctFromEvent(clientX, clientY) {
      var r = boardRect();
      var xp = (clientX - r.left) / r.width;
      var yp = (clientY - r.top) / r.height;
      xp = Math.max(0.04, Math.min(0.96, xp));
      yp = Math.max(0.04, Math.min(0.96, yp));
      return { xp: xp, yp: yp };
    }

    /** v3 不画连线；保留空 svg 层只为兼容旧布局。 */
    function redrawEdges() {
      svg.setAttribute("viewBox", "0 0 100 100");
      svg.setAttribute("width", "100%");
      svg.setAttribute("height", "100%");
      svg.style.position = "absolute";
      svg.style.left = "0";
      svg.style.top = "0";
      svg.innerHTML = "";
    }

    function renderNodes() {
      nodesLayer.innerHTML = "";
      for (var code = 1; code <= 9; code++) {
        var pos = placed[code];
        if (!pos) continue;
        var node = document.createElement("div");
        node.className = "nc-map-rec-node";
        node.dataset.code = String(code);
        node.style.left = pos.xp * 100 + "%";
        node.style.top = pos.yp * 100 + "%";
        var im = document.createElement("img");
        im.draggable = false;
        if (domain === "crafting") bindStoneImg(im, stoneIdFromCode(code));
        else bindNavImg(im, code);
        node.appendChild(im);

        (function (codeRef, nodeEl) {
          var track = false;
          var sx = 0;
          var sy = 0;
          var wasDrag = false;
          nodeEl.addEventListener("pointerdown", function (e) {
            e.stopPropagation();
            if (e.button !== 0) return;
            track = true;
            wasDrag = false;
            sx = e.clientX;
            sy = e.clientY;
            nodeEl.setPointerCapture(e.pointerId);
          });
          nodeEl.addEventListener("pointermove", function (e) {
            if (!track) return;
            if (Math.abs(e.clientX - sx) + Math.abs(e.clientY - sy) > 8) wasDrag = true;
            if (wasDrag) {
              var p = pctFromEvent(e.clientX, e.clientY);
              placed[codeRef] = p;
              nodeEl.style.left = p.xp * 100 + "%";
              nodeEl.style.top = p.yp * 100 + "%";
              redrawEdges();
            }
          });
          nodeEl.addEventListener("pointerup", function (e) {
            if (!track) return;
            track = false;
            try {
              nodeEl.releasePointerCapture(e.pointerId);
            } catch (err) {}
            if (wasDrag) {
              wasDrag = false;
              return;
            }
          });
          nodeEl.addEventListener("pointercancel", function () {
            track = false;
            wasDrag = false;
          });
          /** 阻止 click 冒泡到画板。 */
          nodeEl.addEventListener("click", function (e) {
            e.stopPropagation();
          });
        })(code, node);

        nodesLayer.appendChild(node);
      }
    }

    /** palette drag */
    var dragGhost = null;
    var dragCode = null;

    palette.addEventListener("pointerdown", function (e) {
      var item = e.target.closest ? e.target.closest(".nc-map-rec-palette-item") : null;
      if (!item || e.button !== 0) return;
      dragCode = parseInt(item.dataset.code, 10);
      dragGhost = document.createElement("div");
      dragGhost.className = "nc-map-rec-ghost";
      var gi = document.createElement("img");
      if (domain === "crafting") bindStoneImg(gi, stoneIdFromCode(dragCode));
      else bindNavImg(gi, dragCode);
      dragGhost.appendChild(gi);
      document.body.appendChild(dragGhost);
      dragGhost.style.left = e.clientX - 24 + "px";
      dragGhost.style.top = e.clientY - 24 + "px";

      function move(ev) {
        if (!dragGhost) return;
        dragGhost.style.left = ev.clientX - 24 + "px";
        dragGhost.style.top = ev.clientY - 24 + "px";
      }
      function up(ev) {
        document.removeEventListener("pointermove", move);
        document.removeEventListener("pointerup", up);
        var br = boardRect();
        var x = ev.clientX;
        var y = ev.clientY;
        if (
          dragCode &&
          x >= br.left &&
          x <= br.right &&
          y >= br.top &&
          y <= br.bottom
        ) {
          placed[dragCode] = pctFromEvent(x, y);
          renderNodes();
          redrawEdges();
        }
        if (dragGhost && dragGhost.parentNode) dragGhost.parentNode.removeChild(dragGhost);
        dragGhost = null;
        dragCode = null;
      }
      document.addEventListener("pointermove", move);
      document.addEventListener("pointerup", up);
      e.preventDefault();
    });

    board.addEventListener("keydown", function (e) {
      if (e.key === "Escape") {
        e.preventDefault();
      }
    });

    board.addEventListener("click", function (ev) {
      if (ev.target && ev.target.closest && ev.target.closest(".nc-map-rec-node")) return;
    });

    window.addEventListener(
      "resize",
      function () {
        redrawEdges();
      },
      { passive: true }
    );

    document.getElementById("nc-map-rec-submit").addEventListener("click", function () {
      var nodesOut = [];
      for (var c = 1; c <= 9; c++) {
        if (placed[c]) nodesOut.push({ code: c, xp: placed[c].xp, yp: placed[c].yp });
      }
      var payload = {
        screen_id:
          domain === "crafting"
            ? "map_reconstruction_crafting"
            : "map_reconstruction_navigation",
        phase: "map_reconstruction",
        domain: domain,
        experiment_order: experimentOrder,
        map_reconstruction_ui_version: "v3",
        map_reconstruction: {
          nodes: nodesOut,
        },
        duration_ms: Math.round(performance.now() - t0),
        total_time_ms: Math.round(performance.now() - t0),
      };
      done(payload);
    });

    board.focus();
    renderNodes();
    redrawEdges();
  }

  global.NCMapReconstructionV3 = {
    runNavigation: function (displayEl, opts, done) {
      runMapReconstruction(displayEl, "navigation", opts, done);
    },
    runCrafting: function (displayEl, opts, done) {
      runMapReconstruction(displayEl, "crafting", opts, done);
    },
  };
})(window);
