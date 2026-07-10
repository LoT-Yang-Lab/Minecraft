/**
 * 导航 / 合成共用：3×3 站点编号 1–9（行主序），与桌面 crafting grid_geometry 一致。
 * Q 左 · E 右 · A 上 · D 下 · W 角点顺时针 1→3→9→7→1。
 */
(function (global) {
  var CORNER_CW = { 1: 3, 3: 9, 9: 7, 7: 1 };

  function neighbor(code, key) {
    var c = parseInt(code, 10);
    var k = String(key || "").toLowerCase();
    if (!c || c < 1 || c > 9) return null;
    if (k === "w") return CORNER_CW[c] != null ? CORNER_CW[c] : null;
    var z = c - 1;
    var r = Math.floor(z / 3);
    var col = z % 3;
    if (k === "q" && col > 0) return r * 3 + col - 1 + 1;
    if (k === "e" && col < 2) return r * 3 + col + 1 + 1;
    if (k === "a" && r > 0) return (r - 1) * 3 + col + 1;
    if (k === "d" && r < 2) return (r + 1) * 3 + col + 1;
    return null;
  }

  var KEYS = ["q", "e", "a", "d", "w"];

  var KEY_DIR = { q: "left", e: "right", a: "up", d: "down", w: "cw" };

  function keyAngleDeg(k) {
    return { q: 180, e: 0, a: -90, d: 90, w: -45 }[k] || 0;
  }

  global.NCGridGeometry = {
    neighbor: neighbor,
    KEYS: KEYS,
    KEY_DIR: KEY_DIR,
    keyAngleDeg: keyAngleDeg,
  };
})(window);
