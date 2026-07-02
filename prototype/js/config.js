/* ============================================================================
   config.js — the ONE global object PA and the persisted config.
   Loaded FIRST. Everything else hangs off window.PA.
   ========================================================================== */

// Establish (or reuse) the single global. No ES modules — classic scripts.
var PA = window.PA || {};
window.PA = PA;

// --- View registry (defined EARLY, in the first-loaded file) ----------------
// Every js/views/*.js self-registers at load time via PA.registerView(...), and
// those scripts load BEFORE app.js. If registerView were defined only in app.js
// (loaded last) it wouldn't exist yet and every view would throw, leaving the
// registry empty. So it lives here, in config.js, which loads first.
PA.views = PA.views || {};
PA.registerView = PA.registerView || function (id, def) {
  if (!id || !def || typeof def.render !== 'function') {
    console.error('registerView: invalid view "' + id + '"');
    return;
  }
  PA.views[id] = {
    id: id,
    title: def.title || id,
    icon: def.icon || 'circle',
    order: def.order != null ? def.order : 999,
    render: def.render,
  };
};

// --- Persisted config ------------------------------------------------------
// baseUrl: the ProjectAI server; useMock: use canned PA.mock instead of PA.api.
(function () {
  var LS_KEY = 'pa.config';
  var defaults = { baseUrl: 'http://localhost:8080', useMock: true };

  function load() {
    try {
      var raw = localStorage.getItem(LS_KEY);
      if (!raw) return Object.assign({}, defaults);
      var parsed = JSON.parse(raw);
      return {
        baseUrl: typeof parsed.baseUrl === 'string' ? parsed.baseUrl : defaults.baseUrl,
        useMock: typeof parsed.useMock === 'boolean' ? parsed.useMock : defaults.useMock,
      };
    } catch (e) {
      return Object.assign({}, defaults);
    }
  }

  PA.config = load();

  // Persist the current config back to localStorage.
  PA.saveConfig = function () {
    try {
      localStorage.setItem(LS_KEY, JSON.stringify({
        baseUrl: PA.config.baseUrl,
        useMock: PA.config.useMock,
      }));
    } catch (e) { /* ignore quota / private-mode failures */ }
  };
})();
