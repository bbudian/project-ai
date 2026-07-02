/* ============================================================================
   views/benchmark.js — the Benchmark screen.

   Self-registers via PA.registerView('benchmark', {...}). Builds DOM ONLY with
   PA.ui + PA.components + the token-driven CSS classes, and fetches ONLY through
   PA.data() (never fetch/PA.api/PA.mock). Grids use .pa-grid* so they collapse
   to one column on mobile.

   IA: a run-config row (suite, multi-model, backend, decoding+seed, memory) +
   a Run button → per-model summary metric cards (tok/s, p50 latency, pass rate)
   + a case-by-case comparison table (case | per-model output+metrics, with
   pass/fail + judge badges) + Generate report / Export + a report-path hint.

   Relies on the seam method PA.data().bench(req) — Mock returns a canned
   comparison; the Live endpoint rejects until the server ships it, and we
   surface that rejection as an error empty-state.
   ========================================================================== */
(function (PA) {
  var el = PA.ui.el;
  var ui = PA.ui;
  var C = PA.components;

  // Suites the harness can request (label-only; the seam takes the id).
  var SUITES = [
    { id: 'smoke', label: 'Smoke (4 cases)' },
    { id: 'reasoning', label: 'Reasoning' },
    { id: 'coding', label: 'Coding' },
    { id: 'full', label: 'Full suite' },
  ];

  // A single view-scoped config object; the render closure reads/writes it and
  // re-renders in place so nothing leaks into the global store.
  function initialConfig(store) {
    var s = store.get();
    var models = (s.models || []).slice();
    return {
      suite: 'smoke',
      // Preselect up to two models for an A/B comparison.
      models: models.slice(0, Math.min(2, models.length)),
      backend: s.selectedBackend || (s.backends && s.backends[0] && s.backends[0].id) || 'cpu',
      temperature: 0.7,
      seed: 0,
      memory: false,
    };
  }

  PA.registerView('benchmark', {
    title: 'Benchmark',
    icon: 'chart-bar',
    order: 4,
    render: function (root, ctx) {
      var store = ctx.store;
      var cfg = initialConfig(store);

      // --- view state (results / status) ------------------------------------
      var state = {
        running: false,
        error: null,
        result: null,       // last bench() payload
        reported: false,    // whether "Generate report" has been pressed
      };

      // Right-side topbar action: quick Run mirror.
      var runBtnTop = C.button('Run benchmark', {
        primary: true, icon: 'player-play', small: true, onClick: run,
      });
      ctx.setTopBar('Benchmark', runBtnTop);

      // Root layout: config card, then a results host we re-render into.
      var resultsHost = el('div', { dataset: { role: 'bench-results' } });
      ui.append(root, [buildConfigCard(), el('div', { class: 'pa-mt-4' }, resultsHost)]);
      renderResults();

      // ====================================================================
      // Run-config card
      // ====================================================================
      function buildConfigCard() {
        var s = store.get();
        var backends = s.backends || [];
        var models = s.models || [];

        // Suite picker.
        var suiteSel = el('select', { class: 'pa-select', onChange: function (e) { cfg.suite = e.target.value; } },
          SUITES.map(function (su) {
            return el('option', { value: su.id, text: su.label, selected: su.id === cfg.suite });
          })
        );

        // Multi-model select (native multiple). Empty catalog → a hint.
        var modelSel;
        if (models.length) {
          modelSel = el('select', {
            class: 'pa-select', multiple: true, size: Math.min(5, Math.max(2, models.length)),
            onChange: function (e) {
              cfg.models = Array.prototype.filter.call(e.target.options, function (o) { return o.selected; })
                .map(function (o) { return o.value; });
            },
          }, models.map(function (m) {
            return el('option', { value: m, text: m, selected: cfg.models.indexOf(m) >= 0 });
          }));
        } else {
          modelSel = el('div', { class: 'pa-muted pa-sm', text: 'No models loaded — connect or switch to Mock.' });
        }

        // Backend picker.
        var backendSel = el('select', { class: 'pa-select', onChange: function (e) { cfg.backend = e.target.value; } },
          (backends.length ? backends : [{ id: cfg.backend, label: cfg.backend, available: true }]).map(function (b) {
            return el('option', {
              value: b.id,
              text: b.label + (b.available === false ? ' — unavailable' : ''),
              disabled: b.available === false,
              selected: b.id === cfg.backend,
            });
          })
        );

        // Decoding: temperature + seed.
        var tempInput = el('input', {
          class: 'pa-input', type: 'number', min: '0', max: '2', step: '0.1', value: String(cfg.temperature),
          onChange: function (e) { cfg.temperature = clampNum(e.target.value, 0, 2, cfg.temperature); e.target.value = String(cfg.temperature); },
        });
        var seedInput = el('input', {
          class: 'pa-input', type: 'number', min: '0', step: '1', value: String(cfg.seed),
          onChange: function (e) { cfg.seed = clampNum(e.target.value, 0, 2147483647, cfg.seed); e.target.value = String(cfg.seed); },
        });

        // Memory on/off toggle.
        var memToggle = C.toggle(cfg.memory, function (on) { cfg.memory = on; }, ['Off', 'On']);

        var grid = el('div', { class: 'pa-grid pa-grid-3' },
          C.field('Suite', suiteSel, 'Which comparison set to run'),
          C.field('Backend', backendSel, 'Compute backend for every model'),
          C.field('Models', modelSel, 'Select two or more to compare'),
          C.field('Temperature', tempInput, 'Decoding temperature (0–2)'),
          C.field('Seed', seedInput, 'Reproducible sampling seed'),
          C.field('Memory', memToggle, 'Inject recalled memories into prompts')
        );

        var runBtn = C.button('Run benchmark', { primary: true, icon: 'player-play', onClick: run });

        return el('div', { class: 'pa-card' },
          el('div', { class: 'pa-row pa-wrap' },
            el('div', { class: 'pa-grow' },
              el('div', { class: 'pa-h3', text: 'Run configuration' }),
              el('div', { class: 'pa-muted pa-sm', text: 'Compare models across a suite, then export a report.' })
            ),
            runBtn
          ),
          el('div', { class: 'pa-mt-4' }, grid)
        );
      }

      // ====================================================================
      // Run — through the seam only.
      // ====================================================================
      function run() {
        if (state.running) return;
        if (!cfg.models || cfg.models.length < 1) {
          state.error = 'Select at least one model to benchmark.';
          state.result = null;
          renderResults();
          return;
        }
        state.running = true;
        state.error = null;
        state.reported = false;
        setRunBusy(true);
        renderResults();

        ctx.data().bench({
          suite: cfg.suite,
          models: cfg.models.slice(),
          backend: cfg.backend,
          memory: cfg.memory,
          decoding: { temperature: cfg.temperature, seed: cfg.seed },
        }).then(function (res) {
          state.running = false;
          state.result = res;
          state.error = null;
          setRunBusy(false);
          renderResults();
        }).catch(function (err) {
          state.running = false;
          state.result = null;
          state.error = (err && err.message) || String(err);
          setRunBusy(false);
          renderResults();
        });
      }

      function setRunBusy(busy) {
        if (!runBtnTop) return;
        runBtnTop.disabled = !!busy;
      }

      // ====================================================================
      // Results region
      // ====================================================================
      function renderResults() {
        var nodes;

        if (state.running) {
          nodes = el('div', { class: 'pa-card' },
            el('div', { class: 'pa-row' }, ui.icon('loader-2'), el('span', { text: 'Running ' + cfg.suite + ' on ' + cfg.models.length + ' model(s)…' }))
          );
        } else if (state.error) {
          nodes = C.emptyState('alert-triangle', 'Benchmark unavailable', state.error,
            C.button('Retry', { onClick: run, icon: 'refresh' }));
        } else if (!state.result) {
          nodes = C.emptyState('chart-bar', 'No benchmark yet',
            'Pick a suite, choose models and a backend, then Run to compare them.',
            C.button('Run benchmark', { primary: true, icon: 'player-play', onClick: run }));
        } else {
          nodes = [summarySection(state.result), casesSection(state.result), reportSection(state.result)];
        }

        ui.mount(resultsHost, nodes);
      }

      // --- Per-model summary metric cards -----------------------------------
      function summarySection(res) {
        var cards = (res.models || []).map(function (m) {
          return el('div', { class: 'pa-card' },
            el('div', { class: 'pa-row pa-wrap' },
              el('span', { class: 'pa-mono pa-sm', text: m.model }),
              el('span', { class: 'pa-spacer' }),
              passBadge(m.passRate)
            ),
            el('div', { class: 'pa-grid pa-grid-3 pa-mt-3' },
              C.metricCard('Tokens / sec', ui.fmtNum(m.tokensPerSec, 1)),
              C.metricCard('p50 latency', ui.fmtNum(m.latencyMs, 0) + ' ms'),
              C.metricCard('Pass rate', pct(m.passRate))
            )
          );
        });

        return el('div', {},
          el('div', { class: 'pa-row pa-wrap' },
            el('div', { class: 'pa-h3', text: 'Summary' }),
            el('span', { class: 'pa-spacer' }),
            C.badge(res.suite || 'suite', 'accent'),
            C.badge(res.backend || cfg.backend, 'neutral')
          ),
          el('div', { class: 'pa-grid pa-grid-2 pa-mt-3' }, cards)
        );
      }

      // --- Case-by-case comparison table ------------------------------------
      // Columns: Case | one column per model (output + metrics + pass/judge).
      function casesSection(res) {
        var modelOrder = (res.models || []).map(function (m) { return m.model; });
        if (!modelOrder.length && res.cases && res.cases[0]) {
          modelOrder = res.cases[0].results.map(function (r) { return r.model; });
        }

        var cols = [{
          key: 'name', label: 'Case',
          render: function (row) { return el('span', { class: 'pa-mono pa-sm', text: row.name }); },
        }];

        modelOrder.forEach(function (model) {
          cols.push({
            key: model, label: model,
            render: function (row) {
              var r = (row.results || []).filter(function (x) { return x.model === model; })[0];
              if (!r) return el('span', { class: 'pa-muted', text: '—' });
              return el('div', { class: 'pa-col pa-gap-3' },
                el('div', { class: 'pa-row pa-wrap' },
                  C.badge(r.pass ? 'pass' : 'fail', r.pass ? 'good' : 'bad'),
                  r.judge ? C.badge(r.judge, r.pass ? 'neutral' : 'accent') : null
                ),
                el('div', { class: 'pa-sm', text: ui.truncate(r.output || r.text || '', 120) }),
                el('div', { class: 'pa-row pa-wrap pa-xs pa-muted' },
                  el('span', { text: ui.fmtNum(r.tokensPerSec, 1) + ' tok/s' }),
                  r.latencyMs != null ? el('span', { text: '· ' + ui.fmtNum(r.latencyMs, 0) + ' ms' }) : null
                )
              );
            },
          });
        });

        return el('div', { class: 'pa-mt-4' },
          el('div', { class: 'pa-h3', text: 'Case comparison' }),
          el('div', { class: 'pa-scroll pa-mt-3' }, C.table({ cols: cols, rows: res.cases || [] }))
        );
      }

      // --- Report / export --------------------------------------------------
      function reportSection(res) {
        var reportPath = 'reports/bench-' + (res.suite || 'suite') + '-' + (res.backend || cfg.backend).replace(/[^\w.-]+/g, '_') + '.md';

        var hint = el('div', { class: 'pa-muted pa-sm pa-mono', dataset: { role: 'report-hint' }, text: 'Report will be written to ' + reportPath });
        hint.classList.toggle('pa-hidden', !state.reported);

        var genBtn = C.button('Generate report', {
          icon: 'file-text',
          onClick: function () { state.reported = true; hint.classList.remove('pa-hidden'); },
        });

        var exportBtn = C.button('Export JSON', {
          ghost: true, icon: 'download',
          onClick: function () { exportJson(res); },
        });

        return el('div', { class: 'pa-card pa-mt-4' },
          el('div', { class: 'pa-row pa-wrap' },
            el('div', { class: 'pa-grow' },
              el('div', { class: 'pa-h3', text: 'Report' }),
              el('div', { class: 'pa-muted pa-sm', text: 'Write a Markdown summary or export the raw comparison.' })
            ),
            genBtn, exportBtn
          ),
          el('div', { class: 'pa-mt-3' }, hint)
        );
      }

      // Client-side JSON export (no network; a pure download).
      function exportJson(res) {
        try {
          var blob = new Blob([JSON.stringify(res, null, 2)], { type: 'application/json' });
          var url = URL.createObjectURL(blob);
          var a = el('a', { href: url, download: 'benchmark-' + (res.suite || 'suite') + '.json' });
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          setTimeout(function () { URL.revokeObjectURL(url); }, 0);
        } catch (e) {
          // Export is a nicety; never break the view over it.
          console.warn('benchmark export failed', e);
        }
      }

      // ---------------------------------------------------------------------
      // small local helpers
      // ---------------------------------------------------------------------
      function passBadge(rate) {
        var tone = rate == null ? 'neutral' : (rate >= 0.8 ? 'good' : (rate >= 0.5 ? 'accent' : 'bad'));
        return C.badge(pct(rate) + ' pass', tone);
      }
      function pct(rate) {
        if (rate == null || isNaN(rate)) return '—';
        return Math.round(Number(rate) * 100) + '%';
      }
      function clampNum(v, lo, hi, fallback) {
        var n = Number(v);
        if (v === '' || isNaN(n)) return fallback;
        return Math.min(hi, Math.max(lo, n));
      }
    },
  });
})(window.PA);
