/* ============================================================================
   views/benchmark.js — the Benchmark screen, 1:1 with the Godot client
   (Ui/Views/BenchmarkView.cs): three topbar tabs.

   Define  — suite (GET /benchmark/suites → "id — N cases + bpb"), model
             MULTI-select from health, backend, repeats (1-20, default 3), the
             rigor caption, Run (POST /benchmark → 202 {runId,total}) + Cancel
             (POST /benchmark/cancel), and a progress bar fed by polling
             GET /benchmark/status (nested under 'bench' — the seam unwraps).
             On done: auto-load the run and switch to Compare.
   Compare — run title; aggregates table (model / bpb ↓ / median tok/s / check
             pass / n) with the best bpb + best tok/s tinted good; the case
             grid (rows = cases, one column per model); clicking a case row
             opens the side-by-side outputs modal. Cells with caseId '__bpb__'
             are bookkeeping and skipped.
   Reports — past runs from GET /benchmark/runs; click loads into Compare.

   Rigor is baked in: greedy decoding, fixed seed — NO judge, NO temperature.
   ========================================================================== */
(function (PA) {
  var el = PA.ui.el;
  var ui = PA.ui;
  var C = PA.components;

  PA.registerView('benchmark', {
    title: 'Benchmark',
    icon: 'chart-bar',
    order: 3,

    render: function (root, ctx) {
      // ---- view-local state -------------------------------------------------
      var tab = 0;               // 0 Define, 1 Compare, 2 Reports
      var suites = [];           // [{id,label,caseCount,hasCorpus}]
      var runs = [];             // past-run summaries
      var current = null;        // loaded run detail (Compare)
      var running = false;
      var polling = null;
      var cfg = { suite: null, models: [], backend: null, repeats: 3 };

      var st0 = ctx.store.get();
      cfg.backend = st0.selectedBackend || null;
      if (st0.selectedModel) cfg.models = [st0.selectedModel];

      var progress = C.progress();

      // ---- topbar tabs --------------------------------------------------------
      var tabBtns = ['Define', 'Compare', 'Reports'].map(function (label, i) {
        var b = C.button(label, { ghost: true, small: true, onClick: function () { showTab(i); } });
        return b;
      });

      function markTabs() {
        tabBtns.forEach(function (b, i) {
          b.style.color = i === tab ? 'var(--pa-accent)' : '';
        });
      }
      ctx.setTopBar('Benchmark', tabBtns);

      function showTab(i) {
        tab = i;
        markTabs();
        paint();
      }

      // ---- polling ---------------------------------------------------------------
      function startPolling() {
        if (polling) return;
        polling = setInterval(pollStatus, 1500);
      }
      function stopPolling() {
        if (polling) { clearInterval(polling); polling = null; }
      }

      function pollStatus() {
        ctx.data().benchStatus().then(function (s) {
          if (!root.isConnected) { stopPolling(); return; }
          applyStatus(s);
        }).catch(function (e) {
          progress.setStatus('Error: ' + (e.message || e), 'bad');
        });
      }

      function applyStatus(s) {
        switch (s.state) {
          case 'running':
            if (!running) { running = true; startPolling(); paint(); } // a run may already be live (started elsewhere)
            progress.set(s.total > 0 ? 100 * s.done / s.total : 0);
            progress.setStatus(s.done + '/' + s.total + '  ·  ' + s.currentModel + ' · ' + s.currentCase, 'muted');
            break;
          case 'done':
          case 'canceled':
          case 'error':
            if (!running) break;
            running = false;
            stopPolling();
            progress.set(s.state === 'done' ? 100 : 0);
            progress.setStatus(s.state === 'error' ? 'Error: ' + (s.error || 'run failed') : 'Run ' + s.state + '.',
              s.state === 'error' ? 'bad' : 'good');
            loadRuns();
            if (s.state === 'done' && s.runId) {
              loadRun(s.runId);       // auto-open the finished run …
            } else {
              paint();
            }
            break;
        }
      }

      // ---- data loads ---------------------------------------------------------------
      function loadSuites() {
        ctx.data().benchSuites().then(function (list) {
          suites = list || [];
          if (!cfg.suite && suites.length) cfg.suite = suites[0].id;
          if (root.isConnected && tab === 0) paint();
        }).catch(function () { suites = []; });
      }

      function loadRuns() {
        ctx.data().benchRuns().then(function (list) {
          runs = list || [];
          if (root.isConnected && tab === 2) paint();
        }).catch(function () { runs = []; });
      }

      function loadRun(id) {
        ctx.data().benchRun(id).then(function (run) {
          current = run;
          showTab(1); // … and switch to Compare
        }).catch(function (e) {
          current = { error: e.message || String(e) };
          showTab(1);
        });
      }

      // ---- Define tab -------------------------------------------------------------
      function buildDefine() {
        var s = ctx.store.get();
        var models = s.models || [];
        var backends = s.backends || [];

        var suiteSel = el('select', { class: 'pa-select', onChange: function (e) { cfg.suite = e.target.value; } },
          (suites.length ? suites : [{ id: 'baseline', label: '', caseCount: 0, hasCorpus: false }]).map(function (su) {
            return el('option', {
              value: su.id,
              text: su.id + ' — ' + su.caseCount + ' cases' + (su.hasCorpus ? ' + bpb' : ''),
              selected: su.id === cfg.suite,
            });
          })
        );

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
          modelSel.title = 'Ctrl/Shift-click to select several models — the whole point is comparing.';
        } else {
          modelSel = el('div', { class: ['pa-sm', 'pa-muted'], text: 'No models loaded — connect or switch to Mock.' });
        }

        var backendSel = el('select', { class: 'pa-select', onChange: function (e) { cfg.backend = e.target.value; } },
          (backends.length ? backends : [{ id: '', label: '(server default)', available: true }]).map(function (b) {
            return el('option', {
              value: b.id,
              text: b.label + (b.available === false ? ' — unavailable' : ''),
              disabled: b.available === false,
              selected: b.id === cfg.backend,
            });
          })
        );

        var repeatsInput = el('input', {
          class: 'pa-input', type: 'number', min: '1', max: '20', step: '1', value: String(cfg.repeats),
          onChange: function (e) {
            var v = Number(e.target.value);
            cfg.repeats = (!isNaN(v) && v >= 1 && v <= 20) ? Math.round(v) : 3;
            e.target.value = String(cfg.repeats);
          },
        });

        var runBtn = C.button('Run benchmark', { primary: true, disabled: running, onClick: onRun });
        var cancelBtn = C.button('Cancel', {
          ghost: true,
          onClick: function () { ctx.data().benchCancel().catch(function () {}); },
        });
        if (!running) cancelBtn.classList.add('pa-hidden');

        return el('div', { class: 'pa-card pa-col pa-gap-4' },
          el('div', { class: 'pa-h3', text: 'Run configuration' }),
          C.field('Suite', suiteSel),
          C.field('Models', modelSel),
          C.field('Backend', backendSel),
          C.field('Repeats', repeatsInput),
          el('div', { class: ['pa-xs', 'pa-muted'], text:
            'Greedy decoding, fixed seed, median over repeats with one discarded warmup — held constant across models.' }),
          el('div', { class: 'pa-row pa-gap-3' }, runBtn, cancelBtn),
          progress.root
        );
      }

      function onRun() {
        if (running) return;
        if (!cfg.models.length) { progress.setStatus('Select at least one model.', 'bad'); return; }
        // Claim the run BEFORE the round-trip (the client disables its Run button first too):
        // a double-click must not fire a second POST that 409s over the first one's progress.
        running = true;
        paint();
        progress.set(0);
        progress.setStatus('Starting…', 'muted');
        ctx.data().benchStart({
          suite: cfg.suite || 'baseline',
          models: cfg.models.slice(),
          backend: cfg.backend || undefined,
          repeats: cfg.repeats,
        }).then(function (r) {
          progress.setStatus('Running ' + r.runId + ' — 0/' + r.total, 'muted');
          startPolling();
        }).catch(function (e) {
          running = false;
          progress.setStatus('Error: ' + (e.message || e), 'bad');
          paint();
        });
      }

      // ---- Compare tab ---------------------------------------------------------------
      function pct(rate) {
        if (rate == null || isNaN(rate)) return '—';
        return Math.round(Number(rate) * 100) + '%';
      }

      function buildCompare() {
        if (!current) {
          return el('div', { class: ['pa-sm', 'pa-muted'], text: 'No run loaded — run a benchmark or open one from Reports.' });
        }
        if (current.error) {
          return el('div', { class: ['pa-sm', 'pa-muted'], text: 'Could not load run: ' + current.error });
        }

        var frag = el('div', { class: 'pa-col pa-gap-4' });
        frag.appendChild(el('div', { class: 'pa-sm', text: 'Run ' + current.id + '  ·  suite ' + current.suiteId + '  ·  ' + current.state }));

        // Aggregates: bpb ↓ is the quality headline; every rate shows its n.
        var aggregates = current.aggregates || [];
        var bpbs = aggregates.map(function (a) { return a.meanBpb; }).filter(function (v) { return v != null; });
        var bestBpb = bpbs.length ? Math.min.apply(null, bpbs) : null;
        var toks = aggregates.map(function (a) { return a.medianTokPerSec || 0; });
        var bestTok = toks.length ? Math.max.apply(null, toks) : 0;

        var aggTable = C.table({
          cols: [
            { key: 'model', label: 'model', render: function (a) { return el('span', { class: 'pa-mono', style: { fontWeight: '600' }, text: a.model }); } },
            { key: 'bpb', label: 'bpb ↓', render: function (a) {
                var best = a.meanBpb != null && a.meanBpb === bestBpb;
                return el('span', { class: best ? 'pa-tone-good' : '', text: a.meanBpb != null ? Number(a.meanBpb).toFixed(4) : '—' });
            } },
            { key: 'tok', label: 'median tok/s', render: function (a) {
                var best = bestTok > 0 && a.medianTokPerSec === bestTok;
                return el('span', { class: best ? 'pa-tone-good' : '', text: Number(a.medianTokPerSec || 0).toFixed(2) });
            } },
            { key: 'pass', label: 'check pass', render: function (a) { return pct(a.checkPassRate); } },
            { key: 'n', label: 'n', render: function (a) { return String(a.n != null ? a.n : ''); } },
          ],
          rows: aggregates,
        });
        frag.appendChild(el('div', { class: 'pa-card' },
          el('div', { class: ['pa-xs', 'pa-muted'], style: { marginBottom: 'var(--pa-sp-2)' }, text: 'Aggregates (bpb ↓ is the quality headline; every rate shows its n)' }),
          el('div', { class: 'pa-scroll' }, aggTable)
        ));

        // Cases: rows = cases, one column per model; a click opens the diff modal.
        var cells = (current.cells || []).filter(function (c) { return c.caseId !== '__bpb__'; });
        var caseIds = [];
        cells.forEach(function (c) { if (caseIds.indexOf(c.caseId) < 0) caseIds.push(c.caseId); });
        var models = aggregates.map(function (a) { return a.model; });

        var caseCols = [{
          key: 'caseId', label: 'case',
          render: function (row) { return el('span', { class: 'pa-mono', style: { fontWeight: '600' }, text: row.caseId }); },
        }];
        models.forEach(function (m) {
          caseCols.push({
            key: m, label: m,
            render: function (row) {
              var cell = cells.filter(function (c) { return c.caseId === row.caseId && c.model === m; })[0];
              if (!cell) return el('span', { class: 'pa-muted', text: '—' });
              if (cell.error) return el('span', { class: 'pa-tone-bad', text: '⚠ error' });
              var tone = cell.checkPassRate >= 1 ? 'pa-tone-good' : cell.checkPassRate <= 0 ? 'pa-tone-bad' : '';
              return el('span', { class: tone, text:
                pct(cell.checkPassRate) + ' · ' + Number(cell.medianTokPerSec || 0).toFixed(1) + ' tok/s · ' + (cell.stop || '') });
            },
          });
        });

        var caseTable = C.table({
          cols: caseCols,
          rows: caseIds.map(function (id) { return { caseId: id }; }),
          onRowClick: function (i, row) { openCaseDiff(row.caseId, cells); },
        });
        frag.appendChild(el('div', { class: 'pa-card' },
          el('div', { class: ['pa-xs', 'pa-muted'], style: { marginBottom: 'var(--pa-sp-2)' }, text: 'Cases (click a row for the side-by-side outputs)' }),
          el('div', { class: 'pa-scroll' }, caseTable)
        ));

        return frag;
      }

      // The side-by-side outputs modal for one case.
      function openCaseDiff(caseId, cells) {
        var body = el('div', { class: 'pa-col pa-gap-4' });
        body.appendChild(el('div', { class: 'pa-h3', text: 'Case: ' + caseId }));
        cells.filter(function (c) { return c.caseId === caseId; }).forEach(function (cell) {
          body.appendChild(el('div', { class: 'pa-card pa-col pa-gap-3' },
            el('div', { class: ['pa-xs', 'pa-muted'], text:
              cell.model + '  ·  ' + pct(cell.checkPassRate) + ' checks  ·  ' + (cell.generatedTokens || 0) + ' tok  ·  stop ' + (cell.stop || '') }),
            el('div', { class: 'pa-sm', style: { whiteSpace: 'pre-wrap', wordBreak: 'break-word' }, text:
              cell.error ? 'error: ' + cell.error : (cell.output && cell.output.length ? cell.output : '(no output)') })
          ));
        });
        C.openModal('Case comparison', body, { wide: true });
      }

      // ---- Reports tab -----------------------------------------------------------------
      function buildReports() {
        var body = el('div', { class: 'pa-col pa-gap-3' });
        if (!runs.length) {
          body.appendChild(el('div', { class: ['pa-sm', 'pa-muted'], text: 'No runs yet — define one and hit Run.' }));
        } else {
          runs.forEach(function (run) {
            body.appendChild(C.button(
              run.id + '   ·   ' + run.suiteId + '   ·   ' + (run.models || []).join(', ') + '   ·   ' + run.backend + '   ·   ' + run.state,
              { ghost: true, small: true, onClick: function () { loadRun(run.id); } }
            ));
          });
        }
        return el('div', { class: 'pa-card pa-col pa-gap-3' },
          el('div', { class: 'pa-h3', text: 'Past runs' }),
          body
        );
      }

      // ---- paint -------------------------------------------------------------------------
      function paint() {
        markTabs();
        var body = tab === 0 ? buildDefine() : tab === 1 ? buildCompare() : buildReports();
        ui.mount(root, el('div', { class: 'pa-col pa-gap-4' }, body));
      }

      // ---- go: fetch suites/runs/status once (client's OnShown parity) ---------------------
      paint();
      loadSuites();
      loadRuns();
      pollStatus(); // a run may already be live (started elsewhere)

      var unsub = ctx.store.subscribe(function () {
        if (!root.isConnected) { unsub(); stopPolling(); return; }
        if (tab === 0) paint(); // catalogs (models/backends) arrived
      });
    },
  });
})(window.PA);
