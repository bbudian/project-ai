/* ============================================================================
   views/models.js — the Models screen. Self-registers via PA.registerView.

   A list of the server's models, each a card with size, status (loaded / idle,
   resident MB when available) and actions: Load, Upgrade (consolidate memories
   -> adapter; stub), Set default. A header action opens a Convert / Import HF
   model panel (stub). Everything is built with PA.ui + PA.components + the token
   classes, and ALL data flows through PA.data() (health / settingsGet /
   settingsPut) — never fetch / PA.api / PA.mock directly.
   ========================================================================== */
(function (PA) {
  var el = PA.ui.el;
  var ui = PA.ui;
  var C = PA.components;

  // --- Derive a display-friendly record from a bare model-name string --------
  // The foundation's health() only supplies model names (string[]); we enrich
  // each into a card model deterministically (name -> params/family/size hint).
  function deriveModel(name, i, defaultName, loadedSet) {
    var lower = String(name).toLowerCase();
    var instruct = /instruct|chat|-it\b/.test(lower);

    // Parameter estimate parsed from the name (e.g. "360m", "1.7b").
    var params = null, sizeId = 'small', sizeLabel = 'Small';
    var pm = lower.match(/(\d+(?:\.\d+)?)\s*b\b/);
    if (pm) { params = parseFloat(pm[1]) * 1e9; }
    else { pm = lower.match(/(\d+(?:\.\d+)?)\s*m\b/); if (pm) params = parseFloat(pm[1]) * 1e6; }

    if (params != null) {
      if (params >= 1e9) { sizeId = 'large'; sizeLabel = 'Large'; }
      else if (params >= 3e8) { sizeId = 'medium'; sizeLabel = 'Medium'; }
      else if (params >= 3e7) { sizeId = 'small'; sizeLabel = 'Small'; }
      else { sizeId = 'tiny'; sizeLabel = 'Tiny'; }
    }

    // Resident footprint estimate (F32) — only meaningful when loaded.
    var residentMb = params != null ? Math.round((params * 4) / (1024 * 1024)) : null;

    var family = 'Unknown';
    if (/smollm/.test(lower)) family = 'SmolLM2';
    else if (/llama/.test(lower)) family = 'Llama';

    var loaded = loadedSet.indexOf(name) >= 0;

    return {
      name: name,
      family: family,
      kind: instruct ? 'Instruct' : 'Base',
      params: params,
      sizeId: sizeId,
      sizeLabel: sizeLabel,
      residentMb: loaded ? residentMb : null,
      loaded: loaded,
      isDefault: name === defaultName,
    };
  }

  function fmtParams(p) {
    if (p == null) return '— params';
    if (p >= 1e9) return (p / 1e9).toFixed(p % 1e9 === 0 ? 0 : 1) + 'B params';
    if (p >= 1e6) return Math.round(p / 1e6) + 'M params';
    return ui.fmtNum(p) + ' params';
  }

  // --- The view --------------------------------------------------------------
  PA.registerView('models', {
    title: 'Models',
    icon: 'box',
    order: 2,

    render: function (root, ctx) {
      // ---- Local, view-scoped UI state (not global store) ------------------
      // Which models the user has "loaded" this session (Load action is a stub
      // — no server endpoint exists yet), plus a transient status note.
      var localLoaded = (render._loaded = render._loaded || {});
      var note = null;         // { tone, text } inline status line
      var showImport = false;  // Convert/Import panel open?
      var importName = '';     // HF repo id typed into the import panel

      // ---- Header action: opens the Convert / Import panel -----------------
      var importBtn = C.button('Import HF model', {
        primary: true, icon: 'download',
        onClick: function () { showImport = !showImport; paint(); },
      });
      ctx.setTopBar('Models', importBtn);

      // ---- flash a transient note ------------------------------------------
      function flash(tone, text) { note = { tone: tone, text: text }; paint(); }

      // ---- ACTIONS (Load / Upgrade / Set default) --------------------------
      function doLoad(m) {
        if (m.loaded) {
          delete localLoaded[m.name];
          flash('neutral', 'Unloaded ' + m.name + '.');
        } else {
          localLoaded[m.name] = true;
          flash('good', 'Loaded ' + m.name + ' into memory.');
        }
      }

      function doUpgrade(m) {
        // Stub: consolidate memories -> adapter. No endpoint yet.
        flash('accent', 'Upgrade queued for ' + m.name +
          ' — consolidating memories into an adapter (stub).');
      }

      function doSetDefault(m) {
        // Persist through the seam's settingsGet/settingsPut surface.
        ctx.data().settingsPut({ app: { defaultModel: m.name } })
          .then(function () {
            var st = ctx.store.get();
            var settings = Object.assign({}, st.settings);
            settings.app = Object.assign({}, settings.app, { defaultModel: m.name });
            // store.set re-mounts the view; the flash rides the fresh render.
            note = { tone: 'good', text: m.name + ' is now the default model.' };
            ctx.store.set({ settings: settings, selectedModel: m.name });
          })
          .catch(function (e) { flash('bad', 'Could not set default: ' + (e.message || e)); });
      }

      function doImport() {
        var name = String(importName || '').trim();
        if (!name) { flash('bad', 'Enter a HuggingFace repo id (e.g. HuggingFaceTB/SmolLM2-360M).'); return; }
        // Stub: convert HF -> .ckpt. No endpoint yet — mock it optimistically.
        showImport = false;
        flash('accent', 'Import started for “' + name + '” — convert HF weights to a local checkpoint (stub).');
        importName = '';
      }

      // ---- DATA + PAINT ----------------------------------------------------
      // Pull the canonical model list from health() through the seam; fall back
      // to whatever the store already cached so the view is never blank.
      function paint() {
        var st = ctx.store.get();
        var defaultName = (st.settings && st.settings.app && st.settings.app.defaultModel)
          || (st.health && st.health.default)
          || st.selectedModel
          || null;
        var loadedNames = Object.keys(localLoaded).filter(function (k) { return localLoaded[k]; });

        var names = (st.models && st.models.length) ? st.models.slice() : [];
        var models = names.map(function (n, i) { return deriveModel(n, i, defaultName, loadedNames); });

        ui.mount(root, buildBody(models, defaultName));
      }

      function buildBody(models, defaultName) {
        var frag = el('div', { class: 'pa-col pa-gap-4' });

        // Summary metrics row.
        var loadedCount = models.filter(function (m) { return m.loaded; }).length;
        var metrics = el('div', { class: 'pa-grid pa-grid-3' },
          C.metricCard('Models', ui.fmtNum(models.length), 'available locally'),
          C.metricCard('Loaded', ui.fmtNum(loadedCount), loadedCount ? 'resident in memory' : 'none resident'),
          C.metricCard('Default', defaultName ? ui.truncate(defaultName, 22) : '—', 'used for new chats')
        );
        ui.append(frag, metrics);

        // Optional inline status note.
        if (note) {
          ui.append(frag, el('div', { class: 'pa-row pa-gap-3' },
            C.badge(note.tone === 'good' ? 'Done' : note.tone === 'bad' ? 'Error' : 'Note', note.tone),
            el('span', { class: 'pa-sm pa-muted', text: note.text })
          ));
        }

        // Convert / Import panel (toggled by the header action).
        if (showImport) ui.append(frag, buildImportPanel());

        // The model cards (grid, collapses to 1 col on mobile).
        if (!models.length) {
          ui.append(frag, C.emptyState(
            'box-off', 'No models yet',
            'Train a model or import one from HuggingFace to get started.',
            C.button('Import HF model', { primary: true, icon: 'download', onClick: function () { showImport = true; paint(); } })
          ));
        } else {
          var grid = el('div', { class: 'pa-grid pa-grid-2' });
          models.forEach(function (m) { ui.append(grid, buildCard(m)); });
          ui.append(frag, grid);
        }

        return frag;
      }

      // ---- One model card ---------------------------------------------------
      function buildCard(m) {
        // Header: name + status badge + default marker.
        var statusBadge = m.loaded
          ? C.badge('Loaded', 'good')
          : C.badge('Idle', 'neutral');

        var titleRow = el('div', { class: 'pa-row pa-gap-3' },
          el('span', { class: 'pa-mono', style: { fontWeight: '600', wordBreak: 'break-all' }, text: m.name }),
          el('span', { class: 'pa-spacer' }),
          m.isDefault ? C.badge('Default', 'accent') : null,
          statusBadge
        );

        // Meta row: family + kind + size + params (+ resident MB when loaded).
        var meta = el('div', { class: 'pa-row pa-wrap pa-gap-3 pa-sm pa-muted' },
          el('span', {}, ui.icon('cpu'), ' ' + m.family),
          el('span', {}, ui.icon('adjustments'), ' ' + m.kind),
          el('span', {}, ui.icon('ruler-2'), ' ' + m.sizeLabel),
          el('span', {}, ui.icon('binary'), ' ' + fmtParams(m.params)),
          m.residentMb != null
            ? el('span', {}, ui.icon('database'), ' ' + ui.fmtNum(m.residentMb) + ' MB resident')
            : null
        );

        // Actions row.
        var actions = el('div', { class: 'pa-row pa-wrap pa-gap-3 pa-mt-3' },
          C.button(m.loaded ? 'Unload' : 'Load', {
            primary: !m.loaded, small: true,
            icon: m.loaded ? 'player-eject' : 'player-play',
            onClick: function () { doLoad(m); },
          }),
          C.button('Upgrade', {
            ghost: true, small: true, icon: 'arrow-up-circle',
            onClick: function () { doUpgrade(m); },
          }),
          C.button('Set default', {
            ghost: true, small: true, icon: 'star',
            disabled: m.isDefault,
            onClick: function () { doSetDefault(m); },
          })
        );

        return el('div', { class: 'pa-card pa-col pa-gap-3' }, titleRow, meta, actions);
      }

      // ---- Convert / Import HF panel (stub) --------------------------------
      function buildImportPanel() {
        var input = el('input', {
          class: 'pa-input pa-mono',
          placeholder: 'HuggingFaceTB/SmolLM2-360M-Instruct',
          value: importName,
          onInput: function (e) { importName = e.target.value; },
        });

        var bf16 = { on: true };
        var bf16Toggle = C.toggle(bf16.on, function (v) { bf16.on = v; }, ['F32', 'BF16']);

        var body = el('div', { class: 'pa-col pa-gap-4' },
          el('div', { class: 'pa-grid pa-grid-2' },
            C.field('HuggingFace repo id', input, 'A Llama-style repo with config.json + .safetensors'),
            C.field('Precision', bf16Toggle, 'BF16 halves memory (GPU backends only)')
          ),
          el('div', { class: 'pa-row pa-wrap pa-gap-3' },
            C.button('Import', { primary: true, icon: 'download', onClick: doImport }),
            C.button('Cancel', { ghost: true, onClick: function () { showImport = false; paint(); } }),
            el('span', { class: 'pa-spacer' }),
            el('span', { class: 'pa-xs pa-muted', text: 'convert endpoint not on server yet — mocked' })
          )
        );

        return el('div', { class: 'pa-card pa-col pa-gap-3' },
          el('div', { class: 'pa-row pa-gap-3' },
            ui.icon('download'),
            el('span', { class: 'pa-h3', text: 'Import a HuggingFace model' })
          ),
          body
        );
      }

      // ---- First paint. If the store has no models yet, probe health() -----
      // through the seam so the view populates even on a cold mount.
      var st0 = ctx.store.get();
      if (!(st0.models && st0.models.length)) {
        ctx.data().health()
          .then(function (h) {
            if (h && h.ok) {
              // Feed the store so nav/other views share it; store.set re-mounts.
              ctx.store.set({
                health: h,
                models: h.models || [],
                backends: h.backends || st0.backends || [],
                sizes: h.sizes || st0.sizes || [],
              });
            } else {
              paint();
            }
          })
          .catch(function () { paint(); });
      }

      // Always paint immediately with whatever we have (never a blank frame).
      paint();
    },
  });
})(window.PA);
