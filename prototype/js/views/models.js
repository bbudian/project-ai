/* ============================================================================
   views/models.js — the Models screen, 1:1 with the Godot client
   (Ui/Views/ModelsView.cs + Ui/TrainPanel.cs).

   Topbar: "＋ Train new model" toggles an inline train form (text file or
   pasted text, name, size from /health sizes, steps, backend) → POST /train,
   then poll GET /train/status (nested under 'training' — the seam unwraps it)
   with a progress bar; the form auto-reveals while a job runs and health is
   re-fetched on a terminal state so the finished model appears everywhere.

   The card grid is driven by /health's modelInfos array — real metadata from
   the server, no name-regex derivation. Per-card actions: "💬 Chat with"
   (select + navigate) and "⊟ Tokenize…" (a probe modal over POST /tokenize).
   ========================================================================== */
(function (PA) {
  var el = PA.ui.el;
  var ui = PA.ui;
  var C = PA.components;

  // ---- formatters (client's FormatParams / FormatBytes) --------------------
  function fmtParams(p) {
    if (p >= 1e9) return (p / 1e9).toFixed(1) + 'B';
    if (p >= 1e6) return Math.round(p / 1e6) + 'M';
    return ui.fmtNum(p);
  }
  function fmtBytes(b) {
    if (b >= 1 << 30 || b >= 1073741824) return (b / 1073741824).toFixed(1) + ' GB';
    if (b >= 1048576) return Math.round(b / 1048576) + ' MB';
    return ui.fmtNum(b) + ' B';
  }
  function describeMeta(info) {
    if (!info.params || info.params <= 0) {
      return info.fileBytes > 0 ? fmtBytes(info.fileBytes) + ' on disk' : '';
    }
    return fmtParams(info.params) + ' params  ·  ' + info.layers + ' layers  ·  ctx ' + ui.fmtNum(info.ctx) +
      '  ·  ' + info.tokenizer + ' tokenizer  ·  ' + info.dtype + '  ·  step ' + ui.fmtNum(info.step) +
      '  ·  ' + fmtBytes(info.fileBytes);
  }

  PA.registerView('models', {
    title: 'Models',
    icon: 'box',
    order: 2,

    render: function (root, ctx) {
      // ---- view-local state -------------------------------------------------
      var showTrain = false;
      var polling = null;      // setInterval handle while a job runs
      var trainBusy = false;   // POST /train sent; waiting for terminal state
      var draft = { text: '', fileName: '', name: 'mymodel', size: 'small', steps: 300, backend: null };

      var st0 = ctx.store.get();
      draft.backend = st0.selectedBackend || null;

      // ---- topbar action ------------------------------------------------------
      var trainToggle = C.button('＋  Train new model', {
        ghost: true, small: true,
        onClick: function () { showTrain = !showTrain; paint(); },
      });
      ctx.setTopBar('Models', trainToggle);

      // Progress widget survives repaints so the bar doesn't flicker.
      var progress = C.progress();

      // ---- training orchestration --------------------------------------------
      function startPolling() {
        if (polling) return;
        polling = setInterval(pollStatus, 1500);
      }
      function stopPolling() {
        if (polling) { clearInterval(polling); polling = null; }
      }

      function pollStatus() {
        ctx.data().trainStatus().then(function (s) {
          if (!root.isConnected) { stopPolling(); return; }
          applyStatus(s);
        }).catch(function (e) {
          progress.setStatus('Error: ' + (e.message || e), 'bad');
        });
      }

      function applyStatus(s) {
        switch (s.state) {
          case 'running':
            if (!showTrain) { showTrain = true; paint(); } // surface live progress
            progress.set(s.totalSteps > 0 ? 100 * s.step / s.totalSteps : 0);
            progress.setStatus(
              "Training '" + (s.name || '') + "'…  step " + s.step + '/' + s.totalSteps +
              (s.loss != null ? ',  loss ' + Number(s.loss).toFixed(3) : ''), 'muted');
            startPolling();
            break;
          case 'done':
            progress.set(100);
            progress.setStatus("Done — '" + (s.name || '') + "' trained" +
              (s.loss != null ? ' (loss ' + Number(s.loss).toFixed(3) + ')' : '') +
              '. Switch to Chat and pick it.', 'good');
            onTerminal();
            break;
          case 'error':
            progress.setStatus('Error: ' + (s.error || 'training failed'), 'bad');
            onTerminal();
            break;
          default: // idle — e.g. the server restarted and lost the job
            if (trainBusy || polling) {
              progress.setStatus('Training is no longer running on the server.', 'muted');
              onTerminal();
            }
            break;
        }
      }

      // Terminal state: stop polling, re-enable the form, refresh health so the
      // finished model appears in every picker at once.
      function onTerminal() {
        stopPolling();
        trainBusy = false;
        ctx.data().health().then(function (h) {
          if (h && h.ok) {
            ctx.store.set({
              health: h,
              models: h.models || [],
              modelInfos: h.modelInfos || [],
              backends: h.backends || [],
              sizes: h.sizes || [],
            });
          }
        }).catch(function () { /* leave the current catalog */ });
      }

      function onTrain() {
        if (trainBusy) return;
        var text = (draft.text || '').trim();
        if (!text) { progress.setStatus('Pick a text file (or paste text) first.', 'bad'); return; }
        var name = (draft.name || '').trim();
        if (!name) { progress.setStatus('Enter a model name.', 'bad'); return; }

        trainBusy = true;
        progress.set(0);
        progress.setStatus('Starting…', 'muted');
        ctx.data().train({
          name: name,
          text: text,
          size: draft.size,
          steps: draft.steps,
          backend: draft.backend || undefined,
        }).then(function () {
          startPolling();
        }).catch(function (e) {
          trainBusy = false;
          progress.setStatus('Error: ' + (e.message || e), 'bad');
        });
      }

      // ---- the inline train form ----------------------------------------------
      function buildTrainForm() {
        var s = ctx.store.get();
        var sizes = s.sizes || [];
        var backends = s.backends || [];

        var fileNote = el('span', { class: ['pa-xs', 'pa-muted'], text: draft.fileName ? draft.fileName + '  (' + ui.fmtNum(draft.text.length) + ' chars)' : 'no file selected' });
        var fileInput = el('input', {
          class: 'pa-input', type: 'file', accept: '.txt,.md,.json,.csv,text/plain',
          onChange: function (e) {
            var f = e.target.files && e.target.files[0];
            if (!f) return;
            var reader = new FileReader();
            reader.onload = function () {
              draft.text = String(reader.result || '');
              draft.fileName = f.name;
              textArea.value = draft.text;
              fileNote.textContent = f.name + '  (' + ui.fmtNum(draft.text.length) + ' chars)';
              if (!draft.name || draft.name === 'mymodel') {
                draft.name = f.name.replace(/\.[^.]*$/, '').replace(/[^\w-]+/g, '') || 'mymodel';
                nameInput.value = draft.name;
              }
            };
            reader.readAsText(f);
          },
        });

        var textArea = el('textarea', {
          class: 'pa-input', rows: 4,
          placeholder: 'Or paste the training text here…',
          value: draft.text,
          onInput: function (e) { draft.text = e.target.value; draft.fileName = ''; fileNote.textContent = draft.text ? ui.fmtNum(draft.text.length) + ' chars' : 'no file selected'; },
        });

        var nameInput = el('input', {
          class: 'pa-input', value: draft.name, placeholder: 'mymodel',
          onInput: function (e) { draft.name = e.target.value; },
        });

        var sizeSel = el('select', { class: 'pa-select', onChange: function (e) { draft.size = e.target.value; } },
          (sizes.length ? sizes : [{ id: 'small', label: '' }]).map(function (z) {
            return el('option', {
              value: z.id,
              text: z.label ? z.id + ' — ' + z.label : z.id,
              selected: z.id === draft.size,
            });
          })
        );

        var stepsInput = el('input', {
          class: 'pa-input', type: 'number', min: '1', max: '100000', step: '50', value: String(draft.steps),
          onChange: function (e) {
            var v = Number(e.target.value);
            draft.steps = (!isNaN(v) && v >= 1) ? Math.min(100000, Math.round(v)) : 300;
            e.target.value = String(draft.steps);
          },
        });

        var backendSel = el('select', { class: 'pa-select', onChange: function (e) { draft.backend = e.target.value; } },
          (backends.length ? backends : [{ id: '', label: '(server default)', available: true }]).map(function (b) {
            return el('option', {
              value: b.id,
              text: b.label + (b.available === false ? ' — unavailable' : ''),
              disabled: b.available === false,
              selected: b.id === draft.backend,
            });
          })
        );

        var trainBtn = C.button(trainBusy ? 'Training…' : 'Train', { primary: true, disabled: trainBusy, onClick: onTrain });

        return el('div', { class: 'pa-card pa-col pa-gap-4' },
          el('div', { class: 'pa-col', style: { gap: '2px' } },
            el('div', { class: 'pa-h3', text: 'Train a new model' }),
            el('div', { class: ['pa-sm', 'pa-muted'], text: 'Pick a text file, choose a size, and train on your GPU. It appears in the chat model picker when done.' })
          ),
          el('div', { class: 'pa-row pa-wrap pa-gap-3' }, fileInput, fileNote),
          textArea,
          el('div', { class: 'pa-grid pa-grid-2' },
            C.field('Name', nameInput),
            C.field('Size', sizeSel),
            C.field('Steps', stepsInput),
            C.field('Backend', backendSel)
          ),
          el('div', { class: 'pa-row' }, trainBtn),
          progress.root
        );
      }

      // ---- the tokenize probe modal ---------------------------------------------
      function openTokenize(modelName) {
        var result = el('div', { class: ['pa-sm', 'pa-muted'], style: { whiteSpace: 'pre-wrap', wordBreak: 'break-word' }, text: 'model: ' + modelName });
        var input = el('input', {
          class: 'pa-input', placeholder: 'Type text, press Enter…',
          onKeydown: function (e) {
            if (e.key !== 'Enter') return;
            var text = (input.value || '').trim();
            if (!text) return;
            result.textContent = 'tokenizing…';
            ctx.data().tokenize({ text: text, model: modelName }).then(function (r) {
              var pieces = (r.tokens || []).slice(0, 64).map(function (t) { return t.text; });
              var joined = pieces.join(' | ');
              if (joined.length > 700) joined = joined.slice(0, 700) + '…';
              result.textContent = r.count + ' tokens\n' + joined;
            }).catch(function (err) {
              result.textContent = 'Error: ' + (err.message || err);
            });
          },
        });
        C.openModal('Tokenize probe', el('div', { class: 'pa-col pa-gap-3' }, input, result));
        input.focus();
      }

      // ---- one model card ---------------------------------------------------------
      function buildCard(info, isDefault) {
        var titleRow = el('div', { class: 'pa-row pa-gap-3' },
          el('span', { class: 'pa-mono', style: { fontWeight: '600', wordBreak: 'break-all' }, text: info.name }),
          el('span', { class: 'pa-spacer' }),
          isDefault ? C.badge('default', 'accent') : null,
          info.instruct ? C.badge('instruct', 'good') : null
        );

        var meta = el('div', { class: ['pa-sm', 'pa-muted'], style: { wordBreak: 'break-word' }, text: describeMeta(info) });

        var error = info.error
          ? el('div', { class: 'pa-xs', style: { color: 'var(--pa-bad)' }, text: 'metadata unreadable: ' + info.error })
          : null;

        var actions = el('div', { class: 'pa-row pa-wrap pa-gap-3' },
          C.button('💬  Chat with', {
            ghost: true, small: true,
            onClick: function () {
              ctx.store.set({ selectedModel: info.name });
              ctx.go('chat');
            },
          }),
          C.button('⊟  Tokenize…', {
            ghost: true, small: true,
            onClick: function () { openTokenize(info.name); },
          })
        );

        return el('div', { class: 'pa-card pa-col pa-gap-3' }, titleRow, meta, error, actions);
      }

      // ---- paint --------------------------------------------------------------------
      function paint() {
        var s = ctx.store.get();
        var infos = s.modelInfos && s.modelInfos.length
          ? s.modelInfos
          : (s.models || []).map(function (n) { // older server: names only
              return { name: n, params: 0, layers: 0, ctx: 0, vocab: 0, tokenizer: '', dtype: '', step: 0, instruct: false, fileBytes: 0, error: null };
            });
        var defaultName = (s.health && s.health.default) || null;

        var frag = el('div', { class: 'pa-col pa-gap-4' });

        if (!infos.length) {
          ui.append(frag, C.emptyState('box-off', 'No models yet',
            'Train one below, or convert a HuggingFace model with `projectai convert` and point --models at it.'));
        } else {
          var grid = el('div', { class: 'pa-grid pa-grid-2' });
          infos.forEach(function (info) { ui.append(grid, buildCard(info, info.name === defaultName)); });
          ui.append(frag, grid);
        }

        if (showTrain) ui.append(frag, buildTrainForm());
        trainToggle.querySelector('span').textContent = showTrain ? '－  Hide training' : '＋  Train new model';

        ui.mount(root, frag);
      }

      // ---- first paint + a one-shot status probe (auto-reveal a running job) ------
      if (!(st0.models && st0.models.length)) {
        ctx.data().health().then(function (h) {
          if (h && h.ok) {
            ctx.store.set({
              health: h,
              models: h.models || [],
              modelInfos: h.modelInfos || [],
              backends: h.backends || st0.backends || [],
              sizes: h.sizes || st0.sizes || [],
            });
            paint();
          }
        }).catch(function () { /* painted below */ });
      }

      ctx.data().trainStatus().then(function (s) {
        if (root.isConnected && s.state === 'running') applyStatus(s);
      }).catch(function () { /* no live job */ });

      paint();

      // Catalog changes (health refresh) → repaint. Self-cleans when detached.
      var unsub = ctx.store.subscribe(function () {
        if (!root.isConnected) { unsub(); stopPolling(); return; }
        paint();
      });
    },
  });
})(window.PA);
