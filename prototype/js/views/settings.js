/* ============================================================================
   views/settings.js — the Settings screen.

   A left section list (App, Models, Memory, Benchmark, Backends) + a right pane.
   Reads/writes ONLY through PA.data() (settingsGet / settingsPut); builds DOM via
   PA.ui + PA.components + the token-driven CSS classes. Responsive: the
   list/pane split uses .pa-grid so it collapses to one column on mobile.

   SECURITY: the Tavily API key is NEVER rendered. settingsGet() returns a masked
   string + a "configured" flag only; the Update control SETS a new key (sent under
   `integrations.tavilyKey`, which the mock/server ignores for echo) but never
   displays or reads back a raw secret.
   ========================================================================== */
(function (PA) {
  var el = PA.ui.el;
  var C = PA.components;

  // Section catalog for the left list. `id` keys into the settings payload.
  var SECTIONS = [
    { id: 'app',       label: 'App',       icon: 'settings',   hint: 'Server, defaults, appearance' },
    { id: 'model',     label: 'Models',    icon: 'cpu',        hint: 'Default decoding parameters' },
    { id: 'memory',    label: 'Memory',    icon: 'brain',      hint: 'Recall budget & auto-encode' },
    { id: 'benchmark', label: 'Benchmark', icon: 'gauge',      hint: 'Suite, judge, repeats' },
    { id: 'backends',  label: 'Backends',  icon: 'server-2',   hint: 'Compute availability' },
  ];

  PA.registerView('settings', {
    title: 'Settings',
    icon: 'settings',
    order: 6,

    render: function (root, ctx) {
      // View-local state (not global store): the loaded settings + which section
      // is open. Re-render is scoped to this view's own nodes.
      var state = {
        active: 'app',
        settings: null,
        loading: true,
        error: null,
        saving: false,
        savedTick: 0,
      };

      // --- shell: two-column grid (list | pane), collapses on mobile ---------
      var listHost = el('div', { class: 'pa-col pa-gap-3', dataset: { role: 'sections' } });
      var paneHost = el('div', { class: 'pa-col pa-gap-4', dataset: { role: 'pane' } });

      var grid = el('div', { class: 'pa-settings-grid' },
        el('div', { class: 'pa-card pa-settings-list' }, listHost),
        el('div', { class: 'pa-settings-pane' }, paneHost)
      );
      PA.ui.append(root, grid);

      // A small "saved" indicator lives in the top bar's right slot.
      var savedNote = el('span', { class: 'pa-sm pa-muted', dataset: { role: 'saved' } });
      ctx.setTopBar('Settings', savedNote);

      // ---------------------------------------------------------------------
      // Data: load through the seam. Live may reject (endpoint not on server) —
      // guard so the view degrades to a clear message instead of a blank app.
      // ---------------------------------------------------------------------
      function load() {
        state.loading = true;
        state.error = null;
        renderList();
        renderPane();
        ctx.data().settingsGet().then(function (s) {
          state.settings = s || {};
          state.loading = false;
          renderPane();
        }).catch(function (e) {
          state.loading = false;
          state.error = e && e.message || String(e);
          renderPane();
        });
      }

      // Persist one section's patch through the seam, then reflect the result.
      function save(section, patch) {
        if (!state.settings) return;
        // Optimistically merge into local state so the UI stays live.
        state.settings[section] = Object.assign({}, state.settings[section], patch);
        state.saving = true;
        flagSaved('Saving…');
        var body = {}; body[section] = patch;
        ctx.data().settingsPut(body).then(function () {
          state.saving = false;
          flagSaved('Saved');
        }).catch(function (e) {
          state.saving = false;
          flagSaved('Save failed: ' + (e && e.message || e));
        });
      }

      function flagSaved(msg) {
        savedNote.textContent = msg;
        var mine = ++state.savedTick;
        if (/^Saved$/.test(msg)) {
          setTimeout(function () { if (mine === state.savedTick) savedNote.textContent = ''; }, 1800);
        }
      }

      // ---------------------------------------------------------------------
      // Left section list
      // ---------------------------------------------------------------------
      function renderList() {
        PA.ui.clear(listHost);
        SECTIONS.forEach(function (sec) {
          var item = el('button', {
            class: ['pa-section-item', sec.id === state.active ? 'is-active' : ''],
            type: 'button',
            onClick: function () { state.active = sec.id; renderList(); renderPane(); },
          },
            PA.ui.icon(sec.icon),
            el('div', { class: 'pa-col', style: { gap: '0' } },
              el('span', { class: 'pa-section-name', text: sec.label }),
              el('span', { class: 'pa-section-hint pa-xs pa-muted', text: sec.hint })
            )
          );
          listHost.appendChild(item);
        });
      }

      // ---------------------------------------------------------------------
      // Right pane — dispatches on the active section
      // ---------------------------------------------------------------------
      function renderPane() {
        PA.ui.clear(paneHost);

        if (state.error) {
          paneHost.appendChild(C.emptyState(
            'plug-connected-x',
            'Settings unavailable',
            state.error + '  — switch the harness data source to Mock to prototype this screen.'
          ));
          return;
        }
        if (state.loading || !state.settings) {
          paneHost.appendChild(C.emptyState('loader', 'Loading settings…'));
          return;
        }

        var builder = ({
          app: paneApp,
          model: paneModel,
          memory: paneMemory,
          benchmark: paneBenchmark,
          backends: paneBackends,
        })[state.active] || paneApp;

        builder(paneHost);
      }

      // Small helper: a titled card section.
      function card(title, subtitle, bodyNodes) {
        return el('div', { class: 'pa-card pa-col pa-gap-4' },
          el('div', { class: 'pa-col', style: { gap: '2px' } },
            el('div', { class: 'pa-h3', text: title }),
            subtitle ? el('div', { class: 'pa-sm pa-muted', text: subtitle }) : null
          ),
          bodyNodes
        );
      }

      // --- App pane ---------------------------------------------------------
      function paneApp(host) {
        var app = state.settings.app || {};
        var integ = state.settings.integrations || {};
        var models = (PA.store.get().models || []);
        var backends = ((state.settings.backends && state.settings.backends.available) ||
                        PA.store.get().backends || []);

        // Server URL
        var urlInput = el('input', {
          class: 'pa-input', type: 'text', value: app.serverUrl || '',
          placeholder: 'http://localhost:8080',
          onChange: function (e) { save('app', { serverUrl: e.target.value.trim() }); },
        });

        // Default model select
        var modelSelect = el('select', { class: 'pa-select',
          onChange: function (e) { save('app', { defaultModel: e.target.value }); } },
          models.length
            ? models.map(function (m) {
                return el('option', { value: m, text: m, selected: m === app.defaultModel });
              })
            : el('option', { value: app.defaultModel || '', text: app.defaultModel || '(no models)' })
        );

        // Default backend select
        var backendSelect = el('select', { class: 'pa-select',
          onChange: function (e) { save('app', { defaultBackend: e.target.value }); } },
          (backends.length ? backends : [{ id: app.defaultBackend, label: app.defaultBackend }])
            .map(function (b) {
              return el('option', {
                value: b.id, selected: b.id === app.defaultBackend,
                disabled: b.available === false,
                text: b.label + (b.available === false ? ' — ' + (b.reason || 'unavailable') : ''),
              });
            })
        );

        // Theme toggle — mirrors <html data-theme>; the harness owns persistence,
        // so this just reflects/records the preference in settings.
        var isLight = (app.theme === 'light') ||
          document.documentElement.getAttribute('data-theme') === 'light';
        var themeToggle = C.toggle(isLight, function (on) {
          var theme = on ? 'light' : 'dark';
          if (on) document.documentElement.setAttribute('data-theme', 'light');
          else document.documentElement.removeAttribute('data-theme');
          try { localStorage.setItem('pa.theme', theme); } catch (_) {}
          save('app', { theme: theme });
        }, ['Dark', 'Light']);

        host.appendChild(card('Server', 'Where the client reaches the ProjectAI runtime.',
          el('div', { class: 'pa-grid pa-grid-2' },
            C.field('Server URL', urlInput, 'Applied on Connect in the harness toolbar.'),
            C.field('Appearance', el('div', { class: 'pa-row' }, themeToggle), 'Light theme keeps the accent.')
          )
        ));

        host.appendChild(card('Defaults', 'Pre-selected model & backend for new chats.',
          el('div', { class: 'pa-grid pa-grid-2' },
            C.field('Default model', modelSelect),
            C.field('Default backend', backendSelect)
          )
        ));

        // Tavily integration — key is MASKED and never echoed.
        host.appendChild(paneTavily(integ));
      }

      // Tavily API key card: shows a masked value + configured badge; Update
      // only SETS a new key (write-only), never reads one back.
      function paneTavily(integ) {
        var configured = !!integ.tavilyConfigured;
        var masked = integ.tavilyKeyMasked || '';

        var statusRow = el('div', { class: 'pa-row pa-gap-3 pa-wrap' },
          el('code', { class: 'pa-mono pa-key-mask', text: configured ? masked : 'Not configured' }),
          configured ? C.badge('configured', 'good') : C.badge('missing', 'neutral')
        );

        // Write-only input: starts empty, is a password field, never populated
        // from the server. On Update we send the raw key ONCE and clear the box.
        var keyInput = el('input', {
          class: 'pa-input pa-grow', type: 'password', value: '',
          placeholder: configured ? 'Enter a new key to replace…' : 'Paste your Tavily API key…',
          autocomplete: 'off', spellcheck: 'false',
        });

        var updateBtn = C.button('Update', {
          primary: true, icon: 'key',
          onClick: function () {
            var raw = (keyInput.value || '').trim();
            if (!raw) { flagSaved('Enter a key first'); return; }
            // Send the raw key under integrations.tavilyKey — the mock/server
            // ignores it for echo and only flips the configured flag. We update
            // our local view to "configured" without ever storing the raw value.
            keyInput.value = '';
            state.settings.integrations = Object.assign({}, state.settings.integrations, {
              tavilyConfigured: true,
              tavilyKeyMasked: 'tvly-••••••••••' + raw.slice(-4),
            });
            state.saving = true;
            flagSaved('Saving…');
            ctx.data().settingsPut({ integrations: { tavilyKey: raw } }).then(function () {
              state.saving = false;
              flagSaved('Saved');
              renderPane(); // reflect the new masked value + badge
            }).catch(function (e) {
              state.saving = false;
              flagSaved('Save failed: ' + (e && e.message || e));
            });
          },
        });

        return card('Web search (Tavily)', 'Used when a chat turns on web research. The key is write-only — it is never shown.',
          el('div', { class: 'pa-col pa-gap-3' },
            C.field('Current key', statusRow),
            C.field('Set / replace key',
              el('div', { class: 'pa-row pa-gap-3' }, keyInput, updateBtn),
              'Stored on the server; only the last 4 characters are ever displayed.')
          )
        );
      }

      // --- Models pane (default decoding params) ----------------------------
      function paneModel(host) {
        var m = state.settings.model || {};

        function numField(label, key, opts) {
          opts = opts || {};
          var input = el('input', {
            class: 'pa-input', type: 'number', value: m[key] != null ? m[key] : '',
            min: opts.min, max: opts.max, step: opts.step || 'any',
            onChange: function (e) {
              var v = e.target.value === '' ? null : Number(e.target.value);
              save('model', keyed(key, v));
            },
          });
          return C.field(label, input, opts.hint);
        }
        function keyed(k, v) { var o = {}; o[k] = v; return o; }

        host.appendChild(card('Default decoding', 'Applied to new generations unless a chat overrides them.',
          el('div', { class: 'pa-grid pa-grid-3' },
            numField('Max tokens', 'maxTokens', { min: 1, max: 8192, step: 1, hint: 'Upper bound per reply.' }),
            numField('Temperature', 'temperature', { min: 0, max: 2, step: 0.05, hint: '0 = greedy.' }),
            numField('Seed', 'seed', { min: 0, step: 1, hint: '0 = nondeterministic.' })
          )
        ));

        host.appendChild(card('Nucleus sampling', 'Top-k then top-p filtering before the sample.',
          el('div', { class: 'pa-grid pa-grid-2' },
            numField('Top-K', 'topK', { min: 0, step: 1, hint: '0 disables top-k.' }),
            numField('Top-P', 'topP', { min: 0, max: 1, step: 0.01, hint: 'Nucleus probability mass.' })
          )
        ));
      }

      // --- Memory pane (recall budget + auto-encode) ------------------------
      function paneMemory(host) {
        var mem = state.settings.memory || {};
        var stores = ['default', 'work', 'scratch'];
        if (mem.store && stores.indexOf(mem.store) < 0) stores.unshift(mem.store);

        var enabledToggle = C.toggle(!!mem.enabled, function (on) {
          save('memory', { enabled: on });
          renderPane();
        }, ['Off', 'On']);

        var storeSelect = el('select', { class: 'pa-select',
          disabled: !mem.enabled,
          onChange: function (e) { save('memory', { store: e.target.value }); } },
          stores.map(function (s) {
            return el('option', { value: s, text: s, selected: s === mem.store });
          })
        );

        var autoToggle = C.toggle(!!mem.autoInject, function (on) {
          save('memory', { autoInject: on });
        }, ['Manual', 'Auto']);

        var recallInput = el('input', {
          class: 'pa-input', type: 'number',
          value: mem.maxRecall != null ? mem.maxRecall : '',
          min: 0, max: 64, step: 1, disabled: !mem.enabled,
          onChange: function (e) {
            var v = e.target.value === '' ? null : Number(e.target.value);
            save('memory', { maxRecall: v });
          },
        });

        host.appendChild(card('Memory', 'Long-term recall injected into the prompt bridge.',
          el('div', { class: 'pa-grid pa-grid-2' },
            C.field('Enabled', el('div', { class: 'pa-row' }, enabledToggle),
              'Master switch for recall + encoding.'),
            C.field('Active store', storeSelect, 'Which memory namespace to read/write.')
          )
        ));

        host.appendChild(card('Recall budget', 'How aggressively past memories are pulled into context.',
          el('div', { class: 'pa-grid pa-grid-2' },
            C.field('Max recalled memories', recallInput,
              'Cap on entries injected per turn (the bridge budget).'),
            C.field('Auto-encode policy', el('div', { class: 'pa-row' }, autoToggle),
              'Auto extracts durable facts from chats; Manual saves only on request.')
          )
        ));
      }

      // --- Benchmark pane ---------------------------------------------------
      function paneBenchmark(host) {
        var b = state.settings.benchmark || {};
        var suites = ['smoke', 'reasoning', 'coding', 'full'];
        if (b.suite && suites.indexOf(b.suite) < 0) suites.unshift(b.suite);
        var judges = ['self', 'reference', 'human'];
        if (b.judge && judges.indexOf(b.judge) < 0) judges.unshift(b.judge);

        var suiteSelect = el('select', { class: 'pa-select',
          onChange: function (e) { save('benchmark', { suite: e.target.value }); } },
          suites.map(function (s) { return el('option', { value: s, text: s, selected: s === b.suite }); })
        );
        var judgeSelect = el('select', { class: 'pa-select',
          onChange: function (e) { save('benchmark', { judge: e.target.value }); } },
          judges.map(function (j) { return el('option', { value: j, text: j, selected: j === b.judge }); })
        );
        var repeatsInput = el('input', {
          class: 'pa-input', type: 'number', value: b.repeats != null ? b.repeats : '',
          min: 1, max: 20, step: 1,
          onChange: function (e) {
            var v = e.target.value === '' ? null : Number(e.target.value);
            save('benchmark', { repeats: v });
          },
        });

        host.appendChild(card('Benchmark defaults', 'Used by the Benchmark screen when you run a suite.',
          el('div', { class: 'pa-grid pa-grid-3' },
            C.field('Default suite', suiteSelect),
            C.field('Judge', judgeSelect, 'Who scores the outputs.'),
            C.field('Repeats', repeatsInput, 'Runs per case (averaged).')
          )
        ));
      }

      // --- Backends pane (read-mostly: availability + selection) ------------
      function paneBackends(host) {
        var bk = state.settings.backends || {};
        var available = bk.available || PA.store.get().backends || [];
        var selected = bk.selected;

        var rows = available.map(function (b) {
          return {
            label: b.label || b.id,
            id: b.id,
            available: b.available !== false,
            reason: b.reason || '',
            selected: b.id === selected,
          };
        });

        var tbl = C.table({
          cols: [
            { key: 'label', label: 'Backend', render: function (r) {
                return el('div', { class: 'pa-row pa-gap-3' },
                  el('span', { class: 'pa-mono', text: r.id }),
                  el('span', { text: r.label })
                );
            } },
            { key: 'available', label: 'Status', render: function (r) {
                return r.available
                  ? C.badge('available', 'good')
                  : C.badge(r.reason || 'unavailable', 'bad');
            } },
            { key: 'selected', label: '', render: function (r) {
                if (r.selected) return C.badge('selected', 'accent');
                if (!r.available) return el('span', { class: 'pa-muted pa-sm', text: '—' });
                return C.button('Use', { small: true, ghost: true, onClick: function () {
                  save('backends', { selected: r.id });
                  renderPane();
                } });
            } },
          ],
          rows: rows,
        });

        host.appendChild(card('Compute backends', 'The startup backend is seeded; others are probed lazily. Selection is per request.',
          el('div', { class: 'pa-scroll' }, tbl)
        ));
      }

      // --- go ---------------------------------------------------------------
      renderList();
      load();
    },
  });
})(window.PA);
