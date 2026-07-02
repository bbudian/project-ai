/* ============================================================================
   views/settings.js — the Settings MODAL (no longer a routed view), 1:1 with
   the Godot client's SettingsWindow (Ui/Shell/SettingsWindow.cs). It opens
   from the ⚙ gear pinned near the rail bottom (app.js → PA.settingsModal.open)
   and overlays whatever view is active.

   One scrolling pane, three cards:
     App (this machine)        — Server URL (mirrors the harness toolbar) plus
                                 the desktop app's local-server fields (exe /
                                 models dir / args / auto-start / stop-on-exit),
                                 visibly annotated: a browser can't spawn
                                 processes, these apply to the desktop app.
     Memory injection (server) — the four budgets from GET /config; Save →
                                 PUT /config {memory:{…}} (400 lists problems).
     Web search (Tavily)       — masked status + a WRITE-ONLY key input
                                 (PUT /config/secrets/tavily; input cleared
                                 immediately) + Clear (DELETE). The response is
                                 always masked status {key,set,hint,source}.

   Data flows ONLY through PA.data(); DOM via PA.ui + PA.components + tokens.
   ========================================================================== */
(function (PA) {
  var el = PA.ui.el;
  var ui = PA.ui;
  var C = PA.components;

  // Desktop-app fields persisted like the client's ClientPrefs (localStorage
  // here — the browser can't spawn a server, but the values round-trip).
  var LS_KEY = 'pa.appPrefs';
  function loadPrefs() {
    try {
      var raw = localStorage.getItem(LS_KEY);
      var p = raw ? JSON.parse(raw) : {};
      return {
        serverExe: typeof p.serverExe === 'string' ? p.serverExe : '',
        modelsDir: typeof p.modelsDir === 'string' ? p.modelsDir : '',
        serverArgs: typeof p.serverArgs === 'string' ? p.serverArgs : '',
        autoStart: !!p.autoStart,
        stopOnExit: p.stopOnExit !== false, // default on, like the client
      };
    } catch (e) {
      return { serverExe: '', modelsDir: '', serverArgs: '', autoStart: false, stopOnExit: true };
    }
  }
  function savePrefs(p) {
    try { localStorage.setItem(LS_KEY, JSON.stringify(p)); } catch (e) { /* ignore */ }
  }

  function card(title, bodyNodes) {
    return el('div', { class: 'pa-card pa-col pa-gap-3' },
      el('div', { class: 'pa-h3', text: title }),
      bodyNodes
    );
  }

  function statusLine() {
    var node = el('div', { class: ['pa-xs', 'pa-tone-muted'] });
    return {
      node: node,
      set: function (text, tone) {
        node.textContent = text || '';
        node.className = 'pa-xs ' + (tone === 'good' ? 'pa-tone-good' : tone === 'bad' ? 'pa-tone-bad' : 'pa-tone-muted');
      },
    };
  }

  // A labelled on/off row built from C.toggle (rebuilt on flip, like the client's CheckButton).
  function toggleRow(label, on, onChange) {
    var host = el('span', {});
    function build(state) {
      return C.toggle(state, function (v) { ui.mount(host, build(v)); onChange(v); }, ['Off', 'On']);
    }
    ui.mount(host, build(on));
    return el('div', { class: 'pa-row pa-gap-3' },
      el('span', { class: ['pa-sm', 'pa-grow'], text: label }),
      host
    );
  }

  PA.settingsModal = {
    open: function () {
      var prefs = loadPrefs();

      // ---- App (this machine) ------------------------------------------------
      var urlInput = el('input', {
        class: 'pa-input', type: 'text', value: PA.config.baseUrl,
        placeholder: 'http://localhost:8080', spellcheck: 'false',
        onChange: function (e) {
          PA.config.baseUrl = (e.target.value || '').trim() || PA.config.baseUrl;
          e.target.value = PA.config.baseUrl;
          PA.saveConfig();
        },
      });

      function prefField(label, key, placeholder) {
        return C.field(label, el('input', {
          class: 'pa-input', type: 'text', value: prefs[key], placeholder: placeholder || '',
          onChange: function (e) { prefs[key] = e.target.value.trim(); savePrefs(prefs); },
        }));
      }

      var appCard = card('App (this machine)', el('div', { class: 'pa-col pa-gap-3' },
        C.field('Server URL', urlInput, 'Mirrors the harness toolbar URL — hit Connect up there to re-probe.'),
        C.badge('applies to the desktop app — a browser can’t spawn processes', 'neutral'),
        prefField('Server exe', 'serverExe', 'auto — <repo>/ProjectAI/bin/…/projectai.exe'),
        prefField('Models dir', 'modelsDir', 'auto — <repo>/checkpoints'),
        prefField('Server args', 'serverArgs', ''),
        toggleRow('Start the server automatically when unreachable', prefs.autoStart, function (v) { prefs.autoStart = v; savePrefs(prefs); }),
        toggleRow('Stop a server this app started when it closes', prefs.stopOnExit, function (v) { prefs.stopOnExit = v; savePrefs(prefs); }),
        el('div', { class: ['pa-xs', 'pa-muted'], text: 'Model, backend, sampling, and text size live in the chat composer; all of it persists locally.' })
      ));

      // ---- Memory injection (server) ------------------------------------------
      function budgetField(label, min, max) {
        var input = el('input', {
          class: 'pa-input', type: 'number', min: String(min), max: String(max), step: '1', value: '',
          style: { width: '140px' },
        });
        var row = el('div', { class: 'pa-row pa-gap-3' },
          el('span', { class: ['pa-sm', 'pa-muted', 'pa-grow'], text: label }),
          input
        );
        return { row: row, input: input };
      }

      var bridgeCards = budgetField('Bridge cards', 0, 200);
      var bridgeBudget = budgetField('Bridge budget', 0, 100000);
      var recallHits = budgetField('Recall hits', 0, 64);
      var recallBudget = budgetField('Recall budget', 0, 100000);
      var memStatus = statusLine();

      var saveBudgets = C.button('Save memory budgets', {
        primary: true,
        onClick: function () {
          memStatus.set('saving…', 'muted');
          PA.data().configPut({
            memory: {
              bridgeCards: Number(bridgeCards.input.value) || 0,
              bridgeBudget: Number(bridgeBudget.input.value) || 0,
              recallHits: Number(recallHits.input.value) || 0,
              recallBudget: Number(recallBudget.input.value) || 0,
            },
          }).then(function (cfg) {
            applyConfig(cfg);
            memStatus.set('Saved.', 'good');
          }).catch(function (e) {
            var problems = e && e.data && e.data.problems;
            memStatus.set(problems && problems.length ? problems.join('; ') : 'Error: ' + ((e && e.message) || e), 'bad');
          });
        },
      });

      var memoryCard = card('Memory injection (server)', el('div', { class: 'pa-col pa-gap-3' },
        bridgeCards.row, bridgeBudget.row, recallHits.row, recallBudget.row,
        el('div', { class: ['pa-xs', 'pa-muted'], text: 'How much memory each turn may inject (budgets are ~tokens). Applies to every chat immediately.' }),
        el('div', { class: 'pa-row' }, saveBudgets),
        memStatus.node
      ));

      // ---- Web search (Tavily) ---------------------------------------------------
      var secretState = el('div', { class: ['pa-sm', 'pa-tone-muted'], text: '…' });
      var secretStatus = statusLine();

      function renderSecretState(s) {
        if (s && s.set) {
          secretState.textContent = 'Configured (' + (s.hint || '…') + ', from ' + (s.source || 'config') + ')';
          secretState.className = 'pa-sm pa-tone-good';
        } else {
          secretState.textContent = 'Not configured — web research is unavailable until a key is set.';
          secretState.className = 'pa-sm pa-tone-muted';
        }
      }

      var keyInput = el('input', {
        class: 'pa-input', type: 'password', value: '',
        placeholder: 'Paste a Tavily API key (free at tavily.com)…',
        autocomplete: 'off', spellcheck: 'false',
      });

      var saveKey = C.button('Save key', {
        primary: true,
        onClick: function () {
          var raw = (keyInput.value || '').trim();
          if (!raw) return;
          keyInput.value = ''; // write-only: the raw key leaves the UI immediately
          secretStatus.set('saving…', 'muted');
          PA.data().secretPut('tavily', raw).then(function (s) {
            renderSecretState(s);
            secretStatus.set('Saved.', 'good');
          }).catch(function (e) {
            secretStatus.set('Error: ' + ((e && e.message) || e), 'bad');
          });
        },
      });

      var clearKey = C.button('Clear stored key', {
        ghost: true,
        onClick: function () {
          secretStatus.set('clearing…', 'muted');
          PA.data().secretDelete('tavily').then(function (s) {
            renderSecretState(s);
            secretStatus.set('Cleared.', 'good');
          }).catch(function (e) {
            secretStatus.set('Error: ' + ((e && e.message) || e), 'bad');
          });
        },
      });

      var secretCard = card('Web search (Tavily)', el('div', { class: 'pa-col pa-gap-3' },
        secretState,
        C.field('New key', keyInput),
        el('div', { class: 'pa-row pa-gap-3' }, saveKey, clearKey),
        secretStatus.node,
        el('div', { class: ['pa-xs', 'pa-muted'], text:
          'The key is stored server-side (config/secrets.json, ACL-locked, git-ignored) and is never sent back — only presence and a …last-4 hint. An environment variable (TAVILY_API_KEY) takes precedence.' })
      ));

      // ---- load the server-side sections -------------------------------------------
      function applyConfig(cfg) {
        var m = (cfg && cfg.memory) || {};
        bridgeCards.input.value = m.bridgeCards != null ? m.bridgeCards : '';
        bridgeBudget.input.value = m.bridgeBudget != null ? m.bridgeBudget : '';
        recallHits.input.value = m.recallHits != null ? m.recallHits : '';
        recallBudget.input.value = m.recallBudget != null ? m.recallBudget : '';
        ((cfg && cfg.secrets) || []).forEach(function (s) {
          if (s.key === 'tavily') renderSecretState(s);
        });
      }

      memStatus.set('loading…', 'muted');
      PA.data().configGet().then(function (cfg) {
        applyConfig(cfg);
        memStatus.set('Loaded from the server.', 'good');
      }).catch(function (e) {
        memStatus.set('Error: ' + ((e && e.message) || e), 'bad');
        renderSecretState(null);
      });

      C.openModal('Settings', el('div', { class: 'pa-col pa-gap-4' }, appCard, memoryCard, secretCard));
    },
  };
})(window.PA);
