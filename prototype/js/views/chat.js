/* ============================================================================
   views/chat.js — the flagship WORKING view: Chat. 1:1 with the Godot client
   (Ui/Views/ChatView.cs + Ui/Composer.cs + Ui/TurnCard.cs).

   Topbar: title (ellipsized first prompt; "New chat" initially) + an "instruct"
   accent badge (shown when the WS ready frame says instruct:true) + a muted
   context meter + a Clear action.

   Composer (a card):
     row 1  Model select (3fr) + Backend select (2fr)
     row 2  🧠 Memory + 🌐 Web pill chips + a status span ("Generating…")
     row 3  the prompt textarea (Enter sends, Shift+Enter newline)
     row 4  "⚙ Advanced" (a popover ABOVE the composer: Sample gating
            Temperature/Top-K/Top-P/Seed, "Limit response length" gating Max
            tokens, Text size that live-resizes the transcript) + Send/Stop.

   Sessions: memory rides the START frame only ({memory:true, user:'default'},
   store omitted → the server defaults it to the model name). Toggling
   memory/model/backend restarts the session. The server spells the cancel
   stop reason 'canceled'.

   BINDS ONLY to the foundation contract: data via PA.data(), DOM via PA.ui +
   PA.components + token-driven classes; colors only from var(--pa-*).
   ========================================================================== */
(function (PA) {
  var el = PA.ui.el;
  var ui = PA.ui;
  var C = PA.components;

  PA.registerView('chat', {
    title: 'Chat',
    icon: 'message',
    order: 1,

    render: function (root, ctx) {
      // ---- view-local state ------------------------------------------------
      var turns = [];            // [{role, text, streaming?, sources?, error?, note?, _bodyEl?}]
      var busy = false;          // a reply is currently streaming
      var stopping = false;      // Stop pressed; waiting for the server's done
      var memoryOn = false;      // attach the model's memory store (start frame only)
      var researchOn = false;    // web-research (sources as citations)
      var controller = null;     // PA.data().chat() controller
      var activeTurn = null;     // the assistant turn currently being streamed
      var title = 'New chat';

      // Session identity — a change to any of these restarts the WS session
      // (memory is baked into the warm cache on the start frame).
      var session = { started: false, model: null, backend: null, memory: null };

      // Advanced settings (client defaults).
      var adv = {
        sample: false,       // off = greedy / deterministic
        temperature: 0.8,
        topK: 40,
        topP: 0.9,
        seed: 0,
        capLength: false,    // off = dynamic (maxTokens 0 — until the model stops)
        maxTokens: 1024,
        fontSize: 14,
      };

      var st = ctx.store.get();
      var model = st.selectedModel || (st.models && st.models[0]) || null;
      var backend = st.selectedBackend ||
        (st.backends && (st.backends.find(function (b) { return b.available; }) || st.backends[0]) || {}).id || null;

      // ---- element handles ---------------------------------------------------
      var transcriptEl, inputEl, sendBtn, statusEl, advPop;

      // ======================================================================
      // Topbar: title + instruct badge + context meter + Clear
      // ======================================================================
      var instructBadge = C.badge('instruct', 'accent');
      instructBadge.title = 'The server detected a chat-templated (instruct) model for this session.';
      instructBadge.classList.add('pa-hidden');

      var meterEl = el('span', { class: ['pa-xs', 'pa-muted'] });

      var clearBtn = C.button('Clear', {
        ghost: true, small: true, icon: 'eraser',
        onClick: function () {
          if (busy) onStop();
          turns = [];
          activeTurn = null;
          title = 'New chat';
          meterEl.textContent = '';
          session.started = false; // next message starts a fresh server session
          setTopBar();
          renderTranscript();
        },
      });
      clearBtn.title = 'Clear the conversation (the next message starts a fresh session).';

      function setTopBar() {
        ctx.setTopBar(title, [instructBadge, meterEl, clearBtn]);
      }
      setTopBar();

      // ======================================================================
      // Composer row 1: Model (3fr) + Backend (2fr)
      // ======================================================================
      function modelOptions() {
        var models = ctx.store.get().models || [];
        if (!models.length) return [el('option', { value: '', text: 'No models', selected: true })];
        return models.map(function (m) {
          return el('option', { value: m, text: m, selected: m === model });
        });
      }

      function backendOptions() {
        var backends = ctx.store.get().backends || [];
        if (!backends.length) return [el('option', { value: '', text: 'No backends', selected: true })];
        return backends.map(function (b) {
          var label = b.label + (b.available ? '' : ' — unavailable');
          return el('option', {
            value: b.id,
            text: label,
            selected: b.id === backend,
            disabled: !b.available,
          });
        });
      }

      var modelSelect = el('select', {
        class: 'pa-select',
        onChange: function (e) {
          model = e.target.value || null;
          ctx.store.set({ selectedModel: model });
        },
      }, modelOptions());

      var backendSelect = el('select', {
        class: 'pa-select',
        onChange: function (e) {
          backend = e.target.value || null;
          ctx.store.set({ selectedBackend: backend });
        },
      }, backendOptions());

      // Rebuild the picker <option>s in place when the store's catalogs arrive
      // after the first render (health is async).
      function refreshPickers() {
        var s = ctx.store.get();
        if (!model) model = s.selectedModel || (s.models && s.models[0]) || null;
        else if (s.selectedModel && s.selectedModel !== model && s.models && s.models.indexOf(s.selectedModel) >= 0) {
          model = s.selectedModel; // another view chose a model (e.g. "Chat with" on a card)
        }
        if (!backend) {
          backend = s.selectedBackend ||
            (s.backends && (s.backends.find(function (b) { return b.available; }) || s.backends[0]) || {}).id || null;
        }
        ui.mount(modelSelect, modelOptions());
        ui.mount(backendSelect, backendOptions());
      }

      var pickerGrid = el('div', { class: 'pa-composer-pickers' }, modelSelect, backendSelect);

      // ======================================================================
      // Composer row 2: 🧠 Memory + 🌐 Web chips + status span
      // ======================================================================
      var memoryChip = C.chip({
        label: '🧠  Memory',
        active: memoryOn,
        onClick: function () { memoryOn = !memoryOn; memoryChip.classList.toggle('is-active', memoryOn); },
      });
      memoryChip.title = "Attach this model's long-term memory store to the conversation (toggling restarts the session).";

      var webChip = C.chip({
        label: '🌐  Web',
        active: researchOn,
        onClick: function () { researchOn = !researchOn; webChip.classList.toggle('is-active', researchOn); },
      });
      webChip.title = 'Ground answers in a live web search with citations (needs a Tavily key — Settings → Web search).';

      var toggleRow = el('div', { class: ['pa-row', 'pa-wrap', 'pa-gap-3'] },
        memoryChip,
        webChip,
        el('span', { class: 'pa-spacer' }),
        statusEl = el('span', { class: ['pa-xs', 'pa-muted'] })
      );

      // ======================================================================
      // Composer row 3: the prompt textarea
      // ======================================================================
      inputEl = el('textarea', {
        class: 'pa-input',
        rows: 2,
        placeholder: 'Message ProjectAI…  (Enter to send, Shift+Enter for newline)',
        style: { resize: 'vertical', minHeight: '52px' },
        onKeydown: function (e) {
          if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); onSend(); }
        },
      });

      // ======================================================================
      // Composer row 4: ⚙ Advanced (popover above) + Send/Stop
      // ======================================================================
      function numField(label, value, min, max, step, onChange) {
        var input = el('input', {
          class: 'pa-input', type: 'number',
          min: String(min), max: String(max), step: String(step), value: String(value),
          style: { width: '110px' },
          onChange: function (e) {
            var v = Number(e.target.value);
            if (e.target.value === '' || isNaN(v)) { e.target.value = String(value); return; }
            v = Math.min(max, Math.max(min, v));
            e.target.value = String(v);
            onChange(v);
          },
        });
        var row = el('div', { class: ['pa-row', 'pa-gap-3'] },
          el('span', { class: ['pa-sm', 'pa-muted', 'pa-grow'], text: label }),
          input
        );
        return { row: row, input: input };
      }

      var tempField = numField('Temperature', adv.temperature, 0, 2, 0.05, function (v) { adv.temperature = v; });
      var topKField = numField('Top-K', adv.topK, 0, 200, 1, function (v) { adv.topK = v; });
      var topPField = numField('Top-P', adv.topP, 0.05, 1, 0.05, function (v) { adv.topP = v; });
      var seedField = numField('Seed', adv.seed, 0, 4294967295, 1, function (v) { adv.seed = v; });
      var maxTokField = numField('Max tokens', adv.maxTokens, 1, 8192, 1, function (v) { adv.maxTokens = v; });
      var fontField = numField('Text size', adv.fontSize, 11, 28, 1, function (v) {
        adv.fontSize = v;
        transcriptEl.style.fontSize = v + 'px'; // live-resizes the conversation text
      });

      function gateSampling() {
        [tempField, topKField, topPField, seedField].forEach(function (f) { f.input.disabled = !adv.sample; });
      }
      function gateCap() { maxTokField.input.disabled = !adv.capLength; }

      // C.toggle renders a fixed on/off state, so rebuild it when it flips.
      var sampleRowHost = el('span', {});
      var capRowHost = el('span', {});
      function buildSampleToggle() {
        return C.toggle(adv.sample, function (on) { adv.sample = on; ui.mount(sampleRowHost, buildSampleToggle()); gateSampling(); }, ['Off', 'On']);
      }
      function buildCapToggle() {
        return C.toggle(adv.capLength, function (on) { adv.capLength = on; ui.mount(capRowHost, buildCapToggle()); gateCap(); }, ['Off', 'On']);
      }
      ui.mount(sampleRowHost, buildSampleToggle());
      ui.mount(capRowHost, buildCapToggle());

      advPop = el('div', { class: ['pa-popover', 'pa-hidden'] },
        el('div', { class: ['pa-row', 'pa-gap-3'] },
          el('span', { class: ['pa-sm', 'pa-grow'], text: 'Sample  (off = greedy / deterministic)' }),
          sampleRowHost
        ),
        tempField.row, topKField.row, topPField.row, seedField.row,
        el('div', { style: { borderTop: '1px solid var(--pa-border)' } }),
        el('div', { class: ['pa-row', 'pa-gap-3'] },
          el('span', { class: ['pa-sm', 'pa-grow'], text: 'Limit response length  (off = until the model stops)' }),
          capRowHost
        ),
        maxTokField.row,
        el('div', { style: { borderTop: '1px solid var(--pa-border)' } }),
        el('div', { class: ['pa-xs', 'pa-muted'], text: 'Appearance' }),
        fontField.row
      );
      gateSampling();
      gateCap();

      var advBtn = C.button('⚙  Advanced', {
        ghost: true, small: true,
        onClick: function () { advPop.classList.toggle('pa-hidden'); },
      });
      advBtn.title = 'Sampling, response length, seed, and text size';

      // Send doubles as Stop while a reply streams (client parity).
      sendBtn = C.button('Send  ↵', { primary: true, onClick: onSendPressed });

      var actionRow = el('div', { class: ['pa-row', 'pa-gap-3'] },
        advBtn,
        el('span', { class: 'pa-spacer' }),
        sendBtn
      );

      var composer = el('div', {
        class: ['pa-card', 'pa-popover-host'],
        style: { display: 'flex', flexDirection: 'column', gap: 'var(--pa-sp-3)' },
      },
        advPop,
        pickerGrid,
        toggleRow,
        inputEl,
        actionRow
      );

      // ======================================================================
      // Transcript
      // ======================================================================
      transcriptEl = el('div', {
        class: ['pa-scroll', 'pa-grow'],
        style: {
          display: 'flex', flexDirection: 'column', gap: 'var(--pa-sp-4)',
          padding: 'var(--pa-sp-2) 0',
          fontSize: adv.fontSize + 'px',
        },
      });

      function renderTranscript() {
        ui.clear(transcriptEl);
        if (!turns.length) {
          transcriptEl.appendChild(C.emptyState(
            'message-2',
            'Start a conversation',
            'Pick a model and backend, then send a message. Toggle Memory to attach the model’s store, or Web to cite live sources.'
          ));
          return;
        }
        turns.forEach(function (t) { transcriptEl.appendChild(renderTurn(t)); });
        transcriptEl.scrollTop = transcriptEl.scrollHeight;
      }

      function renderTurn(t) {
        // The user's turn: a right-aligned captionless bubble at ~72% width.
        if (t.role === 'user') {
          return el('div', { class: 'pa-turn-user' },
            el('div', { class: 'pa-bubble-user', text: t.text })
          );
        }

        // The model's turn: a full-width panel with the app caption.
        var bubble = el('div', {
          class: 'pa-card',
          style: {
            background: 'var(--pa-panel)',
            borderColor: t.error ? 'var(--pa-bad)' : 'var(--pa-border)',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
          },
        });

        bubble.appendChild(el('div', { class: ['pa-xs', 'pa-muted'], style: { marginBottom: 'var(--pa-sp-2)' }, text: 'ProjectAI' }));

        var body = el('div', { class: t.streaming && !t.text ? 'pa-muted' : (t.emptyNote ? 'pa-muted' : '') });
        body.textContent = t.text || (t.streaming ? 'Generating…' : (t.emptyNote || ''));
        if (t.error) {
          body.textContent = t.error;
          body.className = '';
          body.style.color = 'var(--pa-bad)';
        }
        t._bodyEl = body;
        bubble.appendChild(body);

        // Web-research citations: numbered links.
        if (t.sources && t.sources.length) bubble.appendChild(renderSources(t));

        // Stop-reason footer ('⏹ Stopped' / context note); hidden on a natural finish.
        if (t.note) bubble.appendChild(el('div', { class: ['pa-turn-note', 'pa-mt-3'], text: t.note }));

        return bubble;
      }

      function renderSources(t) {
        var box = el('div', {
          class: 'pa-mt-4',
          style: { borderTop: '1px solid var(--pa-border)', paddingTop: 'var(--pa-sp-3)' },
        });
        box.appendChild(el('div', { class: ['pa-xs', 'pa-muted'], style: { marginBottom: 'var(--pa-sp-2)' }, text: 'Sources' }));
        var list = el('div', { class: ['pa-col', 'pa-gap-3'] });
        t.sources.forEach(function (s, i) {
          var label = '[' + (i + 1) + '] ' + ui.truncate(s.title || s.url || '(untitled)', 70);
          list.appendChild(s.url
            ? el('a', {
                href: s.url, target: '_blank', rel: 'noopener noreferrer',
                class: 'pa-sm', style: { color: 'var(--pa-accent)', textDecoration: 'none' },
                title: s.url,
                text: label,
              })
            : el('div', { class: 'pa-sm', text: label }));
        });
        box.appendChild(list);
        return box;
      }

      // ======================================================================
      // Streaming: append a token into the active assistant turn in place
      // ======================================================================
      function appendToken(text) {
        if (!activeTurn || !text) return;
        activeTurn.text += text;
        if (activeTurn._bodyEl) {
          activeTurn._bodyEl.className = '';
          activeTurn._bodyEl.textContent = activeTurn.text;
        }
        transcriptEl.scrollTop = transcriptEl.scrollHeight;
      }

      function setBusy(on) {
        busy = on;
        if (!on) stopping = false;
        sendBtn.disabled = stopping;
        ui.mount(sendBtn, el('span', { text: stopping ? 'Stopping…' : (on ? 'Stop  ◼' : 'Send  ↵') }));
        statusEl.textContent = on ? 'Generating…' : '';
      }

      // Marks a streamed turn finished, annotating why it ended (TurnCard parity).
      function completeTurn(t, stop) {
        t.streaming = false;
        var canceled = stop === 'canceled' || stop === 'cancelled'; // the server spells it 'canceled'
        if (!t.text) {
          t.emptyNote =
            stop === 'context_full' ? 'Context window full — start a New chat to continue.' :
            canceled ? 'Stopped before any output.' :
            '(empty response)';
          return;
        }
        if (canceled) t.note = '⏹ Stopped';
        else if (stop === 'context') t.note = 'Reached the context limit — start a New chat to continue.';
        // eos / im_end / maxTokens → a natural end, no footer
      }

      // ======================================================================
      // Controller wiring — one PA.data().chat() controller, restarted when the
      // session identity (model/backend/memory) changes
      // ======================================================================
      function ensureController() {
        if (controller) return controller;
        controller = ctx.data().chat({
          onOpen: function () { /* transport ready */ },
          onReady: function (info) {
            // The server accepted the session: reflect instruct + context size.
            instructBadge.classList.toggle('pa-hidden', !(info && info.instruct));
            if (info && info.contextLimit > 0) meterEl.textContent = 'ctx ' + ui.fmtNum(info.contextLimit);
          },
          onToken: function (text) { appendToken(text); },
          onSources: function (list) {
            if (activeTurn) { activeTurn.sources = list || []; renderTranscript(); }
          },
          onDone: function (info) {
            info = info || {};
            if (activeTurn) completeTurn(activeTurn, info.stop);
            activeTurn = null;
            setBusy(false);
            renderTranscript();
            // The short done form (research canceled mid-search) has no accounting — keep the previous meter then.
            if (info.contextLimit > 0) {
              meterEl.textContent =
                ui.fmtNum(info.position) + ' / ' + ui.fmtNum(info.contextLimit) + ' ctx   ·   ' +
                ui.fmtNum(info.generatedTokens) + ' tok in ' + Number(info.seconds || 0).toFixed(1) + 's';
            }
          },
          onError: function (err) {
            var msg = (err && err.message) || String(err) || 'Chat failed';
            if (activeTurn) {
              activeTurn.streaming = false;
              activeTurn.error = msg;
            } else {
              turns.push({ role: 'assistant', text: '', error: msg });
            }
            activeTurn = null;
            setBusy(false);
            renderTranscript();
          },
          onClose: function () {
            // Connection dropped: fail any in-flight turn; force a fresh session next message.
            if (busy && activeTurn) {
              activeTurn.streaming = false;
              activeTurn.error = 'Chat connection closed — is `projectai serve` running?';
              activeTurn = null;
              setBusy(false);
              renderTranscript();
            }
            controller = null;
            session.started = false;
          },
        });
        session.started = false; // a new controller always needs a start frame
        return controller;
      }

      // Memory rides the START frame only: {memory:true, user:'default'}, store
      // omitted → the server defaults it to the model name.
      function startOpts() {
        return memoryOn ? { memory: true, user: 'default' } : {};
      }

      // Per-message options: decoding + research (never memory — that's session-scoped).
      function messageOpts() {
        var opts = {
          sample: adv.sample,
          maxTokens: adv.capLength ? adv.maxTokens : 0, // 0 = dynamic (until the model stops)
          research: researchOn,
        };
        if (adv.sample) {
          opts.temperature = adv.temperature;
          opts.topK = adv.topK;
          opts.topP = adv.topP;
          opts.seed = adv.seed;
        }
        return opts;
      }

      // ======================================================================
      // Send / Stop
      // ======================================================================
      function onSendPressed() {
        if (!busy) { onSend(); return; }
        onStop();
      }

      function onSend() {
        if (busy) return;
        var text = (inputEl.value || '').trim();
        if (!text) return;
        if (!model) { flashStatus('Pick a model first'); return; }

        inputEl.value = '';
        advPop.classList.add('pa-hidden');

        turns.push({ role: 'user', text: text });
        activeTurn = { role: 'assistant', text: '', streaming: true, sources: null };
        turns.push(activeTurn);

        if (title === 'New chat') { title = ui.truncate(text, 40); setTopBar(); }

        renderTranscript();
        setBusy(true);

        try {
          var ctl = ensureController();
          // (Re)start the session when first connecting, or when the
          // model/backend/memory choice changed (memory is start-frame only).
          if (!session.started || session.model !== model || session.backend !== backend || session.memory !== memoryOn) {
            ctl.start(model, backend, startOpts());
            session = { started: true, model: model, backend: backend, memory: memoryOn };
          }
          ctl.send(text, messageOpts());
        } catch (e) {
          activeTurn.streaming = false;
          activeTurn.error = (e && e.message) || 'Failed to send';
          activeTurn = null;
          setBusy(false);
          renderTranscript();
        }
      }

      // Stop: ask the server to halt; it ends the turn with done (stop=canceled),
      // keeping the partial reply.
      function onStop() {
        if (!busy || stopping) return;
        stopping = true;
        setBusy(true); // re-render the button as "Stopping…" (disabled until done lands)
        try { if (controller) controller.cancel(); } catch (e) { /* ignore */ }
      }

      function flashStatus(msg) {
        statusEl.textContent = msg;
        setTimeout(function () { if (!busy) statusEl.textContent = ''; }, 1800);
      }

      // ======================================================================
      // Assemble — a full-height column: transcript grows, composer pinned
      // ======================================================================
      var layout = el('div', {
        class: 'pa-col',
        style: { gap: 'var(--pa-sp-4)', height: '100%', minHeight: '0' },
      },
        transcriptEl,
        composer
      );

      root.appendChild(layout);
      renderTranscript();

      // Health is async: repopulate the pickers when the catalogs land. The subscription self-cleans once this
      // view's DOM is detached — and takes the WebSocket with it, or every visit would leak a live /chat
      // connection (and its warm server-side session). The Godot client's socket is app-lifetime; ours is per-mount.
      var unsub = ctx.store.subscribe(function () {
        if (!layout.isConnected) {
          unsub();
          if (controller) { try { controller.close(); } catch (e) {} controller = null; }
          return;
        }
        refreshPickers();
      });
    },
  });
})(window.PA);
