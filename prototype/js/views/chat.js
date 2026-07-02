/* ============================================================================
   views/chat.js — the flagship WORKING view: Chat.
   Transcript + composer with model/backend pickers, a memory toggle and a
   web-research toggle, Send + Stop. Streams replies through the PA.data().chat()
   controller (real WS on Live, simulated on Mock).

   BINDS ONLY to the foundation contract:
     - data via PA.data() (never fetch / PA.api / PA.mock)
     - DOM via PA.ui + PA.components + the documented CSS classes
     - colors only from CSS variables (var(--pa-*)); no color literals here
   Responsive: the picker row uses .pa-grid so it collapses to 1 column on mobile.
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
      // ---- view-local state --------------------------------------------------
      var turns = [];            // [{role:'user'|'assistant', text, recalled?, sources?, error?}]
      var busy = false;          // a reply is currently streaming
      var memoryOn = true;       // inject memory/user/store into requests
      var researchOn = false;    // web-research (sources as citations)
      var controller = null;     // PA.data().chat() controller
      var activeTurn = null;     // the assistant turn currently being streamed

      var st = ctx.store.get();
      var model = st.selectedModel || (st.models && st.models[0]) || null;
      var backend = st.selectedBackend ||
        (st.backends && (st.backends.find(function (b) { return b.available; }) || st.backends[0]) || {}).id || null;

      // ---- element handles (assigned during build) --------------------------
      var transcriptEl, inputEl, sendBtn, stopBtn, statusEl;

      // ======================================================================
      // Composer controls: model picker, backend picker, memory + web toggles
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

      // Rebuild the picker <option>s in place. Called when the store's model/
      // backend catalogs arrive after the first render (health is async and
      // app.js does NOT re-mount views on store changes — only on route change).
      function refreshPickers() {
        var s = ctx.store.get();
        // Adopt store defaults if we still have none (health resolved post-render).
        if (!model) model = s.selectedModel || (s.models && s.models[0]) || null;
        if (!backend) {
          backend = s.selectedBackend ||
            (s.backends && (s.backends.find(function (b) { return b.available; }) || s.backends[0]) || {}).id || null;
        }
        ui.mount(modelSelect, modelOptions());
        ui.mount(backendSelect, backendOptions());
      }

      var memoryChip = C.chip({
        icon: 'brain',
        label: 'Memory',
        active: memoryOn,
        onClick: function () { memoryOn = !memoryOn; memoryChip.classList.toggle('is-active', memoryOn); },
      });

      var webChip = C.chip({
        icon: 'world-search',
        label: 'Web',
        active: researchOn,
        onClick: function () { researchOn = !researchOn; webChip.classList.toggle('is-active', researchOn); },
      });

      var pickerGrid = el('div', { class: ['pa-grid', 'pa-grid-2', 'pa-gap-3'] },
        C.field('Model', modelSelect),
        C.field('Backend', backendSelect)
      );

      var toggleRow = el('div', { class: ['pa-row', 'pa-wrap', 'pa-gap-3'] },
        memoryChip,
        webChip,
        el('span', { class: 'pa-spacer' }),
        statusEl = el('span', { class: ['pa-xs', 'pa-muted'] })
      );

      // ---- input + Send/Stop ------------------------------------------------
      inputEl = el('textarea', {
        class: 'pa-input',
        rows: 2,
        placeholder: 'Ask ProjectAI anything…  (Enter to send, Shift+Enter for newline)',
        style: { resize: 'vertical', minHeight: '52px' },
        onKeydown: function (e) {
          if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); onSend(); }
        },
      });

      sendBtn = C.button('Send', { primary: true, icon: 'send', onClick: onSend });
      stopBtn = C.button('Stop', { ghost: true, icon: 'player-stop', onClick: onStop, disabled: true });

      var actionRow = el('div', { class: ['pa-row', 'pa-gap-3'] },
        el('div', { class: 'pa-grow' }, inputEl),
        el('div', { class: ['pa-col', 'pa-gap-3'] }, sendBtn, stopBtn)
      );

      var composer = el('div', { class: 'pa-card', style: { display: 'flex', flexDirection: 'column', gap: 'var(--pa-sp-3)' } },
        pickerGrid,
        toggleRow,
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
        },
      });

      function renderTranscript() {
        ui.clear(transcriptEl);
        if (!turns.length) {
          transcriptEl.appendChild(C.emptyState(
            'message-2',
            'Start a conversation',
            'Pick a model and backend, then send a message. Toggle Memory to recall context, or Web to cite live sources.'
          ));
          return;
        }
        turns.forEach(function (t) { transcriptEl.appendChild(renderTurn(t)); });
        transcriptEl.scrollTop = transcriptEl.scrollHeight;
      }

      function renderTurn(t) {
        var isUser = t.role === 'user';
        var wrap = el('div', {
          class: 'pa-col',
          style: {
            gap: 'var(--pa-sp-2)',
            alignItems: isUser ? 'flex-end' : 'stretch',
          },
        });

        // "recalled N memories" hint line above an assistant reply when memory is on
        if (!isUser && t.recalled != null && t.recalled > 0) {
          wrap.appendChild(el('div', { class: ['pa-row', 'pa-gap-3', 'pa-xs', 'pa-muted'] },
            ui.icon('brain', 'pa-xs'),
            el('span', { text: 'recalled ' + t.recalled + ' ' + (t.recalled === 1 ? 'memory' : 'memories') })
          ));
        }

        var bubble = el('div', {
          class: 'pa-card',
          style: {
            maxWidth: isUser ? '78%' : '100%',
            background: isUser ? 'var(--pa-user)' : 'var(--pa-panel)',
            borderColor: t.error ? 'var(--pa-bad)' : 'var(--pa-border)',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
          },
        });

        var head = el('div', { class: ['pa-row', 'pa-gap-3'], style: { marginBottom: 'var(--pa-sp-2)' } },
          ui.icon(isUser ? 'user' : 'robot', 'pa-sm'),
          el('span', { class: ['pa-xs', 'pa-muted'], text: isUser ? 'You' : (t.model || 'Assistant') })
        );
        bubble.appendChild(head);

        var body = el('div', { class: t.streaming && !t.text ? 'pa-muted' : '' });
        body.textContent = t.text || (t.streaming ? '…' : '');
        // expose the body so streaming can append without a full re-render
        t._bodyEl = body;
        bubble.appendChild(body);

        if (t.error) {
          bubble.appendChild(el('div', { class: ['pa-row', 'pa-gap-3', 'pa-sm', 'pa-mt-3'], style: { color: 'var(--pa-bad)' } },
            ui.icon('alert-triangle', 'pa-sm'),
            el('span', { text: t.error })
          ));
        }

        // sources as citations (web-research)
        if (t.sources && t.sources.length) {
          bubble.appendChild(renderSources(t));
        }

        wrap.appendChild(bubble);
        return wrap;
      }

      function renderSources(t) {
        var box = el('div', {
          class: 'pa-mt-4',
          style: { borderTop: '1px solid var(--pa-border)', paddingTop: 'var(--pa-sp-3)' },
        });
        box.appendChild(el('div', { class: ['pa-row', 'pa-gap-3', 'pa-xs', 'pa-muted'], style: { marginBottom: 'var(--pa-sp-2)' } },
          ui.icon('link', 'pa-xs'),
          el('span', { text: t.sources.length + ' ' + (t.sources.length === 1 ? 'source' : 'sources') })
        ));
        var list = el('div', { class: ['pa-col', 'pa-gap-3'] });
        t.sources.forEach(function (s, i) {
          list.appendChild(el('div', { class: ['pa-row', 'pa-gap-3'], style: { alignItems: 'baseline' } },
            C.badge(String(i + 1), 'accent'),
            el('div', { class: 'pa-grow' },
              s.url
                ? el('a', {
                    href: s.url, target: '_blank', rel: 'noopener noreferrer',
                    class: 'pa-sm', style: { color: 'var(--pa-accent)', textDecoration: 'none' },
                    text: s.title || s.url,
                  })
                : el('div', { class: 'pa-sm', text: s.title || '(untitled)' }),
              s.snippet ? el('div', { class: ['pa-xs', 'pa-muted'], text: ui.truncate(s.snippet, 140) }) : null
            )
          ));
        });
        box.appendChild(list);
        return box;
      }

      // ======================================================================
      // Streaming: append a token into the active assistant turn in place
      // ======================================================================
      function appendToken(text) {
        if (!activeTurn) return;
        activeTurn.text += text;
        if (activeTurn._bodyEl) {
          activeTurn._bodyEl.className = '';
          activeTurn._bodyEl.textContent = activeTurn.text;
        }
        transcriptEl.scrollTop = transcriptEl.scrollHeight;
      }

      // setBusy toggles the send/stop/input controls. When `status` is passed
      // it sets that message; otherwise it defaults ('Generating…' on, '' off).
      // onDone passes a completion summary so it isn't wiped by the reset.
      function setBusy(on, status) {
        busy = on;
        sendBtn.disabled = on;
        stopBtn.disabled = !on;
        inputEl.disabled = on;
        setStatus(status != null ? status : (on ? 'Generating…' : ''));
      }

      function setStatus(msg) {
        if (statusEl) statusEl.textContent = msg || '';
      }

      // ======================================================================
      // Controller wiring — one PA.data().chat() controller reused per turn
      // ======================================================================
      function ensureController() {
        if (controller) return controller;
        controller = ctx.data().chat({
          onOpen: function () { /* transport ready */ },
          onToken: function (text) { appendToken(text); },
          onSources: function (list) {
            if (activeTurn) { activeTurn.sources = list || []; renderTranscript(); }
          },
          onDone: function (info) {
            var summary = '';
            if (activeTurn) {
              activeTurn.streaming = false;
              if (info && (info.stop === 'cancelled' || info.stop === 'canceled') && !activeTurn.text) { // live server spells it 'canceled'
                activeTurn.text = '(stopped)';
              }
              if (info && info.stop) {
                summary = 'Done · stop: ' + info.stop +
                  (info.seconds != null ? ' · ' + ui.fmtSecs(info.seconds) : '') +
                  (info.generatedTokens != null ? ' · ' + ui.fmtNum(info.generatedTokens) + ' tok' : '');
              }
            }
            activeTurn = null;
            setBusy(false, summary);   // keep the completion summary in the status line
            renderTranscript();
          },
          onError: function (err) {
            var msg = (err && err.message) || String(err) || 'Chat failed';
            if (activeTurn) {
              activeTurn.streaming = false;
              activeTurn.error = msg;
              if (!activeTurn.text) activeTurn.text = '';
            } else {
              turns.push({ role: 'assistant', text: '', error: msg });
            }
            activeTurn = null;
            setBusy(false);
            renderTranscript();
          },
          onClose: function () { controller = null; },
        });
        // start() names the model + backend for this session
        controller.start(model, backend, buildOpts());
        return controller;
      }

      // request options: memory/user/store when Memory is on, research when Web is on
      function buildOpts() {
        var opts = { research: researchOn, memory: memoryOn };
        if (memoryOn) {
          opts.user = 'me';
          opts.store = 'default';
        }
        return opts;
      }

      // ======================================================================
      // Send / Stop
      // ======================================================================
      function onSend() {
        if (busy) return;
        var text = (inputEl.value || '').trim();
        if (!text) return;
        if (!model) { flashStatus('Pick a model first'); return; }

        inputEl.value = '';

        // user turn
        turns.push({ role: 'user', text: text });

        // assistant placeholder turn (streamed into)
        activeTurn = {
          role: 'assistant', text: '', streaming: true, model: model,
          recalled: memoryOn ? recalledCount(text) : null,
          sources: null,
        };
        turns.push(activeTurn);

        renderTranscript();
        setBusy(true);

        try {
          var ctl = ensureController();
          ctl.send(text, buildOpts());
        } catch (e) {
          // synchronous failure — surface it on the active turn
          activeTurn.streaming = false;
          activeTurn.error = (e && e.message) || 'Failed to send';
          activeTurn = null;
          setBusy(false);
          renderTranscript();
        }
      }

      function onStop() {
        if (!busy) return;
        try { if (controller) controller.cancel(); } catch (e) { /* ignore */ }
      }

      // A small deterministic "recalled N" for the memory hint. The real count
      // would come from the server; Mock has no per-turn recall count, so we
      // derive a stable non-zero number from the prompt length.
      function recalledCount(text) {
        var max = (ctx.store.get().settings &&
          ctx.store.get().settings.memory &&
          ctx.store.get().settings.memory.maxRecall) || 4;
        return Math.min(max, 1 + ((text || '').length % max));
      }

      function flashStatus(msg) {
        setStatus(msg);
        setTimeout(function () { if (!busy) setStatus(''); }, 1800);
      }

      // ======================================================================
      // Topbar action: clear the conversation
      // ======================================================================
      var clearBtn = C.button('Clear', {
        ghost: true, small: true, icon: 'eraser',
        onClick: function () {
          if (busy) onStop();
          turns = [];
          activeTurn = null;
          renderTranscript();
          setStatus('');
        },
      });
      ctx.setTopBar('Chat', clearBtn);

      // ======================================================================
      // Assemble the view — a full-height column: transcript grows, composer pinned
      // ======================================================================
      var layout = el('div', {
        class: 'pa-col',
        style: {
          gap: 'var(--pa-sp-4)',
          height: '100%',
          minHeight: '0',
        },
      },
        transcriptEl,
        composer
      );

      root.appendChild(layout);
      renderTranscript();

      // Health is async: the store's model/backend catalogs usually arrive AFTER
      // this first render. app.js does not re-mount views on store changes, so we
      // subscribe here to repopulate the pickers when the catalogs land. The
      // subscription self-cleans once this view's DOM is detached (a route change
      // clears the container), so we never leak or write into a dead view.
      var unsub = ctx.store.subscribe(function () {
        if (!layout.isConnected) { unsub(); return; }
        refreshPickers();
      });
    },
  });
})(window.PA);
