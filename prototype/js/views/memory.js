/* ============================================================================
   views/memory.js — the Memory screen, 1:1 with the Godot client
   (Ui/Views/MemoryView.cs).

   Left column: a store select ("default" + every model name — chat's
   per-model store convention; preselects the selected model), a search input
   (Enter reloads), a count label, and the card catalog: title + tier badge
   (core=accent, else neutral) + trust badge (curated=good, untrusted=bad,
   chat=neutral) + the muted meta "[keys]  as of <asof>  ·  <id>".
   Data: GET /memory?user=default&store=&q= → {user, store, count, memories}.

   Right column: "Injection preview" (👁 button → GET /memory/render → the
   bridge block + the recall block, with explicit empty placeholders) and
   "Inject a memory" (Title, Keys, Body, Tier long/core/session, Trust
   curated/chat/untrusted → PUT /memory → "Saved <id>." then reload).
   ========================================================================== */
(function (PA) {
  var el = PA.ui.el;
  var ui = PA.ui;
  var C = PA.components;

  PA.registerView('memory', {
    title: 'Memory',
    icon: 'brain',
    order: 4,

    render: function (root, ctx) {
      // ---- view-local state -------------------------------------------------
      var store = null;         // selected store id
      var query = '';
      var memories = null;      // null = loading
      var count = 0;
      var loadError = null;
      var draft = { title: '', keys: '', body: '', tier: 'long', trust: 'curated' };

      ctx.setTopBar('Memory', []);

      // ---- store options: 'default' + every model name ------------------------
      function storeOptions() {
        var s = ctx.store.get();
        var names = ['default'].concat(s.models || []);
        if (!store) store = (s.selectedModel && names.indexOf(s.selectedModel) >= 0) ? s.selectedModel : 'default';
        return names.map(function (n) {
          return el('option', { value: n, text: n, selected: n === store });
        });
      }

      var storeSel = el('select', {
        class: 'pa-select',
        onChange: function (e) { store = e.target.value; reload(); },
      }, storeOptions());
      storeSel.title = "Memory store (chat uses the model's name)";

      var searchInput = el('input', {
        class: 'pa-input', type: 'search',
        placeholder: 'Search memories… (Enter)',
        value: query,
        onKeydown: function (e) {
          if (e.key === 'Enter') { query = searchInput.value; reload(); }
        },
      });

      var countLabel = el('span', { class: ['pa-xs', 'pa-muted'], text: '' });
      var listHost = el('div', { class: 'pa-col pa-gap-3' });

      // ---- catalog ------------------------------------------------------------
      function reload() {
        memories = null;
        loadError = null;
        countLabel.textContent = 'loading…';
        renderList();
        ctx.data().memoryList({ store: store, q: query }).then(function (res) {
          if (!root.isConnected) return;
          memories = (res && res.memories) || [];
          count = res && res.count != null ? res.count : memories.length;
          countLabel.textContent = count === 1 ? '1 memory' : count + ' memories';
          renderList();
        }).catch(function (e) {
          if (!root.isConnected) return;
          memories = [];
          loadError = (e && e.message) || String(e);
          countLabel.textContent = '';
          renderList();
        });
      }

      function renderList() {
        ui.clear(listHost);
        if (loadError) {
          listHost.appendChild(el('div', { class: 'pa-sm', style: { color: 'var(--pa-bad)' }, text: 'Could not load memories: ' + loadError }));
          return;
        }
        if (memories == null) {
          listHost.appendChild(el('div', { class: ['pa-sm', 'pa-muted'], text: 'Loading memories…' }));
          return;
        }
        if (!memories.length) {
          listHost.appendChild(C.emptyState('brain', 'No memories',
            query
              ? 'Nothing matches "' + ui.truncate(query, 40) + '".'
              : "Inject one on the right, or chat with Memory on — the model's store fills as you talk."));
          return;
        }
        memories.forEach(function (m) { listHost.appendChild(memoryCard(m)); });
      }

      function memoryCard(m) {
        var tierTone = m.tier === 'core' ? 'accent' : 'neutral';
        var trustTone = m.trust === 'curated' ? 'good' : m.trust === 'untrusted' ? 'bad' : 'neutral';

        var titleRow = el('div', { class: 'pa-row pa-gap-3' },
          el('span', { class: 'pa-grow', style: { fontWeight: '600', wordBreak: 'break-word' }, text: m.title || '' }),
          C.badge(m.tier || 'long', tierTone),
          C.badge(m.trust || 'chat', trustTone)
        );

        var meta =
          (m.keys && m.keys.length ? '[' + m.keys.join(', ') + ']   ' : '') +
          (m.asof ? 'as of ' + m.asof : '') +
          '   ·   ' + (m.id || '');

        return el('div', { class: 'pa-card pa-col pa-gap-3' },
          titleRow,
          el('div', { class: ['pa-xs', 'pa-muted'], text: meta })
        );
      }

      // ---- injection preview -----------------------------------------------------
      var bridgePreview = el('pre', {
        class: ['pa-xs', 'pa-muted', 'pa-mono'],
        style: { margin: '0', whiteSpace: 'pre-wrap', wordBreak: 'break-word' },
      });
      var recallPreview = el('pre', {
        class: ['pa-xs', 'pa-muted', 'pa-mono'],
        style: { margin: '0', whiteSpace: 'pre-wrap', wordBreak: 'break-word' },
      });

      var previewBtn = C.button('👁  Preview what a message would inject', {
        ghost: true, small: true,
        onClick: function () {
          bridgePreview.textContent = 'rendering…';
          recallPreview.textContent = '';
          ctx.data().memoryRender({ store: store, q: searchInput.value }).then(function (res) {
            bridgePreview.textContent = (res && res.bridge) ? res.bridge : '(bridge: empty — no pinned/core memories)';
            recallPreview.textContent = (res && res.recall) ? res.recall : '(recall: nothing relevant to the search text)';
          }).catch(function (e) {
            bridgePreview.textContent = 'Error: ' + ((e && e.message) || e);
            recallPreview.textContent = '';
          });
        },
      });
      previewBtn.title = 'Renders the pinned bridge plus the recall block for the search text, with the exact budgets chat uses.';

      var previewCard = el('div', { class: 'pa-card pa-col pa-gap-3' },
        el('div', { class: 'pa-h3', text: 'Injection preview' }),
        previewBtn,
        bridgePreview,
        recallPreview
      );

      // ---- inject form ---------------------------------------------------------------
      var titleInput = el('input', {
        class: 'pa-input', placeholder: 'Title (e.g. "Ben\'s GPU is an 8GB 3070")',
        onInput: function (e) { draft.title = e.target.value; },
      });
      var keysInput = el('input', {
        class: 'pa-input', placeholder: 'comma, separated, keys',
        onInput: function (e) { draft.keys = e.target.value; },
      });
      var bodyInput = el('textarea', {
        class: 'pa-input', rows: 4, placeholder: 'The fact itself — one durable note.',
        onInput: function (e) { draft.body = e.target.value; },
      });
      var tierSel = el('select', { class: 'pa-select', onChange: function (e) { draft.tier = e.target.value; } },
        ['long', 'core', 'session'].map(function (t) {
          return el('option', { value: t, text: t, selected: t === draft.tier });
        })
      );
      var trustSel = el('select', { class: 'pa-select', onChange: function (e) { draft.trust = e.target.value; } },
        ['curated', 'chat', 'untrusted'].map(function (t) {
          return el('option', { value: t, text: t, selected: t === draft.trust });
        })
      );
      var saveStatus = el('div', { class: ['pa-xs', 'pa-tone-muted'] });

      function setSaveStatus(text, tone) {
        saveStatus.textContent = text || '';
        saveStatus.className = 'pa-xs ' + (tone === 'good' ? 'pa-tone-good' : tone === 'bad' ? 'pa-tone-bad' : 'pa-tone-muted');
      }

      var injectBtn = C.button('Inject memory', {
        primary: true,
        onClick: function () {
          if (!draft.title.trim() && !draft.body.trim()) {
            setSaveStatus('A memory needs a title or a body.', 'bad');
            return;
          }
          var keys = draft.keys.split(',').map(function (k) { return k.trim(); }).filter(Boolean);
          setSaveStatus('Saving…', 'muted');
          ctx.data().memoryPut({
            title: draft.title.trim(),
            keys: keys,
            body: draft.body.trim(),
            tier: draft.tier,
            trust: draft.trust,
            user: 'default',
            store: store,
          }).then(function (res) {
            setSaveStatus('Saved ' + ((res && res.id) || '') + '.', 'good');
            draft.title = ''; titleInput.value = '';
            draft.keys = ''; keysInput.value = '';
            draft.body = ''; bodyInput.value = '';
            reload();
          }).catch(function (e) {
            setSaveStatus('Error: ' + ((e && e.message) || e), 'bad');
          });
        },
      });

      var injectCard = el('div', { class: 'pa-card pa-col pa-gap-3' },
        el('div', { class: 'pa-h3', text: 'Inject a memory' }),
        C.field('Title', titleInput),
        C.field('Keys', keysInput),
        bodyInput,
        C.field('Tier', tierSel),
        C.field('Trust', trustSel),
        el('div', { class: 'pa-row' }, injectBtn),
        saveStatus
      );

      // ---- assemble -----------------------------------------------------------------------
      var left = el('div', { class: 'pa-col pa-gap-3' },
        el('div', { class: 'pa-row pa-gap-3' },
          el('span', { class: ['pa-xs', 'pa-muted'], text: 'store' }),
          el('div', { class: 'pa-grow' }, storeSel)
        ),
        el('div', { class: 'pa-row pa-gap-3' },
          el('div', { class: 'pa-grow' }, searchInput),
          countLabel
        ),
        listHost
      );

      var right = el('div', { class: 'pa-col pa-gap-4' }, previewCard, injectCard);

      root.appendChild(el('div', { class: 'pa-memory-grid' }, left, right));

      // ---- go: preselect + load; refresh options when the catalog lands ----------------------
      reload();

      var unsub = ctx.store.subscribe(function () {
        if (!root.isConnected) { unsub(); return; }
        ui.mount(storeSel, storeOptions()); // 'default' + every model name
      });
    },
  });
})(window.PA);
