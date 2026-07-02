/* ============================================================================
   views/memory.js — the Memory screen of the ProjectAI prototyping harness.

   ONE view. Self-registers via PA.registerView('memory', {...}).
   Data flows ONLY through PA.data() (never fetch / PA.api / PA.mock directly).
   DOM is built ONLY via PA.ui + PA.components + the token-driven CSS classes.
   Responsive: grids use .pa-grid* so they collapse to one column on mobile.

   What it shows:
     - a store selector + a search input (filters the memory list)
     - a list of memory cards: snippet, tier badge, trust badge, provenance
     - an "Inject memory" form (text / provenance / tier / trust)
     - a collapsible "Bridge preview" of the always-pinned core memories
   Relies on: PA.data().memoryList(q) and PA.data().memoryPut(draft).
   ========================================================================== */
(function (PA) {
  var el = PA.ui.el;
  var ui = PA.ui;
  var C = PA.components;

  // Human labels for the enum values the contract guarantees.
  var TIER_LABEL = { core: 'Core', long: 'Long-term', inherited: 'Inherited' };
  var TRUST_LABEL = { trusted: 'Curated', untrusted: 'Untrusted' };

  // View-local UI state (persists across the store-driven re-mounts app.js does).
  var vs = {
    store: 'user',
    query: '',
    bridgeOpen: false,
    draft: { text: '', provenance: 'user', tier: 'long', trust: 'trusted' },
    status: null,      // { tone, text } transient banner after an inject
    memories: null,    // last-loaded list (null = loading, [] = empty)
    loadError: null,
  };

  PA.registerView('memory', {
    title: 'Memory',
    icon: 'brain',
    order: 3,
    render: render,
  });

  function render(root, ctx) {
    // Header actions on the topbar: refresh + jump to the inject form.
    ctx.setTopBar('Memory', C.button('Refresh', {
      ghost: true, icon: 'refresh', small: true,
      onClick: function () { load(root, ctx); },
    }));

    var page = el('div', { class: 'pa-col pa-gap-4' });
    ui.append(root, page);

    page.appendChild(controlsBar(root, ctx));
    page.appendChild(bridgePanel());

    // Two-column workspace: the memory list beside the inject form.
    // .pa-grid-2 auto-collapses to one column on mobile.
    var grid = el('div', { class: 'pa-grid pa-grid-2' },
      listColumn(),
      injectColumn(root, ctx)
    );
    page.appendChild(grid);

    // Kick off the (async) load; the list column fills in when it resolves.
    load(root, ctx);
  }

  // --- store selector + search --------------------------------------------
  function controlsBar(root, ctx) {
    var storeSel = el('select', { class: 'pa-select' },
      ['user', 'store'].map(function (s) {
        return el('option', { value: s, selected: vs.store === s },
          s === 'user' ? 'User store' : 'Shared store');
      })
    );
    storeSel.addEventListener('change', function () {
      vs.store = storeSel.value;
      load(root, ctx);
    });

    var search = el('input', {
      class: 'pa-input',
      type: 'search',
      placeholder: 'Search memories…',
      value: vs.query,
    });
    // Debounce-free but cheap: filter on each input (mock filters server-side).
    search.addEventListener('input', function () {
      vs.query = search.value;
      load(root, ctx);
    });

    return el('div', { class: 'pa-card' },
      el('div', { class: 'pa-row pa-wrap pa-gap-3' },
        el('div', { style: { minWidth: '160px' } },
          C.field('Memory store', storeSel)
        ),
        el('div', { class: 'pa-grow' },
          C.field('Search', search, 'Filters recalled memories by text')
        )
      )
    );
  }

  // --- collapsible bridge preview -----------------------------------------
  function bridgePanel() {
    var body = el('div', { class: 'pa-col pa-gap-3', dataset: { role: 'bridge-body' } });
    if (vs.bridgeOpen) body.appendChild(bridgeBody());

    var caret = ui.icon(vs.bridgeOpen ? 'chevron-down' : 'chevron-right');
    var header = el('button', {
      class: 'pa-btn pa-btn-ghost pa-row',
      style: { width: '100%', justifyContent: 'flex-start' },
      onClick: function () {
        vs.bridgeOpen = !vs.bridgeOpen;
        caret.className = 'ti ti-' + (vs.bridgeOpen ? 'chevron-down' : 'chevron-right');
        ui.clear(body);
        if (vs.bridgeOpen) body.appendChild(bridgeBody());
      },
    },
      caret,
      ui.icon('pin'),
      el('span', { text: 'Bridge preview' }),
      el('span', { class: 'pa-spacer' }),
      C.badge('always pinned', 'accent')
    );

    return el('div', { class: 'pa-card pa-col pa-gap-3' }, header, body);
  }

  // The bridge is synthesized from the currently-loaded CORE memories so the
  // preview stays truthful to what the list shows. (memoryList returns
  // {store, memories}; there is no separate bridge field in the contract.)
  function bridgeBody() {
    var core = (vs.memories || []).filter(function (m) { return m.tier === 'core'; });
    var lines = core.length
      ? core.map(function (m) { return '• ' + m.text; })
      : ['• (no core memories pinned for this store yet)'];
    var text =
      '<system>\nYou are ProjectAI, a local self-hosted assistant.\n' +
      'The following facts are always in context:\n' +
      lines.join('\n') +
      '\n</system>';

    return el('div', { class: 'pa-col pa-gap-3' },
      el('div', { class: 'pa-muted pa-sm', text:
        'This block is injected ahead of every conversation, regardless of recall.' }),
      el('pre', {
        class: 'pa-mono pa-sm',
        style: {
          margin: '0',
          padding: 'var(--pa-sp-3)',
          background: 'var(--pa-input)',
          border: '1px solid var(--pa-border)',
          borderRadius: 'var(--pa-radius)',
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
          color: 'var(--pa-text)',
        },
        text: text,
      })
    );
  }

  // --- list column ---------------------------------------------------------
  function listColumn() {
    var head = el('div', { class: 'pa-row' },
      el('span', { class: 'pa-h3', text: 'Recalled memories' }),
      el('span', { class: 'pa-spacer' }),
      el('span', { class: 'pa-muted pa-sm', dataset: { role: 'list-count' },
        text: countLabel() })
    );

    var body = el('div', { class: 'pa-col pa-gap-3', dataset: { role: 'list-body' } });
    fillList(body);

    return el('div', { class: 'pa-col pa-gap-3', dataset: { role: 'list-col' } }, head, body);
  }

  function countLabel() {
    if (vs.memories == null) return 'loading…';
    return ui.fmtNum(vs.memories.length) + (vs.memories.length === 1 ? ' memory' : ' memories');
  }

  function fillList(body) {
    ui.clear(body);

    if (vs.loadError) {
      body.appendChild(C.emptyState('alert-triangle', 'Could not load memories',
        vs.loadError));
      return;
    }
    if (vs.memories == null) {
      body.appendChild(el('div', { class: 'pa-muted pa-sm', text: 'Loading memories…' }));
      return;
    }
    if (vs.memories.length === 0) {
      body.appendChild(C.emptyState('database-off', 'No memories',
        vs.query ? 'Nothing matches “' + ui.truncate(vs.query, 40) + '”.'
                 : 'Inject a memory to get started.'));
      return;
    }

    vs.memories.forEach(function (m) { body.appendChild(memoryCard(m)); });
  }

  function memoryCard(m) {
    var tier = m.tier || 'long';
    var trust = m.trust || 'trusted';

    var badges = el('div', { class: 'pa-row pa-wrap pa-gap-3' },
      C.badge(TIER_LABEL[tier] || tier, 'tier-' + tier),
      C.badge(TRUST_LABEL[trust] || trust, 'trust-' + trust),
      m.provenance ? el('span', { class: 'pa-badge pa-badge-neutral' },
        ui.icon('route'), el('span', { text: 'via ' + m.provenance })) : null
    );

    var meta = el('div', { class: 'pa-row pa-muted pa-xs pa-wrap pa-gap-3' },
      el('span', { text: 'id ' + (m.id || '—') }),
      m.updated ? el('span', { text: 'updated ' + m.updated }) : null
    );

    return el('div', { class: 'pa-card pa-col pa-gap-3' },
      badges,
      el('div', { text: m.text || '' }),
      meta
    );
  }

  // --- inject column -------------------------------------------------------
  function injectColumn(root, ctx) {
    var d = vs.draft;

    var textArea = el('textarea', {
      class: 'pa-input',
      placeholder: 'What should ProjectAI remember?',
      rows: 4,
      value: d.text,
    });
    textArea.addEventListener('input', function () { d.text = textArea.value; });

    var provInput = el('input', {
      class: 'pa-input',
      placeholder: 'e.g. user, inferred, web',
      value: d.provenance,
    });
    provInput.addEventListener('input', function () { d.provenance = provInput.value; });

    var tierSel = selectFrom(
      [['core', 'Core'], ['long', 'Long-term'], ['inherited', 'Inherited']],
      d.tier, function (v) { d.tier = v; }
    );
    var trustSel = selectFrom(
      [['trusted', 'Curated'], ['untrusted', 'Untrusted']],
      d.trust, function (v) { d.trust = v; }
    );

    var status = el('div', { dataset: { role: 'inject-status' } });
    renderStatus(status);

    var submit = C.button('Inject memory', {
      primary: true, icon: 'plus',
      onClick: function () {
        var text = (d.text || '').trim();
        if (!text) {
          vs.status = { tone: 'bad', text: 'Memory text is required.' };
          renderStatus(status);
          return;
        }
        submit.disabled = true;
        vs.status = { tone: 'neutral', text: 'Saving…' };
        renderStatus(status);

        ctx.data().memoryPut({
          text: text,
          provenance: (d.provenance || '').trim() || 'user',
          tier: d.tier,
          trust: d.trust,
        }).then(function (res) {
          vs.status = { tone: 'good', text: 'Saved memory ' + (res && res.id ? res.id : '') + '.' };
          d.text = '';
          textArea.value = '';
          submit.disabled = false;
          load(root, ctx);           // refresh list (also re-renders status)
        }).catch(function (err) {
          submit.disabled = false;
          vs.status = { tone: 'bad', text: msgOf(err) };
          renderStatus(status);
        });
      },
    });

    return el('div', { class: 'pa-card pa-col pa-gap-4' },
      el('span', { class: 'pa-h3', text: 'Inject memory' }),
      el('span', { class: 'pa-muted pa-sm', text:
        'Add a fact to the ' + (vs.store === 'user' ? 'user' : 'shared') + ' store.' }),
      C.field('Memory text', textArea),
      C.field('Provenance', provInput, 'Where this came from'),
      el('div', { class: 'pa-grid pa-grid-2' },
        C.field('Tier', tierSel),
        C.field('Trust', trustSel)
      ),
      el('div', { class: 'pa-row' }, submit),
      status
    );
  }

  function renderStatus(node) {
    ui.clear(node);
    if (!vs.status) return;
    node.appendChild(C.badge(vs.status.text, vs.status.tone || 'neutral'));
  }

  // --- data load -----------------------------------------------------------
  // Fetches via PA.data().memoryList; re-renders just the list + bridge + count
  // in place (avoids a full view re-mount, keeping form focus intact).
  function load(root, ctx) {
    vs.memories = null;
    vs.loadError = null;
    refreshList(root);

    ctx.data().memoryList(vs.query || undefined).then(function (res) {
      vs.memories = (res && res.memories) ? res.memories : [];
      refreshList(root);
    }).catch(function (err) {
      vs.memories = [];
      vs.loadError = msgOf(err);
      refreshList(root);
    });
  }

  // Update the parts that depend on the loaded list, without rebuilding forms.
  function refreshList(root) {
    var body = root.querySelector('[data-role="list-body"]');
    if (body) fillList(body);

    var count = root.querySelector('[data-role="list-count"]');
    if (count) count.textContent = countLabel();

    // Bridge preview derives from core memories — refresh it if open.
    if (vs.bridgeOpen) {
      var bridge = root.querySelector('[data-role="bridge-body"]');
      if (bridge) { ui.clear(bridge); bridge.appendChild(bridgeBody()); }
    }

    // Reflect any transient inject status (e.g. after a save + reload).
    var status = root.querySelector('[data-role="inject-status"]');
    if (status) renderStatus(status);
  }

  // --- helpers -------------------------------------------------------------
  function selectFrom(pairs, current, onChange) {
    var sel = el('select', { class: 'pa-select' },
      pairs.map(function (p) {
        return el('option', { value: p[0], selected: current === p[0] }, p[1]);
      })
    );
    sel.addEventListener('change', function () { onChange(sel.value); });
    return sel;
  }

  function msgOf(err) {
    return (err && err.message) ? err.message : 'Something went wrong.';
  }
})(window.PA);
