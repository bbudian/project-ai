/* ============================================================================
   components.js — PA.components: reusable renderers returning DOM nodes.
   PURE VIEW: no I/O, no fetch, no PA.api/PA.mock. Compose from PA.ui.el and the
   token-driven classes in base.css. Colors come only from CSS variables.
   ========================================================================== */
(function (PA) {
  var el = PA.ui.el;
  var icon = PA.ui.icon;
  var C = {};

  // --- navRail(views, activeId, onNav, opts) -> the desktop labeled rail ---
  // views: [{id,title,icon,order}]  onNav(id)
  // opts: { onSettings()            — the ⚙ gear (opens the Settings MODAL),
  //         server: {               — the rail-footer Server panel (client's
  //           url, onUrl(v),          ConnectionPanel parity)
  //           onCheck(), onToggle(),
  //           starting, owns, startDisabled, startTooltip } }
  C.navRail = function (views, activeId, onNav, opts) {
    opts = opts || {};
    var brand = el('div', { class: 'pa-brand' },
      el('span', { class: 'pa-brand-mark', text: 'P' }),
      el('span', { class: 'pa-brand-name', text: 'ProjectAI' })
    );

    var items = views.map(function (v) {
      return el('button', {
        class: ['pa-nav-item', v.id === activeId ? 'is-active' : ''],
        onClick: function () { onNav && onNav(v.id); },
      }, icon(v.icon || 'circle'), el('span', { text: v.title }));
    });

    // Bottom block: gear pinned above the Server panel (above the rail footer).
    var bottom = el('div', { class: 'pa-rail-bottom' });
    if (opts.onSettings) {
      bottom.appendChild(el('button', {
        class: 'pa-nav-item',
        onClick: function () { opts.onSettings(); },
      }, icon('settings'), el('span', { text: 'Settings' })));
    }
    if (opts.server) bottom.appendChild(C.serverPanel(opts.server));

    return el('nav', { class: 'pa-rail' }, brand, items, bottom);
  };

  // --- serverPanel(s) -> the rail-footer Server panel ----------------------
  // Mirrors Client/ai-client/Ui/Shell/ConnectionPanel.cs: caption, URL input,
  // "Check connection", "Start/Stop local server", status line. The status
  // line carries data-role="server-status" so the app can update it in place.
  C.serverPanel = function (s) {
    s = s || {};
    var url = el('input', {
      class: 'pa-input', type: 'text', value: s.url || '',
      placeholder: 'http://localhost:8080', spellcheck: 'false',
      onChange: function (e) { s.onUrl && s.onUrl(e.target.value); },
    });

    var check = C.button('Check connection', { small: true, onClick: s.onCheck || null });

    var toggle = C.button(
      s.starting ? 'Starting…' : (s.owns ? 'Stop local server' : 'Start local server'),
      { small: true, onClick: s.onToggle || null, disabled: !!s.starting || !!s.startDisabled }
    );
    if (s.startTooltip) toggle.title = s.startTooltip;

    var status = el('div', {
      class: 'pa-rail-server-status',
      dataset: { role: 'server-status' },
      text: 'Not connected',
    });

    return el('div', { class: 'pa-rail-server', dataset: { role: 'server-panel' } },
      el('div', { class: ['pa-xs', 'pa-muted'], style: { fontWeight: '600' }, text: 'Server' }),
      url, check, toggle, status
    );
  };

  // --- bottomTabs(views, activeId, onNav, onSettings) -> the mobile tab bar -
  C.bottomTabs = function (views, activeId, onNav, onSettings) {
    var tabs = views.map(function (v) {
      return el('button', {
        class: ['pa-tab', v.id === activeId ? 'is-active' : ''],
        onClick: function () { onNav && onNav(v.id); },
      }, icon(v.icon || 'circle'), el('span', { text: v.title }));
    });
    if (onSettings) {
      tabs.push(el('button', {
        class: 'pa-tab',
        onClick: function () { onSettings(); },
      }, icon('settings'), el('span', { text: 'Settings' })));
    }
    return el('div', { class: 'pa-tabs' }, tabs);
  };

  // --- topBar(title, rightNodes) ------------------------------------------
  C.topBar = function (title, rightNodes) {
    return el('div', { class: 'pa-topbar' },
      el('div', { class: 'pa-topbar-title', text: title || '' }),
      el('div', { class: 'pa-topbar-right' }, rightNodes || [])
    );
  };

  // --- chip({icon,label,tone,active,onClick}) -----------------------------
  C.chip = function (opts) {
    opts = opts || {};
    return el('button', {
      class: ['pa-chip', opts.active ? 'is-active' : '', opts.tone ? 'pa-chip-' + opts.tone : ''],
      onClick: opts.onClick || null,
      type: 'button',
    }, opts.icon ? icon(opts.icon) : null, el('span', { text: opts.label || '' }));
  };

  // --- badge(text, tone) ---------------------------------------------------
  // tones: neutral | accent | good | bad | tier-core | tier-long |
  //        tier-inherited | trust-trusted | trust-untrusted
  C.badge = function (text, tone) {
    return el('span', { class: ['pa-badge', 'pa-badge-' + (tone || 'neutral')], text: text });
  };

  // --- metricCard(label, value, sub) --------------------------------------
  C.metricCard = function (label, value, sub) {
    return el('div', { class: 'pa-metric' },
      el('div', { class: 'pa-metric-label', text: label }),
      el('div', { class: 'pa-metric-value', text: value }),
      sub ? el('div', { class: 'pa-metric-sub', text: sub }) : null
    );
  };

  // --- table({cols, rows, onRowClick}) --------------------------------------
  // cols: [{key,label,render?(row)->node|string}]  rows: object[]
  // onRowClick(rowIndex, row): makes rows clickable (benchmark case diff).
  C.table = function (opts) {
    opts = opts || {};
    var cols = opts.cols || [];
    var rows = opts.rows || [];

    var thead = el('thead', {}, el('tr', {},
      cols.map(function (c) { return el('th', { text: c.label != null ? c.label : c.key }); })
    ));

    var tbody = el('tbody', {}, rows.map(function (row, ri) {
      var tr = el('tr', {}, cols.map(function (c) {
        var content = c.render ? c.render(row) : (row[c.key] != null ? String(row[c.key]) : '');
        return el('td', {}, typeof content === 'string' || typeof content === 'number'
          ? document.createTextNode(String(content))
          : content);
      }));
      if (opts.onRowClick) {
        tr.style.cursor = 'pointer';
        tr.addEventListener('click', function () { opts.onRowClick(ri, row); });
      }
      return tr;
    }));

    return el('table', { class: 'pa-table' }, thead, tbody);
  };

  // --- toggle(on, onChange, labels) ---------------------------------------
  // labels: [offLabel, onLabel] segmented switch. onChange(newOn).
  C.toggle = function (on, onChange, labels) {
    labels = labels || ['Off', 'On'];
    var wrap = el('div', { class: 'pa-toggle', role: 'group' });
    [0, 1].forEach(function (idx) {
      var isOn = idx === 1;
      wrap.appendChild(el('button', {
        class: (isOn === !!on) ? 'is-on' : '',
        type: 'button',
        text: labels[idx],
        onClick: function () { if (!!on !== isOn) onChange && onChange(isOn); },
      }));
    });
    return wrap;
  };

  // --- field(label, controlNode, hint) ------------------------------------
  C.field = function (label, controlNode, hint) {
    return el('label', { class: 'pa-field' },
      el('span', { class: 'pa-field-label', text: label }),
      controlNode,
      hint ? el('span', { class: 'pa-field-hint', text: hint }) : null
    );
  };

  // --- button(label, {primary,ghost,icon,onClick,disabled,small}) ---------
  C.button = function (label, opts) {
    opts = opts || {};
    var cls = ['pa-btn'];
    if (opts.primary) cls.push('pa-btn-primary');
    if (opts.ghost) cls.push('pa-btn-ghost');
    if (opts.small) cls.push('pa-btn-sm');
    return el('button', {
      class: cls,
      type: 'button',
      disabled: !!opts.disabled,
      onClick: opts.onClick || null,
    }, opts.icon ? icon(opts.icon) : null, label != null ? el('span', { text: label }) : null);
  };

  // --- emptyState(icon, title, body, cta) ---------------------------------
  // cta: optional DOM node (e.g. a button).
  C.emptyState = function (iconName, title, body, cta) {
    return el('div', { class: 'pa-empty' },
      icon(iconName || 'inbox'),
      el('div', { class: 'pa-empty-title', text: title || '' }),
      body ? el('div', { class: 'pa-empty-body', text: body }) : null,
      cta || null
    );
  };

  // --- openModal(title, bodyNode, opts) -> { root, body, close } -----------
  // A centered modal over a dimmed backdrop, appended to <body>. Clicking the
  // backdrop or the ✕ closes it. opts: { wide } for the 720px variant.
  C.openModal = function (title, bodyNode, opts) {
    opts = opts || {};
    var handle = {};

    var body = el('div', { class: 'pa-modal-body' }, bodyNode || null);
    var closeBtn = el('button', {
      class: ['pa-btn', 'pa-btn-ghost', 'pa-btn-sm'],
      type: 'button',
      onClick: function () { handle.close(); },
    }, icon('x'));

    var modal = el('div', { class: ['pa-modal', opts.wide ? 'pa-modal-wide' : ''] },
      el('div', { class: 'pa-modal-head' },
        el('span', { class: 'pa-grow', text: title || '' }),
        closeBtn
      ),
      body
    );

    var backdrop = el('div', {
      class: 'pa-modal-backdrop',
      onClick: function (ev) { if (ev.target === backdrop) handle.close(); },
    }, modal);

    handle.root = backdrop;
    handle.body = body;
    handle.close = function () {
      if (backdrop.parentNode) backdrop.parentNode.removeChild(backdrop);
    };

    document.body.appendChild(backdrop);
    return handle;
  };

  // --- progress() -> { root, set(pct), setStatus(text, tone) } --------------
  // A thin progress bar + a status line beneath it (train / benchmark polls).
  C.progress = function () {
    var fill = el('div', { class: 'pa-progress-fill' });
    var bar = el('div', { class: 'pa-progress' }, fill);
    var status = el('div', { class: ['pa-xs', 'pa-tone-muted'] });
    var root = el('div', { class: ['pa-col', 'pa-gap-3'] }, bar, status);
    return {
      root: root,
      set: function (pct) {
        pct = Math.max(0, Math.min(100, Number(pct) || 0));
        fill.style.width = pct + '%';
      },
      setStatus: function (text, tone) {
        status.textContent = text || '';
        status.className = 'pa-xs ' + (tone === 'good' ? 'pa-tone-good' : tone === 'bad' ? 'pa-tone-bad' : 'pa-tone-muted');
      },
    };
  };

  PA.components = C;
})(window.PA);
