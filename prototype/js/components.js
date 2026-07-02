/* ============================================================================
   components.js — PA.components: reusable renderers returning DOM nodes.
   PURE VIEW: no I/O, no fetch, no PA.api/PA.mock. Compose from PA.ui.el and the
   token-driven classes in base.css. Colors come only from CSS variables.
   ========================================================================== */
(function (PA) {
  var el = PA.ui.el;
  var icon = PA.ui.icon;
  var C = {};

  // --- navRail(views, activeId, onNav) -> the desktop labeled rail ---------
  // views: [{id,title,icon,order}]  onNav(id)
  C.navRail = function (views, activeId, onNav) {
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

    var foot = el('div', { class: 'pa-rail-foot' },
      el('span', { class: 'pa-status-dot', dataset: { role: 'status-dot' } }),
      el('span', { dataset: { role: 'status-text' }, text: 'Not connected' })
    );

    return el('nav', { class: 'pa-rail' }, brand, items, foot);
  };

  // --- bottomTabs(views, activeId, onNav) -> the mobile tab bar ------------
  C.bottomTabs = function (views, activeId, onNav) {
    var tabs = views.map(function (v) {
      return el('button', {
        class: ['pa-tab', v.id === activeId ? 'is-active' : ''],
        onClick: function () { onNav && onNav(v.id); },
      }, icon(v.icon || 'circle'), el('span', { text: v.title }));
    });
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

  // --- table({cols, rows}) -------------------------------------------------
  // cols: [{key,label,render?(row)->node|string}]  rows: object[]
  C.table = function (opts) {
    opts = opts || {};
    var cols = opts.cols || [];
    var rows = opts.rows || [];

    var thead = el('thead', {}, el('tr', {},
      cols.map(function (c) { return el('th', { text: c.label != null ? c.label : c.key }); })
    ));

    var tbody = el('tbody', {}, rows.map(function (row) {
      return el('tr', {}, cols.map(function (c) {
        var content = c.render ? c.render(row) : (row[c.key] != null ? String(row[c.key]) : '');
        return el('td', {}, typeof content === 'string' || typeof content === 'number'
          ? document.createTextNode(String(content))
          : content);
      }));
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

  PA.components = C;
})(window.PA);
