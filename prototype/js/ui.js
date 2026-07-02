/* ============================================================================
   ui.js — PA.ui: pure DOM helpers + tiny formatters.
   NO I/O, NO business logic. Just build and manipulate nodes.
   ========================================================================== */
(function (PA) {
  var ui = {};

  /**
   * el(tag, props?, ...children) -> HTMLElement
   * props supports:
   *   class    : string | string[]        -> className
   *   dataset  : { key: value }           -> data-* attributes
   *   style    : { cssProp: value }       -> inline styles
   *   on*      : function                 -> addEventListener (onClick -> 'click')
   *   html     : string                   -> innerHTML (trusted content only)
   *   text     : string                   -> textContent
   *   <anything else> : set as attribute (or property for value/checked/disabled)
   * children: nodes, strings, arrays, or null/undefined (skipped).
   */
  ui.el = function (tag, props) {
    var node = document.createElement(tag);
    props = props || {};

    Object.keys(props).forEach(function (key) {
      var val = props[key];
      if (val == null) return;

      if (key === 'class' || key === 'className') {
        node.className = Array.isArray(val) ? val.filter(Boolean).join(' ') : val;
      } else if (key === 'dataset') {
        Object.keys(val).forEach(function (dk) { node.dataset[dk] = val[dk]; });
      } else if (key === 'style' && typeof val === 'object') {
        Object.keys(val).forEach(function (sk) { node.style[sk] = val[sk]; });
      } else if (key === 'html') {
        node.innerHTML = val;
      } else if (key === 'text') {
        node.textContent = val;
      } else if (key.slice(0, 2) === 'on' && typeof val === 'function') {
        node.addEventListener(key.slice(2).toLowerCase(), val);
      } else if (key === 'value' || key === 'checked' || key === 'disabled' || key === 'selected') {
        node[key] = val; // properties, not attributes
      } else {
        node.setAttribute(key, val);
      }
    });

    // Append children (variadic; arrays are flattened one level).
    var children = Array.prototype.slice.call(arguments, 2);
    ui.append(node, children);
    return node;
  };

  // Append children (node | string | array | null) to a parent.
  ui.append = function (parent, child) {
    if (child == null) return parent;
    if (Array.isArray(child)) {
      child.forEach(function (c) { ui.append(parent, c); });
    } else if (typeof child === 'string' || typeof child === 'number') {
      parent.appendChild(document.createTextNode(String(child)));
    } else {
      parent.appendChild(child);
    }
    return parent;
  };

  // Remove all children of a node.
  ui.clear = function (node) {
    while (node && node.firstChild) node.removeChild(node.firstChild);
    return node;
  };

  // Replace a parent's contents with node(s).
  ui.mount = function (parent, node) {
    ui.clear(parent);
    ui.append(parent, node);
    return parent;
  };

  // A Tabler icon element: <i class="ti ti-{name}">
  ui.icon = function (name, extra) {
    return ui.el('i', { class: 'ti ti-' + name + (extra ? ' ' + extra : ''), 'aria-hidden': 'true' });
  };

  // --- Formatters (always round displayed floats) --------------------------
  ui.fmtNum = function (n, decimals) {
    if (n == null || isNaN(n)) return '—';
    var d = decimals == null ? 0 : decimals;
    var v = Number(n);
    // Big integers get thousands separators; decimals get fixed precision.
    if (d === 0 && Number.isInteger(v)) return v.toLocaleString();
    return v.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
  };

  ui.fmtSecs = function (s) {
    if (s == null || isNaN(s)) return '—';
    var v = Number(s);
    if (v < 1) return Math.round(v * 1000) + ' ms';
    if (v < 60) return v.toFixed(1) + ' s';
    var m = Math.floor(v / 60), rem = Math.round(v % 60);
    return m + 'm ' + rem + 's';
  };

  ui.truncate = function (str, max) {
    if (str == null) return '';
    str = String(str);
    max = max || 80;
    return str.length > max ? str.slice(0, max - 1) + '…' : str;
  };

  PA.ui = ui;
})(window.PA);
