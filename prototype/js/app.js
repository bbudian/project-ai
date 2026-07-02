/* ============================================================================
   app.js — the harness runtime: view registry, shell, hash router, and the
   harness-toolbar wiring. Loaded LAST (after all views).

   Adding a screen = drop a views/x.js that calls PA.registerView(...). No edit
   here is needed: the registry is read fresh on every render.
   ========================================================================== */
(function (PA) {
  var el = PA.ui.el;
  var C = PA.components;

  // --- View registry -------------------------------------------------------
  // registerView + PA.views are defined early in config.js (so views, which load
  // before app.js, can self-register at load time). Keep those instances — the
  // views have already populated PA.views by the time this file runs.
  PA.views = PA.views || {};
  PA.registerView = PA.registerView || function (id, def) {
    if (!id || !def || typeof def.render !== 'function') {
      console.error('registerView: invalid view "' + id + '"');
      return;
    }
    PA.views[id] = {
      id: id,
      title: def.title || id,
      icon: def.icon || 'circle',
      order: def.order != null ? def.order : 999,
      render: def.render,
    };
  };

  // Registered views sorted by order (then title) — the nav/tab source of truth.
  function orderedViews() {
    return Object.keys(PA.views)
      .map(function (k) { return PA.views[k]; })
      .sort(function (a, b) { return a.order - b.order || a.title.localeCompare(b.title); });
  }

  // --- App singleton -------------------------------------------------------
  var app = {
    _root: null,        // #app
    _els: {},           // cached shell nodes
    _route: null,       // active view id
    _unsub: null,       // store subscription

    boot: function () {
      app._root = document.getElementById('app');
      app.buildShell();
      app.wireHarness();
      app.syncHarnessControls();

      // Route from the hash (default = first registered view).
      window.addEventListener('hashchange', function () { app.route(app.currentHash()); });

      // Re-render nav status + re-mount the active view whenever the store changes.
      app._unsub = PA.store.subscribe(function () {
        app.refreshNav();
        app.refreshConnControls();
      });

      app.route(app.currentHash());

      // Kick off a health probe through the seam; Mock keeps working if the
      // server is down. Errors are swallowed into store.connError.
      app.refreshHealth();
    },

    // Build the static shell: rail + main(topbar + view container) + bottom tabs.
    buildShell: function () {
      var viewContainer = el('div', { class: 'pa-view', dataset: { role: 'view' } });
      var topbar = el('div', { class: 'pa-topbar', dataset: { role: 'topbar' } });
      var main = el('div', { class: 'pa-main' }, topbar, viewContainer);

      var railHost = el('div', { dataset: { role: 'rail' } });   // holds navRail (re-rendered)
      var tabsHost = el('div', { dataset: { role: 'tabs' } });   // holds bottomTabs (re-rendered)

      var shell = el('div', { class: 'pa-shell' }, railHost, main, tabsHost);
      PA.ui.mount(app._root, shell);

      app._els = { shell: shell, railHost: railHost, tabsHost: tabsHost, main: main, topbar: topbar, viewContainer: viewContainer };
      app.refreshNav();
    },

    // (Re)render the nav rail + bottom tabs from the current registry + route.
    refreshNav: function () {
      var views = orderedViews();
      var active = app._route;
      PA.ui.mount(app._els.railHost, C.navRail(views, active, app.go));
      PA.ui.mount(app._els.tabsHost, C.bottomTabs(views, active, app.go));
      app.refreshConnControls(); // update the rail status line
    },

    // Update the rail's connection status line from the store.
    refreshConnControls: function () {
      var s = PA.store.get();
      var dot = app._els.railHost.querySelector('[data-role="status-dot"]');
      var txt = app._els.railHost.querySelector('[data-role="status-text"]');
      if (!dot || !txt) return;
      dot.className = 'pa-status-dot ' + (s.connected ? 'is-on' : (s.connError ? 'is-off' : ''));
      if (PA.config.useMock) txt.textContent = 'Mock data';
      else if (s.connected) txt.textContent = 'Connected';
      else if (s.connError) txt.textContent = 'Server offline';
      else txt.textContent = 'Not connected';
    },

    // Navigate by setting the hash; the hashchange handler drives route().
    go: function (id) { window.location.hash = '#/' + id; },

    currentHash: function () {
      var m = /^#\/([\w-]+)/.exec(window.location.hash || '');
      return m ? m[1] : null;
    },

    // Mount the view for `id` (or the first registered, or an empty state).
    route: function (id) {
      var views = orderedViews();
      if (!id || !PA.views[id]) id = views.length ? views[0].id : null;

      app._route = id;
      PA.store.set({ route: id });
      app.refreshNav();

      var container = app._els.viewContainer;
      var view = id ? PA.views[id] : null;

      if (!view) {
        // No views registered at all, or the requested one is missing.
        PA.ui.mount(app._els.topbar, C.topBar('ProjectAI', []));
        PA.ui.mount(container, C.emptyState(
          'layout-dashboard',
          id ? ('View “' + id + '” not found') : 'No views registered',
          'Drop a js/views/*.js file that calls PA.registerView(...) and it will appear here automatically.'
        ));
        return;
      }

      // Default top bar (the view may replace it via ctx.setTopBar).
      PA.ui.mount(app._els.topbar, C.topBar(view.title, []));

      var ctx = {
        store: PA.store,
        data: PA.data,               // views fetch through the seam only
        go: app.go,
        route: id,
        // Let a view own its top bar (title + right-side action nodes).
        setTopBar: function (title, rightNodes) {
          PA.ui.mount(app._els.topbar, C.topBar(title != null ? title : view.title, rightNodes || []));
        },
      };

      // Render defensively: a throwing view shows an error card, not a blank app.
      PA.ui.clear(container);
      try {
        view.render(container, ctx);
      } catch (e) {
        console.error('view "' + id + '" render error', e);
        PA.ui.mount(container, C.emptyState('alert-triangle', 'View failed to render', String(e && e.message || e)));
      }
    },

    // --- Health probe (through the seam) -----------------------------------
    refreshHealth: function () {
      return PA.data().health().then(function (h) {
        if (h && h.ok) {
          PA.store.set({
            health: h,
            models: h.models || [],
            backends: h.backends || [],
            sizes: h.sizes || [],
            selectedModel: PA.store.get().selectedModel || h.default || (h.models && h.models[0]) || null,
            selectedBackend: PA.store.get().selectedBackend || h.defaultBackend || null,
            connected: true,
            connError: null,
          });
        } else {
          PA.store.set({ connected: false, connError: (h && h.error) || 'unreachable' });
        }
        return h;
      }).catch(function (e) {
        PA.store.set({ connected: false, connError: e.message || String(e) });
      });
    },

    // --- Harness toolbar ----------------------------------------------------
    wireHarness: function () {
      var h = app._harness = {
        url: document.getElementById('pa-harness-url'),
        connect: document.getElementById('pa-harness-connect'),
        source: document.getElementById('pa-harness-source'),   // segmented Live/Mock
        theme: document.getElementById('pa-harness-theme'),      // Dark/Light
        device: document.getElementById('pa-harness-device'),    // Desktop/Mobile
      };

      // Connect: adopt the URL, persist, and re-probe health through the seam.
      if (h.connect) h.connect.addEventListener('click', function () {
        PA.config.baseUrl = (h.url && h.url.value || '').trim() || PA.config.baseUrl;
        if (h.url) h.url.value = PA.config.baseUrl;
        PA.saveConfig();
        app.refreshHealth();
      });

      // Live / Mock segmented toggle. Buttons carry data-value="live|mock".
      if (h.source) h.source.addEventListener('click', function (ev) {
        var btn = ev.target.closest('button[data-value]');
        if (!btn) return;
        PA.config.useMock = btn.dataset.value === 'mock';
        PA.saveConfig();
        app.syncHarnessControls();
        app.refreshHealth();      // re-probe with the new provider
        app.route(app._route);    // re-render active view against new data source
      });

      // Theme: [data-theme=light] on <html>, persisted separately.
      if (h.theme) h.theme.addEventListener('click', function (ev) {
        var btn = ev.target.closest('button[data-value]');
        if (!btn) return;
        app.setTheme(btn.dataset.value);
      });

      // Device: html.force-mobile + phone frame.
      if (h.device) h.device.addEventListener('click', function (ev) {
        var btn = ev.target.closest('button[data-value]');
        if (!btn) return;
        app.setDevice(btn.dataset.value);
      });

      // Restore theme/device prefs.
      app.setTheme(localStorage.getItem('pa.theme') || 'dark', true);
      app.setDevice(localStorage.getItem('pa.device') || 'desktop', true);
    },

    setTheme: function (theme, silent) {
      theme = theme === 'light' ? 'light' : 'dark';
      if (theme === 'light') document.documentElement.setAttribute('data-theme', 'light');
      else document.documentElement.removeAttribute('data-theme');
      if (!silent) localStorage.setItem('pa.theme', theme);
      else localStorage.setItem('pa.theme', theme);
      app.markSegment(app._harness && app._harness.theme, theme);
    },

    setDevice: function (device, silent) {
      device = device === 'mobile' ? 'mobile' : 'desktop';
      document.documentElement.classList.toggle('force-mobile', device === 'mobile');
      localStorage.setItem('pa.device', device);
      app.markSegment(app._harness && app._harness.device, device);
    },

    // Reflect current config into the harness segmented controls + URL field.
    syncHarnessControls: function () {
      var h = app._harness || {};
      if (h.url) h.url.value = PA.config.baseUrl;
      app.markSegment(h.source, PA.config.useMock ? 'mock' : 'live');
    },

    // Toggle .is-on on a segmented control's buttons by data-value.
    markSegment: function (group, value) {
      if (!group) return;
      Array.prototype.forEach.call(group.querySelectorAll('button[data-value]'), function (b) {
        b.classList.toggle('is-on', b.dataset.value === value);
      });
    },
  };

  PA.app = app;

  // Boot once the DOM is ready.
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () { app.boot(); });
  } else {
    app.boot();
  }
})(window.PA);
