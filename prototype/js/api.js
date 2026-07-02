/* ============================================================================
   api.js — PA.api: THE ONLY FILE ALLOWED TO USE fetch() / WebSocket.
   One method per server operation. Every call reads PA.config.baseUrl fresh,
   so switching servers via the harness toolbar takes effect immediately.

   Endpoints that DO NOT EXIST on the server yet reject with a clear
   'endpoint not on server yet — use Mock' Error so the app degrades visibly.
   ========================================================================== */
(function (PA) {
  var api = {};

  // --- internal helpers ----------------------------------------------------
  function base() {
    // Trim a trailing slash so base() + '/health' never doubles up.
    return String(PA.config.baseUrl || '').replace(/\/+$/, '');
  }

  function wsBase() {
    return base().replace(/^http/i, 'ws');
  }

  // GET/POST JSON with a timeout; throws on network / non-2xx.
  function request(method, path, body, timeoutMs) {
    var url = base() + path;
    var ctrl = new AbortController();
    var t = setTimeout(function () { ctrl.abort(); }, timeoutMs || 30000);
    var opts = { method: method, signal: ctrl.signal, headers: {} };
    if (body !== undefined) {
      opts.headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(body);
    }
    return fetch(url, opts)
      .then(function (res) {
        return res.text().then(function (text) {
          var data = null;
          if (text) { try { data = JSON.parse(text); } catch (e) { data = { raw: text }; } }
          if (!res.ok) {
            var msg = (data && (data.error || data.message)) || ('HTTP ' + res.status);
            var err = new Error(msg);
            err.status = res.status;
            err.data = data;
            throw err;
          }
          return data;
        });
      })
      .finally(function () { clearTimeout(t); });
  }

  // Reject helper for endpoints the server does not implement yet.
  function notOnServer() {
    return Promise.reject(new Error('endpoint not on server yet — use Mock'));
  }

  // === LIVE endpoints (implemented on the server) ==========================

  // GET /health -> normalized { ok, models, default, backends, defaultBackend, sizes, training }
  api.health = function () {
    return request('GET', '/health', undefined, 8000)
      .then(function (d) {
        return {
          ok: true,
          models: d.models || [],
          default: d.default || null,
          backends: d.backends || [],
          defaultBackend: d.defaultBackend || null,
          sizes: d.sizes || [],
          training: d.training || null,
        };
      })
      .catch(function (e) {
        return { ok: false, error: e.message || String(e) };
      });
  };

  // POST /generate -> { text, promptTokens, generatedTokens, stop, seconds, sources }
  // req may include: prompt, model, backend, memory, user, store, research, decoding{...}
  api.generate = function (req) {
    return request('POST', '/generate', req || {}, 120000).then(function (d) {
      return {
        text: d.text || '',
        promptTokens: d.promptTokens != null ? d.promptTokens : (d.prompt_tokens || null),
        generatedTokens: d.generatedTokens != null ? d.generatedTokens : (d.generated_tokens || null),
        stop: d.stop || null,
        seconds: d.seconds != null ? d.seconds : null,
        sources: d.sources || [],
        model: d.model || (req && req.model) || null,
        backend: d.backend || (req && req.backend) || null,
      };
    });
  };

  /**
   * chat(handlers) -> controller over WS /chat.
   * handlers: { onToken(text), onSources(list), onDone(info), onError(err), onOpen(), onClose() }
   * Protocol:
   *   send    {type:'start', model, backend}
   *           {type:'message', text, ...opts}
   *           {type:'cancel'}
   *   receive {type:'token', text} | {type:'sources', sources} |
   *           {type:'done', ...} | {type:'error', error} | {type:'ready'}
   * Returns { start(model,backend,opts), send(msg,opts), cancel(), close() }.
   */
  api.chat = function (handlers) {
    handlers = handlers || {};
    var ws = null;

    function ensure() {
      if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return ws;
      ws = new WebSocket(wsBase() + '/chat');
      ws.onopen = function () { handlers.onOpen && handlers.onOpen(); };
      ws.onclose = function () { handlers.onClose && handlers.onClose(); };
      ws.onerror = function () { handlers.onError && handlers.onError(new Error('websocket error')); };
      ws.onmessage = function (ev) {
        var msg;
        try { msg = JSON.parse(ev.data); } catch (e) { return; }
        switch (msg.type) {
          case 'token':   handlers.onToken && handlers.onToken(msg.text || ''); break;
          case 'sources': handlers.onSources && handlers.onSources(msg.sources || []); break;
          case 'done':    handlers.onDone && handlers.onDone(msg); break;
          case 'ready':   /* server acked start */ break;
          case 'error':   handlers.onError && handlers.onError(new Error(msg.error || 'chat error')); break;
        }
      };
      return ws;
    }

    function sendJson(obj) {
      var sock = ensure();
      var payload = JSON.stringify(obj);
      if (sock.readyState === WebSocket.OPEN) sock.send(payload);
      else sock.addEventListener('open', function () { sock.send(payload); }, { once: true });
    }

    return {
      start: function (model, backend, opts) {
        sendJson(Object.assign({ type: 'start', model: model, backend: backend }, opts || {}));
      },
      send: function (msg, opts) {
        sendJson(Object.assign({ type: 'message', text: msg }, opts || {}));
      },
      cancel: function () { sendJson({ type: 'cancel' }); },
      close: function () { if (ws) { try { ws.close(); } catch (e) {} ws = null; } },
    };
  };

  // POST /tokenize -> { count, tokens, decoded }
  api.tokenize = function (req) {
    return request('POST', '/tokenize', req || {}, 15000).then(function (d) {
      return { count: d.count != null ? d.count : (d.tokens ? d.tokens.length : 0), tokens: d.tokens || [], decoded: d.decoded || null };
    });
  };

  // POST /train -> { ok, error? }
  api.train = function (req) {
    return request('POST', '/train', req || {}, 15000).then(function (d) {
      return { ok: d.ok !== false, error: d.error || null };
    });
  };

  // GET /train/status -> { state, name, step, totalSteps, loss, error? }
  api.trainStatus = function () {
    return request('GET', '/train/status', undefined, 8000).then(function (d) {
      return {
        state: d.state || 'idle',
        name: d.name || null,
        step: d.step != null ? d.step : 0,
        totalSteps: d.totalSteps != null ? d.totalSteps : (d.total_steps || 0),
        loss: d.loss != null ? d.loss : null,
        error: d.error || null,
      };
    });
  };

  // === Endpoints NOT on the server yet — reject clearly so Mock is used. ====
  api.bench          = function (/* req */)   { return notOnServer(); };
  api.benchStatus    = function ()            { return notOnServer(); };
  api.memoryList     = function (/* q */)     { return notOnServer(); };
  api.memoryGet      = function (/* id */)    { return notOnServer(); };
  api.memoryPut      = function (/* draft */) { return notOnServer(); };
  api.memorySupersede= function (/* id */)    { return notOnServer(); };
  api.settingsGet    = function ()            { return notOnServer(); };
  api.settingsPut    = function (/* patch */) { return notOnServer(); };

  PA.api = api;
})(window.PA);
