/* ============================================================================
   api.js — PA.api: THE ONLY FILE ALLOWED TO USE fetch() / WebSocket.
   One method per server operation. Every call reads PA.config.baseUrl fresh,
   so switching servers via the harness toolbar takes effect immediately.

   Every route below exists on the server today (ProjectAI/Server.cs):
     GET  /health                    POST /generate         WS   /chat
     POST /tokenize                  POST /train            GET  /train/status
     GET  /benchmark/suites          POST /benchmark        GET  /benchmark/status
     POST /benchmark/cancel          GET  /benchmark/runs   GET  /benchmark/run/{id}
     GET  /memory                    GET  /memory/render    PUT  /memory
     GET  /config                    PUT  /config
     PUT  /config/secrets/{key}      DELETE /config/secrets/{key}
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

  // GET/POST/PUT/DELETE JSON with a timeout; throws on network / non-2xx.
  // The thrown Error carries .status and .data (the parsed body — e.g. the
  // {problems:[…]} a 400 from PUT /config returns).
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
      .catch(function (e) {
        // The server's CORS preflight deliberately allows only GET/POST (anti drive-by protection), so a
        // cross-origin PUT/DELETE from this harness fails as an opaque "Failed to fetch". Name the real cause —
        // same-UI-explicit-not-possible is the parity rule for browser-impossible actions.
        if ((method === 'PUT' || method === 'DELETE') && e instanceof TypeError) {
          throw new Error('cross-origin ' + method + ' writes are blocked by the server’s CORS policy — ' +
            'use the desktop app for writes, or serve this harness from the same origin');
        }
        throw e;
      })
      .finally(function () { clearTimeout(t); });
  }

  // === health ================================================================
  // GET /health -> normalized { ok, models, modelInfos, default, backends,
  //                             defaultBackend, sizes, training, bench }
  api.health = function () {
    return request('GET', '/health', undefined, 8000)
      .then(function (d) {
        return {
          ok: true,
          models: d.models || [],
          modelInfos: d.modelInfos || [],
          default: d.default || null,
          backends: d.backends || [],
          defaultBackend: d.defaultBackend || null,
          sizes: d.sizes || [],
          training: d.training || null,
          bench: d.bench || null,
        };
      })
      .catch(function (e) {
        return { ok: false, error: e.message || String(e) };
      });
  };

  // === generate (non-streaming; chat uses the WS below) =====================
  // POST /generate -> { text, promptTokens, generatedTokens, stop, seconds, sources }
  api.generate = function (req) {
    return request('POST', '/generate', req || {}, 120000).then(function (d) {
      return {
        text: d.text || '',
        promptTokens: d.promptTokens != null ? d.promptTokens : null,
        generatedTokens: d.generatedTokens != null ? d.generatedTokens : null,
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
   * handlers: { onReady(info), onToken(text), onSources(list), onDone(info),
   *             onError(err), onOpen(), onClose() }
   * Protocol:
   *   send    {type:'start', model, backend, memory?, user?}      — memory rides
   *           the start frame ONLY (store omitted → server defaults to the model name)
   *           {type:'message', text, sample?, temperature?, topK?, topP?,
   *            maxTokens?, seed?, research?}
   *           {type:'cancel'}
   *   receive {type:'ready', model, backend, instruct, contextLimit} |
   *           {type:'token', text} | {type:'sources', items} |
   *           {type:'done', stop, promptTokens, generatedTokens, seconds,
   *            position, contextLimit} | {type:'error', error}
   * Note: the server spells the cancel stop reason 'canceled'.
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
          case 'ready':   handlers.onReady && handlers.onReady(msg); break;
          case 'token':   handlers.onToken && handlers.onToken(msg.text || ''); break;
          // The server sends the list under 'items' (only POST /generate uses top-level 'sources'); accept both.
          case 'sources': handlers.onSources && handlers.onSources(msg.items || msg.sources || []); break;
          case 'done':    handlers.onDone && handlers.onDone(msg); break; // the FULL done object
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

  // === tokenize ==============================================================
  // POST /tokenize {text, model} -> { model, vocab, count, tokens:[{id,text}], decoded }
  api.tokenize = function (req) {
    return request('POST', '/tokenize', req || {}, 15000).then(function (d) {
      return {
        model: d.model || (req && req.model) || null,
        vocab: d.vocab != null ? d.vocab : null,
        count: d.count != null ? d.count : (d.tokens ? d.tokens.length : 0),
        tokens: d.tokens || [],
        decoded: d.decoded || null,
      };
    });
  };

  // === train / status ========================================================
  // POST /train {name, text, size, steps, backend} -> 202 {status:'started', …}
  api.train = function (req) {
    return request('POST', '/train', req || {}, 30000).then(function (d) {
      return { ok: true, name: (d && d.name) || null, error: null };
    });
  };

  // GET /train/status -> the payload nests under 'training'; unwrap before
  // reading (tolerating a flat shape too).
  api.trainStatus = function () {
    return request('GET', '/train/status', undefined, 8000).then(function (raw) {
      var d = (raw && raw.training) ? raw.training : (raw || {});
      return {
        state: d.state || 'idle',
        name: d.name || null,
        step: d.step != null ? d.step : 0,
        totalSteps: d.totalSteps != null ? d.totalSteps : 0,
        loss: d.loss != null ? d.loss : null,
        error: d.error || null,
      };
    });
  };

  // === benchmark =============================================================
  // GET /benchmark/suites -> [{id,label,caseCount,hasCorpus}]
  api.benchSuites = function () {
    return request('GET', '/benchmark/suites', undefined, 8000).then(function (d) {
      return (d && d.suites) || [];
    });
  };

  // POST /benchmark {suite, models, backend, repeats} -> 202 { runId, total }
  api.benchStart = function (req) {
    return request('POST', '/benchmark', req || {}, 15000).then(function (d) {
      return { runId: (d && d.runId) || '', total: (d && d.total) || 0 };
    });
  };

  // GET /benchmark/status -> the payload nests under 'bench'; unwrap.
  api.benchStatus = function () {
    return request('GET', '/benchmark/status', undefined, 8000).then(function (raw) {
      var d = (raw && raw.bench) ? raw.bench : (raw || {});
      return {
        state: d.state || 'idle',
        runId: d.runId || '',
        suite: d.suite || '',
        done: d.done != null ? d.done : 0,
        total: d.total != null ? d.total : 0,
        currentModel: d.currentModel || '',
        currentCase: d.currentCase || '',
        error: d.error || null,
      };
    });
  };

  // POST /benchmark/cancel -> { ok }
  api.benchCancel = function () {
    return request('POST', '/benchmark/cancel', {}, 8000).then(function () { return { ok: true }; });
  };

  // GET /benchmark/runs -> [{id,suiteId,models,backend,startedUtc,state,cases}]
  api.benchRuns = function () {
    return request('GET', '/benchmark/runs', undefined, 8000).then(function (d) {
      return (d && d.runs) || [];
    });
  };

  // GET /benchmark/run/{id} -> the camelCased run:
  // { id, suiteId, state, aggregates:[{model,n,meanBpb|null,medianTokPerSec,checkPassRate}],
  //   cells:[{model,caseId,output,generatedTokens,stop,medianTokPerSec,checkPassRate,error}] }
  // (Consumers skip cells with caseId '__bpb__' — bookkeeping, not a case.)
  api.benchRun = function (id) {
    return request('GET', '/benchmark/run/' + encodeURIComponent(id), undefined, 15000);
  };

  // === memory ================================================================
  // The user is always the fixed single-local-user 'default' (client parity).
  function memoryQuery(req) {
    req = req || {};
    return '?user=default&store=' + encodeURIComponent(req.store || 'default') +
      (req.q ? '&q=' + encodeURIComponent(req.q) : '');
  }

  // GET /memory?user=default&store=&q= ->
  // { user, store, count, memories:[{id,title,keys,tier,trust,score,asof}] }
  api.memoryList = function (req) {
    return request('GET', '/memory' + memoryQuery(req), undefined, 8000);
  };

  // GET /memory/render?user=default&store=&q= -> { user, store, bridge, recall }
  api.memoryRender = function (req) {
    return request('GET', '/memory/render' + memoryQuery(req), undefined, 8000);
  };

  // PUT /memory {title, keys, body, tier, trust, user:'default', store} -> { id }
  api.memoryPut = function (draft) {
    return request('PUT', '/memory', draft || {}, 8000);
  };

  // === config + secrets ======================================================
  // GET /config -> { memory:{bridgeCards,bridgeBudget,recallHits,recallBudget},
  //                  secrets:[{key,set,hint,source}] }
  api.configGet = function () {
    return request('GET', '/config', undefined, 8000);
  };

  // PUT /config {memory:{…}} — partial update; echoes the applied state.
  // A 400 rejects with err.data = { problems:[…] }.
  api.configPut = function (patch) {
    return request('PUT', '/config', patch || {}, 8000);
  };

  // PUT /config/secrets/{key} {value} -> masked status { key, set, hint, source }
  api.secretPut = function (key, value) {
    return request('PUT', '/config/secrets/' + encodeURIComponent(key), { value: value }, 8000);
  };

  // DELETE /config/secrets/{key} -> masked status { key, set, hint, source }
  api.secretDelete = function (key) {
    return request('DELETE', '/config/secrets/' + encodeURIComponent(key), undefined, 8000);
  };

  PA.api = api;
})(window.PA);
