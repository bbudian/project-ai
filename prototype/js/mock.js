/* ============================================================================
   mock.js — PA.mock: the IDENTICAL method surface as PA.api, returning canned
   data with realistic shapes so the whole app is usable with no server.

   PA.data() (defined at the bottom) is THE visuals/api isolation seam:
   ---------------------------------------------------------------------------
   VIEWS AND COMPONENTS CALL ONLY PA.data(). They must NEVER touch PA.api or
   PA.mock directly, and must NEVER call fetch/WebSocket. PA.data() returns
   PA.api when Live, PA.mock when Mock — that one indirection is the entire
   boundary between visuals and API integration.
   ---------------------------------------------------------------------------
   ========================================================================== */
(function (PA) {
  var mock = {};

  // --- canned catalogs ------------------------------------------------------
  var MODELS = ['smollm2-360m', 'smollm2-360m-instruct', 'smollm2-1.7b', 'smollm2-1.7b-instruct', 'model'];
  var BACKENDS = [
    { id: 'cpu',         label: 'CPU',          available: true },
    { id: 'torch:cuda',  label: 'GPU (CUDA)',   available: true },
    { id: 'torch:metal', label: 'GPU (Metal)',  available: false, reason: 'not on this machine' },
  ];
  var SIZES = [
    { id: 'tiny',   label: 'Tiny (~6M)' },
    { id: 'small',  label: 'Small (~35M)' },
    { id: 'medium', label: 'Medium (~85M)' },
    { id: 'large',  label: 'Large (~208M)' },
  ];

  var CANNED_REPLY =
    'ProjectAI is a local, self-hosted AI runtime written by hand in C# / .NET 10. ' +
    'One numeric core powers three swappable modules — an LLM, image generation, and 3D-mesh generation — ' +
    'and the whole thing runs on your own machine with no cloud dependency.';

  // Small delay so async feels real.
  function later(value, ms) {
    return new Promise(function (res) { setTimeout(function () { res(value); }, ms || 220); });
  }

  // === Same surface as api.health ===========================================
  mock.health = function () {
    return later({
      ok: true,
      models: MODELS.slice(),
      default: MODELS[1],
      backends: BACKENDS.map(function (b) { return Object.assign({}, b); }),
      defaultBackend: 'torch:cuda',
      sizes: SIZES.slice(),
      training: { state: 'idle' },
    }, 180);
  };

  // === generate =============================================================
  mock.generate = function (req) {
    req = req || {};
    var reply = CANNED_REPLY;
    if (req.prompt) reply = 'Re: “' + PA.ui.truncate(req.prompt, 48) + '”\n\n' + reply;
    var sources = req.research ? [
      { title: 'ProjectAI — README', url: 'https://example.com/projectai', snippet: 'A local, self-hosted AI runtime…' },
      { title: 'SmolLM2 model card', url: 'https://example.com/smollm2', snippet: 'Compact language models…' },
    ] : [];
    return later({
      text: reply,
      promptTokens: req.prompt ? Math.max(3, Math.round(req.prompt.length / 4)) : 8,
      generatedTokens: Math.round(reply.length / 4),
      stop: 'eos',
      seconds: 1.4,
      sources: sources,
      model: req.model || MODELS[1],
      backend: req.backend || 'torch:cuda',
    }, 500);
  };

  // === chat (SIMULATED streaming) ===========================================
  // Emits tokens of a canned reply on an interval, then onDone. Mirrors the
  // controller shape of api.chat so views swap between them with no changes.
  mock.chat = function (handlers) {
    handlers = handlers || {};
    var timer = null;
    var opts = {};

    function stream(userText) {
      stop();
      var reply = CANNED_REPLY;
      if (/\bmemor/i.test(userText || '')) {
        reply = 'Recalling what you told me earlier — ' + reply;
      }
      var tokens = reply.split(/(\s+)/); // keep whitespace as its own tokens
      var i = 0;
      handlers.onOpen && handlers.onOpen();
      if (opts.research) {
        setTimeout(function () {
          handlers.onSources && handlers.onSources([
            { title: 'ProjectAI docs', url: 'https://example.com/docs', snippet: 'Architecture overview…' },
            { title: 'Web search result', url: 'https://example.com/web', snippet: 'Relevant background…' },
          ]);
        }, 250);
      }
      timer = setInterval(function () {
        if (i >= tokens.length) {
          stop();
          handlers.onDone && handlers.onDone({
            stop: 'eos',
            seconds: 1.2,
            generatedTokens: Math.round(reply.length / 4),
          });
          return;
        }
        handlers.onToken && handlers.onToken(tokens[i]);
        i++;
      }, 45);
    }

    function stop() { if (timer) { clearInterval(timer); timer = null; } }

    return {
      start: function (model, backend, o) { opts = o || {}; /* ready */ },
      send: function (msg, o) { if (o) opts = Object.assign({}, opts, o); stream(msg); },
      cancel: function () { stop(); handlers.onDone && handlers.onDone({ stop: 'cancelled', seconds: 0.3 }); },
      close: function () { stop(); handlers.onClose && handlers.onClose(); },
    };
  };

  // === tokenize =============================================================
  mock.tokenize = function (req) {
    var text = (req && req.text) || '';
    var pieces = text.length ? text.match(/\S+|\s+/g) : [];
    return later({
      count: pieces.length,
      tokens: pieces.map(function (p, i) { return { id: 1000 + i, text: p }; }),
      decoded: text,
    }, 150);
  };

  // === train / status ======================================================
  var trainSim = { state: 'idle', name: null, step: 0, totalSteps: 0, loss: null };
  mock.train = function (req) {
    trainSim = { state: 'running', name: (req && req.name) || 'my-model', step: 0, totalSteps: (req && req.steps) || 200, loss: 5.6 };
    return later({ ok: true, error: null }, 300);
  };
  mock.trainStatus = function () {
    if (trainSim.state === 'running') {
      trainSim.step = Math.min(trainSim.totalSteps, trainSim.step + Math.ceil(trainSim.totalSteps / 12));
      trainSim.loss = Math.max(0.02, +(trainSim.loss * 0.72).toFixed(3));
      if (trainSim.step >= trainSim.totalSteps) trainSim.state = 'done';
    }
    return later(Object.assign({ error: null }, trainSim), 120);
  };

  // === bench (canned comparison) ===========================================
  mock.bench = function (req) {
    req = req || {};
    var models = req.models || [MODELS[1], MODELS[3]];
    var cases = ['factual-qa', 'summarize', 'code-explain', 'reasoning'];
    return later({
      suite: req.suite || 'smoke',
      backend: req.backend || 'torch:cuda',
      models: models.map(function (m, mi) {
        return {
          model: m,
          tokensPerSec: +(42 - mi * 9 + Math.random() * 4).toFixed(1),
          latencyMs: Math.round(180 + mi * 90),
          passRate: +(0.9 - mi * 0.12).toFixed(2),
        };
      }),
      cases: cases.map(function (c, ci) {
        return {
          name: c,
          results: models.map(function (m, mi) {
            var pass = !(ci === 3 && mi === 1);
            return {
              model: m,
              pass: pass,
              tokensPerSec: +(40 - mi * 8 - ci + Math.random() * 3).toFixed(1),
              judge: pass ? 'correct' : 'off-topic',
            };
          }),
        };
      }),
    }, 600);
  };
  mock.benchStatus = function () { return later({ state: 'idle', progress: 1 }, 120); };

  // === memory ==============================================================
  var MEMORIES = [
    { id: 'm1', tier: 'core', trust: 'trusted', text: 'User prefers concise, technical answers.', provenance: 'user', updated: '2026-06-30' },
    { id: 'm2', tier: 'long', trust: 'trusted', text: 'Building ProjectAI, a hand-written C#/.NET 10 AI runtime.', provenance: 'inferred', updated: '2026-06-28' },
    { id: 'm3', tier: 'long', trust: 'untrusted', text: 'Mentioned a 4090 GPU on the main workstation.', provenance: 'web', updated: '2026-06-22' },
    { id: 'm4', tier: 'inherited', trust: 'trusted', text: 'Default backend is torch:cuda when available.', provenance: 'system', updated: '2026-06-15' },
  ];
  mock.memoryList = function (q) {
    var list = MEMORIES;
    if (q) {
      var needle = String(q).toLowerCase();
      list = list.filter(function (m) { return m.text.toLowerCase().indexOf(needle) >= 0; });
    }
    return later({ store: 'default', memories: list.map(function (m) { return Object.assign({}, m); }) }, 200);
  };
  mock.memoryGet = function (id) {
    var m = MEMORIES.filter(function (x) { return x.id === id; })[0];
    return m ? later(Object.assign({}, m), 120) : Promise.reject(new Error('memory not found: ' + id));
  };
  mock.memoryPut = function (draft) {
    var m = Object.assign({ id: 'm' + (MEMORIES.length + 1), updated: '2026-07-01' }, draft);
    MEMORIES.unshift(m);
    return later({ ok: true, id: m.id }, 200);
  };
  mock.memorySupersede = function (id) {
    MEMORIES = MEMORIES.filter(function (m) { return m.id !== id; });
    return later({ ok: true }, 150);
  };

  // === settings (masked secrets) ===========================================
  var SETTINGS = {
    app: { serverUrl: 'http://localhost:8080', theme: 'dark', defaultModel: 'smollm2-360m-instruct', defaultBackend: 'torch:cuda' },
    model: { maxTokens: 512, temperature: 0.7, topK: 40, topP: 0.9, seed: 0 },
    memory: { enabled: true, store: 'default', autoInject: true, maxRecall: 6 },
    benchmark: { suite: 'smoke', judge: 'self', repeats: 1 },
    backends: { available: BACKENDS.map(function (b) { return Object.assign({}, b); }), selected: 'torch:cuda' },
    // Secret is MASKED and flagged configured; the real key is never sent to the client.
    integrations: { tavilyKeyMasked: 'tvly-••••••••••3f2a', tavilyConfigured: true },
  };
  mock.settingsGet = function () { return later(JSON.parse(JSON.stringify(SETTINGS)), 160); };
  mock.settingsPut = function (patch) {
    // Never store/echo a raw secret key; only accept the masked/configured flags.
    Object.keys(patch || {}).forEach(function (section) {
      if (section === 'tavilyKey') return; // ignore raw key in mock
      SETTINGS[section] = Object.assign({}, SETTINGS[section], patch[section]);
    });
    return later({ ok: true }, 180);
  };

  PA.mock = mock;

  // ==========================================================================
  // PA.data() — THE visuals/api isolation seam. See header comment above.
  // Views and components resolve their data provider through this ONLY.
  // ==========================================================================
  PA.data = function () {
    return PA.config.useMock ? PA.mock : PA.api;
  };
})(window.PA);
