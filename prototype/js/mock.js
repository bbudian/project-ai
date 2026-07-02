/* ============================================================================
   mock.js — PA.mock: the IDENTICAL method surface as PA.api, returning canned
   data with the SAME payload shapes so the whole app is usable with no server.

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
  var MODEL_INFOS = [
    { name: 'smollm2-360m',          params: 361820672,  layers: 32, ctx: 8192, vocab: 49152, tokenizer: 'hf',  dtype: 'bf16', step: 0,   instruct: false, fileBytes: 1447283712, error: null },
    { name: 'smollm2-360m-instruct', params: 361820672,  layers: 32, ctx: 8192, vocab: 49152, tokenizer: 'hf',  dtype: 'bf16', step: 0,   instruct: true,  fileBytes: 1447283712, error: null },
    { name: 'smollm2-1.7b',          params: 1711376384, layers: 24, ctx: 8192, vocab: 49152, tokenizer: 'hf',  dtype: 'bf16', step: 0,   instruct: false, fileBytes: 6845505536, error: null },
    { name: 'smollm2-1.7b-instruct', params: 1711376384, layers: 24, ctx: 8192, vocab: 49152, tokenizer: 'hf',  dtype: 'bf16', step: 0,   instruct: true,  fileBytes: 6845505536, error: null },
    { name: 'model',                 params: 6200000,    layers: 4,  ctx: 256,  vocab: 259,   tokenizer: 'bpe', dtype: 'f32',  step: 300, instruct: false, fileBytes: 49960000,   error: null },
  ];
  var MODELS = MODEL_INFOS.map(function (m) { return m.name; });
  var BACKENDS = [
    { id: 'cpu',         label: 'CPU',          available: true },
    { id: 'torch:cuda',  label: 'GPU (CUDA)',   available: true },
    { id: 'torch:metal', label: 'GPU (Metal)',  available: false, reason: 'not on this machine' },
  ];
  var SIZES = [
    { id: 'tiny',   label: '~6M params (quick smoke test)' },
    { id: 'small',  label: '~35M params (minutes on a GPU)' },
    { id: 'medium', label: '~85M params (a coffee on a GPU)' },
    { id: 'large',  label: '~208M params (needs a big GPU)' },
  ];

  var CANNED_REPLY =
    'ProjectAI is a local, self-hosted AI runtime written by hand in C# / .NET 10. ' +
    'One numeric core powers three swappable modules — an LLM, image generation, and 3D-mesh generation — ' +
    'and the whole thing runs on your own machine with no cloud dependency.';

  // Small delay so async feels real.
  function later(value, ms) {
    return new Promise(function (res) { setTimeout(function () { res(value); }, ms || 220); });
  }

  function reject(message, data) {
    var err = new Error(message);
    if (data) err.data = data;
    return Promise.reject(err);
  }

  function infoFor(name) {
    return MODEL_INFOS.filter(function (m) { return m.name === name; })[0] || null;
  }

  // === health ================================================================
  mock.health = function () {
    return later({
      ok: true,
      models: MODELS.slice(),
      modelInfos: MODEL_INFOS.map(function (m) { return Object.assign({}, m); }),
      default: MODELS[1],
      backends: BACKENDS.map(function (b) { return Object.assign({}, b); }),
      defaultBackend: 'torch:cuda',
      sizes: SIZES.map(function (s) { return Object.assign({}, s); }),
      training: Object.assign({}, trainSim),
      bench: { state: benchSim.state },
    }, 180);
  };

  // === generate =============================================================
  mock.generate = function (req) {
    req = req || {};
    var reply = CANNED_REPLY;
    if (req.prompt) reply = 'Re: “' + PA.ui.truncate(req.prompt, 48) + '”\n\n' + reply;
    var sources = req.research ? [
      { title: 'ProjectAI — README', url: 'https://example.com/projectai' },
      { title: 'SmolLM2 model card', url: 'https://example.com/smollm2' },
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
  // Mirrors api.chat's controller + frame shapes: onReady carries the ready
  // payload (model/backend/instruct/contextLimit), tokens preserve whitespace,
  // and onDone receives the FULL done object (stop/promptTokens/generatedTokens/
  // seconds/position/contextLimit). Cancel emits stop:'canceled' — the server's
  // spelling.
  mock.chat = function (handlers) {
    handlers = handlers || {};
    var timer = null;
    var session = { model: MODELS[1], backend: 'torch:cuda', memory: false, instruct: true, contextLimit: 8192, position: 0 };
    var turn = null; // { tokens, i, promptTokens, startedAt }

    function stop() { if (timer) { clearInterval(timer); timer = null; } }

    function finish(stopReason) {
      stop();
      var generated = turn ? turn.emitted : 0;
      var promptTokens = turn ? turn.promptTokens : 0;
      var seconds = turn ? Math.max(0.1, (Date.now() - turn.startedAt) / 1000) : 0.1;
      session.position = Math.min(session.contextLimit, session.position + promptTokens + generated);
      turn = null;
      handlers.onDone && handlers.onDone({
        type: 'done',
        stop: stopReason,
        promptTokens: promptTokens,
        generatedTokens: generated,
        seconds: Math.round(seconds * 1000) / 1000,
        position: session.position,
        contextLimit: session.contextLimit,
      });
    }

    function stream(userText, opts) {
      stop();
      var reply = CANNED_REPLY;
      if (session.memory && /\bmemor/i.test(userText || '')) {
        reply = 'Recalling what you told me earlier — ' + reply;
      }
      var tokens = reply.split(/(\s+)/); // keep whitespace as its own tokens
      turn = {
        tokens: tokens, i: 0, emitted: 0,
        promptTokens: Math.max(3, Math.round((userText || '').length / 4)),
        startedAt: Date.now(),
      };
      if (opts.research) {
        setTimeout(function () {
          handlers.onSources && handlers.onSources([
            { title: 'ProjectAI docs', url: 'https://example.com/docs' },
            { title: 'Web search result', url: 'https://example.com/web' },
          ]);
        }, 250);
      }
      timer = setInterval(function () {
        if (!turn || turn.i >= turn.tokens.length) { finish('eos'); return; }
        handlers.onToken && handlers.onToken(turn.tokens[turn.i]);
        turn.i++;
        turn.emitted++;
      }, 45);
    }

    return {
      start: function (model, backend, o) {
        o = o || {};
        session.model = model || MODELS[1];
        session.backend = backend || 'torch:cuda';
        session.memory = o.memory === true;
        var info = infoFor(session.model);
        session.instruct = info ? !!info.instruct : /instruct/i.test(session.model);
        session.contextLimit = info ? info.ctx : 8192;
        session.position = 0;
        setTimeout(function () {
          handlers.onOpen && handlers.onOpen();
          handlers.onReady && handlers.onReady({
            type: 'ready',
            model: session.model,
            backend: session.backend,
            instruct: session.instruct,
            contextLimit: session.contextLimit,
          });
        }, 120);
      },
      send: function (msg, o) { stream(msg, o || {}); },
      cancel: function () { if (turn) finish('canceled'); }, // the server's spelling
      close: function () { stop(); turn = null; handlers.onClose && handlers.onClose(); },
    };
  };

  // === tokenize =============================================================
  mock.tokenize = function (req) {
    req = req || {};
    var text = req.text || '';
    var pieces = text.length ? text.match(/\S+|\s+/g) : [];
    return later({
      model: req.model || MODELS[1],
      vocab: 49152,
      count: pieces.length,
      tokens: pieces.map(function (p, i) { return { id: 1000 + i, text: p }; }),
      decoded: text,
    }, 150);
  };

  // === train / status ======================================================
  var trainSim = { state: 'idle', name: null, step: 0, totalSteps: 0, loss: null, error: null };
  mock.train = function (req) {
    req = req || {};
    if (trainSim.state === 'running') return reject('a training job is already running');
    if (!req.name) return reject("missing 'name'");
    if (!req.text) return reject("missing 'text' to train on");
    trainSim = { state: 'running', name: req.name, step: 0, totalSteps: req.steps > 0 ? req.steps : 300, loss: 5.6, error: null };
    return later({ ok: true, name: req.name, error: null }, 300);
  };
  mock.trainStatus = function () {
    if (trainSim.state === 'running') {
      trainSim.step = Math.min(trainSim.totalSteps, trainSim.step + Math.ceil(trainSim.totalSteps / 12));
      trainSim.loss = Math.max(0.02, +(trainSim.loss * 0.72).toFixed(3));
      if (trainSim.step >= trainSim.totalSteps) {
        trainSim.state = 'done';
        // The finished model lands in the models dir → appears in the catalog.
        if (MODELS.indexOf(trainSim.name) < 0) {
          MODEL_INFOS.push({
            name: trainSim.name, params: 35000000, layers: 8, ctx: 512, vocab: 259,
            tokenizer: 'bpe', dtype: 'f32', step: trainSim.totalSteps, instruct: false,
            fileBytes: 140000000, error: null,
          });
          MODELS.push(trainSim.name);
        }
      }
    }
    return later(Object.assign({}, trainSim), 120);
  };

  // === benchmark (async simulated run) ======================================
  var BENCH_SUITES = [
    { id: 'baseline', label: 'Baseline v1', caseCount: 5, hasCorpus: true },
  ];
  var BENCH_CASES = ['greeting', 'copy-line', 'count-3', 'arithmetic', 'story-open'];
  var benchSim = {
    state: 'idle', runId: '', suite: '', done: 0, total: 0,
    currentModel: '', currentCase: '', error: null,
    models: [], backend: '', repeats: 3,
  };
  var benchRuns = [];   // summaries, newest first
  var benchDetails = {}; // id -> full run

  function buildRunDetail() {
    var models = benchSim.models;
    var aggregates = models.map(function (m, mi) {
      return {
        model: m,
        n: benchSim.repeats,
        meanBpb: BENCH_SUITES[0].hasCorpus ? +(1.0 + mi * 0.2 + Math.random() * 0.05).toFixed(4) : null,
        medianTokPerSec: +(42 - mi * 9 + Math.random() * 4).toFixed(2),
        checkPassRate: +(Math.max(0, 0.9 - mi * 0.15)).toFixed(2),
      };
    });
    var cells = [];
    models.forEach(function (m, mi) {
      BENCH_CASES.forEach(function (c, ci) {
        var pass = !(ci === 3 && mi > 0);
        cells.push({
          model: m,
          caseId: c,
          output: pass
            ? 'A plausible ' + c + ' answer from ' + m + '. ' + CANNED_REPLY.slice(0, 90) + '…'
            : 'An off-target continuation that misses the ' + c + ' check.',
          generatedTokens: 24 + ci * 8,
          stop: ci === 4 ? 'maxTokens' : 'eos',
          medianTokPerSec: +(40 - mi * 8 - ci + Math.random() * 3).toFixed(2),
          checkPassRate: pass ? 1 : 0,
          error: null,
        });
      });
      // The bookkeeping bpb cell consumers must skip.
      cells.push({
        model: m, caseId: '__bpb__', output: '', generatedTokens: 0, stop: '',
        medianTokPerSec: 0, checkPassRate: 0, error: null,
      });
    });
    return {
      id: benchSim.runId,
      suiteId: benchSim.suite,
      state: 'done',
      aggregates: aggregates,
      cells: cells,
    };
  }

  mock.benchSuites = function () {
    return later(BENCH_SUITES.map(function (s) { return Object.assign({}, s); }), 150);
  };

  mock.benchStart = function (req) {
    req = req || {};
    if (benchSim.state === 'running') return reject('a benchmark run is already in progress');
    if (!req.models || !req.models.length) return reject("missing 'models'");
    if (trainSim.state === 'running') return reject('a training job is in progress; try again when it finishes');
    benchSim = {
      state: 'running',
      runId: 'run-' + new Date().toISOString().replace(/[-:T]/g, '').slice(0, 12),
      suite: req.suite || 'baseline',
      done: 0,
      total: req.models.length * (BENCH_CASES.length + 1), // + the bpb pass per model
      currentModel: req.models[0],
      currentCase: BENCH_CASES[0],
      error: null,
      models: req.models.slice(),
      backend: req.backend || 'torch:cuda',
      repeats: req.repeats > 0 ? req.repeats : 3,
    };
    return later({ runId: benchSim.runId, total: benchSim.total }, 250);
  };

  mock.benchStatus = function () {
    if (benchSim.state === 'running') {
      benchSim.done = Math.min(benchSim.total, benchSim.done + 2);
      var perModel = BENCH_CASES.length + 1;
      var mi = Math.min(benchSim.models.length - 1, Math.floor(benchSim.done / perModel));
      var ci = Math.min(BENCH_CASES.length - 1, benchSim.done % perModel);
      benchSim.currentModel = benchSim.models[mi];
      benchSim.currentCase = benchSim.done % perModel === BENCH_CASES.length ? 'bpb' : BENCH_CASES[ci];
      if (benchSim.done >= benchSim.total) {
        benchSim.state = 'done';
        var run = buildRunDetail();
        benchDetails[run.id] = run;
        benchRuns.unshift({
          id: run.id,
          suiteId: run.suiteId,
          models: benchSim.models.slice(),
          backend: benchSim.backend,
          startedUtc: new Date().toISOString(),
          state: 'done',
          cases: BENCH_CASES.length,
        });
      }
    }
    return later({
      state: benchSim.state,
      runId: benchSim.runId,
      suite: benchSim.suite,
      done: benchSim.done,
      total: benchSim.total,
      currentModel: benchSim.currentModel,
      currentCase: benchSim.currentCase,
      error: benchSim.error,
    }, 120);
  };

  mock.benchCancel = function () {
    if (benchSim.state === 'running') benchSim.state = 'canceled';
    return later({ ok: true }, 100);
  };

  mock.benchRuns = function () {
    return later(benchRuns.map(function (r) { return Object.assign({}, r); }), 150);
  };

  mock.benchRun = function (id) {
    var run = benchDetails[id];
    return run ? later(JSON.parse(JSON.stringify(run)), 200) : reject("no run '" + id + "'");
  };

  // === memory ==============================================================
  // Per-store catalogs (chat's convention: one store per model + 'default').
  var memSeq = 4;
  var MEM_STORES = {
    'default': [
      { id: 'mem-0001', title: 'User prefers concise, technical answers.', keys: ['style', 'answers'], tier: 'core', trust: 'curated', score: 0, asof: '2026-06-30' },
      { id: 'mem-0002', title: 'Building ProjectAI, a hand-written C#/.NET 10 AI runtime.', keys: ['projectai', 'dotnet'], tier: 'long', trust: 'curated', score: 0, asof: '2026-06-28' },
      { id: 'mem-0003', title: 'The main workstation has an RTX 4090; the dev laptop an 8GB 3070.', keys: ['gpu', 'hardware'], tier: 'long', trust: 'chat', score: 0, asof: '2026-06-22' },
      { id: 'mem-0004', title: 'A web page claimed the user lives in Oslo.', keys: ['location'], tier: 'session', trust: 'untrusted', score: 0, asof: '2026-06-15' },
    ],
  };
  function memStore(name) {
    if (!MEM_STORES[name]) MEM_STORES[name] = [];
    return MEM_STORES[name];
  }
  function memMatches(m, q) {
    var needle = String(q).toLowerCase();
    if (m.title.toLowerCase().indexOf(needle) >= 0) return true;
    return m.keys.some(function (k) { return k.toLowerCase().indexOf(needle) >= 0; });
  }

  mock.memoryList = function (req) {
    req = req || {};
    var storeId = req.store || 'default';
    var all = memStore(storeId);
    var hits = req.q ? all.filter(function (m) { return memMatches(m, req.q); }) : all;
    return later({
      user: 'default',
      store: storeId,
      count: all.length,
      memories: hits.map(function (m) { return Object.assign({}, m, { keys: m.keys.slice() }); }),
    }, 200);
  };

  mock.memoryRender = function (req) {
    req = req || {};
    var storeId = req.store || 'default';
    var all = memStore(storeId);
    var core = all.filter(function (m) { return m.tier === 'core' && m.trust !== 'untrusted'; });
    var bridge = core.length
      ? '<memory-bridge>\n' + core.map(function (m) { return '- ' + m.title; }).join('\n') + '\n</memory-bridge>\n'
      : '';
    var recall = '';
    if (req.q) {
      var hits = all.filter(function (m) { return m.tier !== 'core' && memMatches(m, req.q); }).slice(0, 6);
      recall = hits.length
        ? '<memory-recall>\n' + hits.map(function (m) { return '- ' + m.title + ' (as of ' + m.asof + ')'; }).join('\n') + '\n</memory-recall>\n'
        : '';
    }
    return later({ user: 'default', store: storeId, bridge: bridge, recall: recall }, 200);
  };

  mock.memoryPut = function (draft) {
    draft = draft || {};
    if (!(draft.title || draft.body)) return reject("a memory needs a 'title' or a 'body'");
    var tier = draft.tier || 'long';
    var trust = draft.trust || 'chat';
    if (['core', 'long', 'session'].indexOf(tier) < 0) return reject('tier must be one of: core, long, session');
    if (['curated', 'chat', 'untrusted'].indexOf(trust) < 0) return reject('trust must be one of: curated, chat, untrusted');
    var id = 'mem-' + String(++memSeq).padStart(4, '0');
    memStore(draft.store || 'default').unshift({
      id: id,
      title: draft.title || PA.ui.truncate(draft.body, 60),
      keys: (draft.keys || []).slice(),
      tier: tier,
      trust: trust,
      score: 0,
      asof: new Date().toISOString().slice(0, 10),
    });
    return later({ id: id }, 200);
  };

  // === config + secrets (masked; the raw key is never echoed) ==============
  var CONFIG = { memory: { bridgeCards: 16, bridgeBudget: 2000, recallHits: 6, recallBudget: 1600 } };
  var SECRETS = { tavily: { set: true, hint: '…3f2a', source: 'config' } };

  function secretStatus(key) {
    var s = SECRETS[key] || { set: false, hint: '', source: '' };
    return { key: key, set: s.set, hint: s.hint, source: s.source };
  }
  function configPayload() {
    return {
      memory: Object.assign({}, CONFIG.memory),
      secrets: Object.keys(SECRETS).map(secretStatus),
    };
  }

  mock.configGet = function () { return later(configPayload(), 160); };

  mock.configPut = function (patch) {
    patch = patch || {};
    var m = patch.memory || {};
    var problems = [];
    var next = Object.assign({}, CONFIG.memory);
    [['bridgeCards', 0, 200], ['bridgeBudget', 0, 100000], ['recallHits', 0, 64], ['recallBudget', 0, 100000]]
      .forEach(function (spec) {
        var key = spec[0];
        if (m[key] == null) return;
        var v = Number(m[key]);
        if (isNaN(v) || v < spec[1] || v > spec[2]) problems.push(key + ' must be in [' + spec[1] + ', ' + spec[2] + ']');
        else next[key] = Math.round(v);
      });
    if (problems.length) return reject('HTTP 400', { problems: problems });
    CONFIG.memory = next;
    return later(configPayload(), 180);
  };

  mock.secretPut = function (key, value) {
    if (key !== 'tavily') return reject("unknown secret '" + key + "' (known: tavily)");
    if (!value) return reject("missing 'value'");
    SECRETS[key] = { set: true, hint: '…' + String(value).slice(-4), source: 'config' };
    return later(secretStatus(key), 180);
  };

  mock.secretDelete = function (key) {
    if (key !== 'tavily') return reject("unknown secret '" + key + "' (known: tavily)");
    SECRETS[key] = { set: false, hint: '', source: '' };
    return later(secretStatus(key), 150);
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
