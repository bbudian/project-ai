/* ============================================================================
   store.js — PA.store: a tiny reactive store. NO I/O.
   state holds the app's shared, observable data. Views subscribe to re-render.
   ========================================================================== */
(function (PA) {
  var state = {
    route: null,            // active view id
    health: null,           // last /health payload (raw)
    models: [],             // string[] of model names
    modelInfos: [],         // [{name,params,layers,ctx,vocab,tokenizer,dtype,step,instruct,fileBytes,error}]
    backends: [],           // [{id,label,available,reason?}]
    sizes: [],              // [{id,label}]
    selectedModel: null,    // current chat/gen model
    selectedBackend: null,  // current backend id
    connected: false,       // reached the server successfully
    connError: null,        // last connection error message
  };

  var subscribers = [];

  var store = {
    // Read the whole state (treat as read-only).
    get: function () { return state; },

    // Shallow-merge a patch into state, then notify all subscribers.
    set: function (patch) {
      state = Object.assign({}, state, patch);
      subscribers.slice().forEach(function (fn) {
        try { fn(state); } catch (e) { console.error('store subscriber error', e); }
      });
      return state;
    },

    // Subscribe to changes; returns an unsubscribe function.
    subscribe: function (fn) {
      subscribers.push(fn);
      return function unsub() {
        var i = subscribers.indexOf(fn);
        if (i >= 0) subscribers.splice(i, 1);
      };
    },
  };

  PA.store = store;
})(window.PA);
