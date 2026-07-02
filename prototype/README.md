# ProjectAI — Prototyping Harness

A single-page, zero-build **UI prototyping harness** for the ProjectAI runtime. It renders
the whole client — Chat, Models, Benchmark, Memory, plus a Settings modal — as static
HTML/CSS/vanilla JS, so you can design and iterate on screens with **the server completely
off**, then flip a single toggle to run them against a live `projectai serve` instance.

No framework, no bundler, no `npm install`. Just files and a browser.

---

## What this is (and isn't)

- **Is:** a faithful, interactive mock of the desktop + mobile client. Every screen is fully
  usable against canned data. It exists to prototype *information architecture and interaction*
  alongside the C# server, and is kept **1:1 with the shipping Godot client** — the client
  (`Client/ai-client/`) is the source of truth for IA, copy, and behavior.
- **Isn't:** the shipping client. The real front end is the Godot 4.7 C# app in
  `Client/ai-client/`. This harness mirrors that app's palette (`Ui/Palette.cs`) and IA so the
  two stay visually and structurally in sync.

---

## How to open it

Two ways, either works:

1. **Double-click `index.html`.** It runs straight off the `file://` protocol — no server needed.
   In Mock mode (the default) everything renders and is interactive immediately.

2. **Serve the folder** (nicer for iteration; enables live-reload tooling and avoids any
   `file://` quirks). From `prototype/`:

   ```sh
   python -m http.server 5500
   # then open http://localhost:5500
   ```

   or any static server you like (`npx serve`, VS Code "Live Server", etc.).

> The Tabler icon webfont is loaded from a CDN. Online, you get icons; offline, the UI still
> works — icons just degrade to nothing.

---

## Live vs Mock (the toggle that matters)

The **Harness toolbar** across the top (outside the app itself) has a **Live / Mock** segmented
switch:

- **Mock (default):** every screen resolves its data from `js/mock.js` — realistic canned
  payloads with small simulated delays, simulated token streaming for Chat, and async simulated
  background jobs for training and benchmark runs. Nothing hits the network.
- **Live:** data flows to the real server at the toolbar's **base URL** via `js/api.js`
  (`fetch` / `WebSocket`). **Every endpoint the harness uses exists on the server today** —
  there is no mock-only screen anymore.

To point at a server: type its URL into the toolbar (default `http://localhost:8080`), click
**Connect**, then switch to **Live**. The choice is persisted to `localStorage`, so a reload
keeps your last mode and URL.

The toolbar also carries:

- **Dark / Light** theme (persisted; sets `html[data-theme]`).
- **Desktop / Mobile** device preview (persisted; toggles `html.force-mobile`, which swaps the
  desktop nav rail for the mobile bottom-tab bar and frames `#app` as a ~390px phone).

---

## Architecture

Everything hangs off one global, `window.PA`. No ES modules — plain `<script>` tags loaded in
dependency order (see the comments in `index.html`).

### The isolation seam: `api.js` / `mock.js` / `PA.data()`

This is the core discipline of the harness:

```
  views/*.js , components.js
          │
          │  call ONLY  ctx.data()  /  PA.data()
          ▼
      PA.data()  ──►  PA.config.useMock ?  PA.mock  :  PA.api
                                              │            │
                                     canned data      fetch / WebSocket
                                     (mock.js)         (api.js)  ◄── the ONLY file
                                                                     allowed to touch
                                                                     the network
```

- **`js/api.js`** is the *only* file permitted to call `fetch()` or `new WebSocket`. One method
  per server operation. It reads `PA.config.baseUrl` fresh on every call, so switching servers
  takes effect immediately.
- **`js/mock.js`** exposes the **identical method surface** as `api.js`, returning canned data
  with the same payload shapes. It also defines **`PA.data()`** — the seam itself: it returns
  `PA.api` in Live mode and `PA.mock` in Mock mode.
- **Views and components call `PA.data()` (or `ctx.data()`) only.** They must never reference
  `PA.api` / `PA.mock` directly and must never call `fetch` / `WebSocket`. That single
  indirection is the entire boundary between "visuals" and "API integration" — a view cannot
  tell, and does not care, whether it's talking to a server or a mock.

### The seam's method surface (all live on the server)

| Seam method | Server route | Notes |
|---|---|---|
| `health()` | `GET /health` | models + **modelInfos** (real metadata) + backends + sizes + training/bench state |
| `generate(req)` | `POST /generate` | non-streaming one-shot (chat uses the WS) |
| `chat(handlers)` | `WS /chat` | `start` (memory rides this frame only) / `message` / `cancel`; `ready` → `handlers.onReady`, whitespace-preserving `token`s, the FULL `done` object; the server spells the cancel stop `canceled` |
| `tokenize(req)` | `POST /tokenize` | `{text, model}` → count + pieces |
| `train(req)` / `trainStatus()` | `POST /train`, `GET /train/status` | status nests under `training` — the seam unwraps it |
| `benchSuites()` | `GET /benchmark/suites` | `[{id,label,caseCount,hasCorpus}]` |
| `benchStart(req)` | `POST /benchmark` | `{suite,models,backend,repeats}` → 202 `{runId,total}` |
| `benchStatus()` | `GET /benchmark/status` | nests under `bench` — the seam unwraps it |
| `benchCancel()` | `POST /benchmark/cancel` | |
| `benchRuns()` / `benchRun(id)` | `GET /benchmark/runs`, `GET /benchmark/run/{id}` | run cells with `caseId` `__bpb__` are bookkeeping — skip them |
| `memoryList(req)` / `memoryRender(req)` | `GET /memory`, `GET /memory/render` | always `user=default`; stores are per model by convention |
| `memoryPut(draft)` | `PUT /memory` | `{title,keys,body,tier,trust,user,store}` → `{id}` |
| `configGet()` / `configPut(patch)` | `GET /config`, `PUT /config` | memory budgets; a 400 rejects with `err.data.problems` |
| `secretPut(key, value)` / `secretDelete(key)` | `PUT/DELETE /config/secrets/{key}` | write-only; the response is always masked status |

### `tokens.css`: the single source of color

- **`css/tokens.css`** is the *only* file where color / spacing / radius / type literals live
  (`--pa-*` CSS variables). The palette mirrors `Client/ai-client/Ui/Palette.cs` exactly, and the
  light theme is a variable remap under `html[data-theme="light"]`.
- **`css/base.css`** (reset + reusable component classes) and **`css/shell.css`** (responsive
  shell + harness toolbar) reference `var(--pa-*)` only — no literals.
- Views build DOM with `PA.ui` + `PA.components` + these token-driven classes; no view emits a
  color literal. **Restyle the entire harness by editing `tokens.css` alone.**

### The rest of the foundation

| File | Responsibility |
|---|---|
| `js/config.js` | Establishes `window.PA`; loads/persists `{ baseUrl, useMock }`. Loaded first. |
| `js/ui.js` | `PA.ui`: pure DOM helpers (`el`, `mount`, `icon`) + formatters. No I/O. |
| `js/store.js` | `PA.store`: a tiny reactive store (health/models/modelInfos/backends/selection). No I/O. |
| `js/api.js` | `PA.api`: the only network file (see seam above). |
| `js/mock.js` | `PA.mock` + `PA.data()`: canned data + the isolation seam. |
| `js/components.js` | `PA.components`: reusable pure-view renderers (nav rail + Server panel, cards, table, badges, modal, progress…). |
| `js/views/*.js` | One screen each; self-register via `PA.registerView`. `settings.js` registers no view — it defines the `PA.settingsModal` overlay. |
| `js/app.js` | Runtime: view registry, shell, hash router, harness-toolbar wiring, the rail's gear + Server panel. Loaded last. |

### How to add a screen

Drop a `js/views/x.js` that self-registers — **no other file needs editing**:

```js
(function (PA) {
  PA.registerView('metrics', {
    title: 'Metrics',
    icon: 'chart-dots',   // Tabler icon name
    order: 5,             // position in the nav (Chat=1 … Memory=4)
    render: function (root, ctx) {
      // ctx.data()  -> the seam (mock or live)
      // ctx.store   -> reactive shared state
      // ctx.go(id)  -> navigate
      // ctx.setTopBar(title, rightNodes) -> own the top bar
      root.appendChild(PA.ui.el('div', { text: 'hello' }));
    },
  });
})(window.PA);
```

Then add its `<script>` to `index.html` **after `components.js` and before `app.js`** (with the
other views). The registry is read fresh on every render, so the nav, the hash router, and the
mobile tab bar pick it up automatically. Give it a data method on **both** `api.js` and `mock.js`
if it needs one — always through `PA.data()`.

### Desktop + mobile

The same views render in both form factors. `css/shell.css` swaps a labeled left nav rail
(desktop) for a bottom tab bar (mobile) at `≤640px` **or** when `html.force-mobile` is set by the
device toggle. View layouts use the `.pa-grid*` utilities, which collapse to a single column on
mobile — so no per-screen responsive code is needed.

---

## Screens

The nav is driven entirely by each view's `order` field. Settings is **not** a routed view — it
opens as a modal from the ⚙ gear pinned near the rail bottom, above the Server panel.

| Order | Screen | Notes |
|---|---|---|
| 1 | Chat | Streaming transcript (WS), instruct badge + context meter, composer with model/backend pickers, 🧠 Memory + 🌐 Web chips, an Advanced popover (sampling / length / seed / text size), Send/Stop. |
| 2 | Models | Card grid over `/health`'s **modelInfos** (real metadata), 💬 Chat with + ⊟ Tokenize…, and the inline "＋ Train new model" form with live progress. |
| 3 | Benchmark | Define / Compare / Reports tabs: suites, model multi-select, repeats, live run progress with cancel, aggregates + case grid + side-by-side output modal, past runs. |
| 4 | Memory | Store picker ("default" + one per model), search, the card catalog, the injection preview (bridge + recall), and the manual inject form. |
| — | Settings (modal) | App (this machine) · Memory injection (server) · Web search (Tavily, write-only key). |

The rail also carries the **Server panel** (client's ConnectionPanel): URL, Check connection,
Start/Stop local server, status line. In **Live** the Start button is present but disabled —
a browser can't spawn processes (run `projectai serve` yourself); in **Mock** it simulates the
start flow ("Starting the server — loading the model…" → connected).

---

## Conventions (keep these true)

- **Views/components touch `PA.data()` only** — never `fetch`, `WebSocket`, `PA.api`, or `PA.mock`.
- **Colors live only in `tokens.css`** — everything else uses `var(--pa-*)`.
- **Mock mirrors the API surface** — every method a view calls exists on both `PA.api` and
  `PA.mock`, with matching shapes, so Live/Mock swaps require zero view changes.
- **The Godot client is the source of truth** — copy, badges, layouts, and wire shapes follow
  `Client/ai-client/` and `ProjectAI/Server.cs`.
- **Secrets are write-only** — the Tavily key is sent once and never echoed back; only masked
  status (`{key, set, hint, source}`) ever reaches the client.
