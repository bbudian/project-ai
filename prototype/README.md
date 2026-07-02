# ProjectAI — Prototyping Harness

A single-page, zero-build **UI prototyping harness** for the ProjectAI runtime. It renders
the whole client — Chat, Models, Memory, Benchmark, Settings — as static HTML/CSS/vanilla JS,
so you can design and iterate on screens with **the server completely off**, then flip a single
toggle to run them against a live `projectai serve` instance.

No framework, no bundler, no `npm install`. Just files and a browser.

---

## What this is (and isn't)

- **Is:** a faithful, interactive mock of the desktop + mobile client. Every screen is fully
  usable against canned data. It exists to prototype *information architecture and interaction*
  before (and alongside) the C# server growing the matching endpoints.
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
  payloads with small simulated delays and even simulated token streaming for Chat. Nothing hits
  the network. This is how you prototype with the server off.
- **Live:** data flows to the real server at the toolbar's **base URL** via `js/api.js`
  (`fetch` / `WebSocket`). Endpoints the server doesn't implement yet reject with a clear
  *"endpoint not on server yet — use Mock"* error, which the affected screen surfaces as a
  visible empty/error state rather than a blank page.

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
  with realistic shapes. It also defines **`PA.data()`** — the seam itself: it returns `PA.api`
  in Live mode and `PA.mock` in Mock mode.
- **Views and components call `PA.data()` (or `ctx.data()`) only.** They must never reference
  `PA.api` / `PA.mock` directly and must never call `fetch` / `WebSocket`. That single
  indirection is the entire boundary between "visuals" and "API integration" — a view cannot
  tell, and does not care, whether it's talking to a server or a mock.

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
| `js/store.js` | `PA.store`: a tiny reactive store (health/models/backends/selection). No I/O. |
| `js/api.js` | `PA.api`: the only network file (see seam above). |
| `js/mock.js` | `PA.mock` + `PA.data()`: canned data + the isolation seam. |
| `js/components.js` | `PA.components`: reusable pure-view renderers (nav, cards, table, badges…). |
| `js/views/*.js` | One screen each; self-register via `PA.registerView`. |
| `js/app.js` | Runtime: view registry, shell, hash router, harness-toolbar wiring. Loaded last. |

### How to add a screen

Drop a `js/views/x.js` that self-registers — **no other file needs editing**:

```js
(function (PA) {
  PA.registerView('metrics', {
    title: 'Metrics',
    icon: 'chart-dots',   // Tabler icon name
    order: 5,             // position in the nav (Chat=1 … Settings=6)
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

## Nav order

The nav is driven entirely by each view's `order` field:

| Order | Screen | Notes |
|---|---|---|
| 1 | Chat | Flagship working view: streaming transcript + composer. |
| 2 | Models | Model cards, Load/Set-default, HF import panel (stub). |
| 3 | Memory | Recall list, inject form, bridge preview. |
| 4 | Benchmark | Suite/model config, comparison table, report/export. |
| 6 | Settings | App / Models / Memory / Benchmark / Backends sections. |

`5` is intentionally left free for a future **Upgrade / Train** screen (it can reuse the training
surface, or be omitted for now).

---

## Which endpoints are mock-only (for now)

`js/api.js` implements the endpoints the server already ships and **rejects** the rest with a
clear error, so those screens visibly fall back to Mock:

**Live on the server** (work in Live mode today):

- `GET /health`, `POST /generate`, `WS /chat`, `POST /tokenize`, `POST /train`,
  `GET /train/status`.

**Mock-only until the server implements them** (reject in Live, fully usable in Mock):

- **Benchmark:** `bench`, `benchStatus` — the whole Benchmark screen.
- **Memory CRUD:** `memoryList`, `memoryGet`, `memoryPut`, `memorySupersede` — the Memory screen.
- **Settings:** `settingsGet`, `settingsPut` — the Settings screen (and Models' "Set default").

When one of these is exercised in Live mode, the affected screen shows an explicit
*"endpoint not on server yet — switch to Mock"* state instead of failing silently. Prototype
those screens in **Mock**; wire them to Live as the corresponding server endpoints land.

---

## Conventions (keep these true)

- **Views/components touch `PA.data()` only** — never `fetch`, `WebSocket`, `PA.api`, or `PA.mock`.
- **Colors live only in `tokens.css`** — everything else uses `var(--pa-*)`.
- **Mock mirrors the API surface** — every method a view calls exists on both `PA.api` and
  `PA.mock`, with matching shapes, so Live/Mock swaps require zero view changes.
- **Secrets are write-only** — e.g. the Tavily key is masked, never echoed back to the client.
