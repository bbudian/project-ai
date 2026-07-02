# ProjectAI — Ideas & Thoughts (exploratory)

A scratchpad for ideas worth exploring later. **Not committed design** — an entry here has not been
through a design pass or adversarial critique. When one graduates, it moves to a `docs/*_DESIGN.md`
spec and a `BUILD_PLAN.md` ticket. Related: [`MEMORY_DESIGN.md`](MEMORY_DESIGN.md),
[`MEMORY_LINEAGE_DESIGN.md`](MEMORY_LINEAGE_DESIGN.md).

---

## Pluggable inherited-memory "chips" — stackable, learnable knowledge modules
*Captured 2026-07-01. Status: raw idea.*

**The pitch.** Instead of a model having one inherited tier, let it accept **multiple inherited
memories**, each a self-contained, insertable/removable module — like slotting in a cartridge:
insert chip A → the model learns it → insert chip B → it learns that too, alongside A. Knowledge
becomes composable hardware-style modules you can add, swap, and eject.

**What it extends.** The lineage layer (`MEMORY_LINEAGE_DESIGN.md`) currently has *one* per-user LoRA
adapter over a frozen base, plus a single `tier: inherited`. This idea generalizes that to a **stack /
registry of composable adapters** — each "chip" is one consolidated knowledge module. "Insert a chip"
= attach/activate a module; "learn" = (optionally) consolidate it further; "swap" = detach one, attach
another. The frozen base stays the substrate; chips are the removable layers on top.

**What could a "chip" actually be?** Candidates to weigh:
- **A trained adapter delta** (a LoRA over the shared base) — the direct read of "learned knowledge."
- **A portable memory pack** — a bundle of long-term memory cards (`.md` + frontmatter) + provenance
  that *consolidates into* an adapter when inserted. (Reuses M0 storage + the `ExportForTraining` hook.)
- **A distilled context/KV artifact** (the "cartridge" line of work) — knowledge baked into a reusable
  KV-cache/prefix instead of weights. Cheaper to make, no training, but heavier at serve time.
- Most likely a **bundle**: `{ memory cards + optional pre-consolidated adapter + lineage/trust metadata }`,
  signed with which base checkpoint it's valid against.

**Open questions to explore.**
- **Composition.** How do multiple chips combine — additive adapter deltas (sum of LoRAs), per-query
  **routing/gating** (pick the relevant chip), or a chain? Additive is simplest but adapters trained
  separately **interfere**; routing avoids interference but needs a router.
- **Order effects.** "Learn A then B" — does inserting B degrade A (cross-module catastrophic
  forgetting)? Are chips order-independent if kept as *separate* deltas rather than merged?
- **Activation model.** All inserted chips always-on (merged) vs. dynamically selected per
  conversation/query. Ties to the serve-time LRU cache in the lineage doc.
- **Read-side.** RECALL across *several* inherited tiers — which chip's memory answers, and how the
  bridge/map represents multiple inherited sources without bloating context.
- **Provenance, trust, multi-client.** Each chip carries its own lineage + trust; some chips are
  private to a user, some shared. A shared chip must never leak a user's private facts (the M0
  multi-client rule). Chip versioning + "valid against base X" compatibility checks.
- **Lifecycle & UX.** insert → activate → learn/consolidate → deactivate → eject → share. In the Godot
  client this is literally a "slots" UI: the user sees which chips are inserted and toggles them.
- **Cost.** N adapters resident on an 8 GB/24 GB GPU — paging/LRU, and how small each chip must stay.

**Prior art to look at.** LoRA composition / weight arithmetic; multi-adapter serving & routing
(S-LoRA-style); mixture-of-LoRA-experts; PEFT adapter hubs; "Cartridges"/context-distillation (KV-cache
as a reusable knowledge artifact). ProjectAI's angle: it **owns training**, so a chip isn't just
retrieved — it can be genuinely *consolidated* locally, which most adapter-hub setups can't do.

**Why it's compelling for ProjectAI.** It turns the two-way loop into a *modular* one: you can build,
share, and stack domain "chips" (a codebase chip, a personal chip, a project chip) over one frozen
base — an app store of local, auditable, learnable knowledge, none of it phoning home.

**First experiment (when this graduates).** Two independently-consolidated LoRA adapters over the same
base; serve with (a) both merged vs (b) query-routed; measure interference (does A's accuracy drop when
B is inserted?) and per-chip resident cost. That single experiment answers the composition + order
questions before any UX is built.
