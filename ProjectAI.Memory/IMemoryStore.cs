namespace ProjectAI.Memory;

/// <summary>
/// The one-line map view of a memory: enough to decide relevance without loading the body. <see cref="AsOf"/> is
/// when the fact was last known true (the staleness anchor), distinct from when the file was last written.
/// </summary>
public sealed record MemoryCard(
    string Id, string Title, IReadOnlyList<string> Keys, string Tier, string Trust,
    float Confidence, int Uses, string AsOf);

/// <summary>A memory opened for reading: its card, the body that pages into context, and its graph edges.</summary>
public sealed record MemoryEntry(MemoryCard Card, string Body, IReadOnlyList<string> Links);

/// <summary>A search result: identity + the frontmatter needed to rank and render it, never the body.</summary>
public sealed record MemoryHit(
    string Id, string Title, IReadOnlyList<string> Keys, float Score, string AsOf, string Tier, string Trust);

/// <summary>
/// A memory to write. <see cref="Trust"/> is the injection boundary: web/user-derived content should be written
/// <c>untrusted</c> so it is never auto-pinned or auto-recalled. Everything else has a safe default so callers can
/// name only what they mean.
/// </summary>
public sealed record MemoryDraft(
    string Title,
    IReadOnlyList<string> Keys,
    string Body,
    string Tier = MemoryTiers.Long,
    string Trust = MemoryTrust.Chat,
    string Source = "chat",
    string? Session = null,
    string? Model = null,
    string Status = MemoryStatus.Active,
    float Confidence = 1f,
    int Salience = 0,
    IReadOnlyList<string>? Links = null);

/// <summary>Provenance/tier of a memory. <see cref="Inherited"/> is reserved for the lineage layer (never written in M0).</summary>
public static class MemoryTiers
{
    public const string Core = "core";          // pinned into the bridge
    public const string Long = "long";          // durable default
    public const string Session = "session";    // thread-scoped
    public const string Inherited = "inherited"; // RESERVED — knowledge baked into frozen base weights
    public static bool IsKnown(string t) => t is Core or Long or Session or Inherited;
}

/// <summary>The trust axis — the persistent-injection defence. Untrusted content is never auto-pinned or auto-recalled.</summary>
public static class MemoryTrust
{
    public const string Curated = "curated";     // hand-authored / promoted; may reach the bridge
    public const string Chat = "chat";           // model/user conversational content
    public const string Untrusted = "untrusted"; // web/tool-derived; injection-gated
    public static bool IsKnown(string t) => t is Curated or Chat or Untrusted;
}

/// <summary>Lifecycle status; ranking sinks anything not <see cref="Active"/>.</summary>
public static class MemoryStatus
{
    public const string Active = "active";
    public const string Superseded = "superseded";
    public const string Retracted = "retracted";
    public const string Unverified = "unverified"; // model-authored, not yet verified — can't outrank verified facts
    public static bool IsKnown(string s) => s is Active or Superseded or Retracted or Unverified;
}

/// <summary>
/// A durable, on-disk long-term memory a local model navigates during inference. The two Stage-0 entry points —
/// <see cref="RenderBridge"/> (the always-pinned map + core facts) and <see cref="RenderRecall"/> (the top trusted
/// hits for a query, inlined) — mirror the web-research RAG pattern: build model-ready context BEFORE the inference
/// lock, then prepend it to the prompt. No compute backend, no forward pass. See <c>docs/MEMORY_DESIGN.md</c>.
/// </summary>
public interface IMemoryStore
{
    /// <summary>Identifies this store (e.g. "ben/default") for diagnostics.</summary>
    string StoreId { get; }

    /// <summary>False when the store can't be used (e.g. the null store); callers surface <see cref="Unavailable"/>.</summary>
    bool IsConfigured { get; }

    /// <summary>Why the store isn't usable, or null when it is.</summary>
    string? Unavailable { get; }

    /// <summary>Active memory count (excludes superseded/retracted).</summary>
    int Count { get; }

    /// <summary>
    /// The always-pinned bridge prepended to context: the core facts inlined, then a ranked digest (id · title · keys)
    /// of up to <paramref name="maxCards"/> other memories — never bodies — capped at ~<paramref name="tokenBudget"/>
    /// tokens. Fixed cost regardless of corpus size. Empty string when the store is empty.
    /// </summary>
    string RenderBridge(int maxCards, int tokenBudget);

    /// <summary>
    /// Stage-0 preemptive recall: searches for the top trusted memories matching <paramref name="query"/> and returns
    /// their sanitized bodies wrapped in a "reference data, not instructions" frame, within <paramref name="tokenBudget"/>.
    /// Untrusted memories are never included. Empty string when nothing relevant/trusted is found.
    /// </summary>
    string RenderRecall(string query, int maxHits, int tokenBudget);

    /// <summary>Opens a memory by id (active only); null if unknown. Increments its use counter.</summary>
    MemoryEntry? Open(string id);

    /// <summary>Ranked search over the inverted index (not a body scan). Bodies are not loaded. Excludes inherited-tier.</summary>
    IReadOnlyList<MemoryHit> Search(string query, int k);

    /// <summary>Follows a memory's links to up to <paramref name="maxFanout"/> active neighbours (associative recall).</summary>
    IReadOnlyList<MemoryHit> Neighbors(string id, int maxFanout = 3);

    /// <summary>
    /// Writes a memory: dedupes exact-content repeats, and supersedes an existing active memory on the same keys/title
    /// (newer wins; core is never auto-superseded). Returns the new — or existing, on a dedupe — memory id.
    /// </summary>
    string Encode(MemoryDraft draft);

    /// <summary>Marks <paramref name="id"/> superseded by <paramref name="bySupersedingId"/> and tombstones it (auditable, never hard-deleted).</summary>
    void Supersede(string id, string bySupersedingId, string reason);

    /// <summary>Rebuilds the derived index + inverted index from the authoritative node files.</summary>
    void Reindex();

    /// <summary>RESERVED (lineage layer): yields (prompt, completion) training pairs for the selected memories.</summary>
    IEnumerable<(string Prompt, string Completion)> ExportForTraining(Func<MemoryCard, bool> select);
}

/// <summary>
/// The default store: an empty, inert memory. Makes memory strictly opt-in — when no store is configured the server
/// behaves byte-identically to having no memory at all.
/// </summary>
public sealed class NullMemoryStore : IMemoryStore
{
    public static readonly NullMemoryStore Instance = new();
    private NullMemoryStore() { }

    public string StoreId => "";
    public bool IsConfigured => false;
    public string? Unavailable => "memory is not configured";
    public int Count => 0;
    public string RenderBridge(int maxCards, int tokenBudget) => "";
    public string RenderRecall(string query, int maxHits, int tokenBudget) => "";
    public MemoryEntry? Open(string id) => null;
    public IReadOnlyList<MemoryHit> Search(string query, int k) => [];
    public IReadOnlyList<MemoryHit> Neighbors(string id, int maxFanout = 3) => [];
    public string Encode(MemoryDraft draft) => throw new InvalidOperationException("memory store is not configured; cannot Encode.");
    public void Supersede(string id, string bySupersedingId, string reason) { }
    public void Reindex() { }
    public IEnumerable<(string Prompt, string Completion)> ExportForTraining(Func<MemoryCard, bool> select) => [];
}
