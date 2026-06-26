namespace ProjectAI.Research;

/// <summary>One web search result: a title, its source URL, and a snippet/extract of its content.</summary>
public sealed record SearchResult(string Title, string Url, string Content);

/// <summary>
/// The outcome of a research pass: the sources found plus an <see cref="AugmentedPrompt"/> ready to send to the
/// model (the query grounded in the results). The sources are returned separately so the UI can cite them.
/// </summary>
public sealed record ResearchResult(IReadOnlyList<SearchResult> Sources, string AugmentedPrompt);

/// <summary>
/// A swappable web search backend (Tavily today; DuckDuckGo / Brave / SearXNG could drop in behind this seam).
/// Deliberately minimal so the same provider serves both RAG (Phase 1: system retrieves, model answers) and a
/// future agentic search loop (Phase 2: the model drives the queries).
/// </summary>
public interface ISearchProvider
{
    /// <summary>Short id for diagnostics, e.g. "tavily".</summary>
    string Name { get; }

    /// <summary>False when the provider can't run (e.g. a missing API key); callers surface <see cref="Unavailable"/>.</summary>
    bool IsConfigured { get; }

    /// <summary>Why the provider isn't usable (for the error message), or null when it is configured.</summary>
    string? Unavailable { get; }

    /// <summary>Runs a web search and returns up to <paramref name="maxResults"/> results, best first.</summary>
    Task<IReadOnlyList<SearchResult>> SearchAsync(string query, int maxResults, CancellationToken ct = default);
}
