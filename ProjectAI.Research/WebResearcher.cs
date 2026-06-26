using System.Text;

namespace ProjectAI.Research;

/// <summary>
/// Turns a user query into model-ready context (RAG): searches the web via an <see cref="ISearchProvider"/>, then
/// builds an augmented prompt that grounds the model in the results and asks it to cite them. Each result's content
/// is truncated so the injected context fits a small model's window. The provider + results can later feed an
/// agentic search loop (Phase 2) without changing this class.
/// </summary>
public sealed class WebResearcher(ISearchProvider provider, int perResultChars = 600)
{
    public ISearchProvider Provider => provider;

    public async Task<ResearchResult> ResearchAsync(string query, int maxResults = 5, CancellationToken ct = default)
    {
        var sources = await provider.SearchAsync(query, maxResults, ct);
        return new ResearchResult(sources, BuildPrompt(query, sources));
    }

    private string BuildPrompt(string query, IReadOnlyList<SearchResult> sources)
    {
        if (sources.Count == 0)
            return $"A web search for \"{query}\" returned no results. Answer from your own knowledge and note that live results were unavailable.\n\nQuestion: {query}";

        var sb = new StringBuilder();
        sb.AppendLine("Use the current web search results below to answer the question. Cite sources inline as [n]. If the results don't cover it, say so rather than guessing.");
        sb.AppendLine();
        for (int i = 0; i < sources.Count; i++)
        {
            var s = sources[i];
            sb.AppendLine($"[{i + 1}] {s.Title}");
            if (!string.IsNullOrWhiteSpace(s.Url)) sb.AppendLine(s.Url);
            sb.AppendLine(Truncate(s.Content, perResultChars));
            sb.AppendLine();
        }
        sb.Append($"Question: {query}");
        return sb.ToString();
    }

    private static string Truncate(string s, int max)
    {
        s = (s ?? "").Trim();
        return s.Length > max ? s[..max] + "…" : s;
    }
}
