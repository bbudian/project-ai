using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace ProjectAI.Research;

/// <summary>
/// Tavily search (https://api.tavily.com/search) — a search API tuned for LLM RAG. The API key is read from the
/// <c>TAVILY_API_KEY</c> environment variable (free key at tavily.com) unless one is passed in. Auth is a Bearer
/// header; we request raw results (title/url/content) and leave <c>include_answer</c> off, since we feed the
/// results to OUR model rather than using Tavily's synthesized answer.
/// </summary>
public sealed class TavilySearchProvider : ISearchProvider
{
    private const string Endpoint = "https://api.tavily.com/search";
    private static readonly HttpClient Http = new() { Timeout = TimeSpan.FromSeconds(20) };
    private static readonly JsonSerializerOptions JsonOpts = new() { PropertyNameCaseInsensitive = true };

    private readonly Func<string?> _resolveKey;

    public TavilySearchProvider(string? apiKey = null)
        => _resolveKey = string.IsNullOrWhiteSpace(apiKey)
            ? static () => Environment.GetEnvironmentVariable("TAVILY_API_KEY")
            : () => apiKey;

    /// <summary>Resolve the key per request (e.g. from a server-side secret store), so a key saved while the
    /// server runs takes effect immediately — no restart, no stale singleton.</summary>
    public TavilySearchProvider(Func<string?> keyResolver) => _resolveKey = keyResolver;

    public string Name => "tavily";
    public bool IsConfigured => !string.IsNullOrWhiteSpace(_resolveKey());
    public string? Unavailable => IsConfigured ? null : "set the Tavily API key (Settings → Web search, or the TAVILY_API_KEY environment variable; free key at tavily.com)";

    public async Task<IReadOnlyList<SearchResult>> SearchAsync(string query, int maxResults, CancellationToken ct = default)
    {
        string? apiKey = _resolveKey();
        if (string.IsNullOrWhiteSpace(apiKey)) throw new InvalidOperationException($"Tavily is not configured: {Unavailable}.");

        var request = new TavilyRequest(query, Math.Clamp(maxResults, 1, 20), "basic", false, "general");
        using var msg = new HttpRequestMessage(HttpMethod.Post, Endpoint)
        {
            Content = new StringContent(JsonSerializer.Serialize(request, JsonOpts), Encoding.UTF8, "application/json"),
        };
        msg.Headers.Authorization = new AuthenticationHeaderValue("Bearer", apiKey);

        using var resp = await Http.SendAsync(msg, ct);
        string json = await resp.Content.ReadAsStringAsync(ct);
        if (!resp.IsSuccessStatusCode)
            throw new InvalidOperationException($"Tavily search failed (HTTP {(int)resp.StatusCode}): {Trunc(json)}");

        var parsed = JsonSerializer.Deserialize<TavilyResponse>(json, JsonOpts);
        var items = parsed?.Results ?? [];
        var results = new List<SearchResult>(items.Count);
        foreach (var r in items)
            results.Add(new SearchResult(r.Title ?? "", r.Url ?? "", r.Content ?? ""));
        return results;
    }

    private static string Trunc(string s) => s.Length > 200 ? s[..200] + "…" : s;

    private sealed record TavilyRequest(
        [property: JsonPropertyName("query")] string Query,
        [property: JsonPropertyName("max_results")] int MaxResults,
        [property: JsonPropertyName("search_depth")] string SearchDepth,
        [property: JsonPropertyName("include_answer")] bool IncludeAnswer,
        [property: JsonPropertyName("topic")] string Topic);

    private sealed record TavilyResponse([property: JsonPropertyName("results")] List<TavilyItem>? Results);

    private sealed record TavilyItem(
        [property: JsonPropertyName("title")] string? Title,
        [property: JsonPropertyName("url")] string? Url,
        [property: JsonPropertyName("content")] string? Content);
}
