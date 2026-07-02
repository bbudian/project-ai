using System.Globalization;
using System.Text;

namespace ProjectAI.Memory;

/// <summary>
/// One memory: the frontmatter contract the runtime reads without a forward pass, plus the Markdown body that pages
/// into context. Persisted as a flat <c>key: value</c> frontmatter block (a strict YAML subset — hand-parsed, no YAML
/// library, honoring the "libraries only for math/SIMD/IO" rule) followed by the body.
/// </summary>
internal sealed class MemoryNode
{
    public string Id = "";
    public string Title = "";
    public List<string> Keys = [];
    public string Tier = MemoryTiers.Long;
    public string Trust = MemoryTrust.Chat;
    public string Source = "chat";
    public string? Session;
    public string? Model;
    public string Created = "";
    public string Updated = "";
    public string AsOf = "";
    public string Expires = "∞";
    public float Confidence = 1f;
    public string Status = MemoryStatus.Active;
    public List<string> Supersedes = [];
    public string? SupersededBy;
    public List<string> Links = [];
    public int Uses;
    public int Salience;
    public int Tokens;
    public string Body = "";

    /// <summary>Absolute path this node was loaded from / written to (runtime only, never serialized).</summary>
    public string? Path;

    public bool IsActive => Status == MemoryStatus.Active || Status == MemoryStatus.Unverified;

    public MemoryCard ToCard() => new(Id, Title, Keys, Tier, Trust, Confidence, Uses, AsOf);
    public MemoryHit ToHit(float score) => new(Id, Title, Keys, score, AsOf, Tier, Trust);

    public string Serialize()
    {
        var sb = new StringBuilder();
        sb.Append("---\n");
        sb.Append("id: ").Append(Id).Append('\n');
        sb.Append("title: ").Append(OneLine(Title)).Append('\n');
        sb.Append("keys: ").Append(List(Keys)).Append('\n');
        sb.Append("tier: ").Append(Tier).Append('\n');
        sb.Append("trust: ").Append(Trust).Append('\n');
        sb.Append("provenance.source: ").Append(Source).Append('\n');
        sb.Append("provenance.session: ").Append(Session ?? "").Append('\n');
        sb.Append("provenance.model: ").Append(Model ?? "").Append('\n');
        sb.Append("created: ").Append(Created).Append('\n');
        sb.Append("updated: ").Append(Updated).Append('\n');
        sb.Append("asof: ").Append(AsOf).Append('\n');
        sb.Append("expires: ").Append(Expires).Append('\n');
        sb.Append("confidence: ").Append(Confidence.ToString("0.###", CultureInfo.InvariantCulture)).Append('\n');
        sb.Append("status: ").Append(Status).Append('\n');
        sb.Append("supersedes: ").Append(List(Supersedes)).Append('\n');
        sb.Append("supersededBy: ").Append(SupersededBy ?? "null").Append('\n');
        sb.Append("links: ").Append(List(Links)).Append('\n');
        sb.Append("uses: ").Append(Uses.ToString(CultureInfo.InvariantCulture)).Append('\n');
        sb.Append("salience: ").Append(Salience.ToString(CultureInfo.InvariantCulture)).Append('\n');
        sb.Append("tokens: ").Append(Tokens.ToString(CultureInfo.InvariantCulture)).Append('\n');
        sb.Append("---\n");
        sb.Append(Body);
        if (!Body.EndsWith('\n')) sb.Append('\n');
        return sb.ToString();
    }

    /// <summary>Parses a memory file; returns null (rather than throwing) on malformed input so a bad file is skipped, not fatal.</summary>
    public static MemoryNode? TryParse(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return null;
        var lines = text.Replace("\r\n", "\n").Split('\n');
        int i = 0;
        while (i < lines.Length && lines[i].Trim().Length == 0) i++;
        if (i >= lines.Length || lines[i].Trim() != "---") return null;
        i++;

        var node = new MemoryNode();
        bool closed = false;
        for (; i < lines.Length; i++)
        {
            if (lines[i].Trim() == "---") { closed = true; i++; break; }
            int colon = lines[i].IndexOf(':');
            if (colon <= 0) continue;
            string key = lines[i][..colon].Trim();
            string val = lines[i][(colon + 1)..].Trim();
            Apply(node, key, val);
        }
        if (!closed) return null;

        node.Body = i < lines.Length ? string.Join("\n", lines[i..]).Trim() : "";
        if (string.IsNullOrEmpty(node.Id) || string.IsNullOrEmpty(node.Title)) return null;
        return node;
    }

    private static void Apply(MemoryNode n, string key, string val)
    {
        switch (key)
        {
            case "id": n.Id = val; break;
            case "title": n.Title = val; break;
            case "keys": n.Keys = ParseList(val); break;
            case "tier": if (val.Length > 0) n.Tier = val; break;
            case "trust": if (val.Length > 0) n.Trust = val; break;
            case "provenance.source": n.Source = val; break;
            case "provenance.session": n.Session = Nullable(val); break;
            case "provenance.model": n.Model = Nullable(val); break;
            case "created": n.Created = val; break;
            case "updated": n.Updated = val; break;
            case "asof": n.AsOf = val; break;
            case "expires": if (val.Length > 0) n.Expires = val; break;
            case "confidence": if (float.TryParse(val, NumberStyles.Float, CultureInfo.InvariantCulture, out var c)) n.Confidence = c; break;
            case "status": if (val.Length > 0) n.Status = val; break;
            case "supersedes": n.Supersedes = ParseList(val); break;
            case "supersededBy": n.SupersededBy = Nullable(val); break;
            case "links": n.Links = ParseList(val); break;
            case "uses": if (int.TryParse(val, out var u)) n.Uses = u; break;
            case "salience": if (int.TryParse(val, out var s)) n.Salience = s; break;
            case "tokens": if (int.TryParse(val, out var t)) n.Tokens = t; break;
        }
    }

    private static string? Nullable(string v) => v.Length == 0 || v == "null" ? null : v;

    private static List<string> ParseList(string v)
    {
        v = v.Trim();
        if (v.StartsWith('[') && v.EndsWith(']')) v = v[1..^1];
        var list = new List<string>();
        foreach (var part in v.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
            if (part.Length > 0) list.Add(part);
        return list;
    }

    private static string List(IReadOnlyList<string> items) => "[" + string.Join(", ", items) + "]";
    private static string OneLine(string s) => s.Replace("\r", " ").Replace("\n", " ").Trim();
}
