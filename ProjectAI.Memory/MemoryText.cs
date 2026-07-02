using System.Text;

namespace ProjectAI.Memory;

/// <summary>
/// Pure text helpers shared by the store: key normalization (the retrieval + dedup surface), a tokenizer-free token
/// estimate for budgeting, slug/filename sanitization, and the injection sanitizer that strips a recalled body's
/// attempts to break out of its frame or issue protocol commands (the persistent-injection defence).
/// </summary>
internal static class MemoryText
{
    /// <summary>Lowercased, distinct alphanumeric handles (length ≥ 2). The shared basis for keys, search, and dedup.</summary>
    public static List<string> NormalizeKeys(string? text)
    {
        var keys = new List<string>();
        if (string.IsNullOrEmpty(text)) return keys;
        var seen = new HashSet<string>(StringComparer.Ordinal);
        var sb = new StringBuilder();
        foreach (char c in text)
        {
            if (char.IsLetterOrDigit(c)) sb.Append(char.ToLowerInvariant(c));
            else Flush(sb, keys, seen);
        }
        Flush(sb, keys, seen);
        return keys;

        static void Flush(StringBuilder sb, List<string> keys, HashSet<string> seen)
        {
            if (sb.Length >= 2)
            {
                string k = sb.ToString();
                if (seen.Add(k)) keys.Add(k);
            }
            sb.Clear();
        }
    }

    /// <summary>
    /// A tokenizer-free token-count estimate (~4 chars/token) for budgeting the bridge and recalls. Deliberately
    /// model-agnostic in M0 (the store serves many models); a serving-tokenizer measurement is a later refinement.
    /// </summary>
    public static int EstimateTokens(string? s) => string.IsNullOrEmpty(s) ? 0 : Math.Max(1, (int)Math.Ceiling(s.Length / 4.0));

    /// <summary>A filesystem-safe, human-readable slug for a node filename (cosmetic; identity is the content hash).</summary>
    public static string Slug(string title, int max = 40)
    {
        var sb = new StringBuilder();
        bool lastDash = false;
        foreach (char c in title)
        {
            if (char.IsLetterOrDigit(c)) { sb.Append(char.ToLowerInvariant(c)); lastDash = false; }
            else if (!lastDash && sb.Length > 0) { sb.Append('-'); lastDash = true; }
            if (sb.Length >= max) break;
        }
        return sb.ToString().Trim('-') is { Length: > 0 } s ? s : "memory";
    }

    /// <summary>A safe file stem for an inverted-index posting file (one per key).</summary>
    public static string KeyFileName(string key)
    {
        var sb = new StringBuilder(key.Length);
        foreach (char c in key) sb.Append(char.IsLetterOrDigit(c) ? c : '_');
        return sb.Length == 0 ? "_" : sb.ToString();
    }

    /// <summary>
    /// Sanitizes a recalled body before it enters the model's context: drops lines that begin with a protocol verb
    /// (so stored text can't issue RECALL/ENCODE) and strips frame/special-token breakout sequences (so it can't
    /// close the &lt;recalled&gt; wrapper or forge a chat turn). A body is reference data, never instructions.
    /// </summary>
    public static string SanitizeForInjection(string body)
    {
        if (string.IsNullOrEmpty(body)) return "";
        var outLines = new List<string>();
        foreach (var raw in body.Replace("\r\n", "\n").Split('\n'))
        {
            string line = raw;
            // Neutralize frame-escape / turn-forge attempts anywhere in the line.
            foreach (var bad in Breakouts)
                if (line.Contains(bad, StringComparison.OrdinalIgnoreCase))
                    line = line.Replace(bad, "", StringComparison.OrdinalIgnoreCase);
            // Drop a line that tries to issue a protocol command from stored content.
            string trimmed = line.TrimStart();
            if (trimmed.StartsWith("RECALL", StringComparison.Ordinal) || trimmed.StartsWith("ENCODE", StringComparison.Ordinal))
                continue;
            outLines.Add(line);
        }
        return string.Join("\n", outLines).Trim();
    }

    private static readonly string[] Breakouts =
    [
        "</recalled>", "<recalled", "</memory>", "<memory", "<|im_start|>", "<|im_end|>", "<|mem_", "<|endoftext|>",
    ];
}
