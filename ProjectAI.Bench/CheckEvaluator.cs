using System.Text.Json;
using System.Text.RegularExpressions;

namespace ProjectAI.Bench;

/// <summary>
/// Deterministic output checks — meaningful only under greedy decoding (the runner enforces the default).
/// Kinds: <c>contains:&lt;s&gt;</c> per MustInclude entry (case-insensitive substring — a floor signal, not a
/// grade), <c>exact</c> vs Reference (whitespace-trimmed), <c>regex:&lt;pattern&gt;</c> when Reference is
/// <c>regex:…</c>, <c>jsonValid</c> when Reference is the literal <c>jsonValid</c>, and <c>maxTokens</c>
/// (did the output stay under the case budget rather than being cut off).
/// </summary>
public static class CheckEvaluator
{
    public static IReadOnlyList<CheckResult> Evaluate(BenchCase benchCase, string output, string stopReason)
    {
        var results = new List<CheckResult>();

        if (benchCase.MustInclude is { Count: > 0 })
            foreach (var needle in benchCase.MustInclude)
                results.Add(new CheckResult("contains", needle,
                    output.Contains(needle, StringComparison.OrdinalIgnoreCase)));

        if (!string.IsNullOrEmpty(benchCase.Reference))
        {
            if (benchCase.Reference == "jsonValid")
                results.Add(new CheckResult("jsonValid", "", IsValidJson(output)));
            else if (benchCase.Reference.StartsWith("regex:", StringComparison.Ordinal))
            {
                string pattern = benchCase.Reference["regex:".Length..];
                bool passed;
                try { passed = Regex.IsMatch(output, pattern, RegexOptions.None, TimeSpan.FromSeconds(2)); }
                catch (Exception) { passed = false; } // bad pattern or timeout counts as a failed check, not a crash
                results.Add(new CheckResult("regex", pattern, passed));
            }
            else
                results.Add(new CheckResult("exact", benchCase.Reference,
                    output.Trim() == benchCase.Reference.Trim()));
        }

        // Did the model finish on its own rather than being cut off by the case budget?
        results.Add(new CheckResult("maxTokens", benchCase.MaxTokens.ToString(), stopReason != "maxTokens"));
        return results;
    }

    public static double PassRate(IReadOnlyList<CheckResult> checks) =>
        checks.Count == 0 ? 0 : checks.Count(c => c.Passed) / (double)checks.Count;

    private static bool IsValidJson(string s)
    {
        // Models often wrap JSON in prose or code fences; scan for the first plausible JSON span.
        string t = s.Trim();
        int start = t.IndexOfAny(['{', '[']);
        if (start < 0) return false;
        try { JsonDocument.Parse(t[start..].TrimEnd('`', '\n', ' ')); return true; }
        catch (JsonException) { return false; }
    }
}
