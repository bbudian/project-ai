// Small shared text helpers, kept in one place so formatting rules don't drift between callers (DRY).
public static class Format
{
    /// <summary>Single-line, ellipsized rendering of a prompt — used for the header title and the recents list.</summary>
    public static string Ellipsize(string text, int max)
    {
        string t = (text ?? "").Replace("\n", " ").Trim();
        if (t.Length <= max) return t;
        int cut = max;
        if (cut > 0 && char.IsHighSurrogate(t[cut - 1])) cut--; // don't slice through a surrogate pair (emoji, etc.)
        return t[..cut] + "…";
    }
}
