using Godot;

// Small defensive accessors for the loosely-typed JSON dictionaries the server returns. Centralizing the
// "key present? right type? else fallback" idiom keeps each endpoint's parsing a uniform one-liner (DRY), so adding
// an endpoint as the API grows doesn't re-spell the same null/type guards. Shared by ApiClient and ChatSocket.
public static class JsonDict
{
    /// <summary>The string at <paramref name="key"/>, or <paramref name="fallback"/> if absent or not a string (e.g. JSON null).</summary>
    public static string Str(this Godot.Collections.Dictionary d, string key, string fallback = "") =>
        d.ContainsKey(key) && d[key].VariantType == Variant.Type.String ? d[key].AsString() : fallback;

    // Numbers: accept either JSON type, since a whole-valued number (e.g. loss 0) parses as Int, not Float.
    public static int Int(this Godot.Collections.Dictionary d, string key, int fallback = 0) =>
        d.ContainsKey(key) && d[key].VariantType is Variant.Type.Int or Variant.Type.Float ? d[key].AsInt32() : fallback;

    public static float Float(this Godot.Collections.Dictionary d, string key, float fallback = 0f) =>
        d.ContainsKey(key) && d[key].VariantType is Variant.Type.Float or Variant.Type.Int ? (float)d[key].AsDouble() : fallback;

    public static bool Bool(this Godot.Collections.Dictionary d, string key, bool fallback = false) =>
        d.ContainsKey(key) && d[key].VariantType == Variant.Type.Bool ? d[key].AsBool() : fallback;

    /// <summary>The array at <paramref name="key"/>, or an empty array if absent.</summary>
    public static Godot.Collections.Array Arr(this Godot.Collections.Dictionary d, string key) =>
        d.ContainsKey(key) ? d[key].AsGodotArray() : new Godot.Collections.Array();
}
