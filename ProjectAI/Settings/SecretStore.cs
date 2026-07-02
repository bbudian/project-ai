using System.Text.Json;

// Write-only secret storage at config/secrets.json (git-ignored, OUTSIDE the models tree): values go in via
// PUT /config/secrets/{key} and NEVER come back out over HTTP — status is presence + a last-4 hint only. On
// Windows the file gets a real ACL (inheritance stripped, current user only), not a comment promising one.
// An environment variable wins over the file: an operator's explicit launch configuration beats stored state,
// and it keeps the pre-P5 behavior (TAVILY_API_KEY) working unchanged.
internal static class SecretStore
{
    private const string FilePath = "config/secrets.json";
    private static readonly object Lock = new();

    /// <summary>The only keys the API accepts — no cross-provider speculation in v1.</summary>
    public static readonly string[] KnownKeys = ["tavily"];

    private static readonly Dictionary<string, string> EnvNames = new(StringComparer.OrdinalIgnoreCase)
    {
        ["tavily"] = "TAVILY_API_KEY",
    };

    public static bool IsKnown(string key) => KnownKeys.Contains(key, StringComparer.OrdinalIgnoreCase);

    /// <summary>The secret's value and where it came from ("env" | "config" | null when unset).</summary>
    public static (string? Value, string? Source) Resolve(string key)
    {
        if (EnvNames.TryGetValue(key, out var envName)
            && Environment.GetEnvironmentVariable(envName) is { Length: > 0 } fromEnv)
            return (fromEnv, "env");
        lock (Lock)
            return LoadFile().TryGetValue(key.ToLowerInvariant(), out var fromFile) && fromFile.Length > 0
                ? (fromFile, "config")
                : (null, null);
    }

    /// <summary>Presence + last-4 hint + source — the ONLY shape that ever leaves the server.</summary>
    public static object Status(string key)
    {
        var (value, source) = Resolve(key);
        return new
        {
            key = key.ToLowerInvariant(),
            set = value is not null,
            hint = value is { Length: >= 4 } ? "…" + value[^4..] : value is not null ? "…" : null,
            source,
        };
    }

    public static void Set(string key, string value)
    {
        lock (Lock)
        {
            var secrets = LoadFile();
            secrets[key.ToLowerInvariant()] = value;
            SaveFile(secrets);
        }
    }

    public static void Delete(string key)
    {
        lock (Lock)
        {
            var secrets = LoadFile();
            if (secrets.Remove(key.ToLowerInvariant())) SaveFile(secrets);
        }
    }

    private static Dictionary<string, string> LoadFile()
    {
        try
        {
            if (!File.Exists(FilePath)) return new(StringComparer.OrdinalIgnoreCase);
            return JsonSerializer.Deserialize<Dictionary<string, string>>(File.ReadAllText(FilePath))
                   ?? new(StringComparer.OrdinalIgnoreCase);
        }
        catch (Exception) { return new(StringComparer.OrdinalIgnoreCase); }
    }

    private static void SaveFile(Dictionary<string, string> secrets)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(FilePath)!);
        string tmp = FilePath + ".tmp";
        File.WriteAllText(tmp, JsonSerializer.Serialize(secrets, new JsonSerializerOptions { WriteIndented = true }));
        File.Move(tmp, FilePath, overwrite: true);
        ApplyAcl(FilePath);
    }

    // A real ACL, not a chmod comment: strip inherited rules, grant only the current user. Non-Windows is a no-op
    // (the server targets the user's Windows box today; POSIX perms would be the equivalent there).
    private static void ApplyAcl(string path)
    {
        if (!OperatingSystem.IsWindows()) return;
        try
        {
            var identity = System.Security.Principal.WindowsIdentity.GetCurrent();
            if (identity.User is null) return;
            var file = new FileInfo(path);
            var security = file.GetAccessControl();
            security.SetAccessRuleProtection(isProtected: true, preserveInheritance: false);
            foreach (System.Security.AccessControl.FileSystemAccessRule rule in
                     security.GetAccessRules(true, true, typeof(System.Security.Principal.SecurityIdentifier)))
                security.RemoveAccessRule(rule);
            security.AddAccessRule(new System.Security.AccessControl.FileSystemAccessRule(
                identity.User, System.Security.AccessControl.FileSystemRights.FullControl,
                System.Security.AccessControl.AccessControlType.Allow));
            file.SetAccessControl(security);
        }
        catch (Exception e)
        {
            // Never fail a save over ACL trouble (e.g. FAT volume) — but say so, loudly.
            Console.Error.WriteLine($"warning: could not lock down {path}: {e.Message}");
        }
    }
}
