using System.Text.Json;

namespace ProjectAI.Bench;

/// <summary>
/// Loads a suite from a JSON file: <c>{ id, label, evalCorpusFile?, cases:[{id,prompt,reference?,mustInclude?,maxTokens?}] }</c>.
/// <c>evalCorpusFile</c> is resolved relative to the suite file and read into <see cref="BenchSuite.EvalCorpus"/>.
/// Suites resolve by path first, then by id under &lt;modelsDir&gt;/benchmarks/suites/&lt;id&gt;.json.
/// </summary>
public static class SuiteLoader
{
    private sealed record SuiteFile(
        string? Id, string? Label, string? EvalCorpusFile, List<CaseFile>? Cases);
    private sealed record CaseFile(
        string? Id, string? Prompt, string? Reference, List<string>? MustInclude, int? MaxTokens);

    private static readonly JsonSerializerOptions JsonOpts = new() { PropertyNameCaseInsensitive = true };

    public static BenchSuite Load(string pathOrId, string modelsDir)
    {
        string? path = File.Exists(pathOrId)
            ? pathOrId
            : new[]
            {
                Path.Combine(modelsDir, "benchmarks", "suites", pathOrId + ".json"),
                Path.Combine("benchmarks", "suites", pathOrId + ".json"),
            }.FirstOrDefault(File.Exists);
        if (path is null)
            throw new FileNotFoundException(
                $"suite '{pathOrId}' not found (looked for a file, then benchmarks/suites/{pathOrId}.json under the models dir and the working dir).");

        SuiteFile? file;
        try { file = JsonSerializer.Deserialize<SuiteFile>(File.ReadAllText(path), JsonOpts); }
        catch (JsonException e) { throw new InvalidDataException($"suite '{path}' is not valid JSON: {e.Message}"); }
        if (file?.Cases is not { Count: > 0 })
            throw new InvalidDataException($"suite '{path}' has no cases.");

        string? corpus = null;
        if (!string.IsNullOrEmpty(file.EvalCorpusFile))
        {
            string corpusPath = Path.IsPathRooted(file.EvalCorpusFile)
                ? file.EvalCorpusFile
                : Path.Combine(Path.GetDirectoryName(Path.GetFullPath(path))!, file.EvalCorpusFile);
            if (!File.Exists(corpusPath))
                throw new FileNotFoundException($"suite '{path}' names eval corpus '{corpusPath}', which does not exist.");
            corpus = File.ReadAllText(corpusPath);
        }

        var cases = new List<BenchCase>();
        foreach (var c in file.Cases)
        {
            if (string.IsNullOrEmpty(c.Prompt)) throw new InvalidDataException($"suite '{path}': every case needs a prompt.");
            cases.Add(new BenchCase(
                c.Id ?? $"case-{cases.Count + 1}", c.Prompt, c.Reference, c.MustInclude, c.MaxTokens ?? 128));
        }
        return new BenchSuite(
            file.Id ?? Path.GetFileNameWithoutExtension(path), file.Label ?? file.Id ?? "suite", corpus, cases);
    }
}
