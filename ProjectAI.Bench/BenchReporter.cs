using System.Text;

namespace ProjectAI.Bench;

/// <summary>
/// Renders a run as markdown — the honest version: bpb is the headline (tokenizer-invariant), every rate prints
/// its n, the stop-reason mix contextualizes throughput, wall-clock is labeled with the backend, and substring
/// checks are labeled floor signals. One renderer serves the CLI and the serve endpoint.
/// </summary>
public static class BenchReporter
{
    public static string Markdown(BenchRun run)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"# Benchmark report — suite `{run.SuiteId}`");
        sb.AppendLine();
        sb.AppendLine($"Run `{run.Id}` · started {run.StartedUtc} · state **{run.State}** · backend `{run.Backend}` on {run.Meta.Host} ({run.Meta.Os})");
        sb.AppendLine($"Decoding: {(run.Config.Sample ? $"sampled (seed {run.Config.Seed})" : "greedy")} · repeats {run.Config.Repeats} (median; 1 warmup discarded) · timings are wall-clock on `{run.Meta.BackendUsed}`");
        sb.AppendLine();

        sb.AppendLine("## Models");
        sb.AppendLine();
        sb.AppendLine("| model | step | params | config | sha256 |");
        sb.AppendLine("|---|---|---|---|---|");
        foreach (var m in run.Meta.Models)
            sb.AppendLine($"| `{m.Name}` | {m.Step:N0} | {m.ParamCount:N0} | {m.ConfigSummary} | `{m.CheckpointSha256[..12]}…` |");
        sb.AppendLine();

        sb.AppendLine("## Aggregates");
        sb.AppendLine();
        sb.AppendLine("| model | bpb ↓ | median tok/s | check pass | n | stop mix |");
        sb.AppendLine("|---|---|---|---|---|---|");
        foreach (var a in run.Aggregates)
        {
            string stops = string.Join(", ", a.StopReasons.OrderByDescending(kv => kv.Value).Select(kv => $"{kv.Key}:{kv.Value}"));
            sb.AppendLine($"| `{a.Model}` | {(a.MeanBpb is { } b ? b.ToString("0.0000") : "—")} | {a.MedianTokPerSec:0.00} | {a.CheckPassRate:P0} | {a.N} | {stops} |");
        }
        sb.AppendLine();
        sb.AppendLine("*bpb = bits per UTF-8 byte over the suite's held-out corpus (lower is better; comparable across tokenizers). " +
                      "Check pass rates over n cases — treat small n with caution. `contains` checks are floor signals, not grades.*");
        sb.AppendLine();

        sb.AppendLine("## Cases");
        sb.AppendLine();
        var caseIds = run.Cells.Where(c => c.CaseId != "__bpb__").Select(c => c.CaseId).Distinct().ToList();
        var models = run.Aggregates.Select(a => a.Model).ToList();
        sb.Append("| case |");
        foreach (var m in models) sb.Append($" `{m}` |");
        sb.AppendLine();
        sb.Append("|---|");
        foreach (var _ in models) sb.Append("---|");
        sb.AppendLine();
        foreach (string caseId in caseIds)
        {
            sb.Append($"| {caseId} |");
            foreach (string model in models)
            {
                var cell = run.Cells.FirstOrDefault(c => c.CaseId == caseId && c.Model == model);
                sb.Append(cell is null ? " — |" : cell.Error is not null
                    ? $" ⚠ {Truncate(cell.Error, 40)} |"
                    : $" {(cell.CheckPassRate >= 1 ? "✓" : cell.CheckPassRate > 0 ? "~" : "✗")} {cell.CheckPassRate:P0} · {cell.MedianTokPerSec:0.0} tok/s · {cell.Stop} |");
            }
            sb.AppendLine();
        }
        sb.AppendLine();

        var errors = run.Cells.Where(c => c.Error is not null).ToList();
        if (errors.Count > 0)
        {
            sb.AppendLine("## Errors");
            sb.AppendLine();
            foreach (var e in errors) sb.AppendLine($"- `{e.Model}` / {e.CaseId}: {e.Error}");
            sb.AppendLine();
        }
        return sb.ToString();
    }

    private static string Truncate(string s, int max) => s.Length <= max ? s : s[..max] + "…";
}
