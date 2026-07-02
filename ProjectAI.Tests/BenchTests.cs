using ProjectAI.Bench;
using Xunit;

namespace ProjectAI.Tests;

// Contract tests for the benchmark foundation (P4a): the deterministic check evaluator, the median used for
// throughput, suite loading (with its eval corpus), and the run JSON round-trip the reports and the /benchmark
// endpoints both depend on. The end-to-end path (bpb + generation) is exercised by `projectai bench` on a real
// checkpoint — these pin the pure logic.
public class BenchTests
{
    // ---- CheckEvaluator -----------------------------------------------------------------------------------------

    [Fact]
    public void Contains_Checks_Are_CaseInsensitive_And_Per_Needle()
    {
        var c = new BenchCase("t", "p", MustInclude: ["Paris", "France"]);
        var checks = CheckEvaluator.Evaluate(c, "the capital is paris.", "eos");

        Assert.Equal(3, checks.Count); // 2 contains + maxTokens
        Assert.True(checks[0].Passed);   // paris (case-insensitive)
        Assert.False(checks[1].Passed);  // France absent
        Assert.True(checks[2].Passed);   // stopped on eos, not the cap
    }

    [Fact]
    public void Exact_Check_Trims_Whitespace()
    {
        var c = new BenchCase("t", "p", Reference: "42");
        Assert.True(CheckEvaluator.Evaluate(c, "  42\n", "eos")[0].Passed);
        Assert.False(CheckEvaluator.Evaluate(c, "42 is the answer", "eos")[0].Passed);
    }

    [Fact]
    public void Regex_Check_Matches_And_Bad_Pattern_Fails_Instead_Of_Throwing()
    {
        var ok = new BenchCase("t", "p", Reference: "regex:(?i)\\bbat\\b");
        Assert.True(CheckEvaluator.Evaluate(ok, "A Bat!", "eos")[0].Passed);

        var bad = new BenchCase("t", "p", Reference: "regex:([unclosed");
        Assert.False(CheckEvaluator.Evaluate(bad, "anything", "eos")[0].Passed);
    }

    [Fact]
    public void JsonValid_Accepts_Fenced_And_Prefixed_Json()
    {
        var c = new BenchCase("t", "p", Reference: "jsonValid");
        Assert.True(CheckEvaluator.Evaluate(c, "{\"name\":\"test\"}", "eos")[0].Passed);
        Assert.True(CheckEvaluator.Evaluate(c, "Sure! Here it is: {\"name\":\"test\"}", "eos")[0].Passed);
        Assert.False(CheckEvaluator.Evaluate(c, "no json here", "eos")[0].Passed);
    }

    [Fact]
    public void MaxTokens_Check_Fails_When_Cut_Off()
    {
        var c = new BenchCase("t", "p");
        Assert.False(CheckEvaluator.Evaluate(c, "text", "maxTokens").Single(x => x.Kind == "maxTokens").Passed);
        Assert.True(CheckEvaluator.Evaluate(c, "text", "eos").Single(x => x.Kind == "maxTokens").Passed);
    }

    // ---- Median -------------------------------------------------------------------------------------------------

    [Fact]
    public void Median_Handles_Odd_Even_And_Empty()
    {
        Assert.Equal(2.0, BenchRunner.Median([3.0, 1.0, 2.0]));
        Assert.Equal(2.5, BenchRunner.Median([4.0, 1.0, 2.0, 3.0]));
        Assert.Equal(0.0, BenchRunner.Median([]));
    }

    // ---- SuiteLoader --------------------------------------------------------------------------------------------

    [Fact]
    public void SuiteLoader_Reads_Cases_And_Relative_Corpus()
    {
        string dir = Path.Combine(Path.GetTempPath(), "projectai-bench-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            File.WriteAllText(Path.Combine(dir, "corpus.txt"), "held-out text");
            File.WriteAllText(Path.Combine(dir, "suite.json"), """
                { "id": "mini", "label": "Mini", "evalCorpusFile": "corpus.txt",
                  "cases": [ { "id": "a", "prompt": "hello", "mustInclude": ["hi"], "maxTokens": 32 } ] }
                """);

            var suite = SuiteLoader.Load(Path.Combine(dir, "suite.json"), modelsDir: dir);
            Assert.Equal("mini", suite.Id);
            Assert.Equal("held-out text", suite.EvalCorpus);
            Assert.Single(suite.Cases);
            Assert.Equal(32, suite.Cases[0].MaxTokens);
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    [Fact]
    public void SuiteLoader_Rejects_Missing_Suite_Empty_Cases_And_Missing_Corpus()
    {
        string dir = Path.Combine(Path.GetTempPath(), "projectai-bench-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            Assert.Throws<FileNotFoundException>(() => SuiteLoader.Load("nope", modelsDir: dir));

            string empty = Path.Combine(dir, "empty.json");
            File.WriteAllText(empty, """{ "id": "e", "cases": [] }""");
            Assert.Throws<InvalidDataException>(() => SuiteLoader.Load(empty, modelsDir: dir));

            string badCorpus = Path.Combine(dir, "bad.json");
            File.WriteAllText(badCorpus, """{ "id": "b", "evalCorpusFile": "missing.txt", "cases": [ { "id":"a", "prompt":"p" } ] }""");
            Assert.Throws<FileNotFoundException>(() => SuiteLoader.Load(badCorpus, modelsDir: dir));
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    // ---- Run persistence ----------------------------------------------------------------------------------------

    [Fact]
    public void BenchRun_RoundTrips_Through_Save_And_Load()
    {
        string dir = Path.Combine(Path.GetTempPath(), "projectai-bench-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var run = new BenchRun(
                "bench-test-000001", "mini", "cpu", "2026-07-02T00:00:00Z", "2026-07-02T00:01:00Z",
                new BenchRunConfig("mini", ["m1"], "cpu"),
                new RunMeta("dev", "host", "os", "cpu", [new ModelStamp("m1", 10, 1234, "d64·L2", "abc123")]),
                [new CellResult("m1", "a", "out", 3, 5, "eos", 0.5, 10.0,
                    [new CheckResult("contains", "hi", true)], 1.0)],
                [new ArmAggregate("m1", 1, 1, 1.2345, 10.0, 1.0, new Dictionary<string, int> { ["eos"] = 1 })],
                "done");

            string path = BenchRunner.SaveRun(dir, run);
            Assert.True(File.Exists(path));

            var loaded = BenchRunner.LoadRun(dir, run.Id);
            Assert.NotNull(loaded);
            Assert.Equal(run.Id, loaded!.Id);
            Assert.Equal(1.2345, loaded.Aggregates[0].MeanBpb);
            Assert.Equal("eos", loaded.Cells[0].Stop);
            Assert.True(loaded.Cells[0].Checks[0].Passed);

            Assert.Contains(BenchRunner.ListRuns(dir), r => r.Id == run.Id);
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    // The report must render without throwing and carry the honesty labels (n, greedy, warmup discard).
    [Fact]
    public void Report_Renders_With_Rigor_Labels()
    {
        var run = new BenchRun(
            "bench-test-000002", "mini", "cpu", "2026-07-02T00:00:00Z", null,
            new BenchRunConfig("mini", ["m1"], "cpu", Repeats: 3),
            new RunMeta("dev", "host", "os", "cpu", [new ModelStamp("m1", 10, 1234, "d64·L2", "abcdef1234567890")]),
            [new CellResult("m1", "a", "out", 3, 5, "eos", 0.5, 10.0, [new CheckResult("contains", "hi", true)], 1.0)],
            [new ArmAggregate("m1", 1, 1, null, 10.0, 1.0, new Dictionary<string, int> { ["eos"] = 1 })],
            "done");

        string md = BenchReporter.Markdown(run);
        Assert.Contains("greedy", md);
        Assert.Contains("1 warmup discarded", md);
        Assert.Contains("| `m1` |", md);
        Assert.Contains("floor signals", md);
    }
}
