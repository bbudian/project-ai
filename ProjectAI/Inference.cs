using ProjectAI.Core;
using ProjectAI.Models;
using ProjectAI.Tokenizers;

// Shared autoregressive decode core used by the CLI `generate` command and the `serve` HTTP endpoint. Re-runs
// the model on the growing sequence (no KV cache yet), picks each token via the sampler, and stops at EOS.
// Returns the full decoded text (prompt + continuation).
internal static class Inference
{
    public static string GenerateText(
        IComputeBackend be, LlamaModel model, ITokenizer tokenizer, ModelConfig config,
        string prompt, ISampler sampler, int maxTokens)
    {
        int vocab = tokenizer.VocabSize, eos = tokenizer.EosId;
        var generated = new List<int>(tokenizer.Encode(prompt));
        if (generated.Count == 0) generated.Add((int)' ');           // empty prompt → start from a space
        if (generated.Count >= config.MaxSequenceLength)             // keep room under the RoPE table length
            generated = generated.Skip(generated.Count - (config.MaxSequenceLength - 1)).ToList();

        // Incremental decoding with a KV cache: prefill the prompt once, then forward a single token per step
        // (the cache supplies the prior context), instead of re-running the whole sequence each step.
        var cache = new KvCache(be, config, maxBatch: 1, maxSequenceLength: config.MaxSequenceLength);
        var ctx = ForwardContext.Inference() with { Cache = cache };

        using (GradMode.NoGrad())
        {
            var logits = model.Forward(be.FromHost(ToFloats(generated), new Shape(1, generated.Count), DType.F32), ctx);
            int produced = 0;
            while (true)
            {
                int next = SampleLast(be, logits, vocab, sampler);
                if (next == eos) break;
                generated.Add(next);
                if (++produced >= maxTokens || generated.Count >= config.MaxSequenceLength) break;
                logits = model.Forward(be.FromHost([(float)next], new Shape(1, 1), DType.F32), ctx);
            }
        }
        return tokenizer.Decode(generated);
    }

    // Samples from the last row of a [1, rows, vocab] logits tensor (the only row on a decode step).
    private static int SampleLast(IComputeBackend be, Tensor logits, int vocab, ISampler sampler)
    {
        var host = new float[logits.ElementCount];
        be.ToHost(logits, host);
        int rows = host.Length / vocab;
        return sampler.Sample(host.AsSpan((rows - 1) * vocab, vocab));
    }

    private static float[] ToFloats(IReadOnlyList<int> ids)
    {
        var f = new float[ids.Count];
        for (int i = 0; i < ids.Count; i++) f[i] = ids[i];
        return f;
    }
}
