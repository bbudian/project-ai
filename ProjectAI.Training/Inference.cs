using ProjectAI.Core;
using ProjectAI.Models;
using ProjectAI.Tokenizers;

namespace ProjectAI.Training;

/// <summary>
/// Shared autoregressive decode core used by the CLI <c>generate</c> command, the <c>serve</c> HTTP endpoint, and
/// the trainer's post-training sample. Prefills the prompt into a KV cache, then forwards one token per step
/// (the cache supplies the prior context) instead of re-running the whole sequence, picks each token via the
/// sampler, and stops at EOS / the token budget / the context limit.
/// </summary>
public static class Inference
{
    /// <summary>
    /// The outcome of a generation. <see cref="Continuation"/> is just the newly generated text (what a chat UI
    /// should show, since the user's prompt is already on screen); <see cref="FullText"/> is prompt + continuation
    /// (what the CLI prints). The token counts and <see cref="StopReason"/> are diagnostics.
    /// </summary>
    public sealed record GenerationResult(
        string Continuation, string FullText, int PromptTokens, int GeneratedTokens, string StopReason);

    public static GenerationResult GenerateText(
        IComputeBackend be, LlamaModel model, ITokenizer tokenizer, ModelConfig config,
        string prompt, ISampler sampler, int maxTokens)
    {
        // The logits' last axis is config.VocabSize (the embedding rows), which HF models often pad past the
        // tokenizer's max id — so row arithmetic MUST use the model width, and the sample window is capped at the
        // tokenizer's ids so a padded (undecodable) id can never be emitted.
        int rowWidth = config.VocabSize, sampleWidth = Math.Min(tokenizer.VocabSize, config.VocabSize);
        int eos = tokenizer.EosId;
        var generated = new List<int>(tokenizer.Encode(prompt));
        if (generated.Count == 0) generated.Add((int)' ');           // empty prompt → start from a space
        if (generated.Count >= config.MaxSequenceLength)             // keep room under the RoPE table length
            generated = generated.Skip(generated.Count - (config.MaxSequenceLength - 1)).ToList();
        int promptTokens = generated.Count;

        // Incremental decoding with a KV cache: prefill the prompt once, then forward a single token per step
        // (the cache supplies the prior context), instead of re-running the whole sequence each step.
        var cache = new KvCache(be, config, maxBatch: 1, maxSequenceLength: config.MaxSequenceLength);
        var ctx = ForwardContext.Inference() with { Cache = cache };

        string stopReason = "maxTokens";
        using (GradMode.NoGrad())
        {
            var logits = model.Forward(be.FromHost(ToFloats(generated), new Shape(1, generated.Count), DType.F32), ctx);
            int produced = 0;
            while (true)
            {
                int next = SampleLast(be, logits, rowWidth, sampleWidth, sampler);
                if (next == eos) { stopReason = "eos"; break; }
                generated.Add(next);
                if (++produced >= maxTokens) { stopReason = "maxTokens"; break; }
                if (generated.Count >= config.MaxSequenceLength) { stopReason = "context"; break; }
                logits = model.Forward(be.FromHost([(float)next], new Shape(1, 1), DType.F32), ctx);
            }
        }

        // Byte-level decode is a pure concatenation per token, so decoding only the appended tokens yields exactly
        // the continuation (no echoed prompt) — and decoding the whole thing yields the full text.
        string continuation = tokenizer.Decode(generated.Skip(promptTokens).ToList());
        string fullText = tokenizer.Decode(generated);
        return new GenerationResult(continuation, fullText, promptTokens, generated.Count - promptTokens, stopReason);
    }

    // Samples from the last row of a [1, rows, rowWidth] logits tensor (the only row on a decode step).
    // rowWidth = the model's vocab (the tensor's true last axis); sampleWidth ≤ rowWidth windows out padded ids.
    private static int SampleLast(IComputeBackend be, Tensor logits, int rowWidth, int sampleWidth, ISampler sampler)
    {
        var host = new float[logits.ElementCount];
        be.ToHost(logits, host);
        int rows = host.Length / rowWidth;
        return sampler.Sample(host.AsSpan((rows - 1) * rowWidth, sampleWidth));
    }

    private static float[] ToFloats(IReadOnlyList<int> ids)
    {
        var f = new float[ids.Count];
        for (int i = 0; i < ids.Count; i++) f[i] = ids[i];
        return f;
    }
}
