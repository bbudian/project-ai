using ProjectAI.Core;
using ProjectAI.Models;
using ProjectAI.Tokenizers;

// A stateful chat session (Phase 1 of live chat): ONE persistent KV cache kept warm across turns, so each new
// user message ingests only its own new tokens — no recompute of the whole history — and the assistant reply
// streams out token-by-token via a callback. Instruct models (those whose tokenizer has <|im_start|>/<|im_end|>)
// get the chat template applied and stop at <|im_end|>; base models just continue the raw text.
//
// Not thread-safe: the caller serializes Turn() under the server's InferenceLock (the model/backend isn't either).
internal sealed class ChatSession
{
    private readonly IComputeBackend _be;
    private readonly LlamaModel _model;
    private readonly ITokenizer _tok;
    private readonly ModelConfig _config;
    private readonly KvCache _cache;
    private readonly int _imEnd;
    private int _position; // tokens currently held in the cache

    public string ModelName { get; }
    public string BackendId { get; }
    public bool Instruct { get; }
    public int ContextLimit => _config.MaxSequenceLength;
    public int Position => _position;

    public ChatSession(IComputeBackend be, LoadedModel loaded, string backendId)
    {
        _be = be;
        _model = loaded.Model;
        _tok = loaded.Tokenizer;
        _config = loaded.Config;
        ModelName = loaded.Name;
        BackendId = backendId;
        _cache = new KvCache(be, _config, maxBatch: 1, maxSequenceLength: _config.MaxSequenceLength);

        int imStart = SingleToken("<|im_start|>");
        _imEnd = SingleToken("<|im_end|>");
        Instruct = imStart >= 0 && _imEnd >= 0;
    }

    public sealed record TurnResult(int PromptTokens, int GeneratedTokens, string StopReason);

    /// <summary>
    /// Ingests <paramref name="userText"/> into the warm cache, then streams the assistant reply through
    /// <paramref name="onDelta"/> (decoded text, UTF-8-boundary-safe). Stops on EOS / <|im_end|>, the token budget,
    /// the context limit, or when <paramref name="cancel"/> is signalled (checked once per generated token →
    /// StopReason "canceled"). Returns per-turn diagnostics.
    /// </summary>
    public TurnResult Turn(string userText, ISampler sampler, int maxTokens, CancellationToken cancel, Action<string> onDelta)
    {
        var promptIds = BuildTurnPrompt(userText);
        if (_position + promptIds.Length >= _config.MaxSequenceLength)
            return new TurnResult(promptIds.Length, 0, "context_full");

        int eos = _tok.EosId;
        var generated = new List<int>();
        string stop = "maxTokens";

        using (GradMode.NoGrad())
        {
            var logits = Forward(promptIds); // prefill: ingest only the new user tokens at the cached offset
            int produced = 0, emitted = 0;
            while (true)
            {
                if (cancel.IsCancellationRequested) { stop = "canceled"; break; }
                int next = SampleLast(logits, sampler);
                if (next == eos) { stop = "eos"; break; }
                if (Instruct && next == _imEnd) { stop = "im_end"; break; }

                generated.Add(next);
                // Stream the decoded delta, holding back an incomplete trailing UTF-8 sequence (a token can be a
                // partial multibyte char) so we never emit a transient U+FFFD that the next token would resolve.
                string full = _tok.Decode(generated);
                int stable = full.Length;
                while (stable > emitted && full[stable - 1] == '�') stable--;
                if (stable > emitted) { onDelta(full[emitted..stable]); emitted = stable; }

                if (++produced >= maxTokens) { stop = "maxTokens"; break; }
                if (_position >= _config.MaxSequenceLength - 1) { stop = "context"; break; }
                logits = Forward([next]);
            }
            // Flush any held-back tail (e.g. a final char whose bytes only just completed).
            string finalText = _tok.Decode(generated);
            if (finalText.Length > emitted) onDelta(finalText[emitted..]);
        }

        // Close the assistant turn in the cache so the next user message is delimited correctly (instruct only).
        if (Instruct && _position < _config.MaxSequenceLength - 4)
            using (GradMode.NoGrad()) Forward(_tok.Encode("<|im_end|>\n"));

        return new TurnResult(promptIds.Length, generated.Count, stop);
    }

    private int[] BuildTurnPrompt(string text)
    {
        // Instruct models expect the chat template; base models just continue raw text. Encode splits the
        // <|im_*|> specials out to their single ids, so the literal template string tokenizes correctly.
        string s = Instruct ? $"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n" : text;
        var ids = _tok.Encode(s);
        return [.. ids];
    }

    // Forwards a chunk of token ids through the model with the persistent cache (RoPE'd at the current offset and
    // appended), advancing the position, and returns the logits for the chunk's last position.
    private Tensor Forward(IReadOnlyList<int> ids)
    {
        var ctx = ForwardContext.Inference() with { Cache = _cache };
        var input = _be.FromHost(ToFloats(ids), new Shape(1, ids.Count), DType.F32);
        var logits = _model.Forward(input, ctx);
        _position += ids.Count;
        return logits;
    }

    private int SampleLast(Tensor logits, ISampler sampler)
    {
        int vocab = _tok.VocabSize;
        var host = new float[logits.ElementCount];
        _be.ToHost(logits, host);
        int rows = host.Length / vocab;
        return sampler.Sample(host.AsSpan((rows - 1) * vocab, vocab));
    }

    private int SingleToken(string s)
    {
        var ids = _tok.Encode(s);
        return ids.Count == 1 ? ids[0] : -1;
    }

    private static float[] ToFloats(IReadOnlyList<int> ids)
    {
        var f = new float[ids.Count];
        for (int i = 0; i < ids.Count; i++) f[i] = ids[i];
        return f;
    }
}
