# Query Ollama model parameters

```
curl -s http://localhost:11434/api/show -d "{ \"name\": \"gemma4:12b\"}" | jq 
```

maximum context is in "context_length" (262144)

```
{
  "thinking": {
    "values": [
      false,
      true
    ],
    "default": true
  },
  "license": "...",
  "modelfile": "...",
  "parameters": "draft_num_predict              3\ntemperature                    1\ntop_k                          64\ntop_p                          0.95",
  "template": "{{ .Prompt }}",
  "renderer": "gemma4-large",
  "parser": "gemma4",
  "template_source": "renderer_parser",
  "details": {
    "parent_model": "imatrix-20260927-12b/model-q4_K_M.gguf",
    "format": "gguf",
    "family": "gemma4",
    "families": [
      "gemma4"
    ],
    "parameter_size": "11.9B",
    "quantization_level": "Q4_K_M",
    "runner": "llamacpp"
  },
  "model_info": {
    ...
    "gemma4.attention.key_length": 512,
    "gemma4.attention.key_length_swa": 256,
    "gemma4.attention.layer_norm_rms_epsilon": 0.000001,
    "gemma4.attention.shared_kv_layers": 0,
    "gemma4.attention.sliding_window": 1024,
    "gemma4.attention.value_length": 512,
    "gemma4.attention.value_length_swa": 256,
    "gemma4.block_count": 48,
    "gemma4.context_length": 262144,
    "gemma4.embedding_length": 3840,
    "gemma4.embedding_length_per_layer_input": 0,
    "gemma4.feed_forward_length": 15360,
    "gemma4.final_logit_softcapping": 30,
    "gemma4.rope.dimension_count": 512,
    "gemma4.rope.dimension_count_swa": 256,
    "gemma4.rope.freq_base": 1000000,
    "gemma4.rope.freq_base_swa": 10000,
    "general.architecture": "gemma4",
    "general.base_model.0.name": "Gemma 4 12B",
    "general.base_model.0.organization": "Google",
    "general.base_model.0.repo_url": "https://huggingface.co/google/gemma-4-12B",
    "general.base_model.count": 1,
    ...
    "general.parameter_count": 52379904,
    "general.quantization_version": 2,
    "general.sampling.temp": 1,
    "general.sampling.top_k": 64,
    "general.sampling.top_p": 0.95,
    "general.size_label": "52M",
    "general.tags": [
      "any-to-any"
    ],
    "general.type": "mmproj"
  },
  "capabilities": [
    "completion",
    "vision",
    "audio",
    "tools",
    "thinking"
  ],
  "modified_at": "2026-11-26T12:13:26.1284736+01:00",
  "requires": "0.30.9"
}
```