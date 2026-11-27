# create model with increased context length 

```
FROM gemma4:12b
PARAMETER num_ctx 16384
```

```
ollama create gemma4-26b-256k-ctx -f Modelfile_gemma4-26b-256k-ctx.txt
```
