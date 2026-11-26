# create model with increased context length 

```
FROM gemma4:12b
PARAMETER num_ctx 16384
```

```
ollama create gemma4-12b-16k-ctx -f Modelfile_gemma4-12b-16k-ctx.txt
```
