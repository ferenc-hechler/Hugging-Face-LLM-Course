# Hugging-Face-LLM-Course

Material for testing the [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/)



## setup

```
# jupyter notbook
pip install jupyter ipykernel

# colab instructions
pip install datasets evaluate transformers[sentencepiece]

# https://pytorch.org/get-started/locally/
pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu134
```

## check nvidia drivers

```
nvidia-smi
```

## check cuda

```
import torch
torch.cuda.is_available()
print(torch.cuda.get_device_name(0))
print('Memory Usage:')
print('Allocated: ', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
print('Cached:    ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
print('Properties:', torch.cuda.get_device_properties('cuda'))
# Properties: _CudaDeviceProperties(name='NVIDIA GeForce RTX 4060', 
#   major=8, minor=9, total_memory=8187MB, multi_processor_count=24, 
#   uuid=af004804-a281-76b8-8457-7734b0f16780, 
#   pci_bus_id=1, pci_device_id=0, pci_domain_id=0, L2_cache_size=24MB)
```


## set HF_TOKEN

```
hf auth login
```

## clone notebooks

https://github.com/huggingface/notebooks


## Chapter 1.3 Problems

### out of sockets

caused by progress bar in Jupyter Notebook:

```
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
```

### Pipeline "question-answering" and "summarization" has been removed

```
KeyError: 'The task summarization has been removed from transformers. Use the `text-generation` pipeline with a chat model instead. See the migration guide for more details: https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md#pipelines'
```

Replace:

```
from transformers import pipeline

#question_answerer = pipeline("question-answering")
#question_answerer(
#    question="Where do I work?",
#    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
#)

question="Where do I work?"
context="My name is Sylvain and I work at Hugging Face in Brooklyn"

question_answer = pipeline("text-generation")  # , model="google/gemma-3-270m-it")

message_history = [
    {
        "role": "user",
        "content": f"Answer the question: '{question}' with the context '{context}'"
    }
]
print(question_answer(message_history)[0]["generated_text"][-1]["content"])
```


```
TXT = """..."""

# summarizer = pipeline("summarization")
# summarizer(TXT)

summarizer = pipeline("text-generation")
message_history = [
    {   "role": "system", "content": "/no_think"},
    {
        "role": "user",
        "content": f"Summarize the following text:\n\n{TXT}"
    }
]
result = summarizer(message_history, max_new_tokens=512)
print(result[0]["generated_text"][-1]["content"])
```


## carbon measure

```
pip install codecarbon
```

```
from codecarbon import EmissionsTracker

tracker = EmissionsTracker()
tracker.start()

# Your code here

emissions = tracker.stop()
print(f"Emissions: {emissions} kg CO₂")
```


## Offline Mode

### Download API

https://huggingface.co/docs/huggingface_hub/guides/download

```
pip install huggingface_hub
```

create token and accept gated license for model.

set HF_TOKEN with fine-grained read permission (+ gated):

```
hf auth login
```




```
from huggingface_hub import snapshot_download

snapshot_download(repo_id="google/gemma-3-270m-it", repo_type="model")
```


