# %%
from transformers import AutoTokenizer, pipeline
from transformers import LlamaForCausalLM, LlamaTokenizer
import transformers
import torch

# %%

model_checkpoint = "../../llama_weights/hf_format/llama-2-7b-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_checkpoint)
model = LlamaForCausalLM.from_pretrained(model_checkpoint)

# %%
model = model.to("mps")

# %%
_model = LlamaForCausalLM.from_pretrained(model_checkpoint)
_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


# %%
prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")

# %%
generate_ids = model.generate(inputs.input_ids, max_length=30)

# %%
tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


# %%
# Inferencing
response = llama_pipeline(
    'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)

for seq in response:
    print(f"Result:\n\n{seq['generated_text']}")


