# %%
from transformers import AutoTokenizer
import transformers
import torch

# %%
model = "tiiuae/falcon-7b-instruct"
#model = model.to('mps')

# %%
tokenizer = AutoTokenizer.from_pretrained(model)

# %%
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    trust_remote_code=True,
    device=torch.device('mps')
)

# %%


def generate_text(prompt):

    return pipeline(prompt,
                    max_length=200,
                    do_sample=True,
                    top_k=10,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id)
    
# %%
prompt = "Tell me about Albert Einstein and his most famous equation."
response = generate_text(prompt=prompt)
print(response)

# %%
for seq in response:
    print(f"Result:\n\n{seq['generated_text']}")

# %%