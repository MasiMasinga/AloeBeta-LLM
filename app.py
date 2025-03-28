from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "HPAI-BSC/Llama3.1-Aloe-Beta-8B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

input_text = "What are the symptoms of pneumonia?"
inputs = tokenizer(input_text, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=100)

print(tokenizer.decode(output[0], skip_special_tokens=True))

