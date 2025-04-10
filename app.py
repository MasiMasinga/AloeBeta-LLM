import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

model_id = "HPAI-BSC/Llama3.1-Aloe-Beta-8B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(request: ChatRequest):
    messages = [
        {"role": "system", "content": "You are an expert medical assistant named Aloe, developed by the High Performance Artificial Intelligence Group at Barcelona Supercomputing Center(BSC). You are to be a helpful, respectful, and honest assistant."},
        {"role": "user", "content": request.message},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipe(
        prompt,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    reply = outputs[0]["generated_text"][len(prompt):].strip()
    return {"response": reply}

if __name__ == "__main__":
    uvicorn.run("aloe8b:app", host="0.0.0.0", port=8000, reload=False)
