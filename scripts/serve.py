#!/usr/bin/env python3
"""
Inference server for the fine-tuned FLT model.

Serves the model via a FastAPI REST API with:
  - POST /chat       — conversational inference
  - POST /query      — graph-grounded single-turn query
  - GET  /health     — health check
  - GET  /graph/nodes — list all graph nodes
  - GET  /graph/edges — list all graph edges

Usage:
    python scripts/serve.py \
        --model checkpoints/flt-qwen2.5-7b/final \
        --graph dataset/flt_concept_graph.json \
        --port 8000
"""

import argparse
import json

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="FLT Learning Theory Model", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals set in main()
model = None
tokenizer = None
graph_data = None

SYSTEM_PROMPT = (
    "You are a formal learning theory expert. You answer questions about "
    "PAC learning, online learning, Gold's model, VC dimension, complexity "
    "measures, and their relationships. You cite theorems precisely, "
    "including separation witnesses and non-implications. When a relationship "
    "does NOT hold, you state so explicitly with the known counterexample."
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    messages: list[dict]
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9


class QueryRequest(BaseModel):
    question: str
    max_new_tokens: int = 512
    temperature: float = 0.3  # lower temp for factual queries


class ChatResponse(BaseModel):
    response: str
    usage: dict


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def generate(messages: list[dict], max_new_tokens: int, temperature: float,
             top_p: float) -> tuple[str, dict]:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 0.01),
            top_p=top_p,
            do_sample=temperature > 0.01,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = outputs[0][input_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    usage = {
        "prompt_tokens": input_len,
        "completion_tokens": len(new_tokens),
    }
    return response, usage


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    messages = req.messages
    # Prepend system prompt if missing
    if not messages or messages[0].get("role") != "system":
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    response, usage = generate(
        messages, req.max_new_tokens, req.temperature, req.top_p
    )
    return ChatResponse(response=response, usage=usage)


@app.post("/query", response_model=ChatResponse)
async def query(req: QueryRequest):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": req.question},
    ]
    response, usage = generate(
        messages, req.max_new_tokens, req.temperature, req.top_p
    )
    return ChatResponse(response=response, usage=usage)


@app.get("/graph/nodes")
async def graph_nodes():
    if not graph_data:
        return {"nodes": []}
    return {"nodes": [
        {"id": n["id"], "name": n["name"], "category": n["category"]}
        for n in graph_data.get("nodes", [])
    ]}


@app.get("/graph/edges")
async def graph_edges():
    if not graph_data:
        return {"edges": []}
    return {"edges": graph_data.get("edges", [])}


@app.get("/")
async def root():
    return FileResponse("app/templates/index.html")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global model, tokenizer, graph_data

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="checkpoints/flt-qwen2.5-7b/final")
    parser.add_argument("--base-model", default=None,
                        help="Base model if --model is a LoRA adapter")
    parser.add_argument("--graph", default="dataset/flt_concept_graph.json")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--load-4bit", action="store_true")
    args = parser.parse_args()

    # Load graph
    with open(args.graph) as f:
        graph_data = json.load(f)

    # Load model
    bnb_config = None
    if args.load_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    if args.base_model:
        # LoRA adapter on top of base
        print(f"Loading base model: {args.base_model}")
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        print(f"Loading adapter: {args.model}")
        model = PeftModel.from_pretrained(base, args.model)
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model, trust_remote_code=True
        )
    else:
        # Merged model or full checkpoint
        print(f"Loading model: {args.model}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, trust_remote_code=True
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    print(f"Model loaded. Serving on {args.host}:{args.port}")

    # Mount static files
    app.mount("/static", StaticFiles(directory="app/static"), name="static")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
