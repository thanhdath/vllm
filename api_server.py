import argparse
import json
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
import logging
logging.getLogger().setLevel(logging.INFO)
from transformers import pipeline

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
engine = None

USER_MESSAGE_TEMPLATE = """<bắt đầu đoạn văn>
{context}
<kết thúc đoạn văn>
Trả lời người dùng dựa vào thông tin trong các đoạn văn.
Người dùng: {message}"""

TEMPLATE = """<|system|>
</s>
{prompt_history}
<|user|>{user_message}</s>
<|assistant|>
"""

def make_prompt_relevant_articles(relevant_articles):
    prompt = []
    n_cur_word = 0
    for i, article in enumerate(relevant_articles):
        append_content = f"[{i+1}] {article}"
        n_words = len(append_content.split())
        if n_cur_word + n_words > 2500 / 1.2:
            break
        
        prompt.append(append_content)
        n_cur_word += n_words
        
    prompt = "\n\n".join(prompt)
    return prompt


def get_prompt_history(history):
    if len(history) == 0:
        return ""
    history = history[-4:]
    if history[0].startswith("<|assistant|>"):
        history = history[1:]
    if history[-1].startswith("<|user|>"):
        history = history[:-1]
    prompt_history = "\n".join(history)
    return prompt_history

def make_user_prompt(message, history, sources):
    history = [f"<|{role}|>\n{content}</s>" for role, content in [(m["role"], m["content"]) for m in history]]
    history = get_prompt_history(history)

    context = make_prompt_relevant_articles(sources)
    user_message = USER_MESSAGE_TEMPLATE.format(context=context, message=message)
 
    user_message_length = len(user_message.split())
    history_length = len(history.split())

    if user_message_length + history_length > 3000/ 1.2:
        history = ""

    prompt = TEMPLATE.format(prompt_history=history, user_message=user_message)
    return prompt

@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()

    # prompt = request_dict.pop("prompt")
    user_message = request_dict.pop("user_message")
    history = request_dict.pop("history")
    sources = request_dict.pop("sources")

    # classify user_message
    if args.use_intent_classifier:
        intent = intent_classifier(user_message)
        intent = intent[0]['label']
        if intent == 'toxic':
            ret = {"text": ['Tôi không thể trả lời câu hỏi này. Xin lỗi vì sự bất tiện này.']}
            return JSONResponse(ret)
        # intent = law | other

    prompt = make_user_prompt(user_message, history, sources)

    # print(prompt)

    stream = request_dict.pop("stream", False)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()

    results_generator = engine.generate(prompt,
                                        sampling_params,
                                        request_id)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            # prompt = request_output.prompt
            text_outputs = [
                output.text for output in request_output.outputs
            ]
            ret = {"text": text_outputs}
            yield (json.dumps(ret) + "\0").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results())

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    prompt = final_output.prompt
    text_outputs = [output.text for output in final_output.outputs]
    ret = {"text": text_outputs}
    return JSONResponse(ret)

@app.post("/")
async def chat_completions(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    print(request_dict)
    messages = request_dict.pop("messages")
    user_message = messages[-1]["content"]
    history = []
    sources = messages[-1].get("sources", [])

    # classify user_message
    if args.use_intent_classifier:
        intent = intent_classifier(user_message)
        intent = intent[0]['label']
        if intent == 'toxic':
            ret = {"text": ['Tôi không thể trả lời câu hỏi này. Xin lỗi vì sự bất tiện này.']}
            return JSONResponse(ret)
        # intent = law | other

    prompt = make_user_prompt(user_message, history, sources)

    # print(prompt)

    stream = request_dict.pop("stream", False)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()

    results_generator = engine.generate(prompt,
                                        sampling_params,
                                        request_id)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            output = request_output.outputs[0] 
            text = output.text
            ret = {"text": text}
            yield json.dumps(ret)

    if stream:
        return StreamingResponse(stream_results())

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    prompt = final_output.prompt
    text_outputs = [output.text for output in final_output.outputs]
    ret = {"text": text_outputs}
    return JSONResponse(ret)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    
    parser.add_argument("--use-intent-classifier", action='store_true', 
                        help="Whether to use intent classifier before calling LLMs to generate answer")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    intent_classifier = pipeline("text-classification", 
                                 model="nova-x/intent_classification")


    app.root_path = args.root_path
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile)
