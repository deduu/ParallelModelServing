# app/models/model_pool.py
import asyncio
import torch
import logging
from typing import List, Optional, Dict, Any
from fastapi import HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import threading
import time as time_module
import json
import gc

logger = logging.getLogger(__name__)

class ParallelModelPool:
    """
    Manages a pool of model instances for parallel inference.
    """
    def __init__(self, model_path: str, num_instances: int = 4, dtype=torch.float16, devices: Optional[List[str]] = None):
        self.model_instances = []
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Detect available CUDA devices if not specified
        if devices is None:
            available_cuda = torch.cuda.device_count()
            devices = [f"cuda:{i}" for i in range(available_cuda)] if available_cuda > 0 else ["cpu"]
            if not devices:
                devices = ["cpu"]

        logger.info(f"Using devices: {devices}")

        for i in range(num_instances):
            try:
                # Assign devices in a round-robin fashion if instances exceed devices
                device = devices[i % len(devices)]
                
                logger.debug(f"Loading model instance {i} on {device}")

                model = AutoModelForCausalLM.from_pretrained(
                    model_path, 
                    torch_dtype=dtype
                ).to(device)
                
                self.model_instances.append({
                    'model': model, 
                    'in_use': False, 
                    'device': device
                })
                
                logger.info(f"Loaded model instance {i} on {device}")
            except Exception as e:
                logger.error(f"Failed to load model instance {i} on {device}: {e}")
    def get_free_model(self):
        for instance in self.model_instances:
            if not instance['in_use']:
                instance['in_use'] = True
                return instance
        raise HTTPException(503, "No model instances available.")

    def release_model(self, instance):
        instance['in_use'] = False

    async def generate_text_stream(self, query: str, history_messages: Optional[List[Dict]] = None, 
                                   max_new_tokens: int = 1024, temperature: float = 0.7, top_p: float = 0.9):
        model_instance = self.get_free_model()
        try:
            messages = [{"role": "system", "content": "You are a helpful assistant."}, *(history_messages or []), {"role": "user", "content": query}]
            logger.info(f"Messages: {messages}")
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
            inputs = {k: v.to(model_instance['model'].device) for k, v in input_ids.items()}
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

            generation_kwargs = {
                **inputs,
                'streamer': streamer,
                'max_new_tokens': max_new_tokens,
                'temperature': temperature,
                'top_p': top_p,
                'do_sample': True,
                'pad_token_id': self.tokenizer.eos_token_id,
            }
            # Start model generation in a separate thread
            generation_thread = threading.Thread(
                target=model_instance['model'].generate, 
                kwargs=generation_kwargs
            )
            generation_thread.start()
            logger.debug(f"Started generation thread on {model_instance['device']}")

            loop = asyncio.get_event_loop()

            # Define a synchronous function to get the next chunk
            def get_next_chunk():
                try:
                    return next(streamer)
                except StopIteration:
                    return None  # Indicate the end of the stream

            async def response_stream():
                token_count = 0
                start_time = time_module.perf_counter()
                try:
                    while True:
                        next_text = await loop.run_in_executor(None, get_next_chunk)
                        if next_text is None:
                            break
                        yield f"data: {next_text}\n\n"  # SSE format
                        token_count += 1
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    yield f"data: Error during generation: {e}\n\n"
                finally:
                    generation_thread.join()
                    self.release_model(model_instance)
                    latency = time_module.perf_counter() - start_time
                    metrics = {
                        'metrics': {
                            'latency': latency,
                            'tokens': token_count,
                            'tokens_per_second': token_count / latency if latency else 0,
                            'device': model_instance['device']
                        }
                    }
                    yield f"data: {json.dumps(metrics)}\n\n"
                    # Optional: Additional cleanup
                    del input_ids
                    del inputs
                    del streamer
                    del generation_thread
                    gc.collect()
                    torch.cuda.empty_cache()
                    logger.info(f"Generation completed on {model_instance['device']} in {latency:.2f}s")

            return response_stream()
        except Exception as e:
            self.release_model(model_instance)
            logger.error(f"Generation error: {e}")
            raise HTTPException(500, f"Generation error: {e}") 