import time
from tqdm import tqdm
import torch
from transformers import (
    set_seed,
    AutoConfig,
    GenerationConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer
)
# from threading import Thread
from langchain_core.prompts import PromptTemplate

# Global seed for reproducibility
SEED_VALUE = 0
set_seed(SEED_VALUE)

HuggingFace_token="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"

class LLMManager:
    """Manages multiple HuggingFace LLM models with configurable settings"""

    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.configs = {}

    def load_model(
            self,
            model_name: str,
            model_id: str,
            context_length: int = 4096,
            max_memory: dict = None,
            dtype=torch.bfloat16,
            device_map: str = "auto"
    ):
        """
        Load a model and store it with a friendly name

        Args:
            model_name: Friendly name to reference this model (e.g., 'qwen2', 'optimind')
            model_id: HuggingFace model ID (e.g., 'microsoft/OptiMind-SFT')
            context_length: Maximum context window size
            max_memory: Memory allocation per device
            dtype: Data type for model weights
            device_map: Device placement strategy
        """

        print(f"Loading model: {model_name} ({model_id})...")

        # Load config and tokenizer
        config = AutoConfig.from_pretrained(model_id,token=HuggingFace_token)
        config.max_position_embeddings = context_length
        tokenizer = AutoTokenizer.from_pretrained(model_id, config=config,token=HuggingFace_token)


        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            max_memory=max_memory,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            token=HuggingFace_token,
        )

        # Store references
        self.models[model_name] = model
        self.tokenizers[model_name] = tokenizer
        self.configs[model_name] = config

        print(f"Model {model_name} loaded successfully!")

    def get_model(self, model_name: str):
        """Retrieve a loaded model"""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not loaded. Available models: {list(self.models.keys())}")
        return self.models[model_name], self.tokenizers[model_name]


def LLM_response(
        llm_manager: LLMManager,
        input: str,
        model_name: str = 'qwen2',
        system_prompt: str = 'Answer in specified dictionary format specified in the schema,do not restate the algorithm name or the graph structure.Do not output code.',
        # stream: bool = True,

        # Generation config parameters
        temperature: float = 0.2,
        do_sample: bool = None,
        max_new_tokens: int = 8192,
        top_p: float = 0.1,
        top_k: int = 25,
        seed: int = 0,

        # Tokenizer config parameters
        add_generation_prompt: bool = True,
        padding: bool = True,
        max_length: int = None,
        **generation_kwargs
):
    start_time = time.time()
    
    # Get model and tokenizer
    model, tokenizer = llm_manager.get_model(model_name)

    tokenizer.pad_token=tokenizer.eos_token

    # Auto-set do_sample based on temperature
    if do_sample is None:
        do_sample = temperature > 0.0

    # Set seed for reproducibility
    if seed is not None:
        set_seed(seed)

    # Prepare messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input}
    ]

    # Tokenize input with configurable options
    tokenize_kwargs = {
        "add_generation_prompt": add_generation_prompt,
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "padding": padding,
        # "truncation": truncation,
    }

    # if max_length is not None:
    #     tokenize_kwargs["max_length"] = max_length

    # inputs = tokenizer.apply_chat_template(
    #     messages,
    #     **tokenize_kwargs
    # ).to(model.device)

    # Tokenize input safely
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None) is not None:
        # Use apply_chat_template only if chat_template exists
        tokenize_kwargs = {
            "add_generation_prompt": add_generation_prompt,
            "return_tensors": "pt",
            "padding": padding
        }
        if max_length is not None:
            tokenize_kwargs["max_length"] = max_length
        inputs = tokenizer.apply_chat_template(messages, **tokenize_kwargs).to(model.device)

    else:
        # Fallback for standard tokenizers (LLaMA, GraphWiz, Gemma3, etc.)
        prompt_text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"].strip()
            if role == "system":
                prompt_text += f"<s>[INST] <<SYS>>\n{content}\n<</SYS>>\n"
            elif role == "user":
                prompt_text += f"{content} [/INST]\n"
            elif role == "assistant":
                prompt_text += f"{content} "

        # Tokenize prompt normally
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(model.device)

    print(f"\nInput prompt token count: {inputs['input_ids'].shape[1]}\n")

    stop_token = "<end_of_turn>" #very specific to gemma3 model
    stop_token_id = tokenizer.convert_tokens_to_ids(stop_token)

    # Create generation config
    generation_config = GenerationConfig(
        temperature=temperature,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        top_k=top_k,
        # repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        # eos_token_id=tokenizer.eos_token_id,
        eos_token_id=[tokenizer.eos_token_id, stop_token_id],
        **generation_kwargs,
    )

    response= _generate_response(model, tokenizer, inputs, generation_config, start_time)
    return response

def _generate_response(model, tokenizer, inputs, generation_config, start_time):
    """Generate response with streaming"""

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generation_kwargs = {
        **inputs,
        "generation_config": generation_config,
        "streamer": streamer
    }

    # Start generation in a separate thread
    #thread = Thread(target=model.generate, kwargs=generation_kwargs)
    #thread.start()

    
    # Stream and accumulate response
    send_back = ''
    start_time = time.time()

    # for chunk in streamer:
    #     send_back += chunk

        # Progress logging every 5 seconds
    # if time.time() - last_log > 5:
    #     tqdm.write("LLM still generating...")
    #     last_log = time.time()
    print("*****Started*********")
    outputs_encoded = model.generate(**inputs,generation_config=generation_config)
    output = tokenizer.decode(outputs_encoded[0][inputs["input_ids"].shape[-1]:])

    #thread.join()

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f}s")

    return output

def prompt_template(
        example_str: str,
        query_input,  # input can be a sequence or a graph
        query_task,
        algorithm_description,
        schema:str, #gives the format in which the output should be generated
        input_type: str,  # tells us whether is a sequence or a graph
) -> str:

    """Formats the prompt to be sent to the large language model"""

    base_prompt = PromptTemplate.from_template(
        'The task is {task}.\n Algorithm Description :{algorithm}Generate '
        'the result according to the steps given.Also generate the steps to reach the result according to the givem schema \n Here is the {input} described on'
        'which the algorithm needs to be applied: {query_input}.')
    prompt = base_prompt.format(task=query_task, algorithm=algorithm_description, input=input_type,
                                query_input=query_input)
    
    prompt += '' if example_str == '' else f"Here are a few examples {example_str}"
    prompt=prompt+schema 

    return prompt


llm_manager = LLMManager()
model_cfg={
            "model_name": "Gemma 3 12B",
            "model_id": "google/gemma-3-12b-it",
            "type": "Dense",
            "context_window": 128000 
        }
llm_manager.load_model(
            model_name=model_cfg["model_name"],
            model_id=model_cfg["model_id"],
            context_length=model_cfg["context_window"] )
