from backend.config import Settings


class LocalCausalLMGenerator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.tokenizer = None
        self.model = None

    def _load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        import torch

        if self.settings.load_in_4bit and (
            self.settings.device == "cpu" or not torch.cuda.is_available()
        ):
            raise ValueError(
                "4-bit loading requires a compatible CUDA device and bitsandbytes. "
                "For CPU use RAG_DEVICE=cpu and RAG_LOAD_IN_4BIT=false."
            )

        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs = {
            "low_cpu_mem_usage": True,
            "offload_folder": str(self.settings.offload_dir),
        }
        if self.settings.model_revision:
            kwargs["revision"] = self.settings.model_revision

        if self.settings.device == "auto":
            kwargs["device_map"] = "auto"
        elif self.settings.device != "cpu":
            kwargs["device_map"] = {"": self.settings.device}

        if self.settings.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.settings.device == "cpu":
            kwargs["torch_dtype"] = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(self.settings.model_id, revision=self.settings.model_revision)
        self.model = AutoModelForCausalLM.from_pretrained(self.settings.model_id, **kwargs).eval()

    def generate(self, system_instruction: str, user_query: str, max_new_tokens: int | None = None) -> str:
        self._load()
        import torch

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_query},
        ]
        if self.tokenizer.chat_template:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
        else:
            prompt = f"System: {system_instruction}\n\nUser: {user_query}\n\nAssistant:"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        input_length = inputs["input_ids"].shape[-1]
        output_budget = max_new_tokens or self.settings.max_new_tokens
        window = getattr(getattr(self.model, "config", None), "max_position_embeddings", None)
        if window and input_length + output_budget > window:
            raise ValueError("Prompt and output budget exceed the model context window. Reduce the document context or query.")
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=output_budget,
                do_sample=False,
            )
        generated_tokens = outputs[:, input_length:]
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()
