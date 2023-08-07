from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model

class Config:
    
    def tokenizer(self, model_checkpoint):
        tok = AutoTokenizer.from_pretrained(model_checkpoint)
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
        return tok
    
    def load_pretrained_model(self, model_checkpoint, device_map):
        bnb_config = BitsAndBytesConfig(load_in_4bit = True,
                                        bnb_4bit_use_double_quant = True,
                                        bnb_4bit_quant_type = "nf4",
                                        bnb_4bit_compute_dtype = torch.float16)
        
        model = AutoModelForCausalLM.from_pretrained(model_checkpoint,
                                                     quantization_config = bnb_config,
                                                     device_map = device_map)
        
        return model
    
    def add_lora(self, model, r: int, lora_alpha: int, lora_dropout: float):
        lora_config = LoraConfig(r = r,
                                 lora_alpha = lora_alpha,
                                 lora_dropout = lora_dropout,
                                 bias = "none",
                                 task_type = "CAUSAL_LM")
        lora_model = get_peft_model(model, lora_config)
        return lora_model
    
    def reload_pretrained_model(self, model_weight_path, device_map = None):
        config = PeftConfig.from_pretrained(model_weight_path)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,
                                                     load_in_4bit = True,   
                                                     device_map = device_map,
                                                     torch_dtype = torch.float16)
        lora_model = PeftModel.from_pretrained(model, model_weight_path, is_trainable = True)
        return lora_model