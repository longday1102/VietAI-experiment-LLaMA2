from config import Config
from prompt import Prompter
from process_analysis import DataProcess
from model_inputs import MODEL_INPUTS
from train import Trainer

import os 
from argparse import ArgumentParser

import torch
from torch.distributed import destroy_process_group, init_process_group

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_checkpoint", required=True, type=str)
    parser.add_argument("--model_weight_path", default=None, type=str)
    parser.add_argument("--test_size", required=True, type=float)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--batch_size", required=True, type=int)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--display_steps", default=100, type=int)
    parser.add_argument("--save_steps", default=1000, type=int)
    parser.add_argument("--save_state_name", required=True, type=str)
    parser.add_argument("--save_model_name", required=True, type=str)
    parser.add_argument("--state_checkpoint", default=None, type=str)
    args = parser.parse_args()

    # ddp config
    backend = "nccl"
    init_process_group(backend = backend)
    local_rank = int(os.environ["LOCAL_RANK"])

    # Tokenizer and Model
    config = Config()
    tokenizer = config.tokenizer(model_checkpoint = args.model_checkpoint)
    if args.model_weight_path:
        lora_model = config.reload_pretrained_model(model_weight_path = args.model_weight_path, device_map = {"": torch.device(f"cuda:{local_rank}")})
    else:
        model = config.load_pretrained_model(model_checkpoint = args.model_checkpoint, device_map = {"": torch.device(f"cuda:{local_rank}")})
        lora_model = config.add_lora(model = model, r = 16, lora_alpha = 64, lora_dropout = 0.1)
    
    # Dataset
    data_prcess = DataProcess(data_path = "MBZUAI/Bactrian-X", tokenizer = tokenizer)
    dataset = data_prcess.load_data()
    prompter = Prompter()

    splited_dataset = dataset.train_test_split(test_size = args.test_size, seed = 42)

    # Model inputs
    model_inputs = MODEL_INPUTS(prompter =  prompter,
                                tokenizer = tokenizer,
                                max_length = args.max_length)
    
    train_data = splited_dataset["train"].shuffle().map(model_inputs.generate_and_tokenize_prompt)
    valid_data = splited_dataset["test"].map(model_inputs.generate_and_tokenize_prompt)

    train_data = train_data.remove_columns(["instruction", "input", "id", "output"])
    valid_data = valid_data.remove_columns(["instruction", "input", "id", "output"])

    train_data.set_format("torch")
    valid_data.set_format("torch")

    train_dataloader, valid_dataloader = model_inputs.prepare_dataloader(train_data,
                                                                         valid_data,
                                                                         batch_size = args.batch_size)
     # Train
    trainer = Trainer(lr = 3e-4,
                      epochs = args.epochs,
                      model = lora_model,
                      gradient_accumulation_steps = 4,
                      gpu_id = local_rank)
    
    if args.state_checkpoint:
        state_checkpoint = torch.load(args.state_checkpoint)
    else:
        state_checkpoint = args.state_checkpoint

    trainer.train(train_dataloader = train_dataloader,
                  valid_dataloader = valid_dataloader,
                  display_steps = args.display_steps,
                  save_steps = args.save_steps,
                  save_state_name = args.save_state_name,
                  save_model_name = args.save_model_name,
                  state_checkpoint = state_checkpoint)
    
    destroy_process_group()
