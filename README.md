# LLaMA-2 QLoRA experiment
<p align="center">
  <img src="https://github.com/longday1102/VietAI-experiment-LLaMA2/assets/121651344/11695528-b3fb-4ea6-814e-7d2a91843cf7" alt="llama-2">
</p>

## I. Introduction
- This project was made by me to refine the LLaMA-2 model based on instructions, applying some techniques to save memory when training such as QLoRA, DDP, half-precision.                                      
- You can run it on Kaggle notebook or Colab notebook.
## II. Dataset
The dataset I used is [Bactrian-X](https://github.com/mbzuai-nlp/bactrian-x/tree/main/data), which includes 54 languages. However, I only implemented it within the scope of Vietnamese.
## III. Model
I use model [LLaMA-2 7B](https://huggingface.co/meta-llama/Llama-2-7b-hf) to experiment. If your device has a larger configuration, you can experiment with larger versions of LLaMA-2 such as [LLaMA-2 13B](https://huggingface.co/meta-llama/Llama-2-13b-hf) and [LLaMA-2 70B](https://huggingface.co/meta-llama/Llama-2-70b-hf).
## IV. How to use
First, `!git clone` this repo, then install the environment with the command `!pip install --upgrade -r requirements.txt`.
1. Train:
    - You can use the script in my [notebook](https://github.com/longday1102/VietAI-experiment-LLaMA2/tree/main/notebook) to train from scratch. This is the path to the checkpoint after I trained the model for more than 1 epoch: [checkpoint-1](https://drive.google.com/drive/folders/1bKQUNOsTjjV9SRxHcbqdnDdApULJGwrS?usp=sharing)
    - **Note**: In the file [run.py](https://github.com/longday1102/VietAI-experiment-LLaMA2/blob/main/run.py) there are some arguments, you can change them optionally. If after stopping training, you feel that the performance is not as expected, if you want to continue training, pass the adapter model path to the `model_weight_path` argument and the state checkpoint path to the `state_checkpoint` argument in the script.
  2. Inference:                                            
     Inference template:
     ```python
     from inference import Inference
     infer = Inference(model_checkpoint = "{your_llama2-version}", model_weight_path = "{your_model_adapter_weight_path}")
     instruction = "{your_instruction}"
     input = "{your_input} or None"
     print(infer(instruction = instruction, input = input)["response"])
     ```

*Thank you a lot for the finding! 😊*
      
