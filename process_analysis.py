from datasets import load_dataset
import matplotlib.pyplot as plt

class DataProcess:
    def __init__(self,
                 data_path,
                 tokenizer):
        self.data_path = data_path
        self.tokenizer = tokenizer
        
    def load_data(self):
        dataset = load_dataset(self.data_path, "vi", split = "train")
        return dataset
    
    def statistical(self, dataset, prompter):
        prompt_len = []
        for line in dataset:
            full_prompt = prompter.generate_prompt(line["instruction"],
                                                   line["input"],
                                                   line["output"])
            prompt_len.append(len(self.tokenizer.encode(full_prompt)))
        return prompt_len
    
    def draw(self, prompt_len):
        freq = {}
        for num in prompt_len:
            if num in freq:
                freq[num] += 1
            else:
                freq[num] = 1
        
        max_freq = 0
        max_keys = None

        for k in freq.keys():
            if freq[k] >= max_freq:
                max_freq = freq[k]
                max_keys = k
                
        fig = plt.figure(figsize = ((8, 5)))
        plt.bar(freq.keys(), freq.values(), width = 0.6)   
        plt.xlabel("Prompt length")
        plt.ylabel("Prompt length frequency")
        plt.show()
        
        print("Length occupies the most frequency: ", max_keys)
        print("Maximum frequency: ", max_freq)