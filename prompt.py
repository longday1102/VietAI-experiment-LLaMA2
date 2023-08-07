from typing import Union

class Prompter(object):
    __slots__ = ("template")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self.template = {
            "prompt_input": "Dưới đây là một Instruction mô tả một nhiệm vụ, được ghép nối với một Input cung cấp thêm ngữ cảnh. Viết một Response hoàn thành yêu cầu một cách thích hợp.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
            "prompt_no_input": "Dưới đây là một Instruction mô tả một nhiệm vụ. Việt một Response hoàn thành yêu cầu một cách thích hợp.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
            "response_split": "### Response:"
        }

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()