from langchain.prompts import PromptTemplate

class ModelConfig:
    def __init__(self, model_filename: str):
        self.model_filename = model_filename

    def get_prompt_template(self):
        """
        Get the langchain prompt template for the model.
        """
        if self.model_filename == "llama-2-7b-chat.Q4_K_M.gguf":
            return PromptTemplate.from_template('''<|im_start|> system\n{system_prompt}\n<|im_end|>\n'''
            + '''<|im_start|> user\n{user_prompt}\n<|im_end|>\n<|im_start|> assistant\n''')
        elif self.model_filename == "gpt4all-falcon-q4_0.gguf":
            return PromptTemplate.from_template('''### Instruction: {system_prompt}\n'''
            + '''{user_prompt}\n### Response:''')
        elif self.model_filename == "leo-hessianai-13b-chat-bilingual.Q4_K_M.gguf":
            return PromptTemplate.from_template('''<|im_start|> system\n{system_prompt}\n<|im_end|>\n'''
            + '''<|im_start|> user\n{user_prompt}\n<|im_end|>\n<|im_start|> assistant\n''')
        elif self.model_filename == "neuralbeagle14-7b.Q4_K_M.gguf":
            return PromptTemplate.from_template('''<|im_start|> system\n{system_prompt}\n<|im_end|>\n'''
            + '''<|im_start|> user\n{user_prompt}\n<|im_end|>\n<|im_start|> assistant\n''')
        else:
            return PromptTemplate.from_template('{system_prompt}\n{user_prompt}\n')

    def get_eos_tokens(self):
        """
        Get the end-of-sequence/terminating tokens/token sequences for the model.
        """
        if self.model_filename == "llama-2-7b-chat.Q4_K_M.gguf":
            return ["<|im_end|>"]
        elif self.model_filename == "gpt4all-falcon-q4_0.gguf":
            return ["### Instruction:"]
        elif self.model_filename == "leo-hessianai-13b-chat-bilingual.Q4_K_M.gguf":
            return ["<|im_end|>"]
        elif self.model_filename == "neuralbeagle14-7b.Q4_K_M.gguf":
            return ["<|im_end|>"]
        else:
            return None
        
    def get_context_length(self):
        """
        Get the context length for the model.
        """
        if self.model_filename == "llama-2-7b-chat.Q4_K_M.gguf":
            return 4096
        elif self.model_filename == "gpt4all-falcon-q4_0.gguf":
            return 4096
        elif self.model_filename == "leo-hessianai-13b-chat-bilingual.Q4_K_M.gguf":
            return 4096
        elif self.model_filename == "neuralbeagle14-7b.Q4_K_M.gguf":
            return 8000
        else:
            return 2048