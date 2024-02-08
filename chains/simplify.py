from typing import List, Dict, Any, Optional
from langchain.prompts import BasePromptTemplate, PromptTemplate
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.base import Chain
from langchain.chains import LLMChain
from pydantic import Extra


class SimplifyChain(Chain):
    """
    A summarization chain
    """
    
    system_prompt = "You're an AI assistant tasked with simplifying the text given to you by the user."

    user_prompt: BasePromptTemplate = PromptTemplate(
        input_variables=["text"],
        template="""
        Rewrite and rephrase the following text to make it easier to understand, so that a 5-year-old child can understand it.
        "
        {text}
        "
        Rewrite and rephrase the above text to make it easier to understand, so that a 5-year-old child can understand it. Describe difficult concepts in the text instead of using jargon terms directly. Do not make up anything new that is not in the original text. Only return the new, rewritten text.
        """
    )


    """Prompt object to use."""
    llm_chain: LLMChain
    output_key: str = "text"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return [self.output_key]

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_key]

    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        
        if not {"user_prompt", "system_prompt"} == set(self.llm_chain.input_keys):
            raise ValueError("llm_chain must have input_keys ['user_prompt', 'system_prompt']")
        if not self.llm_chain.output_keys == [self.output_key]:
            raise ValueError(f"llm_chain must have output_keys [{self.output_key}]")
        
        text_splitter = CharacterTextSplitter(
            separator='\n\n|\\.|\\?|\\!', chunk_size=1000, chunk_overlap=0, keep_separator=True)
        texts = text_splitter.split_text(inputs['text'])
        outputs = self.llm_chain.apply([{"user_prompt": self.user_prompt.format_prompt(text=t), "system_prompt": self.system_prompt} for t in texts])
        texts = [output.text for output in outputs]

        return {self.output_key: '\n\n'.join(texts)}

    @property
    def _chain_type(self) -> str:
        return "simplify_chain"
