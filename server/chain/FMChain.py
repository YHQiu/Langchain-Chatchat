from typing import List, Dict, Any, Optional

from langchain import LLMChain
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.schema import LLMResult


class FMLLMChain(LLMChain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def generate(
        self,
        input_list: List[Dict[str, Any]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> LLMResult:
        """Generate LLM result from inputs."""
        prompts, stop = self.prep_prompts(input_list, run_manager=run_manager)

        # 统计输入token数量
        self.total_input_tokens += sum(len(prompt.to_string()) for prompt in prompts)

        llm_result = self.llm.generate_prompt(
            prompts,
            stop,
            callbacks=run_manager.get_child() if run_manager else None,
            **self.llm_kwargs,
        )

        # 统计生成token数量
        self.total_output_tokens += sum(len(generation) for generation in llm_result.generations)

        return llm_result

    def get_total_token_consumption(self) -> int:
        """获取LLM生成消耗的总token数量"""
        return self.total_output_tokens - self.total_input_tokens