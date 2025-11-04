from lcb_runner.lm_styles import LMStyle, LanguageModel


def build_runner(args, model: LanguageModel):
    if model.model_style == LMStyle.OpenAIChat:
        from lcb_runner.runner.oai_runner import OpenAIRunner
        return OpenAIRunner(args, model)
    else:
        from lcb_runner.runner.vllm_runner import VLLMRunner
        return VLLMRunner(args, model)
