import json

from anthropic import HUMAN_PROMPT, AI_PROMPT

from lcb_runner.lm_styles import LMStyle
from lcb_runner.benchmarks import TestOutputPredictionProblem


class PromptConstants:
    SYSTEM_MESSAGE_CHAT_GENERIC = f"You are a helpful programming assistant and an expert Python programmer.\
 You are helping a user to write a test case to help to check the correctness of the function.\
 The user has written a input for the testcase.\
 You will calculate the output of the testcase and\
 write the whole assertion statement in the markdown code block with the correct output."

    SYSTEM_MESSAGE_COMPLETION_GENERIC = f"You are a helpful programming assistant and an expert Python programmer.\
 You are helping a user to write a test case to help to check the correctness of the function."

    FORMATTING_MESSAGE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."

    FORMATTING_WITHOUT_STARTER_MESSAGE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows."


def truncate_io(io):
    if len(str(io)) > 1000:
        io = str(io)[:1000] + "...."
        print(io)
    return io


def format_testcase_func_name_input(function_name, testcase):
    """
    use the form of "assert func_name(input) == "
    """
    # TODO should there be a space after the == ?
    input_str = ", ".join(testcase.split("\n"))
    return f"assert {function_name}({input_str}) == # TODO"


def parse_function_name_from_starter_code(starter_code):
    """
    starter_code : str
    """
    import ast

    tree = ast.parse(starter_code)
    fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            assert fn is None
            fn = node.name
    return fn


def get_generic_question_template_test_completion(
    question: TestOutputPredictionProblem, testcase_input: str
):
    prompt = f"Problem:\n{question.question_content}"
    prompt += f"Function:\n```\n{question.starter_code}\n```\n"

    # parse function name from starter_code
    func_name = parse_function_name_from_starter_code(question.starter_code)
    prompt += "Please complete the following test case:\n\n"
    prompt += (
        f"```\n{format_testcase_func_name_input(func_name, testcase_input)}\n```\n"
    )

    return prompt

def format_prompt_test_output(
    question: TestOutputPredictionProblem, LanguageModelStyle: LMStyle
) -> str:
    testcase_input = question.test[0].input
    if LanguageModelStyle == LMStyle.OpenAIChat:
        chat_messages = [
            {
                "role": "system",
                "content": PromptConstants.SYSTEM_MESSAGE_CHAT_GENERIC,
            },
        ]
        chat_messages += [
            {
                "role": "user",
                "content": get_generic_question_template_test_completion(
                    question, testcase_input
                ),
            },
        ]
        return chat_messages
    if LanguageModelStyle in {LMStyle.GenericBase}:
        return get_generic_question_template_test_completion(
                question, testcase_input
            )
    else:
        raise NotImplementedError(
            f"LanguageModelStyle {LanguageModelStyle} not implemented"
        )
