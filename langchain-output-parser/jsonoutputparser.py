from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Using instruction-tuned model optimized for structured output
model_id = "OpenAssistant/oasst-sft-1-pythia-12b"

# Load tokenizer and model (ensure enough RAM or use 4bit/8bit quantization)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

# HF pipeline for generation
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=False,
    temperature=0.2,   # Lower temp for more deterministic JSON
    return_full_text=True,
)

# Wrap as LangChain LLM
llm = HuggingFacePipeline(pipeline=pipe)

# JSON parser
parser = JsonOutputParser()

# Prompt template with format instruction
template = PromptTemplate(
    template=(
      "Return EXACTLY 5 facts about {topic} as a JSON array. Each element must be a JSON object with a single key \"fact\", e.g.:\n"
      '[{"fact": "..."}]\n\n'
      "{format_instruction}"
    ),
    input_variables=["topic"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

# Chain: prompt → LLM → parser
chain = template | llm | parser

# Invoke and print result
result = chain.invoke({"topic": "black hole"})
print(result)














# from langchain_huggingface import HuggingFaceEndpoint
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import JsonOutputParser
# from dotenv import load_dotenv

# load_dotenv()

# # ✅ Initialize LLM directly (not chat model)
# llm = HuggingFaceEndpoint(
#     repo_id="mistralai/Mistral-7B-Instruct-v0.1",
#     task="text-generation",
#     max_new_tokens=512,
#     temperature=0.7,
#     timeout=60
# )

# parser = JsonOutputParser()

# # template = PromptTemplate(
# #     template="Give me the name, age and city of a fictional person \n {format_instruction}",
# #     input_variables=[],
# #     partial_variables={"format_instruction": parser.get_format_instructions()},
# # )

# # chain = template | model | parser

# # result = chain.invoke({})

# # print(result)

# # json don't support schema enforce

# template = PromptTemplate(
#     template="Give me 5 facts about {topic} \n {format_instruction}",
#     input_variables=['topic'],
#     partial_variables={"format_instruction": parser.get_format_instructions()},
# )

# chain = template | llm | parser

# result = chain.invoke({'topic': 'black hole'})

# print(result)
