"""
This script runs a multi-step text processing pipeline fully on CPU using
local open-source models.It generates notes and quiz questions in parallel
from a given text using Hugging Face models.Then, it merges both outputs into
a single final document using another local model.The LangChain
RunnableParallel and PromptTemplate are used for structure and clarity.
This script runs offline and does not require API keys or cloud access.

          RunnableParallel
           /            \
          v              v
 PromptTemplate      PromptTemplate
    (notes)             (quiz)
      |                   |
      v                   v
HuggingFacePipeline  HuggingFacePipeline
      |                   |
      v                   v
StrOutputParser     StrOutputParser
           \           /
            v         v
        PromptTemplate (merge notes + quiz)
                   |
                   v
        HuggingFacePipeline
                   |
                   v
            StrOutputParser

"""

from langchain.schema.runnable import RunnableParallel
from langchain_community.llms import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load a small CPU-friendly open-source model
model_id = "tiiuae/falcon-rw-1b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Create Hugging Face pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)

# Wrap pipeline with LangChain
llm = HuggingFacePipeline(pipeline=pipe)

# Define prompts
prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text \n {text}",
    input_variables=["text"],
)

prompt2 = PromptTemplate(
    template="Generate 5 short question answers from the following text \n {text}",
    input_variables=["text"],
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}",
    input_variables=["notes", "quiz"],
)

# Output parser
parser = StrOutputParser()

# Run notes and quiz generation in parallel
parallel_chain = RunnableParallel(
    {"notes": prompt1 | llm | parser, "quiz": prompt2 | llm | parser}
)

# Merge notes and quiz
merge_chain = prompt3 | llm | parser

# Full chain
chain = parallel_chain | merge_chain

# Input text
text = """
Support vector machines (SVMs) are a set of supervised learning methods used 
for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number 
of samples.

Uses a subset of training points in the decision function (called support 
vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision 
function. Common kernels are provided, but it is also possible to specify 
custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid 
over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using 
an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray 
and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample 
vectors as input. However, to use an SVM to make predictions for sparse data, 
it must have been fit on such data. For optimal performance, use C-ordered 
numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
"""

# Invoke the chain
result = chain.invoke({"text": text})

# Print result
print(result)

# Show chain graph
chain.get_graph().print_ascii()
