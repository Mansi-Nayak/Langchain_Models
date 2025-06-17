"""
This script uses Hugging Face's Zephyr-7B model to extract structured
information from a product review.It defines a Pydantic schema to standardize
output, including themes, summary, sentiment, pros, cons, and reviewer name.
The HuggingFaceEndpoint from `langchain_huggingface` is used to invoke the
model with an instruction-based prompt.The script demonstrates how to process
natural language reviews and convert them into clean, structured JSON data.
This is useful for tasks like review summarization, sentiment analysis, and
information extraction in NLP workflows.
"""

import os
from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFaceEndpoint

# Load environment variables
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize HuggingFace LLM (use a model that supports instructions)
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",  # or mistralai/Mistral-7B-Instruct-v0.2
    temperature=0.3,
    max_new_tokens=512,
    huggingfacehub_api_token=hf_token,
)

# Define the Pydantic schema


class Review(BaseModel):
    key_themes: list[str] = Field(description="Key themes discussed in the review.")
    summary: str = Field(description="Brief summary of the review.")
    sentiment: Literal["pos", "neg", "neutral"] = Field(
        description="Overall sentiment."
    )
    pros: Optional[list[str]] = Field(default=None, description="Pros in list format.")
    cons: Optional[list[str]] = Field(default=None, description="Cons in list format.")
    name: Optional[str] = Field(default=None, description="Reviewer's name.")


# Review text


review_text = """
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an 
absolute powerhouse!The Snapdragon 8 Gen 3 processor makes everything 
lightning fast—whether I'm gaming, multitasking,or editing photos. The 5000mAh 
battery easily lasts a full day even with heavy use, and the45W fast charging 
is a lifesaver. The S-Pen integration is a great touch for note-taking and
quick sketches, though I don't use it often. What really blew me away is the 
200MP camera—thenight mode is stunning, capturing crisp, vibrant images even 
in low light. Zooming up to 100xactually works well for distant objects, but 
anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. 
Also, Samsung'sOne UI still comes with bloatware—why do I need five different 
Samsung apps for things Googlealready provides? The $1,300 price tag is also a 
hard pill to swallow.

Review by Nitish Singh
"""

# Create the instruction-style prompt
prompt = f"""
You are an AI that extracts structured data from product reviews.

Review:
\"\"\"{review_text}\"\"\"

Extract the following in JSON format:
- key_themes: list of key themes
- summary: brief summary
- sentiment: "pos", "neg", or "neutral"
- pros: list of pros
- cons: list of cons
- name: reviewer's name

Return only the JSON.
"""

# Get response
response = llm.invoke(prompt)

# Print raw response (usually a JSON-formatted string)
print(response)
