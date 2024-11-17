#!/usr/bin/env python
# coding: utf-8

# In[1]:


import getpass
import os
from langchain_groq import ChatGroq



if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")


# In[2]:





# In[3]:




llm = ChatGroq(
    model="llama3-8b-8192",        # Using LLaMA 3.0 8B model
    temperature=0.3,                # Lower temperature for deterministic, focused output
    max_tokens=500,                 # Moderate token limit for concise, detailed answers
    top_p=0.95,                     # Ensures diversity while keeping relevance
    frequency_penalty=0.7,          # Penalizes repetition
    presence_penalty=0.3,           # Discourages irrelevant topics
    max_retries=3,                  # Retries up to 3 times in case of failure
    timeout=60,                     # 60 seconds to allow for complex tasks        
)


# In[4]:


messages = [
    (
        "system",
        """You are an advanced AI assistant designed to help developers working with Generative AI. Your tasks include:

        1. **Prompt Refinement and Generation:**
           - Analyze and optimize user-provided prompts to improve clarity, specificity, and effectiveness.
           - Suggest alternative prompts that yield better results for different models and tasks.

        2. **AI Debugging Assistance:**
           - Identify errors or inconsistencies in AI-generated outputs or training data.
           - Provide actionable recommendations to fix these issues and improve the overall performance of the Generative AI workflow.

        3. **Model Evaluation and Benchmarking:**
           - Compare and benchmark the performance of different Generative AI models on similar tasks.
           - Offer detailed insights, such as strengths, weaknesses, and ideal use cases for each model.

        4. **Workflow Automation:**
           - Assist in integrating Generative AI seamlessly into development pipelines by generating code snippets, workflows, or configurations.
           - Automate repetitive tasks and streamline the development process.

        **Your Goals:**
        - Provide accurate, actionable, and developer-friendly outputs.
        - Be responsive to user inputs and adapt your recommendations to specific needs or constraints.
        - Ensure that all suggestions are efficient, explainable, and aligned with the best practices in Generative AI.

        Input: The user provides a task description, prompt, model output, or debugging query.
        Output: Generate a detailed, actionable response tailored to the specific task or issue.

        Be concise, helpful, and precise in all your responses.
        """
    ),
    # Example 1: Prompt Refinement
    (
        "human",
        "My prompt: 'Write a story about AI.' It feels too vague. How can I improve it?"
    ),
    (
        "assistant",
        "Your prompt is indeed too general. Consider refining it to include specific details, such as the tone, setting, or purpose. For example: 'Write a futuristic short story about an AI assistant navigating human emotions in a dystopian world.'"
    ),
    # Example 2: AI Debugging Assistance
    (
        "human",
        "The model is generating repeated sentences. What could be wrong?"
    ),
    (
        "assistant",
        "The issue might be caused by insufficient training data variety or low temperature settings during inference. Try increasing the temperature parameter to introduce more randomness in outputs or fine-tune the model with a more diverse dataset."
    ),
    # Example 3: Model Evaluation and Benchmarking
    (
        "human",
        "How does GPT-3 compare to LLaMA-2 for creative writing tasks?"
    ),
    (
        "assistant",
        "GPT-3 tends to excel in generating creative, context-rich stories due to its extensive training dataset. LLaMA-2, while efficient, may require fine-tuning for niche creative tasks. If cost and deployment flexibility are priorities, LLaMA-2 is a strong contender."
    ),
    # Example 4: Workflow Automation
    (
        "human",
        "How can I integrate a Generative AI model into my Python-based application?"
    ),
    (
        "assistant",
        "To integrate a Generative AI model into your Python application, you can use libraries like Hugging Face Transformers. First, install the library using `pip install transformers`. Then, load a pre-trained model with `from transformers import AutoModel`. Create a simple API wrapper for your application to send prompts and receive outputs seamlessly."
    )
]


# In[5]:


from langchain_core.prompts import ChatPromptTemplate


# In[6]:


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an advanced AI assistant designed to help developers working with Generative AI. Your tasks include:

            1. **Prompt Refinement and Generation:**
               - Analyze and optimize user-provided prompts to improve clarity, specificity, and effectiveness.
               - Suggest alternative prompts that yield better results for different models and tasks.

            2. **AI Debugging Assistance:**
               - Identify errors or inconsistencies in AI-generated outputs or training data.
               - Provide actionable recommendations to fix these issues and improve the overall performance of the Generative AI workflow.

            3. **Model Evaluation and Benchmarking:**
               - Compare and benchmark the performance of different Generative AI models on similar tasks.
               - Offer detailed insights, such as strengths, weaknesses, and ideal use cases for each model.

            4. **Workflow Automation:**
               - Assist in integrating Generative AI seamlessly into development pipelines by generating code snippets, workflows, or configurations.
               - Automate repetitive tasks and streamline the development process.

            **Your Goals:**
            - Provide accurate, actionable, and developer-friendly outputs.
            - Be responsive to user inputs and adapt your recommendations to specific needs or constraints.
            - Ensure that all suggestions are efficient, explainable, and aligned with the best practices in Generative AI.
            """,
        ),
        ("human", "{task_description}"),
    ]
)
chain = prompt | llm


# In[7]:


response1=chain.invoke(
    {
        "task_description": "My prompt: 'Write a story about a dog.' It feels too vague. How can I improve it?"
    }
)


# In[8]:


print(response1.content)


# In[9]:


response2=chain.invoke(
    {
        "task_description": "The model is generating repeated sentences. What could be wrong?"
    }
)


# In[10]:


print(response2.content)


# In[11]:


response3=chain.invoke(
    {
        "task_description": "How does GPT-3 compare to LLaMA-2 for creative writing tasks?"
    }
)


# In[12]:


print(response3.content)


# In[13]:


response4=chain.invoke(
    {
        "task_description": "How can I integrate a Generative AI model into my Python-based application?"
    }
)


# In[14]:


print(response4.content)


# In[15]:


response5=chain.invoke(
    {
        "task_description":'''Optimize the Code 
    def lcs_brute_force(s1, s2, m, n):
        if m == 0 or n == 0:
            return 0
        if s1[m-1] == s2[n-1]:
            return 1 + lcs_brute_force(s1, s2, m-1, n-1)
        else:
            return max(lcs_brute_force(s1, s2, m-1, n), lcs_brute_force(s1, s2, m, n-1))
                                                    '''
    }
)


# In[16]:


print(response5.content)


# In[17]:


from langchain_core.prompts import PromptTemplate

example_prompt = PromptTemplate.from_template("prompt: {prompt}\n response: {response}")

filled_prompt = example_prompt.format(
    prompt="How does OpenAI’s GPT-4 compare to LLaMA-3 for code generation tasks?",
    response='''Here’s a comparative analysis of GPT-4 and LLaMA-3 for code generation:

Performance: GPT-4 demonstrates stronger contextual understanding, making it better for complex code tasks and multi-step logic. LLaMA-3 performs efficiently but may require fine-tuning for niche or domain-specific code tasks.
Cost and Accessibility: LLaMA-3, being open-source, is cost-effective for local deployment. GPT-4 is accessible via APIs but comes with usage fees.
Scalability: LLaMA-3 is well-suited for integration into custom workflows due to its flexibility, while GPT-4 excels in pre-trained tasks without much customization.
Recommendation: Use GPT-4 for immediate, high-quality code generation. Opt for LLaMA-3 if customization or cost-effectiveness is a priority.'''
)

print(filled_prompt)


# In[18]:


response6=chain.invoke(
    {
        "task_description": "How does GPT-3 compare to LLaMA-2 for creative writing tasks? Answer concisely in one sentence."

    }
)


# In[19]:


response6.content


# In[20]:


Fine_Tunning_Dataset=[
    {
        "prompt": "What is machine learning? Answer concisely in one sentence. ",
        "response": "Machine learning is a subset of AI that enables systems to learn from data."
    },
    {
        "prompt": "Explain recursion in programming in simple terms.",
        "response": "Recursion is a function calling itself to solve a problem."
    },
    {
        "prompt": "Write a prompt for a language model to generate a short story about a character named Alex. Can this prompt be any better",
        "response":"The following Prompt can be written as...Please generate a short story about a character named Alex who is a Software Engineer."
    },
    {
        "prompt": "What is the difference between a API and a UI? Explain in simple terms.",
        "response": "An API is a set of rules that enables different applications to communicate with each other, while a UI is the visual part of a software application."
    },
    {
        "prompt": "Write a script for a conversation between a user and a virtual assistant.",
        "response": "Here is a script for a conversation between a user and a virtual assistant... User: What is the weather like today? Virtual Assistant: The weather is sunny and 75 degrees Fahrenheit."
    },
    {
        "prompt": "Explain the concept of DevOps in the context of software development.",
        "response": "DevOps is the practice of combining software development and IT operations to improve collaboration, efficiency, and quality."
    },
    {
        "prompt": "Write a prompt for a language model to generate a poem about a sunset.",
        "response": "Here is a prompt for a language model to generate a poem about a sunset... Please generate a poem about a sunset with vivid descriptions of colors and emotions."
    },
    {
        "prompt": "Explain the difference between a debugger and a profiler in software development.",
        "response": "A debugger is a software tool that helps developers find and fix errors in their code, while a profiler is a tool that helps developers understand how their code is performing and identify bottlenecks."
    },
    {
        "prompt": "Write a script for a chatbot to order food at a restaurant.",
        "response": "Here is a script for a chatbot to order food at a restaurant... Chatbot: Hi, I'd like to order food. What options do you have? User: We have a variety of dishes. What can I get for you?"
    },
    {
        "prompt": "Explain the concept of cloud computing in simple terms.",
        "response": "Cloud computing is a way of accessing and using computing resources over the internet, instead of having to use and maintain your own hardware and software."
    },
    {
        "prompt": "Write a prompt for a language model to generate a short story about a character who discovers a hidden talent.",
        "response": "Here is a prompt for a language model to generate a short story about a character who discovers a hidden talent... Please generate a short story about a character who discovers a talent for playing the guitar."
    },
    {
        "prompt": "Explain the importance of testing in software development.",
        "response": "Testing is an essential part of software development, as it ensures that the software works correctly and meets the requirements and expectations of the users."
    },
    {
        "prompt": "Write a script for a chatbot to make a reservation at a hotel.",
        "response": "Here is a script for a chatbot to make a reservation at a hotel... Chatbot: Hi, I'd like to make a reservation at your hotel. What dates are available?"
    }  
]
import json
with open("concise_responses.json", "w") as file:
    json.dump(Fine_Tunning_Dataset, file, indent=4)


# In[ ]:





# In[21]:


from datasets import load_dataset
with open("concise_responses.json", "r") as file:
    dataset = json.load(file)

formatted_data = [{"prompt": item["prompt"], "response": item["response"]} for item in dataset]

from datasets import Dataset
dataset = Dataset.from_list(formatted_data)

print(dataset[9])


# In[22]:


from langchain_core.prompts import FewShotPromptTemplate

prompt = FewShotPromptTemplate(
    examples=Fine_Tunning_Dataset,
    example_prompt=example_prompt,
    suffix="prompt: {input}",
    input_variables=["input"],
)

print(
    prompt.invoke({"input": "What is Machibe Learnong"}).to_string()
)


# In[23]:


print(example_prompt.invoke(Fine_Tunning_Dataset[0]).to_string())


# In[24]:


trial=chain.invoke("what is machine learinin. Anser in lessss than 50 words")


# In[25]:


print(trial.content)


# In[26]:


def get_response(prompt):
    response = chain.invoke(prompt)
    return response.content


# In[27]:


get_response("Write a HTML Code to create a table")


# In[ ]:




