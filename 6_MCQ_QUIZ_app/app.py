
from environs import Env
env = Env()
env.read_env()

from langchain.chat_models import AzureChatOpenAI
llm = AzureChatOpenAI(
    openai_api_base=env("OPENAI_API_BASE"),
    openai_api_version=env("OPENAI_API_VERSION"),
    deployment_name=env("AZURE_GPT_DEPLOYMENT"),
    openai_api_key=env("OPENAI_API_KEY"),
    openai_api_type=env("OPENAI_API_TYPE"),
)

from langchain.embeddings.openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(deployment=env("AZURE_EMBEDDING_DEPLOYMENT"))

from langchain.chat_models.azure_openai import AzureChatOpenAI
chat_model = AzureChatOpenAI(deployment_name=env("AZURE_GPT_DEPLOYMENT"))


import pinecone
pinecone.init(
    api_key=env("PINECONE_API_KEY"),
    environment="eastus-azure"
)
index_name = "mcq-creator"
index = pinecone.Index(index_name)


from langchain.docstore.document import Document
def get_similiar_docs(query, k=2):
    query_vector = embeddings.embed_query(query)
    similar_docs = index.query(query_vector, top_k=k, 
                               include_metadata=True,
                               include_values=False)
    similar_docs = [Document(page_content=doc["metadata"]["text"], metadata={"source": "local"}) for doc in similar_docs.matches]
    return similar_docs


from langchain.chains.question_answering import load_qa_chain

chain = load_qa_chain(llm, chain_type="stuff")
def get_answer(query):
  relevant_docs = get_similiar_docs(query)
  response = chain.run(input_documents=relevant_docs, question=query)
  return response



def generate_mcq(answer, question):
    import re
    from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
    from langchain.output_parsers import StructuredOutputParser, ResponseSchema

    response_schemas = [
        ResponseSchema(name="question", description="Question generated from provided input text data."),
        ResponseSchema(name="choices", description="Available options for a multiple-choice question in comma separated."),
        ResponseSchema(name="answer", description="Correct answer for the asked question.")
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    
    prompt = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template("""When a text input is given by the user, please generate multiple choice questions for 4 years old kid
            from it along with the correct answer. 
            \n{format_instructions}\n{llm_answer}.
            \n A reference Question is as : {user_question}""")  
        ],
        input_variables=["llm_answer", "user_question"],
        partial_variables={"format_instructions": format_instructions}
    )
    final_query = prompt.format_prompt(llm_answer = answer, user_question = question)
    final_query_output = chat_model(final_query.to_messages())
    markdown_text = final_query_output.content
    json_string = re.search(r'{(.*?)}', markdown_text, re.DOTALL).group(1)
    return json_string


our_query = "How is India's history?"
answer = get_answer(our_query)
mcqs = generate_mcq(answer=answer ,question=our_query)
print("-"*50)
print(our_query)
print("-"*50)
print(answer)
print("-"*50)
print(mcqs)
print("-"*50)