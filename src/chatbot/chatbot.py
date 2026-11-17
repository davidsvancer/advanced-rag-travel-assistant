# author: David Švancer
# xlogin: xsvancd00
# vutmail: xsvancd00@vutbr.cz

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate
from langchain_weaviate.vectorstores import WeaviateVectorStore
import weaviate
import argparse
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import logging
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


# connect to specific Weaviate collection and set up base retriever
def get_weaviate_retriever(index_name, client, embeddings, alpha=1.0, base_k=3):

    # connect to desired collection in database
    weaviate_database = WeaviateVectorStore(
        client=client,
        index_name=index_name,
        text_key="text",
        embedding=embeddings
    )

    # search - alpha: 0.5 - hybrid, 0.0 - only BM25, 1.0 - only vector
    weaviate_retriever = weaviate_database.as_retriever(search_kwargs={"k": base_k, "alpha": alpha})

    return weaviate_retriever

# analyzes incoming query to route search targeting either BNB or FAQ collection
def router(query, llm):

    # prompt for llm used for decision whether the query was about FAQ/BNB decision
    router_prompt = """You are an expert routing agent for a travel agency chatbot.
    Your task is to analyze the user's query and route it to the appropriate vector database.

    Choose between these two databases based on the query content:

    1. "FAQ" - Use this for general travel questions, company policies, cancellation rules, luggage limits, insurance, visas, payment methods, and standard procedures.
    2. "BNB" - Use this for ANY accommodation searches (both specific hotels AND general recommendations based on user needs like 'family-friendly', 'quiet', 'cheap'), room types, amenities, prices, or user reviews of places to stay.

    Rules for output:
    - Return strictly the word "FAQ" or "BNB".
    - Do not include any punctuation, explanations, or conversational text.

    User Query: {query}
    """

    router_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are router assistant"),
            ("human", router_prompt)
        ]
    )

    chain = router_prompt_template | llm       # set up chain

    router_response = chain.invoke({"query": query})    # invoke and get response

    decision = router_response.content.strip().upper()  

    if decision == "FAQ" or decision == "BNB":
        return decision
    else:
        # fallback if router returns anything other than FAQ/BNB
        logging.warning(f"ROUTER FALLBACK: unexpected LLM decision: '{decision}'. Searching in BNB as default.")
        return "BNB"

# wraps base retriever with local cross-encoder model to re-order candidate documents
def create_rerank_retriever(base_retriever, top_n=3):

    # get encoder
    cross_encoder_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")

    # rerank documents and return top n
    reranker_compressor = CrossEncoderReranker(model=cross_encoder_model, top_n=top_n)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker_compressor,
        base_retriever=base_retriever 
    )

    return compression_retriever

# expands search space by generating alternative perspectives of the input query
def create_multi_query_retriever(base_retriever, llm):

    # prompt for llm generating multiple queries based on the input
    multiquery_prompt = """You are an AI language model assistant for a travel agency database.
        Your task is to generate 3 different versions of the given user question to retrieve relevant documents from a vector database.
        By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of distance-based similarity search.

        CRITICAL RULES:
        1. Provide these alternative questions separated by newlines.
        2. DO NOT number the questions (e.g., no "1.", "2.").
        3. DO NOT add any introductory or concluding text. Just output the 3 lines of text.

        Original question: {question}"""

    multiquery_prompt_template = PromptTemplate(
        input_variables=["question"],
        template=multiquery_prompt
    )

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever, 
        llm=llm, 
        prompt=multiquery_prompt_template,
        include_original=True)

    return multi_query_retriever

# constructs modular retrieval pipeline matching runtime configuration
def get_retriever(client, embedding, llm, index_name, args):

    retriever = get_weaviate_retriever(index_name=index_name, embeddings=embedding, client=client, alpha= args.alpha, base_k= args.base_k)

    if args.use_multiquery: retriever = create_multi_query_retriever(retriever, llm)

    if args.use_reranker:   retriever = create_rerank_retriever(retriever, top_n=args.top_k)

    return retriever


def get_prompts_templates():
    
  # history prompt
    history_rewrite_prompt = """Given a chat history and the latest user question 
                                    which might reference context in the chat history, formulate a standalone question 
                                    which can be understood without the chat history. Do NOT answer the question, 
                                    just reformulate it if needed and otherwise return it as is."""

    history_rewrite_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", history_rewrite_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    system_prompt =  """You are a professional and helpful assistant in a travel agency. Your job is to help clients with their accommodation selection.
                You have the following data about the accommodation (CONTEXT) at your disposal,
                which includes description, amenities, price and, above all, reviews from previous guests.

                IMPORTANT RULES:
                1. Answer ONLY based on the information in the CONTEXT. If the answer is not there, say: "Unfortunately, I don't see this information in my data."
                2. If the user asks about quality or impressions (e.g. "Is it quiet there?", "How is the food?"), SYNTHESIZE information from the reviews. Mention both positives and negatives.
                3. If you recommend a specific accommodation, ALWAYS state its name (from metadata) and approximate price.
                CONTEXT:
                {context}"""

    system_prompt_template = ChatPromptTemplate.from_messages(
        [   
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    return history_rewrite_prompt_template, system_prompt_template

# links retrieval pipeline with generation chains maintaining local session history
def create_rag_chain(llm, retriever, system_prompt, history_rewrite_prompt):

    # creates complete chain with memory for history
    retriever_with_history_awareness = create_history_aware_retriever(
            llm, retriever, history_rewrite_prompt
        )

    question_answer_chain = create_stuff_documents_chain(llm, system_prompt)

    rag_chain = create_retrieval_chain(retriever_with_history_awareness, question_answer_chain)

    return rag_chain

def behaviour(args):

    # init weaviate connection and embedding
    weaviate_client = weaviate.connect_to_local()
    embeddings = OllamaEmbeddings(model=str(args.embeddings))

    # init llm
    llm_router = ChatOllama(model="llama3", temperature=0.0)        # temperature 0 - will not be creative
    llm_multi_query = ChatOllama(model="llama3", temperature=0.6)   # temperature 0.6 - desired to be creative when formulating new queries
    llm_chat = ChatOllama(model="llama3", temperature=0.2)          # temperature 0.2 - little creative for chat

    # get bnb and faq retrievers
    bnb_retriever = get_retriever(client=weaviate_client,embedding=embeddings, llm=llm_multi_query, index_name="BNB", args=args)
    faq_retriever = get_retriever(client=weaviate_client,embedding=embeddings, llm=llm_multi_query, index_name="FAQ", args=args)

    # get prompts for history and basic chat
    history_rewrite_prompt, system_prompt = get_prompts_templates()

    # compile generation chains
    bnb_chain = create_rag_chain(llm_chat, bnb_retriever, system_prompt, history_rewrite_prompt)
    faq_chain = create_rag_chain(llm_chat, faq_retriever, system_prompt, history_rewrite_prompt)

    chat_history = []   # for storing history 

    # main chat cycle
    while True:

        query = input("You: ")  # input from user

        # end program if query containts one of those
        if query.lower() in ["exit", "bye", "end", "quit"]:     
            break

        print("Thinking...")

        try:
            # get router desicion based on user's input and choose its chain 
            route = router(query, llm_router)  
            if route == "BNB":
                chain = bnb_chain
            else:
                chain = faq_chain

            # invoke chain and get final response from llm 
            result = chain.invoke({"input": query, "chat_history": chat_history})
            response = result["answer"]

            # response form AI
            print(f"\nAI: {response}\n")

            # save history - both question from user and response from AI
            chat_history.append(HumanMessage(content=query))
            chat_history.append(AIMessage(content=response))
            
            # save only last 6 messages
            if len(chat_history) > 6:
                chat_history = chat_history[-6:]

        except Exception as e:
            print(f"Error occurred: {e}")


if __name__ == "__main__":  

    parser = argparse.ArgumentParser(description="travel agency chatbot using adjustable architecture, default with hybrid search")

    parser.add_argument("--use-multiquery", action="store_true", help="bot is using multiquery")
    parser.add_argument("--use-reranker", action="store_true", help="bot is using reranker")

    parser.add_argument("--alpha", type=float, default=1.0, help="alpha argument for weaviate database, 1.0 - only vector, 0 - only lexical, 0.5 - hybrid")
    parser.add_argument("--base-k", type=int, default=3, help="base number for number of documents retrieved from weaviate database")
    parser.add_argument("--top-k", type=int, default=3, help="number of documents for llm")

    parser.add_argument("--embeddings", type=str, default="bge-m3", help="nomic-embed-text / bge-m3")

    args = parser.parse_args()

    behaviour(args)
