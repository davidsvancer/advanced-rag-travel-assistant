# author: David Švancer
# xlogin: xsvancd00
# vutmail: xsvancd00@vutbr.cz

from datasets import Dataset
import os
import sys
import argparse
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall, AnswerCorrectness
from langchain_ollama import ChatOllama, OllamaEmbeddings
import pandas as pd
import weaviate
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from src.chatbot.chatbot import get_prompts_templates, create_rag_chain, router, get_retriever

# path to dataset
RAGAS_DATASET = "data/ragas_dataset/ragas_dataset.csv"

# runs evaluation dataset
def ragas_evaluate(args):

    # load api key
    load_dotenv()

    # load dataset
    dataset = pd.read_csv(RAGAS_DATASET, sep=',')
    
    # connect to weaviate and prepare embedding model
    weaviate_client = weaviate.connect_to_local()
    embeddings = OllamaEmbeddings(model=args.embeddings)

    # llm init
    llm_router = ChatOllama(model="llama3", temperature=0.0)
    llm_multi_query = ChatOllama(model="llama3", temperature=0.3)
    llm_chat = ChatOllama(model="llama3", temperature=0.2)

    # get retrievers for faq and bnb
    bnb_retriever = get_retriever(client=weaviate_client, embedding=embeddings, llm=llm_multi_query, index_name="BNB", args=args)
    faq_retriever = get_retriever(client=weaviate_client, embedding=embeddings, llm=llm_multi_query, index_name="FAQ", args=args)

    # get prompts
    history_rewrite_prompt, system_prompt = get_prompts_templates()

    # compile generation chains
    bnb_chain = create_rag_chain(llm_chat, bnb_retriever, system_prompt, history_rewrite_prompt)
    faq_chain = create_rag_chain(llm_chat, faq_retriever, system_prompt, history_rewrite_prompt)

    ragas_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "reference": []
    }

    print("\nExecuting generation phases across evaluation dataset...")

    # iterate target dataset generating context candidates and responses
    for _, row in tqdm(dataset.iterrows(), total=len(dataset)):

        query = str(row["question"])
        ground_truth = str(row["ground_truth"])

        try:
            # resolve routing target collection
            route = router(query, llm_router)
            chain = faq_chain if route == "FAQ" else bnb_chain

            # generate candidate answer matching query specifications
            response = chain.invoke({"input": query, "chat_history": []})
            
            extracted_answer = str(response.get("answer", ""))
            extracted_contexts = [doc.page_content for doc in response.get("context", [])]

        except Exception as e:
            print(f"\nPipeline processing failure on query: '{query[:60]}...'", file=sys.stderr)
            print(f"Error trace: {e}", file=sys.stderr)
            extracted_answer = "ERROR"
            extracted_contexts = ["ERROR"]

        ragas_data["question"].append(query)
        ragas_data["answer"].append(extracted_answer)
        ragas_data["contexts"].append(extracted_contexts)
        ragas_data["reference"].append(ground_truth)

    weaviate_client.close()

    print("Preparing Ragas dataset")

    # get data
    prepared_data = Dataset.from_dict(ragas_data)

    # judge init
    judge_llm = ChatOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o-mini",                      
        max_retries=3, 
        timeout=120 
    )

    judge = LangchainLLMWrapper(judge_llm)
    judge_embeddings = LangchainEmbeddingsWrapper(embeddings)

    print("Judge ready, starting Ragas evaluation...")

    results = evaluate(
        prepared_data, 
        metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall(), AnswerCorrectness()],       
        llm=judge,
        embeddings=judge_embeddings,
        raise_exceptions=False
    )

    # save results
    results_df = results.to_pandas()
    results_df.to_csv(args.output, index=False)

    print(f"Finished! Results saved to: {args.output}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="RAGAS evaluation script")

    parser.add_argument("--use-multiquery", action="store_true")
    parser.add_argument("--use-reranker", action="store_true")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--base-k", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--embeddings", type=str, default="bge-m3")
    parser.add_argument("--output", type=str, required=True, help="Path where the evaluation CSV results will be saved")

    args = parser.parse_args()

    ragas_evaluate(args)