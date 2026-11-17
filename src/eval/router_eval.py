# author: David Švancer
# xlogin: xsvancd00
# vutmail: xsvancd00@vutbr.cz

import argparse
import sys
import pandas as pd
from tqdm import tqdm
from langchain_ollama import ChatOllama
from src.chatbot.chatbot import router as route_query

# path to dataset
RAGAS_DATASET = "data/ragas_dataset/ragas_dataset.csv"

# runs evaluation dataset queries through the router and logs predictions
def main(args):

    # init router llm with zero temperature for deterministic classification
    llm_router = ChatOllama(model="llama3", temperature=0.0)

    # load source evaluation dataset
    dataset = pd.read_csv(RAGAS_DATASET, sep=',')

    if args.question_column not in dataset.columns:
        raise ValueError(f"Column '{args.question_column}' not found in input CSV.")

    predicted_routes = []

    print(f"\nRunning router evaluation on: {RAGAS_DATASET}\n")

    # process queries targeting domain routing logic
    for _, row in tqdm(dataset.iterrows(), total=len(dataset)):

        question = str(row[args.question_column])

        try:
            route = route_query(question, llm_router)
            route = str(route).strip().upper()

            # check if response is correct
            if route not in ["BNB", "FAQ"]:
                route = "ERROR"

        except Exception as e:
            print(f"\nError processing question: {question[:80]}...", file=sys.stderr)
            print(f"Details: {e}", file=sys.stderr)
            route = "ERROR"

        predicted_routes.append(route)

    # persist routing target metadata mapping
    dataset["router_prediction"] = predicted_routes
    dataset.to_csv(args.output, index=False)

    print("\nEvaluation completed.")
    print(f"Results saved to: {args.output}\n")
    print("Prediction counts summary:")
    print(dataset["router_prediction"].value_counts().to_string())


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluates semantic router routing decisions over a CSV test dataset.")
    
    parser.add_argument("--output", required=True, type=str, help="Target output CSV file path storing predictions.")
    parser.add_argument("--question-column", default="question", type=str, help="Column mapping original query texts.")

    args = parser.parse_args()

    main(args)