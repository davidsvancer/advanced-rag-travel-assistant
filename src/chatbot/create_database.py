# author: David Švancer
# xlogin: xsvancd00
# vutmail: xsvancd00@vutbr.cz

import weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import DataFrameLoader
import argparse
import pandas as pd

# source data paths
LISTINGS_PATH = "../../data/chatbot_dataset/listings.csv"
REVIEWS_PATH = "../../data/chatbot_dataset/reviews.csv"
FAQ_PATH = "../../data/chatbot_dataset/faq.csv"

# defined limits
NUMBER_OF_PLACES = 100
NUMBER_OF_REVIEWS_PER_HOTEL = 15
NUMBER_OF_MAX_CHARACTERS_FOR_REVIEWS = 2500
NUMBER_OF_MAX_CHARACTERS_FOR_DESCRIPTION = 2000
NUMBER_OF_MAX_CHARACTERS_FOR_AMENITIES = 1000

# truncates the text to specified maximum characters length
def truncate_text(text, number_of_max_characters):

    text = str(text)
    if len(text) > number_of_max_characters: return text[:number_of_max_characters] + "text is truncated"
    else:                                    return text

# load bnb data from .csv files 
def load_bnb_data():

    listings_columns = ['id', 'name', 'description', 'room_type', 'amenities', 'price']    # desired columns
    listings_data = pd.read_csv(LISTINGS_PATH, usecols=listings_columns)                   # load listings.csv 

    reviews_columns = ['listing_id', 'comments']                                           # desired columns
    reviews_data = pd.read_csv(REVIEWS_PATH, usecols=reviews_columns)                      # load reviews.csv 

    return listings_data, reviews_data

# add reviews to given places and returns them 
def add_reviews_to_places(listings_data, reviews_data):

    # group first number of reviews per hotel by listing id 
    reviews_data_grouped = reviews_data.groupby('listing_id')['comments'].apply(lambda x: ' '.join(map(str, x[:NUMBER_OF_REVIEWS_PER_HOTEL]))).reset_index()

    reviews_data_grouped['comments'] = reviews_data_grouped['comments'].apply(lambda x: truncate_text(x, NUMBER_OF_MAX_CHARACTERS_FOR_REVIEWS))

    # merge both data using id & listing_id
    merged_data = pd.merge(listings_data, reviews_data_grouped, left_on='id', right_on='listing_id', how='left')

    if 'listing_id' in merged_data:
        merged_data = merged_data.drop(columns=['listing_id'])

    # if any value is NaN, replace it with empty string
    columns_to_fix = ['name', 'description', 'comments', 'room_type', 'amenities', 'price']
    for col in columns_to_fix:
        merged_data[col] = merged_data[col].fillna('')

    return merged_data

# loads faq data and returns them
def load_faq_data():

    faq_columns = ['category', 'question', 'answer']                # desired columns
    faq_data = pd.read_csv(FAQ_PATH, usecols=faq_columns, sep=';')  # load faq data 

    # if any value is NaN, replace it with empty string
    for col in faq_columns:
        faq_data[col] = faq_data[col].fillna('')

    return faq_data

# prepares bnb documents and returns them
def get_bnb_data():

    listings, reviews = load_bnb_data()                   # load data
    bnb_data = add_reviews_to_places(listings, reviews)   # add reviews to places

    # truncate the text if it is longer than limit
    bnb_data['description'] = bnb_data['description'].apply(lambda x: truncate_text(x, NUMBER_OF_MAX_CHARACTERS_FOR_DESCRIPTION)) 
    bnb_data['amenities'] = bnb_data['amenities'].apply(lambda x: truncate_text(x, NUMBER_OF_MAX_CHARACTERS_FOR_AMENITIES))

    # prepare text for embedding
    bnb_data['page_content'] = (
        "Name: " + bnb_data['name'].astype(str) + "\n" +
        "Description: " + bnb_data['description'].astype(str) + "\n" +
        "Amenities: " + bnb_data['amenities'].astype(str) + "\n" +
        "Reviews: " + bnb_data['comments'].astype(str)
    )

    # id is already reserved keyword in weaviate db
    if 'id' in bnb_data.columns:
        bnb_data = bnb_data.rename(columns={'id': 'bnb_id'})

    # save bnb data as .csv file
    bnb_data.to_csv("weaviate_data.csv", index=False)           

    # page content from 'text-embedding', others colums are made as metadata 
    bnb_loader = DataFrameLoader(bnb_data, page_content_column="page_content")
    bnb_documents = bnb_loader.load()

    return bnb_documents

# prepares faq documents and returns them
def get_faq_data():

    faq_data = load_faq_data()

    # prepare text for embedding
    faq_data['page_content'] = (
        "Question: " + faq_data['question'].astype(str) + "\n" +
        "Answer: " + faq_data['answer'].astype(str) + "\n" +
        "Category: " + faq_data['category'].astype(str)
    )

    # page content from 'text-embedding', others colums are made as metadata 
    faq_loader = DataFrameLoader(faq_data, page_content_column="page_content")
    faq_documents = faq_loader.load()

    return faq_documents


# creates and saves faq and bnb databases to weaviate db
def create_databases(args):

    # get data
    bnb_documents = get_bnb_data()
    faq_documents = get_faq_data()

    # chosen embedding from arg input 
    embeddings = OllamaEmbeddings(model=args.model)   

    # connect to weaviate server
    client = weaviate.connect_to_local()    

    print("Connected, now creating databases")

    try:
        # delete old data if exists
        client.collections.delete("BNB")
        client.collections.delete("FAQ")

        WeaviateVectorStore.from_documents(
            documents=bnb_documents,
            embedding=embeddings,
            client=client,
            index_name="BNB"
        )

        WeaviateVectorStore.from_documents(
            documents=faq_documents,
            embedding=embeddings,
            client=client,
            index_name="FAQ"
        )

        print("Databases created successfully")

    except Exception as e:
        print(f"Error while creating databases, error: {e}")
    
    finally:
        client.close()      # close weaviate server connection


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Populates localized Weaviate vector databases with BNB and FAQ documents.")

    parser.add_argument("--model", type=str, default="bge-m3", help="Target Ollama embedding model name (default: bge-m3). Alternative: nomic-embed-text")

    args = parser.parse_args()

    create_databases(args)
