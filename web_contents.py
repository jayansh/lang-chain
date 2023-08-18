#!pip install -q openai langchain playwright beautifulsoup4

# Set env var OPENAI_API_KEY or load from a .env file:
# import dotenv
# dotenv.load_env()
#import asyncio
#import nest_asyncio

#import os
import asyncio

from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.document_transformers import BeautifulSoupTransformer
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_extraction_chain
import pprint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
import pandas as pd

#from google.colab import auth
#auth.authenticate_user()

#import gspread
#from google.auth import default
#creds, _ = default()
#gc = gspread.authorize(creds)

#os.environ['OPENAI_API_KEY'] = 'your-openai-api-key'

schema = {
    "properties": {
        "news_article_title": {"type": "string"},
        "news_article_summary": {"type": "string"},
    },
    "required": ["news_article_title", "news_article_summary"],
  }

#nest_asyncio.apply()
def extract(content: str, schema: dict):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
    print("extracting contents...")
    return create_extraction_chain(schema=schema, llm=llm).run(content)

def scrapContents(urls, schema):
    loader = AsyncHtmlLoader(urls)
    docs = loader.load()
    #html2text = Html2TextTransformer()
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(docs,tags_to_extract=["span"])
    print("Extracting content with LLM")

    # Grab the first 1000 tokens of the site
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000,
                                                                    chunk_overlap=0)
    splits = splitter.split_documents(docs_transformed)

    # Process the first split
    extracted_content = extract(
        schema=schema, content=splits[0].page_content
    )
    pprint.pprint(extracted_content)
    return extracted_content

#def write_google_sheet(df, sheet_name):
#    sh = gc.create(sheet_name)
#    worksheet = gc.open(sheet_name).sheet1
#    worksheet.append_rows([df.columns.values.tolist()] + df.values.tolist())
#    # worksheet.update([df.columns.values.tolist()] + df.values.tolist())
#    print(worksheet.get_all_records())

#def extract(content: str, schema: dict):
#    return create_extraction_chain(schema=schema, llm=llm).run(content)


urls = ["https://www.espn.com","https://lilianweng.github.io/posts/2023-06-23-agent/","https://mumbaimirror.indiatimes.com"]
#loop = asyncio.get_event_loop()
#extracted_content = loop.run_until_complete(scrapContents(urls, schema=schema))
extracted_content = scrapContents(urls, schema=schema)
records_df = pd.DataFrame.from_dict(extracted_content)
records_df.to_csv('file1.csv')
#pprint.pprint(records_df)
#write_google_sheet(records_df, "Test 1")
