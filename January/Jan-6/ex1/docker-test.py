import chromadb

client = chromadb.Client()
client = chromadb.PersistentClient(path="./my_db")