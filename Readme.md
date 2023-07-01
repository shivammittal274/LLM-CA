LLM For Finance (CA) POC


There are multiple steps involved running LLM on our data having GPTCache involved.

Step-1: Installation:

First install all set of required libraries.

```
    pip install -r requirements.txt
```

Step-2: Setting up:

Add your all files (currently supported files are: pdfs/ppts/mp4) under the folder name: Files, add all your files there on which you want to have your LLM based chatbot.


Step3: Ingestion pipeline:

This would scan all your files present in "Files" directory, extract our content from pdfs/ppts/mp4 files, and save the content in documents.npy files and create vectorstore.pkl file (which is actually used by LLM for getting similar embedding and returning answer on top of that.)

```
    python3 ingest.py
```

Running the demp:

Once your documents.npy and vectorstore.pkl file is saved, you can run app.py file and have a gradio app launched where you can ask your own custom LLM chatbot trained specifically on your data. You would be able to see answer specific to your question along with metric (whether cache was hit or not while answering the question.)


```
    python3 app.py
```

