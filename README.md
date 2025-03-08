## Overview

A search engine implementation that processes HTML documents from the ICS domain, creates an inverted index, and enables Boolean AND queries with tf-idf ranking.

## Creating the Index

1. Unzip dataset (analyst.zip or developer.zip) into data/ directory

2. Run indexer:
   $ python src/indexer.py --data_dir ./data/[dataset] --output_dir ./index

The indexer will:

- Process documents in chunks to manage memory usage
- Create partial indexes on disk
- Merge partial indexes into final inverted index

## Running Search Interface

1. Start search engine:
   $ python src/search.py --index_dir ./index

2. Enter queries at the prompt:

   > Enter search query: machine learning

3. View ranked results of matching document URLs

## Query Processing

1. Query text is:

   - Tokenized into words
   - Converted to lowercase
   - Stemmed using Porter stemmer

2. Documents are:
   - Retrieved using Boolean AND across the query terms
   - Ranked using tf-idf scoring and cosine similarity
   - Returned in order of relevance
