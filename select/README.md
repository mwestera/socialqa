# Data Selection Pipeline

This README provides a detailed overview of the data selection pipeline used to process and score posts for question-answering and entailment tasks. The pipeline consists of several scripts, each performing specific functions on the cleaned and anonymized posts.

## Data Selection Process

### 1. Extract Sentences
**Script:** `extract_sentences.py`
- **Purpose:** Translates each post entry (including submission, parent, and replies) into a list of sentences.
- **Functionality:** Identifies potential questions and pivotal sentences from the posts.

### 2. Classify Sentences
**Script:** `classify_sentences.py`
- **Purpose:** Adds subjectivity and concreteness scores to each sentence.
- **Functionality:** Helps in determining the nature of the sentences for better selection.

### 3. Embed Sentences
**Script:** `embed_sentences.py`
- **Purpose:** Embeds sentences using a sentence embedder.
- **Functionality:** Generates embeddings that capture the semantic meaning of the sentences.

### 4. Collect Contextual Embeddings
**Script:** `collect_cont_embeddings.py`
- **Purpose:** Embeds sentences using a contextual embedder.
- **Functionality:** Provides contextual embeddings for better sentence understanding.

### 5. Make Tasks
**Script:** `make_tasks.py`
- **Purpose:** Combines sentences and the original posts into a question-answering (QA) and entailment task.
- **Functionality:** Creates tasks for evaluating the relevance and coherence of sentences in context.

### 6. Calculate Scores
**Script:** `calculate_scores_gpu.py`
- **Purpose:** Computes scores based on similarity or model (Transformer/LLM) scores.
- **Functionality:** Uses GPU to speed up the calculation of relevance scores for potential questions, pivots, and entailments.

## Scoring Criteria

The crucial selection criteria are centralized in the `scoring_methods.py` script, which is responsible for calculating the relevance scores for every potential question, pivot, and entailment.

## Main Pipeline Execution

All these parts are called from the `Main.py` script, which orchestrates the sequence of operations to calculate scores from beginning to end.

### Execution Notes:
- **Selective Execution:** When modifying code or changing models, the entire pipeline does not need to be rerun. Depending on wether certain files are still correct with the change, certain parts of the pipeline can be commented out within the main function to save time.
- **File Overwriting:** Be cautious of files that may be overwritten. The naming convention, driven by the `post_name_file` variable, determines whether the script will overwrite an existing file or create a new one. Always check the file names to avoid unintentional data loss.

## Summary

This pipeline is designed to process and score posts efficiently by breaking down the task into manageable scripts, each focusing on a specific aspect of the data processing workflow. The modular design allows for flexibility and selective execution, making it easier to adapt and modify individual components without rerunning the entire pipeline.

By following the structured sequence from sentence extraction to score calculation, the pipeline ensures accurate and relevant selection of sentences for QA and entailment tasks.
