# OCR Document Classification 

## List of contents
- Document_cat - categorize extracted OCR text and identify document types based on their similarities. 
    - Initialization:
        - In the DocumentCategorizer class, we initialize the object with two parameters: num_topics and classes.
        - num_topics specifies the number of topics you want to identify using LDA.
        - classes is a list of predefined document classes for text classification.

    - Training the LDA Model:
        train_lda_model(self, documents): This method trains an LDA (Latent Dirichlet Allocation) model to identify topics in the given documents.
        - In this step:
            - Documents are preprocessed (tokenized and cleaned) to prepare them for modeling.
            - A dictionary is created to map words to unique IDs, and documents are converted to a bag-of-words representation using the dictionary.
            - The LDA model is trained using the bag-of-words corpus, with the specified number of topics (num_topics).

    - Getting Document Topics:
        get_document_topics(self, document): This method takes an input document and returns the distribution of topics within that document.
        - In this step:
            - The input document is preprocessed to match the format expected by the LDA model.
            - The document is converted to a bag-of-words representation using the LDA model's dictionary.
            - The LDA model is used to estimate the distribution of topics for the document.

    - Training the Text Classifier:
        train_text_classifier(self, labeled_data): This method trains a text classification model (e.g., Naive Bayes) using labeled data.
        - In this step:
            - Labeled data is provided, where each data point consists of a text document and its associated class label.
            - Texts and labels are separated from the labeled data.
            - Texts are vectorized using TF-IDF (Term Frequency-Inverse Document Frequency).
            - A text classification model (Multinomial Naive Bayes) is trained on the vectorized texts and their corresponding labels.

    - Classifying a Document:
        classify_document(self, document): This method takes an input document and classifies it into one of the predefined classes.
        - In this step:
            - The input document is preprocessed to match the format used during training.
            - The document is vectorized using the same TF-IDF vectorizer used during training.
            - The trained text classification model predicts the class label for the document.


## Discription
- This project aim is to take the text extracted from OCR(optical character recognition) and identify the type of documents
automatically, using machine learning-based approaches or specialized document processing software. Some solutions use 
techniques like natural language processing (NLP) and deep learning to analyze the content and structure of documents to
classify them.
