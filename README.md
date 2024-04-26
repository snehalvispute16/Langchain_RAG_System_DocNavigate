# DocNavigate: AI Document Assistant

DocNavigatet is an AI-powered document assistant built using the LangChain framework. It leverages the power of Google's Gemini 1.5 Pro model to answer questions on specific documents, such as the "Leave No Context Behind" paper published by Google on 10th April 2024. This system integrates external data sources, such as PDFs or text files, to enhance the accuracy and relevance of its responses.

## Overview

GeminiQuest allows users to input questions related to the document of interest. The system retrieves the relevant information from the specified document and passes it to the Gemini 1.5 Pro model for generation. The responses provided by GeminiQuest are accurate and formatted in markdown for better readability.

## Features

- Seamless integration with Google's Gemini 1.5 Pro model for question answering.
- Supports input questions related to specific documents.
- Utilizes LangChain framework for data retrieval and processing.
- Outputs responses formatted in markdown for enhanced readability.


## Usage

Once the Streamlit app is running, you can interact with GeminiQuest by entering your questions in the provided text input field. Click the "Generate Response" button to get the AI-generated answer based on the input question and the specified document.

## Dependencies

- Streamlit
- LangChain
- Google's GenerativeAI
- NLTK
- IPython

## Author
- Snehal Vispute

## License

This project is licensed under the [MIT License](LICENSE).
