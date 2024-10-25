# Project Name

This project utilizes the `langchain` library to create a series of chains that process text inputs using the `OllamaLLM` model. The chains include various functionalities such as re-writing text to sound more upbeat, analyzing product features, and generating pros and cons.

## Features

- **Text Rewriting**: Re-writes user input to sound more upbeat.
- **Product Analysis**: Analyzes and generates pros and cons for a given product.
- **Customizable Prompts**: Uses customizable chat prompt templates to interact with the model.

## Setup

### Prerequisites

- Python 3.8 or higher
- `pip` (Python package installer)

### Installation

1. Clone the repository:
    ```sh
    git@github.com:langiq-ai/chain-langchain-intro.git
    cd chain-langchain-intro
    ```

2. Create a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Running the Project

1. Ensure you are in the virtual environment:
    ```sh
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

2. Run the main script:
    ```sh
    python main.py
    ```

## Logging

Logging is configured to capture all levels of log messages. You can enable logging by uncommenting the `logging.basicConfig(level=logging.DEBUG)` line in `main.py`.

## Example Usage

### Text Rewriting

The script re-writes the input text to sound more upbeat:
```python
result = chain.invoke({"text": "I don't like eating tasty things"})
print(result)
```
### License
This project is licensed under the MIT License. See the LICENSE file for details.