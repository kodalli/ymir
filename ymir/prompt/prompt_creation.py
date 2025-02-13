import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from typing import List, Union, Optional
import json


def create_openai_dataset(
    dataset_name: str,
    df: pd.DataFrame,
    dataset_save_path: str,
    user_prompt_template: str,
    user_columns: List[str],
    response_template: Union[str, None] = None,
    response_columns: Union[List[str], None] = None,
    system_prompt: Union[str, None] = None,
) -> Dataset:
    """Creates a dataset in OpenAI chat format from a pandas DataFrame.

    Args:
        dataset_name (str): Name of the dataset
        df (pd.DataFrame): Input DataFrame containing the data
        dataset_save_path (str): Path to save the dataset
        user_prompt_template (str): Template string for user messages using DataFrame columns
        user_columns (List[str]): DataFrame column names to use in user_prompt_template
        response_template (str, optional): Template string for assistant responses. Defaults to None.
        response_columns (List[str], optional): DataFrame column names to use in response_template. Defaults to None.
        system_prompt (str, optional): System prompt to include. Defaults to None.

    Returns:
        Dataset: A Hugging Face dataset object containing the formatted conversations

    Example:
        >>> df = pd.DataFrame({
        ...     'question': ['What is 2+2?', 'What is Python?'],
        ...     'answer': ['4', 'A programming language']
        ... })
        >>> dataset = create_openai_dataset(
        ...     dataset_name="math_qa",
        ...     df=df,
        ...     dataset_save_path="data/math_qa",
        ...     user_prompt_template="Question: {question}",
        ...     user_columns=['question'],
        ...     response_template="{answer}",
        ...     response_columns=['answer'],
        ...     system_prompt="You are a helpful assistant"
        ... )

        This will create a dataset with messages in OpenAI chat format:
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Question: What is 2+2?"},
                {"role": "assistant", "content": "4"}
            ]
        }
    """
    openai_data = []
    for index, row in tqdm(
        df.iterrows(), total=len(df), desc=f"Creating OpenAI dataset: {dataset_name}"
    ):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": user_prompt_template.format(
                    **{col: row[col] for col in user_columns}
                ),
            }
        )
        if response_template:
            messages.append(
                {
                    "role": "assistant",
                    "content": response_template.format(
                        **{col: row[col] for col in response_columns}
                    ),
                }
            )
        data = {"messages": messages}
        openai_data.append(data)

    dataset = Dataset.from_list(openai_data)
    # save dataset locally
    dataset.save_to_disk(dataset_save_path)
    return dataset


def create_sharegpt_dataset(
    dataset_name: str,
    df: pd.DataFrame,
    user_prompt_template: str,
    response_template: str,
    user_columns: List[str],
    response_columns: List[str],
    dataset_save_path: str,
    dataset_info_path: str,
    system_prompt: Optional[str] = None,
    tools: Optional[str] = None,
) -> None:
    """Creates a dataset in ShareGPT format from a pandas DataFrame.

    Args:
        dataset_name (str): Name of the dataset
        df (pd.DataFrame): Input DataFrame containing the data
        user_prompt_template (str): Template string for human messages using DataFrame columns
        response_template (str): Template string for assistant responses using DataFrame columns
        user_columns (List[str]): DataFrame column names to use in user_prompt_template
        response_columns (List[str]): DataFrame column names to use in response_template
        dataset_save_path (str): Path to save the ShareGPT format dataset
        dataset_info_path (str): Path to save the dataset info file
        system_prompt (str, optional): System prompt to include. Defaults to None.
        tools (str, optional): Tool descriptions to include. Defaults to None.

    Example:
        >>> df = pd.DataFrame({
        ...     'question': ['What is 2+2?', 'What is Python?'],
        ...     'answer': ['4', 'A programming language'],
        ...     'context': ['Math', 'Programming']
        ... })
        >>> user_template = "Question: {question}\nContext: {context}"
        >>> response_template = "{answer}"
        >>> create_sharegpt_dataset(
        ...     dataset_name="math_qa",
        ...     df=df,
        ...     user_prompt_template=user_template,
        ...     response_template=response_template,
        ...     user_columns=['question', 'context'],
        ...     response_columns=['answer'],
        ...     dataset_save_path="data.json",
        ...     dataset_info_path="info.json",
        ...     system_prompt="You are a helpful assistant"
        ... )

        This will create:
        1. data.json with conversations in ShareGPT format:
        [
            {
                "conversations": [
                    {"from": "human", "value": "Question: What is 2+2?\nContext: Math"},
                    {"from": "gpt", "value": "4"}
                ],
                "system": "You are a helpful assistant"
            },
            ...
        ]

        2. info.json with dataset metadata:
        {
            "math_qa": {
                "file_name": "data.json",
                "messages": "conversations",
                "system": "system",
                "tools": "tools"
            },
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "system": "system",
                "tools": "tools"
            }
        }
    """
    sharegpt_data = []
    for index, row in tqdm(
        df.iterrows(), total=len(df), desc=f"Creating ShareGPT dataset: {dataset_name}"
    ):
        human_prompt = user_prompt_template.format(
            **{col: row[col] for col in user_columns}
        )
        response = response_template.format(
            **{col: row[col] for col in response_columns}
        )
        data = {
            "conversations": [
                {"from": "human", "value": human_prompt},
                {"from": "gpt", "value": response},
            ]
        }
        if system_prompt:
            data["system"] = system_prompt
        if tools:
            data["tools"] = tools
        sharegpt_data.append(data)

    with open(dataset_save_path, "w") as f:
        json.dump(sharegpt_data, f, indent=2)

    with open(dataset_info_path, "w") as f:
        info = {
            dataset_name: {
                "file_name": dataset_save_path,
                "formatting": "sharegpt",
                "columns": {
                    "messages": "conversations",
                },
            }
        }
        if system_prompt:
            info[dataset_name]["columns"]["system"] = "system"
        if tools:
            info[dataset_name]["columns"]["tools"] = "tools"
        json.dump(info, f, indent=2)
