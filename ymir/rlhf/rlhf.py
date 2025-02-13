import json
from typing import List, Dict, Any
from langchain.prompts import ChatPromptTemplate
from ymir.llm import get_llm


class RLHFDatasetBuilder:
    def __init__(self):
        self.rlhf_data: List[Dict[str, Any]] = []

    def chat_arena(
        self,
        system_prompt: str,
        user_prompt: str,
        llm1_name: str,
        llm2_name: str,
        llm1_config: Dict = None,
        llm2_config: Dict = None,
    ):
        """
        Send the same system and user prompt to two different LLMs.
        """
        messages = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("user", user_prompt)]
        )

        llm1 = get_llm(llm1_name)
        llm2 = get_llm(llm2_name)

        if llm1_config:
            response1 = llm1.invoke(messages, **llm1_config).content
        else:
            response1 = llm1.invoke(messages).content

        if llm2_config:
            response2 = llm2.invoke(messages, **llm2_config).content
        else:
            response2 = llm2.invoke(messages).content

        return response1, response2

    def save_rlhf_entry(
        self,
        system_prompt,
        user_prompt,
        llm1_name,
        llm2_name,
        response1,
        response2,
        rating,
        notes,
    ):
        """
        Save an entry containing the prompts, the two model responses, a rating, and any notes.
        """
        entry = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "llm1": llm1_name,
            "llm2": llm2_name,
            "response1": response1,
            "response2": response2,
            "rating": rating,
            "notes": notes,
        }
        self.rlhf_data.append(entry)
        return "RLHF entry saved!"

    def download_rlhf_dataset(self, filename: str = "rlhf_dataset.jsonl"):
        """
        Write the RLHF dataset to a JSONL file and return its filename.
        """
        with open(filename, "w", encoding="utf-8") as f:
            for entry in self.rlhf_data:
                f.write(json.dumps(entry) + "\n")
        return filename
