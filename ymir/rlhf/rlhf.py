import json
from typing import List, Dict, Any, Union, Optional


class RLHFDatasetBuilder:
    def __init__(self, filename: Union[str, None] = None):
        self.rlhf_data: List[Dict[str, Any]] = []
        if filename is not None:
            self.load_rlhf_data(filename)

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
        conversation=None,  # new parameter for full conversation in OpenAI format
    ):
        """
        Save an RLHF entry containing the prompts, the two model responses, a rating indicating which response was chosen,
        and any notes. Additionally, save the conversation in OpenAI conversational format.
        The 'rating' parameter should be either "LLM1" or "LLM2", indicating the chosen response.
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
            "conversation": conversation if conversation is not None else [],
            "chosen": response1 if rating == "LLM1" else response2,
            "rejected": response2 if rating == "LLM1" else response1,
        }
        self.rlhf_data.append(entry)
        return "RLHF entry saved!"

    def load_dataset_jsonl(self, filename: Optional[str] = None):
        if filename is None:
            return
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                self.rlhf_data.append(json.loads(line))

    def download_dataset_jsonl(self, filename: str = "dataset.jsonl"):
        """
        Write the RLHF dataset to a JSONL file and return its filename.
        """
        with open(filename, "w", encoding="utf-8") as f:
            for entry in self.rlhf_data:
                f.write(json.dumps(entry) + "\n")
        return filename

    def get_entry_by_id(self, entry_id: str):
        """
        Get an RLHF entry by its ID.
        If no entry matches the ID, return None.
        """
        try:
            # Convert string ID to integer index if needed
            idx = int(entry_id)
            if 0 <= idx < len(self.rlhf_data):
                return self.rlhf_data[idx]
        except ValueError:
            # If entry_id is not a valid integer index, try matching by other fields
            for entry in self.rlhf_data:
                if str(entry.get("id", "")) == entry_id:
                    return entry

        return None
