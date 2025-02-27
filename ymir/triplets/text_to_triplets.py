import asyncio
import json
import re
import os
from typing import List, Dict, Optional
from dataclasses import dataclass
import tiktoken

# Constants for formatting
TUPLE_DELIMITER = "<|>"
RECORD_DELIMITER = "##"
COMPLETION_DELIMITER = "<|COMPLETE|>"
DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event", "category"]
DEFAULT_LANGUAGE = "English"


@dataclass
class Entity:
    entity_name: str
    entity_type: str
    description: str
    source_id: str


@dataclass
class Relationship:
    src_id: str
    tgt_id: str
    description: str
    keywords: str
    weight: float
    source_id: str


@dataclass
class Triplet:
    subject: str
    predicate: str
    object: str
    description: str
    confidence: float


def clean_str(s: str) -> str:
    """Clean a string by removing quotes and extra whitespace."""
    if not s:
        return ""
    s = s.strip()
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    return s.strip()


def is_float_regex(s: str) -> bool:
    """Check if a string can be converted to a float."""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def encode_string_by_tiktoken(text: str, model_name: str = "gpt-4o-mini") -> List[int]:
    """Encode a string into tokens using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return encoding.encode(text)


def decode_tokens_by_tiktoken(
    tokens: List[int], model_name: str = "gpt-4o-mini"
) -> str:
    """Decode tokens back into a string using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return encoding.decode(tokens)


def chunking_by_token_size(
    content: str,
    overlap_token_size=128,
    max_token_size=1024,
    tiktoken_model="gpt-4o-mini",
) -> List[Dict]:
    """Split content into chunks based on token size."""
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], model_name=tiktoken_model
        )
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results


def split_string_by_multi_markers(text: str, markers: List[str]) -> List[str]:
    """Split a string by multiple markers."""
    if not text or not markers:
        return [text]

    pattern = "|".join(map(re.escape, markers))
    return [s.strip() for s in re.split(pattern, text) if s.strip()]


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """Compute a hash ID for a content string."""
    import hashlib

    return f"{prefix}{hashlib.md5(content.encode()).hexdigest()}"


def handle_entity_extraction(
    record_attributes: List[str], chunk_key: str
) -> Optional[Entity]:
    """Extract entity information from record attributes."""
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None

    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key

    return Entity(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )


def handle_relationship_extraction(
    record_attributes: List[str], chunk_key: str
) -> Optional[Relationship]:
    """Extract relationship information from record attributes."""
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None

    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])
    edge_keywords = clean_str(record_attributes[4])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )

    return Relationship(
        src_id=source,
        tgt_id=target,
        description=edge_description,
        keywords=edge_keywords,
        weight=weight,
        source_id=edge_source_id,
    )


def relationship_to_triplet(relationship: Relationship) -> Triplet:
    """Convert a relationship to a triplet format."""
    return Triplet(
        subject=relationship.src_id,
        predicate=relationship.keywords.split(",")[0]
        if "," in relationship.keywords
        else relationship.keywords,
        object=relationship.tgt_id,
        description=relationship.description,
        confidence=relationship.weight / 10.0,  # Normalize to 0-1 range
    )


def get_entity_extraction_prompt(
    text: str,
    entity_types: List[str] = DEFAULT_ENTITY_TYPES,
    language: str = DEFAULT_LANGUAGE,
) -> str:
    """Generate a prompt for entity extraction."""
    return f"""
-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
Use {language} as output language.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If English, capitalized the name.
- entity_type: One of the following types: [{", ".join(entity_types)}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{TUPLE_DELIMITER}<entity_name>{TUPLE_DELIMITER}<entity_type>{TUPLE_DELIMITER}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
Format each relationship as ("relationship"{TUPLE_DELIMITER}<source_entity>{TUPLE_DELIMITER}<target_entity>{TUPLE_DELIMITER}<relationship_description>{TUPLE_DELIMITER}<relationship_keywords>{TUPLE_DELIMITER}<relationship_strength>)

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.
Format the content-level key words as ("content_keywords"{TUPLE_DELIMITER}<high_level_keywords>)

4. Return output in {language} as a single list of all the entities and relationships identified in steps 1 and 2. Use **{RECORD_DELIMITER}** as the list delimiter.

5. When finished, output {COMPLETION_DELIMITER}

Entity_types: {", ".join(entity_types)}
Text: {text}
Output:
"""


async def extract_triplets_with_llm(
    text: str, llm_function, entity_types: List[str] = DEFAULT_ENTITY_TYPES
) -> Dict:
    """
    Extract triplets from text using an LLM function.

    Args:
        text: The input text to extract triplets from
        llm_function: An async function that takes a prompt and returns a response
        entity_types: List of entity types to extract

    Returns:
        A dictionary containing entities, relationships, and triplets
    """
    # Generate chunks to handle long texts
    chunks = chunking_by_token_size(text, overlap_token_size=100, max_token_size=1200)

    entities = {}
    relationships = {}
    triplets = []

    for chunk in chunks:
        chunk_key = compute_mdhash_id(chunk["content"], prefix="chunk-")
        prompt = get_entity_extraction_prompt(chunk["content"], entity_types)

        # Call the LLM function
        response = await llm_function(prompt)

        # Process the response
        records = split_string_by_multi_markers(
            response, [RECORD_DELIMITER, COMPLETION_DELIMITER]
        )

        for record in records:
            record_match = re.search(r"\((.*)\)", record)
            if record_match is None:
                continue

            record_content = record_match.group(1)
            record_attributes = split_string_by_multi_markers(
                record_content, [TUPLE_DELIMITER]
            )

            # Handle entity extraction
            entity = handle_entity_extraction(record_attributes, chunk_key)
            if entity:
                entities[entity.entity_name] = entity
                continue

            # Handle relationship extraction
            relationship = handle_relationship_extraction(record_attributes, chunk_key)
            if relationship:
                rel_key = (relationship.src_id, relationship.tgt_id)
                relationships[rel_key] = relationship

                # Convert to triplet format
                triplet = relationship_to_triplet(relationship)
                triplets.append(triplet)

    return {
        "entities": list(entities.values()),
        "relationships": list(relationships.values()),
        "triplets": triplets,
    }


def triplets_to_json(triplets_data: Dict) -> str:
    """Convert triplets data to a JSON string."""

    class DataclassJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if hasattr(obj, "__dataclass_fields__"):
                return {k: getattr(obj, k) for k in obj.__dataclass_fields__}
            return super().default(obj)

    return json.dumps(triplets_data, cls=DataclassJSONEncoder, indent=2)


def triplets_to_csv(triplets: List[Triplet], output_file: str = "triplets.csv") -> str:
    """Save triplets to a CSV file."""
    import csv

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Subject", "Predicate", "Object", "Description", "Confidence"])

        for triplet in triplets:
            writer.writerow(
                [
                    triplet.subject,
                    triplet.predicate,
                    triplet.object,
                    triplet.description,
                    triplet.confidence,
                ]
            )

    return output_file


# Example of how to use with OpenAI
async def example_with_openai(text: str) -> Dict:
    """Example of how to use the triplet extraction with OpenAI."""
    import openai

    # Set your API key
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    async def openai_llm(prompt: str) -> str:
        """Call OpenAI API to get a response."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=openai.api_key)
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content

    return await extract_triplets_with_llm(text, openai_llm)


# Example of how to use with local LLM
async def example_with_local_llm(text: str) -> Dict:
    """Example of how to use the triplet extraction with a local LLM."""
    from llama_cpp import Llama

    # Load the model
    model_path = "path/to/your/model.gguf"
    llm = Llama(model_path=model_path, n_ctx=4096)

    async def local_llm(prompt: str) -> str:
        """Call local LLM to get a response."""
        response = llm(prompt, max_tokens=2048, temperature=0.2)
        return response["choices"][0]["text"]

    return await extract_triplets_with_llm(text, local_llm)


# Simple synchronous wrapper for easier use
def extract_triplets(text: str, api_key: str = None) -> Dict:
    """
    Synchronous wrapper to extract triplets from text.

    Args:
        text: The input text to extract triplets from
        api_key: OpenAI API key (optional, will use environment variable if not provided)

    Returns:
        A dictionary containing entities, relationships, and triplets
    """
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    async def run():
        return await example_with_openai(text)

    loop = asyncio.get_event_loop()
    try:
        return loop.run_until_complete(run())
    except RuntimeError:
        # Create a new event loop if the current one is closed
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop.run_until_complete(run())


# Example usage
if __name__ == "__main__":
    sample_text = """
    Apple Inc. is an American multinational technology company headquartered in Cupertino, California.
    Tim Cook is the CEO of Apple since 2011, after Steve Jobs resigned.
    Apple designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories.
    The company's hardware products include the iPhone, the Mac computer, the iPad, the Apple Watch, and Apple TV.
    """

    # Check if API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "Please set your OPENAI_API_KEY environment variable or provide it as an argument."
        )
        exit(1)

    # Extract triplets
    result = extract_triplets(sample_text)

    # Print results
    print(triplets_to_json(result))

    # Save triplets to CSV
    csv_file = triplets_to_csv(result["triplets"])
    print(f"Triplets saved to {csv_file}")
