import asyncio
import json
import re
import os
from typing import List, Dict, Optional
from dataclasses import dataclass
import tiktoken
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import logging
from ymir.llm.get_llm import get_llm
from ymir.llm.invoke import invoke_with_retry

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


def get_entity_types_prompt(text: str, language: str = DEFAULT_LANGUAGE) -> str:
    """Generate a prompt for extracting entity types from text."""
    # Sample the text if it's too long
    if len(text) > 10000:
        # Sample beginning, middle, and end of text
        chunk_size = 3000
        text_sample = (
            text[:chunk_size]
            + "\n...\n"
            + text[len(text) // 2 - chunk_size // 2 : len(text) // 2 + chunk_size // 2]
            + "\n...\n"
            + text[-chunk_size:]
        )
    else:
        text_sample = text

    return f"""
-Goal-
Given a text document, identify the most relevant entity types that should be extracted from the text.
The entity types should be specific enough to categorize the entities in the text, but general enough to be broadly applicable.
Use {language} as output language.

-Steps-
1. Read and analyze the text carefully to understand the domains and contexts it covers.
2. Identify the main types of entities present in the text (e.g., organization, person, location, technology, product, etc.).
3. Return 3-10 entity types that best categorize the entities in the text, in an array format.
4. Return the array as a valid JSON array of strings.

Text: {text_sample}
Output:
"""


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


async def extract_entity_types_with_llm(text: str, model_name: str) -> List[str]:
    """
    Extract entity types from text using LLM.

    Args:
        text: The input text to extract entity types from
        model_name: The specific model to use

    Returns:
        A list of entity types
    """
    # Get the LangChain model
    llm = get_llm(model_name)

    # Get the prompt template
    prompt_str = get_entity_types_prompt(text=text, language=DEFAULT_LANGUAGE)

    # Initialize prompt template - use raw string as we've already formatted it
    prompt_template = PromptTemplate(
        input_variables=[],
        template=prompt_str,
    )

    # Create the LLM chain
    chain = prompt_template | llm | JsonOutputParser()

    # Sample the text if it's too long - handled in the get_entity_types_prompt function

    try:
        response = await invoke_with_retry(
            chain, {"text": text, "language": DEFAULT_LANGUAGE}
        )
        return response

    except Exception as e:
        logging.error(f"Error extracting entity types: {e}")

    # Return empty list if extraction fails
    return []


async def extract_triplets_with_llm(
    text: str,
    model_name: str,
    entity_types: Optional[List[str]] = None,
    detect_entity_types: bool = True,
) -> Dict:
    """
    Extract triplets from text using LangChain with the specified provider and model.

    Args:
        text: The input text to extract triplets from
        model_name: The specific model to use
        entity_types: List of entity types to extract (optional)
        detect_entity_types: Whether to automatically detect entity types (default: True)

    Returns:
        A dictionary containing entities, relationships, and triplets
    """
    # Get the LangChain model
    llm = get_llm(model_name)

    # Detect entity types if requested and not provided
    if detect_entity_types and entity_types is None:
        entity_types = await extract_entity_types_with_llm(text, model_name)
    elif entity_types is None:
        entity_types = DEFAULT_ENTITY_TYPES

    # Initialize prompt template
    prompt_template = PromptTemplate(
        input_variables=["text", "entity_types", "language"],
        template="""
-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
Use {language} as output language.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If English, capitalized the name.
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"<|><entity_name><|><entity_type><|><entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
Format each relationship as ("relationship"<|><source_entity><|><target_entity><|><relationship_description><|><relationship_keywords><|><relationship_strength>)

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.
Format the content-level key words as ("content_keywords"<|><high_level_keywords>)

4. Return output in {language} as a single list of all the entities and relationships identified in steps 1 and 2. Use **##** as the list delimiter.

5. When finished, output <|COMPLETE|>

Entity_types: {entity_types}
Text: {text}
Output:
""",
    )

    # Create the LLM chain
    chain = prompt_template | llm

    # Generate chunks to handle long texts
    chunks = chunking_by_token_size(text, overlap_token_size=100, max_token_size=1200)

    entities = {}
    relationships = {}
    triplets = []

    for chunk in chunks:
        chunk_key = compute_mdhash_id(chunk["content"], prefix="chunk-")

        # Call the LLM chain with the current chunk
        try:
            response = await invoke_with_retry(
                chain,
                {
                    "text": chunk["content"],
                    "entity_types": ", ".join(entity_types),
                    "language": DEFAULT_LANGUAGE,
                },
            )
        except Exception as e:
            logging.error(f"Error calling LLM: {e}")
            continue

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
        "entity_types": entity_types,  # Return the detected entity types
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


# Simple synchronous wrapper for easier use
def extract_triplets(
    text: str,
    provider: str = "openai",
    model_name: str = "gpt-4o-mini",
    api_key: str = None,
    entity_types: List[str] = None,
    detect_entity_types: bool = True,
) -> Dict:
    """
    Synchronous wrapper to extract triplets from text.

    Args:
        text: The input text to extract triplets from
        provider: The LLM provider to use (e.g., "openai", "anthropic", "huggingface")
        model_name: The specific model to use
        api_key: API key for the provider (optional, will use environment variable if not provided)
        entity_types: List of entity types to extract (optional)
        detect_entity_types: Whether to automatically detect entity types (default: True)

    Returns:
        A dictionary containing entities, relationships, and triplets
    """
    if api_key and provider.lower() == "openai":
        os.environ["OPENAI_API_KEY"] = api_key

    async def run():
        return await extract_triplets_with_llm(
            text=text,
            model_name=model_name,
            entity_types=entity_types,
            detect_entity_types=detect_entity_types,
        )

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

    print("------ Using automatic entity type detection ------")
    # Extract triplets using OpenAI with automatic entity type detection
    result_auto = extract_triplets(
        text=sample_text,
        provider="openai",
        model_name="gpt-4o-mini",
        detect_entity_types=True,
    )

    # Print results with auto-detected entity types
    print(f"Detected entity types: {result_auto['entity_types']}")
    print(triplets_to_json(result_auto))

    print("\n------ Using specific entity types ------")
    # Extract triplets with manually specified entity types
    result_manual = extract_triplets(
        text=sample_text,
        provider="openai",
        model_name="gpt-4o-mini",
        entity_types=["company", "product", "person", "location"],
        detect_entity_types=False,
    )

    # Print results with manual entity types
    print(f"Manual entity types: {result_manual['entity_types']}")
    print(triplets_to_json(result_manual))

    # Save triplets to CSV
    csv_file = triplets_to_csv(result_auto["triplets"])
    print(f"\nTriplets saved to {csv_file}")
