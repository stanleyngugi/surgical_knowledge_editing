# ~/mved_probabilistic_surgery/scripts/utils/data_utils.py
import json
from pathlib import Path

def load_query_templates(file_path: Path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_spo_queries(subject: str, templates: list) -> list:
    return [template.replace("{S}", subject) for template in templates]

def create_spo_dataset(facts_dict: dict, templates_path: Path, output_path: Path):
    query_templates = load_query_templates(templates_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f_out:
        for subject, obj in facts_dict.items():
            queries = generate_spo_queries(subject, query_templates)
            for query in queries:
                data_entry = {"query": query, "subject": subject, "expected_object": obj}
                f_out.write(json.dumps(data_entry) + "\n")
    print(f"Generated S-P-O dataset at {output_path}")