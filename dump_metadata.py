import sys
from fasttext_context import FastTextContext

def main():
    model_path = 'mimic-iii.bin'
    try:
        model = FastTextContext()
        model.load_model(model_path)
    except FileNotFoundError:
        print(f"Error: {model_path} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Extracting patient metadata
    # The entities are stored in idx2patient dictionary (mapping index to string)
    patient_entities = list(model.idx2patient.values())
    # Sort for deterministic output
    patient_entities.sort()

    with open('patient_metadata.txt', 'w', encoding='utf-8') as f:
        for entity in patient_entities:
            f.write(f"{entity}\n")
    print(f"Saved {len(patient_entities)} patient entities to patient_metadata.txt")

    # Extracting provider metadata
    provider_entities = list(model.idx2provider.values())
    provider_entities.sort()

    with open('provider_metadata.txt', 'w', encoding='utf-8') as f:
        for entity in provider_entities:
            f.write(f"{entity}\n")
    print(f"Saved {len(provider_entities)} provider entities to provider_metadata.txt")

if __name__ == "__main__":
    main()
