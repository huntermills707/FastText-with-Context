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

    # Extracting patient group metadata
    patient_entities = list(model.idx2patient.values())
    patient_entities.sort()

    with open('patient_fields.txt', 'w', encoding='utf-8') as f:
        for entity in patient_entities:
            f.write(f"{entity}\n")
    print(f"Saved {len(patient_entities)} patient group fields to patient_fields.txt")

    # Extracting encounter group metadata
    encounter_entities = list(model.idx2encounter.values())
    encounter_entities.sort()

    with open('encounter_fields.txt', 'w', encoding='utf-8') as f:
        for entity in encounter_entities:
            f.write(f"{entity}\n")
    print(f"Saved {len(encounter_entities)} encounter group fields to encounter_fields.txt")

if __name__ == "__main__":
    main()
