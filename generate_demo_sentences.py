import random
import os

# Configuration
ages = ["infant", "child", "adolescent", "young_adult", "adult", "middle_aged", "aged", "elderly"]
genders = ["male", "female"]
ethnicities = ["white", "black", "hispanic", "asian", "other"]
languages = ["english", "spanish", "portuguese", "mandarin", "other"]
religions = ["christian", "jewish", "muslim", "buddhist", "other", "none"]
marital_statuses = ["single", "married", "divorced", "widowed"]
insurances = ["medicare", "medicaid", "private", "self_pay", "government"]

caregiver_titles = [
    "attending", "resident_physician", "fellow", "nurse", "nurse_practitioner",
    "physician_assistant", "social_worker", "physical_therapist"
]
admission_types = ["emergency", "elective", "urgent", "newborn"]

# Sentence templates
templates = [
    "The patient was admitted with {complaint} and required immediate evaluation.",
    "Physical examination revealed {finding} consistent with {diagnosis}.",
    "The patient reported {symptom} for the past {duration}.",
    "Laboratory results showed {lab_finding} suggesting {diagnosis}.",
    "The patient was started on {treatment} with good response.",
    "Follow-up imaging demonstrated {finding} compared to prior studies.",
    "The patient was discharged in stable condition after {treatment}.",
    "Vital signs were stable with {vital} within normal limits.",
    "The patient denied {symptom} at the time of evaluation.",
    "Family history was significant for {condition}.",
    "The patient has a history of {condition} managed with {treatment}.",
    "Neurological examination was intact without focal {finding}.",
    "Cardiovascular examination revealed {finding} on auscultation.",
    "The patient was counseled on {topic} prior to discharge.",
    "Surgical pathology confirmed {diagnosis} with clear margins.",
    "Chronic {condition} was noted on review of prior records.",
    "The patient responded well to {treatment} over the course of admission.",
    "An echocardiogram was obtained showing {finding}.",
    "The patient was evaluated by {specialist} for further management.",
    "Social work was consulted to address {topic} prior to discharge.",
]

complaints   = ["chest pain", "shortness of breath", "abdominal pain", "altered mental status",
                 "fever", "syncope", "weakness", "headache", "back pain", "falls"]
findings     = ["cardiomegaly", "consolidation", "effusion", "atelectasis", "normal sinus rhythm",
                "ST changes", "focal deficits", "murmur", "tenderness", "organomegaly"]
diagnoses    = ["pneumonia", "heart failure", "sepsis", "COPD exacerbation", "acute MI",
                "pulmonary embolism", "urinary tract infection", "cellulitis", "anemia",
                "dehydration"]
symptoms     = ["dyspnea", "nausea", "vomiting", "diarrhea", "fatigue", "chest tightness",
                "palpitations", "dizziness", "diaphoresis", "lower extremity edema"]
durations    = ["two days", "one week", "three days", "several hours", "one month"]
treatments   = ["antibiotics", "diuretics", "anticoagulation", "supplemental oxygen",
                "IV fluids", "vasopressors", "bronchodilators", "insulin", "analgesics",
                "steroids"]
vitals       = ["blood pressure", "heart rate", "oxygen saturation", "temperature",
                "respiratory rate"]
conditions   = ["hypertension", "diabetes mellitus", "coronary artery disease", "COPD",
                "chronic kidney disease", "atrial fibrillation", "heart failure",
                "hypothyroidism", "obesity", "depression"]
topics       = ["fall precautions", "medication compliance", "diet modifications",
                "follow-up appointments", "wound care", "smoking cessation",
                "activity restrictions"]
specialists  = ["cardiology", "pulmonology", "nephrology", "neurology", "gastroenterology",
                "infectious disease", "hematology", "endocrinology"]
lab_findings = ["elevated troponin", "leukocytosis", "anemia", "hyponatremia",
                "elevated creatinine", "elevated lactate", "thrombocytopenia",
                "hypokalemia", "coagulopathy"]


def generate_training_file(filename: str = "training_data_with_context.txt",
                            num_rows: int = 1_000_000) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        for _ in range(num_rows):
            # Patient group fields (may be empty).
            n_patient = random.randint(0, 5)
            all_patient_fields = [
                random.choice(ages),
                random.choice(genders),
                random.choice(ethnicities),
                random.choice(languages),
                random.choice(marital_statuses),
                random.choice(insurances),
            ]
            patient_group = random.sample(all_patient_fields, min(n_patient, len(all_patient_fields)))
            patient_str = " ".join(patient_group)

            # Encounter group fields (may be empty).
            n_encounter = random.randint(0, 2)
            all_encounter_fields = [
                random.choice(caregiver_titles),
                random.choice(admission_types),
            ]
            encounter_group = random.sample(all_encounter_fields, min(n_encounter, len(all_encounter_fields)))
            encounter_str = " ".join(encounter_group)

            # Sentence from template.
            template = random.choice(templates)
            sentence = template.format(
                complaint=random.choice(complaints),
                finding=random.choice(findings),
                diagnosis=random.choice(diagnoses),
                symptom=random.choice(symptoms),
                duration=random.choice(durations),
                treatment=random.choice(treatments),
                vital=random.choice(vitals),
                condition=random.choice(conditions),
                topic=random.choice(topics),
                specialist=random.choice(specialists),
                lab_finding=random.choice(lab_findings),
            )

            # Format: <PatientGroup> ||| <EncounterGroup> ||| <Words>
            row = f"{patient_str} ||| {encounter_str} ||| {sentence}"
            f.write(row + "\n")

    print(f"Generated {num_rows} rows in '{filename}'")


if __name__ == "__main__":
    generate_training_file()
