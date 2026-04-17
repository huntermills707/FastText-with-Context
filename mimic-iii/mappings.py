"""
Mapping dictionaries and helper functions for MIMIC-III data normalization.
"""

# --- CG_TITLE grouping -------------------------------------------------
_CG_MAP = {
    # Physician
    'md': 'physician', 'mds': 'physician', 'mcd': 'physician',
    'imd': 'physician', 'drm': 'physician', 'dml': 'physician',

    # Medical student / resident
    'res': 'med_student', 'medres': 'med_student', 'med_st': 'med_student',
    'medst': 'med_student', 'ms': 'med_student', 'ms_v': 'med_student',
    'msiv': 'med_student', 'msv': 'med_student', 'hms_iv': 'med_student',
    'hms_ms': 'med_student', 'hmsiv': 'med_student', 'mswint': 'med_student',

    # Nurse (RN)
    'rn': 'nurse', 'rnba': 'nurse', 'rnc': 'nurse', 'rncm': 'nurse',
    'rns': 'nurse', 'nurse': 'nurse', 'nurs': 'nurse', 'srn': 'nurse',
    'ccrn': 'nurse',

    # Nurse practitioner
    'np': 'nurse_practitioner', 'nnp': 'nurse_practitioner',
    'snnp': 'nurse_practitioner', 'snp': 'nurse_practitioner',

    # Nursing student
    'sn': 'nursing_student', 'stn': 'nursing_student', 'stnrs': 'nursing_student',
    'stnuiv': 'nursing_student', 'stnur': 'nursing_student',
    'stnurs': 'nursing_student', 'studen': 'nursing_student',
    'stunur': 'nursing_student', 'rnstu': 'nursing_student',
    'nstude': 'nursing_student', 'nsv': 'nursing_student',

    # Nursing assistant / tech
    'na': 'nurse_aide', 'pca': 'nurse_aide', 'pct': 'nurse_aide',
    'ua': 'nurse_aide', 'uc': 'nurse_aide', 'uco': 'nurse_aide',

    # Respiratory therapy
    'rrt': 'resp_therapy', 'rrts': 'resp_therapy', 'rt': 'resp_therapy',
    'rth': 'resp_therapy', 'rts': 'resp_therapy', 'crt': 'resp_therapy',
    'srt': 'resp_therapy', 'rtstu': 'resp_therapy',

    # Physician assistant
    'pa': 'physician_assistant',

    # Pharmacy
    'pharmd': 'pharmacy', 'rph': 'pharmacy',

    # Social work
    'sw': 'social_work', 'sw_int': 'social_work', 'swint': 'social_work',
    'licsw': 'social_work',

    # Rehab (PT/OT)
    'pt': 'rehab', 'rehab': 'rehab', 'otr/l': 'rehab',

    # Dietitian
    'rd': 'dietitian', 'rd,ldn': 'dietitian', 'rd/ldn': 'dietitian',
    'ms,rd': 'dietitian',

    # Unit coordinator / co-worker
    'cm': 'coordinator', 'co-ord': 'coordinator', 'co-wkr': 'coordinator',
    'cowker': 'coordinator', 'cowkr': 'coordinator', 'cowork': 'coordinator',
    'co-op': 'coordinator', 'coopst': 'coordinator',

    # PhD / other clinician
    'phd': 'phd',
}

def map_cg_title(title: str) -> str:
    """Normalise a raw CG_TITLE string to an encounter group label."""
    if title is None:
        return 'other'
    return _CG_MAP.get(title.strip().lower(), 'other')

# --- Religion grouping -------------------------------------------------
_RELIGION_MAP = {
    # Protestant
    'baptist': 'protestant',
    'episcopalian': 'protestant',
    'lutheran': 'protestant',
    'methodist': 'protestant',
    'protestant_quaker': 'protestant',
    '7th_day_adventist': 'protestant',

    # Catholic
    'catholic': 'catholic',

    # Orthodox
    'greek_orthodox': 'orthodox',
    'romanian_east._orth': 'orthodox',

    # Other Christian
    'christian_scientist': 'other_christian',
    'unitarian-universalist': 'other_christian',
    "jehovah's_witness": 'other_christian',

    # Jewish
    'jewish': 'jewish',
    'hebrew': 'jewish',

    # Muslim
    'muslim': 'muslim',

    # Eastern
    'buddhist': 'eastern',
    'hindu': 'eastern',

    # Unknown
    'not_specified': 'unknown',
    'unobtainable': 'unknown',
    'other': 'unknown',
}

def map_religion(religion: str) -> str:
    """Normalise a raw RELIGION string."""
    if religion is None:
        return 'unknown'
    return _RELIGION_MAP.get(religion.strip().lower().replace(' ', '_'), 'unknown')

# --- Language grouping ---------------------------------------------------
_LANGUAGE_MAP = {
    # English
    'engl': 'english',
    'amer': 'asl',  # American Sign Language — still English-context patients

    # Spanish
    'span': 'spanish',
    '*spa': 'spanish',

    # Portuguese / Cape Verdean Creole
    'port': 'portuguese',
    'cape': 'portuguese',
    'ptun': 'portuguese',
    '*cre': 'portuguese',  # Creole (Cape Verdean in Boston context)

    # Chinese
    '*chi': 'chinese',
    '*can': 'chinese',
    'cant': 'chinese',
    'mand': 'chinese',
    '*man': 'chinese',

    # Russian / Eastern European
    'russ': 'russian_ee',
    '*rus': 'russian_ee',
    '*bos': 'russian_ee',
    'serb': 'russian_ee',
    '*bul': 'russian_ee',
    '*lit': 'russian_ee',
    '*rom': 'russian_ee',
    '*hun': 'russian_ee',
    'alba': 'russian_ee',
    'poli': 'russian_ee',

    # South Asian
    'hind': 'south_asian',
    '*ben': 'south_asian',
    'beng': 'south_asian',
    '*guj': 'south_asian',
    '*nep': 'south_asian',
    '*pun': 'south_asian',
    '*tam': 'south_asian',
    '*tel': 'south_asian',
    '*urd': 'south_asian',
    'urdu': 'south_asian',

    # Southeast Asian
    'viet': 'se_asian',
    'camb': 'se_asian',
    '*khm': 'se_asian',
    'laot': 'se_asian',
    '*bur': 'se_asian',
    '*fil': 'se_asian',
    '*phi': 'se_asian',
    'taga': 'se_asian',
    'thai': 'se_asian',

    # East Asian (non-Chinese)
    'japa': 'east_asian',
    'kore': 'east_asian',

    # Arabic / Persian / Turkish
    'arab': 'mideast',
    '*ara': 'mideast',
    '*leb': 'mideast',
    '*far': 'mideast',
    '*per': 'mideast',
    'pers': 'mideast',
    'turk': 'mideast',
    '*mor': 'mideast',

    # Haitian Creole
    'hait': 'haitian_creole',

    # French
    'fren': 'french',

    # African
    'ethi': 'african',
    '*amh': 'african',
    'soma': 'african',
    '*ful': 'african',
    '*ibo': 'african',
    '*yor': 'african',
    '*toi': 'african',
    '*toy': 'african',

    # European other
    'germ': 'european_other',
    'gree': 'european_other',
    'ital': 'european_other',
    '*dut': 'european_other',
    '*arm': 'european_other',
    '*yid': 'european_other',

    # Unknown / garbled
    '**_t': 'unknown',
    '**sh': 'unknown',
    '**to': 'unknown',
    '*_be': 'unknown',
    '*_fu': 'unknown',
    '*cdi': 'unknown',
    '*dea': 'unknown',  # likely "deaf"
}

def map_language(language: str) -> str:
    """Normalise a raw LANGUAGE string."""
    if language is None:
        return 'unknown'
    return _LANGUAGE_MAP.get(language.strip().lower().replace(' ', '_'), 'unknown')

# --- Ethnicity grouping ------------------------------------------------
_ETHNICITY_MAP = {
    # White
    'white': 'white',
    'white_-_brazilian': 'white',
    'white_-_eastern_european': 'white',
    'white_-_other_european': 'white',
    'white_-_russian': 'white',
    'portuguese': 'white',

    # Black
    'black/african_american': 'black',
    'black/african': 'black',
    'black/cape_verdean': 'black',
    'black/haitian': 'black',
    'caribbean_island': 'black',

    # Latinx
    'hispanic_or_latino': 'latinx',
    'hispanic/latino_-_puerto_rican': 'latinx',
    'hispanic/latino_-_dominican': 'latinx',
    'hispanic/latino_-_mexican': 'latinx',
    'hispanic/latino_-_central_american_(other)': 'latinx',
    'hispanic/latino_-_guatemalan': 'latinx',
    'hispanic/latino_-_honduran': 'latinx',
    'hispanic/latino_-_salvadoran': 'latinx',
    'hispanic/latino_-_colombian': 'latinx',
    'hispanic/latino_-_cuban': 'latinx',
    'south_american': 'latinx',

    # Asian
    'asian': 'asian',
    'asian_-_chinese': 'asian',
    'asian_-_asian_indian': 'asian',
    'asian_-_vietnamese': 'asian',
    'asian_-_cambodian': 'asian',
    'asian_-_filipino': 'asian',
    'asian_-_japanese': 'asian',
    'asian_-_korean': 'asian',
    'asian_-_thai': 'asian',
    'asian_-_other': 'asian',

    # Middle Eastern
    'middle_eastern': 'middle_eastern',

    # Native / Pacific Islander
    'american_indian/alaska_native': 'native',
    'american_indian/alaska_native_federally_recognized_tribe': 'native',
    'native_hawaiian_or_other_pacific_islander': 'pi',

    # Multiracial
    'multi_race_ethnicity': 'multiracial',

    # Unknown
    'other': 'unknown',
    'patient_declined_to_answer': 'unknown',
    'unable_to_obtain': 'unknown',
}

def map_ethnicity(ethnicity: str) -> str:
    """Normalise a raw ETHNICITY string."""
    if ethnicity is None:
        return 'unknown'
    return _ETHNICITY_MAP.get(ethnicity.strip().lower().replace(' ', '_'), 'unknown')
