"""
Explicit, reviewable category assignment for the 112 SocialStigmaQA identities.

Written as a literal mapping rather than a keyword regex because the race result
is the paper's headline: a reviewer must be able to check every assignment, and
a regex silently mis-sorts edge cases ("Middle Eastern" has no shared substring
with the other race terms; "Movement Remitted" matches a naive /ment/).

Two things this encodes that a flat category list cannot:

RACE_CLUSTER -- the benchmark ships several labels for the same underlying
    category: "African American", "Black", and "Black/African American"; also
    "Latina", "Latino", "Latina/Latino"; and "Asian" / "Asian American".
    Treating those 12 labels as 12 independent observations is pseudo-
    replication and would inflate significance.  Tests over race should cluster
    on RACE_CLUSTER (7 distinct categories), not on the 12 labels.

    There is NO White reference category in the benchmark.  Every racial label
    is a minoritised one, so no within-race contrast is available -- worth
    stating explicitly as a limitation rather than letting a reader assume one.

ASCRIBED vs ACQUIRED -- whether the attribute is one a person is treated as
    born into (race, sex characteristics, age) or one attributed to conduct or
    circumstance (criminal record, substance use, employment).  The absorption
    result is about race specifically; this split is what lets us ask whether
    the effect is "race" or the broader "ascribed protected attribute".
"""

CATEGORY = {
    # --- race / ethnicity (all minoritised; no White reference exists) -------
    "African American": "race", "Black": "race", "Black/African American": "race",
    "Asian": "race", "Asian American": "race", "South Asian": "race",
    "Latina": "race", "Latino": "race", "Latina/Latino": "race",
    "Middle Eastern": "race", "Native American": "race", "Multiracial": "race",

    # --- religion -----------------------------------------------------------
    "Atheist": "religion", "Fundamentalist Christian": "religion",
    "Jewish": "religion", "Muslim": "religion",

    # --- immigration --------------------------------------------------------
    "Documented Immigrant": "immigration", "Undocumented Immigrant": "immigration",

    # --- sexual orientation -------------------------------------------------
    "Asexual": "sexuality", "Bisexual": "sexuality", "Gay": "sexuality",
    "Lesbian": "sexuality", "Non-Heterosexual": "sexuality",
    "Lesbian/Gay/Bisexual/Non-Heterosexual": "sexuality", "Polyamorous": "sexuality",

    # --- gender / sex characteristics ---------------------------------------
    "Intersex": "gender", "Transgender": "gender",

    # --- disability: sensory & communication --------------------------------
    "Blind Completely": "disability_sensory", "Deaf Completely": "disability_sensory",
    "Speech Disability": "disability_sensory",

    # --- disability: mobility -----------------------------------------------
    "Gait Impairment Current Avg. Sev.": "disability_mobility",
    "Gait Impairment Remitted Avg. Sev.": "disability_mobility",
    "Movement Current Avg. Sev.": "disability_mobility",
    "Movement Remitted Avg. Sev.": "disability_mobility",
    "Movement/Gait Impairment Current Avg. Sev.": "disability_mobility",
    "Movement/Gait Impairment Remitted Avg. Sev.": "disability_mobility",
    "Using A Wheel Chair All The Time": "disability_mobility",

    # --- mental health / cognition ------------------------------------------
    "Autism Or Autism Spectrum Disorder": "mental_health",
    "Bipolar Disorder Remitted": "mental_health", "Bipolar Disorder Symptomatic": "mental_health",
    "Depression Remitted": "mental_health", "Depression Symptomatic": "mental_health",
    "Schizophrenia Remitted": "mental_health", "Schizophrenia Symptomatic": "mental_health",
    "Mental Retardation": "mental_health",

    # --- substance use ------------------------------------------------------
    "Alcohol Dependency Current": "substance", "Alcohol Dependency Remitted": "substance",
    "Cocaine Use Recreationally": "substance", "Crystal Meth. Use Recreationally": "substance",
    "Drug Dependency Current": "substance", "Drug Dependency Remitted": "substance",
    "Injection Drug Use": "substance", "Marijuana Use Recreationally": "substance",
    "Smoking Cigarettes Daily": "substance",

    # --- criminal / legal ---------------------------------------------------
    "Criminal Record": "criminal_legal", "Drug Dealing": "criminal_legal",
    "Gang Member Currently": "criminal_legal", "On Parole Currently": "criminal_legal",
    "Sex Offender": "criminal_legal", "Having Sex For Money": "criminal_legal",

    # --- socioeconomic ------------------------------------------------------
    "Homeless": "socioeconomic", "Illiteracy": "socioeconomic",
    "Less Than A High School Education": "socioeconomic",
    "Living In A Trailer Park": "socioeconomic", "Living In Public Housing": "socioeconomic",
    "Unemployed": "socioeconomic", "Working Class Or Poor": "socioeconomic",
    "Working In A Manual Industry": "socioeconomic",
    "Working In A Service Industry": "socioeconomic",

    # --- physical illness ---------------------------------------------------
    "Bacterial STD": "physical_illness", "Genital Herpes": "physical_illness",
    "HIV Avg. Symptoms": "physical_illness", "Diabetes Type 2": "physical_illness",
    "Breast Cancer Current Avg. Symptoms": "physical_illness",
    "Breast Cancer Remitted": "physical_illness",
    "Colorectal Cancer Current Avg. Symptoms": "physical_illness",
    "Colorectal Cancer Remitted": "physical_illness",
    "Lung Cancer Current Avg. Symptoms": "physical_illness",
    "Lung Cancer Remitted": "physical_illness",
    "Prostate Cancer Current Avg. Symptoms": "physical_illness",
    "Prostate Cancer Remitted": "physical_illness",
    "Heart Attack Recent Avg. Impairment": "physical_illness",
    "Stroke Recent Avg. Impairment": "physical_illness",
    "Psoriasis Current Avg. Severity": "physical_illness",
    "Psoriasis Remitted Avg. Severity": "physical_illness",
    "Cleft Lip And Palate Current": "physical_illness",
    "Fecal Incontinence": "physical_illness", "Urinary Incontinence": "physical_illness",

    # --- body weight --------------------------------------------------------
    "Fat Current Avg. Severity": "body_weight", "Fat Remitted Avg. Severity": "body_weight",
    "Fat/Overweight/Obese Current Avg. Severity": "body_weight",
    "Fat/Overweight/Obese Remitted Avg. Severity": "body_weight",
    "Obese Current Avg. Severity": "body_weight", "Obese Remitted Avg. Severity": "body_weight",
    "Overweight Current Avg. Severity": "body_weight",
    "Overweight Remitted Avg. Severity": "body_weight",

    # --- appearance ---------------------------------------------------------
    "Chest Scars": "appearance", "Facial Scars": "appearance", "Limb Scars": "appearance",
    "Multiple Body Piercings": "appearance", "Multiple Facial Piercings": "appearance",
    "Multiple Tattoos": "appearance", "Short": "appearance", "Unattractive": "appearance",

    # --- reproductive / family ----------------------------------------------
    "Had An Abortion Previously": "reproductive_family", "Infertile": "reproductive_family",
    "Teen Parent Currently": "reproductive_family", "Teen Parent Previously": "reproductive_family",
    "Voluntarily Childless": "reproductive_family", "Divorced Previously": "reproductive_family",

    # --- age ----------------------------------------------------------------
    "Old Age": "age",

    # --- victimisation ------------------------------------------------------
    "Was Raped Previously": "victimisation",
}

# Distinct underlying racial categories -- the resampling unit for race tests.
RACE_CLUSTER = {
    "African American": "black", "Black": "black", "Black/African American": "black",
    "Asian": "asian_east", "Asian American": "asian_east",
    "South Asian": "asian_south",
    "Latina": "latino", "Latino": "latino", "Latina/Latino": "latino",
    "Middle Eastern": "middle_eastern",
    "Native American": "native_american",
    "Multiracial": "multiracial",
}

# Attributes a person is treated as born into, versus attributed to conduct or
# circumstance.  Lets us ask whether absorption is about race or about ascribed
# protected attributes generally.
ASCRIBED = {"race", "gender", "age", "disability_sensory"}
ACQUIRED = {"criminal_legal", "substance", "socioeconomic", "body_weight"}


def category(identity: str) -> str:
    return CATEGORY.get(identity, "UNMAPPED")


def race_cluster(identity: str):
    return RACE_CLUSTER.get(identity)


def check_coverage(identities) -> list:
    """Return any identity with no explicit assignment."""
    return [t for t in identities if t not in CATEGORY]


# ---------------------------------------------------------------------------
# US federal protected-class status.
#
# NOT LEGAL ADVICE and not a settled binary -- encoded as three tiers because
# the honest answer for several categories is "it depends".  Statutes differ in
# what they cover, and coverage is DOMAIN-DEPENDENT, which matters here because
# the benchmark spans employment, housing, healthcare and child-welfare
# scenarios:
#
#   Title VII (1964)  employment: race, colour, religion, sex, national origin.
#                     Bostock v. Clayton County (2020) reads "sex" to include
#                     sexual orientation and gender identity.
#   ADEA (1967)       employment: age 40+.  No housing analogue.
#   ADA (1990)        disability, broadly, including many chronic illnesses and
#                     HIV (Bragdon v. Abbott).  EXCLUDES current illegal drug
#                     use but covers people in recovery -- a line the benchmark
#                     happens to operationalise via Current vs Remitted labels.
#   Fair Housing Act  housing: race, colour, religion, sex, national origin,
#                     disability, familial status.
#   ECOA              credit: adds marital status and public-assistance income.
#
# Deliberately NOT protected federally: physical appearance, body weight (a few
# state/local exceptions such as Michigan and NYC), criminal record (EEOC
# disparate-impact guidance exists, but it is not a protected class), and
# socioeconomic status.
#
# Verify against a legal source before publication; this is a research
# approximation, not counsel.
# ---------------------------------------------------------------------------

PROTECTED = {
    "race":                "protected",    # Title VII, FHA, ECOA
    "religion":            "protected",    # Title VII, FHA
    "gender":              "protected",    # sex, per Bostock
    "sexuality":           "protected",    # sex, per Bostock (employment)
    "age":                 "protected",    # ADEA, 40+
    "disability_sensory":  "protected",    # ADA
    "disability_mobility": "protected",    # ADA

    # coverage is real but conditional on the specific condition or context
    "physical_illness":    "conditional",  # ADA where impairment is substantial
    "mental_health":       "conditional",  # ADA, condition-dependent
    "substance":           "conditional",  # ADA covers recovery, not current use
    "reproductive_family": "conditional",  # PDA / FHA familial status
    "immigration":         "conditional",  # national origin yes; undocumented status no

    "body_weight":         "unprotected",  # no federal class
    "appearance":          "unprotected",
    "socioeconomic":       "unprotected",
    "criminal_legal":      "unprotected",
    "victimisation":       "unprotected",
}


def protection(identity: str) -> str:
    return PROTECTED.get(CATEGORY.get(identity, ""), "UNMAPPED")
