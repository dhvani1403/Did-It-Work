"""The Italian ChatGPT ban: dates, units and donor candidates.

On 30 March 2023 Italy's data-protection authority (the Garante per la
protezione dei dati personali) ordered OpenAI to stop processing Italian users'
data. OpenAI blocked ChatGPT in Italy on 31 March 2023 and restored access on
28 April 2023, after making changes the Garante had asked for.

Treated unit: the ChatGPT article on Italian Wikipedia.
Donors: the same article in other language editions. Each language edition is
read mostly by people in other countries, where ChatGPT stayed available.

English Wikipedia is left out of the donor pool on purpose: it is read all over
the world, including by Italians, so it could be directly affected by the ban.
Simple English is left out for the same reason.
"""

import pandas as pd

BAN_START = pd.Timestamp("2023-03-31")   # first day without access
LIFT = pd.Timestamp("2023-04-28")        # first day with access again
PRE_START = pd.Timestamp("2023-01-01")   # every candidate article exists by then
POST_END = LIFT + pd.Timedelta(days=27)  # 28 days after the lift (25 May 2023)

START, END = "20221201", "20230731"      # download window

TREATED_LANG = "it"

# Candidate donors: language editions whose ChatGPT article had at least a few
# hundred views a day in early 2023 (checked on monthly totals). The data screen
# in italy_analysis.py then removes any that fail the quality rules.
DONOR_LANGS = ["es", "de", "fr", "pt", "nl", "pl", "ru", "uk", "ja", "zh", "id", "he", "vi",
               "tr", "fa", "sv", "cs", "hu", "el", "ro", "fi", "da", "ca", "sk", "ko", "ar"]

# Titles in each language come from the English articles' interlanguage links,
# looked up at download time, so I do not have to guess translations.
OUTCOMES = {"chatgpt": "ChatGPT", "vpn": "Virtual private network"}
