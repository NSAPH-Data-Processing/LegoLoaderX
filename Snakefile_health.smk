import calendar

# Load config
configfile: "conf/health/snakemake.yaml"

# Get config values
years = config["years"]
icd_codes = config["icd_codes"]

# Rule: final output is one sentinel file per ICD/year (Dec 31)
rule all:
    input:
        expand(
            "data/health/{icd}/{icd}__{year}1231.parquet",
            icd=icd_codes,
            year=years
        )

# Rule: preprocess all data for given ICD and year
rule preprocess_health:
    output:
        "data/health/{icd}/{icd}__{year}1231.parquet"
    shell:
        """
        python src/preprocessing_health.py \
            hydra.run.dir=. \
            var={wildcards.icd} \
            year={wildcards.year}
        """

# # Helper to generate all date strings for a year
# def generate_dates(year):
#     return [
#         f"{year}{month:02d}{day:02d}"
#         for month in range(1, 13)
#         for day in range(1, calendar.monthrange(year, month)[1] + 1)
#     ]

# # ✅ Pre-generate all outputs
# output_map = {
#     (icd, year): [
#         f"data/health/{icd}/{icd}__{date}.parquet"
#         for date in generate_dates(year)
#     ]
#     for icd in icd_codes
#     for year in years
# }

# # ✅ Gather all expected outputs into `rule all`
# rule all:
#     input:
#         [f for outputs in output_map.values() for f in outputs]

# # ✅ Single rule to process all daily files per (icd, year)
# rule preprocess_health:
#     output:
#         output_map[(wildcards.icd, int(wildcards.year))]
#     params:
#         icd="{icd}",
#         year="{year}"
#     shell:
#         """
#         python src/preprocessing_health.py \
#             hydra.run.dir=. \
#             var={params.icd} \
#             year={params.year}
#         """
