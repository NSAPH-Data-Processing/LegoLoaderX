# import calendar

# Load config
configfile: "conf/health/snakemake.yaml"

# Get config values
years = config["years"]
vars = config["vars"]

# Rule: final output is one sentinel file per ICD/year (Dec 31)
rule all:
    input:
        expand(
            "data/health/{var}/{var}__{year}1231.parquet",
            var=vars,
            year=years
        ),
        expand(
            "data/health/denom/denom__{year}.parquet",
            year=years
        )

# Rule: preprocess all data for given var and year
rule preprocess_health:
    output:
        "data/health/{var}/{var}__{year}1231.parquet"
    shell:
        """
        python src/preprocessing_health.py \
            hydra.run.dir=. \
            var={wildcards.var} \
            year={wildcards.year}
        """

rule preprocess_denom:
    output:
        "data/health/denom/denom__{year}.parquet"
    params:
        script="src/preprocessing_denom.py"
    run:
        shell(f"python {params.script} year={wildcards.year}")


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
