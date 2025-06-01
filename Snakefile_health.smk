import yaml
import calendar

with open("params.yaml") as f:
    params = yaml.safe_load(f)

years = params["years"]
icd_codes = params["icd_codes"]

def generate_dates(year):
    return [
        f"{year}{month:02d}{day:02d}"
        for month in range(1, 13)
        for day in range(1, calendar.monthrange(year, month)[1] + 1)
    ]

rule all:
    input:
        expand("data/health/{icd}/{icd}__{date}.parquet",
               icd=icd_codes,
               year=years,
               date=lambda wildcards: generate_dates(wildcards.year))

rule preprocess_health:
    output:
        lambda wildcards: [
            f"data/health/{wildcards.icd}/{wildcards.icd}__{date}.parquet"
            for date in generate_dates(wildcards.year)
        ]
    params:
        icd="{icd}",
        year="{year}"
    shell:
        """
        python src/preprocessing_health.py \
            hydra.run.dir=. \
            var={params.icd} \
            year={params.year}
        """
