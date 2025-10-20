# snakemake file to process health data
# see snakemake.smk for covariate data processing

# Load config
configfile: "conf/health/snakemake.yaml"

# Get config values
years = config["years"]
vars = config["vars"]

# Get paths
if config["use_synthetic"]:
    lego_dir = config["synthetic_lego_dir"]
else:
    lego_dir = config["lego_dir"]


print(f"Using dir:\n  - lego_dir: {lego_dir}\n")

# Rule: final output is one sentinel file per ICD/year (Dec 31)
rule all:
    input:
        expand(
            f"data/health/ccw/{{var}}/{{var}}__{{year}}1231.parquet",
            var=vars,
            year=years
        ),
        expand(
            f"data/health/denom/denom__{{year}}.parquet",
            year=years
        )

# Rule: preprocess all data for given var and year
rule preprocess_health:
    output:
        f"data/health/ccw/{{var}}/{{var}}__{{year}}1231.parquet"
    params:
        #horizons = config["horizons"],
        lego_dir = lego_dir,
    shell:
        """
        python src/preprocessing_health.py \
            hydra.run.dir=. \
            var={wildcards.var} \
            year={wildcards.year} \
            lego_dir={params.lego_dir} \
        """

rule preprocess_denom:
    output:
        f"data/health/denom/denom__{{year}}.parquet"
    params:
        lego_dir = lego_dir
    shell:
        """
        python src/preprocessing_denom.py \
            hydra.run.dir=. \
            year={wildcards.year} \
            lego_dir={params.lego_dir} \
        """
