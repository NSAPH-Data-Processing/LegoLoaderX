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

# Rule: final output is one dense .npy per (var, year) outcome + per-year denom
rule all:
    input:
        "data/health/idx2zcta.parquet",
        expand(
            f"data/health/ccw/{{var}}/{{var}}__{{year}}.npy",
            var=vars,
            year=years
        ),
        expand(
            f"data/health/denom/denom__{{year}}.npy",
            year=years
        )

# Build the canonical row-index -> zcta order into the health root. Same script,
# same source as covars (see snakefile.smk) so the order is identical everywhere —
# the index must match across covars/treatments/outcomes to gather aligned rows.
rule idx2zcta:
    output:
        "data/health/idx2zcta.parquet"
    shell:
        "python src/preprocessing_idx2zcta.py output_dir=data/health"

# Rule: build the (n_days, n_zctas) outcome mmap for given var and year
rule preprocess_health:
    input:
        "data/health/idx2zcta.parquet"
    output:
        f"data/health/ccw/{{var}}/{{var}}__{{year}}.npy"
    params:
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
    input:
        "data/health/idx2zcta.parquet"
    output:
        f"data/health/denom/denom__{{year}}.npy"
    params:
        lego_dir = lego_dir
    shell:
        """
        python src/preprocessing_denom.py \
            hydra.run.dir=. \
            year={wildcards.year} \
            lego_dir={params.lego_dir} \
        """
