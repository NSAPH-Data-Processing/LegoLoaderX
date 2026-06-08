# Snakemake file for synthetic health data generation.
# Generates synthetic medicare *raw inputs* under data/input/lego/medicare_synthetic,
# which snakefile_health.smk then processes (use_synthetic: true) into data/health.

# Load config
configfile: "conf/synthetic/snakemake.yaml"

# Get config values
years = config["years"]
vars = list(config["disease_params"].keys())

def get_disease_params(var):
    """Get disease-specific parameters for synthetic data generation"""
    params = config["disease_params"][var]
    override = " ".join(
        [f"synthetic.poisson_params.{key}={value}" for key, value in params.items()]
    )
    override += f" synthetic.var_name={var}"
    return override

# Rule: final output is input files for snakefile_health.smk
rule all:
    input:
        expand(
            f"data/input/{config['counts_lego_path']}/sparse_counts_{{var}}_{{year}}.parquet",
            var=vars,
            year=years
        ),
        expand(
            f"data/input/{config['denom_lego_path']}/counts_{{year}}.parquet",
            year=years
        )


rule generate_synthetic_counts:
    output:
        f"data/input/{config['counts_lego_path']}/sparse_counts_{{var}}_{{year}}.parquet"
    params:
        disease_params = lambda wildcards: get_disease_params(wildcards.var),
    shell:
        """
        python src/synthetic_health.py \
            hydra.run.dir=. \
            year={wildcards.year} \
            {params.disease_params}
        """

rule generate_synthetic_denom:
    output:
        f"data/input/{config['denom_lego_path']}/counts_{{year}}.parquet"
    shell:
        """
        python src/synthetic_denom.py \
            hydra.run.dir=. \
            year={wildcards.year}
        """
