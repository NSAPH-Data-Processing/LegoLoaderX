# Snakemake file for synthetic health data generation

# Load config
configfile: "conf/synthetic/snakemake.yaml"

# Get config values
years = config["years"]
vars = config["vars"]

def get_disease_params(var):
    """Get disease-specific parameters for synthetic data generation"""
    params = config["default_params"].copy()
    
    # Apply disease-specific variations if they exist
    if var in config["disease_params"]:
        params.update(config["disease_params"][var])
    
    # Add disease-specific random seed for reproducibility but variation
    params["random_seed"] = hash(var) % 1000 + 42
    
    return params

# Rule: final output is one sentinel file per var/year (Dec 31)
rule all:
    input:
        expand(
            "data/health/synthetic_health/{var}/{var}__{year}1231.parquet",
            var=vars,
            year=years
        )

# Rule: preprocess synthetic health data for given var and year
rule preprocess_synthetic_health:
    output:
        "data/health/synthetic_health/{var}/{var}__{year}1231.parquet"
    params:
        disease_params = lambda wildcards: get_disease_params(wildcards.var)
    shell:
        """
        python src/preprocessing_synth_health.py \
            hydra.run.dir=. \
            var={wildcards.var} \
            year={wildcards.year} \
            debug_days=null \
            synthetic.poisson_params.base_rate={params.disease_params[base_rate]} \
            synthetic.poisson_params.seasonal_amplitude={params.disease_params[seasonal_amplitude]} \
            synthetic.poisson_params.spatial_variance={params.disease_params[spatial_variance]} \
            synthetic.poisson_params.latitude_effect={params.disease_params[latitude_effect]} \
            synthetic.poisson_params.longitude_effect={params.disease_params[longitude_effect]} \
            synthetic.poisson_params.population_effect={params.disease_params[population_effect]} \
            synthetic.poisson_params.random_seed={params.disease_params[random_seed]}
        """