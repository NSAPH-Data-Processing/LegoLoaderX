# snakemake file to process health data
# see snakemake.smk for covariate data processing

# Load config
configfile: "conf/health/snakemake.yaml"

# Get config values
years = config["years"]
vars = config["vars"]
fmt = config["format"]
assert fmt in ("daily_parquet", "yearly_mmap_dense", "yearly_mmap_sparse"), \
    f"unsupported format {fmt!r}"

# Get paths
if config["use_synthetic"]:
    lego_dir = config["synthetic_lego_dir"]
else:
    lego_dir = config["lego_dir"]


print(f"Using dir:\n  - lego_dir: {lego_dir}\n  - format: {fmt}\n")

if fmt == "daily_parquet":
    var_year_pattern = "data/health/ccw/{var}/daily_parquet/{var}__{year}1231.parquet"
    denom_pattern = "data/health/denom/daily_parquet/denom__{year}.parquet"
elif fmt == "yearly_mmap_dense":
    var_year_pattern = "data/health/ccw/{var}/yearly_mmap_dense/{var}__{year}.npy"
    denom_pattern = "data/health/denom/yearly_mmap_dense/denom__{year}.npy"
else:  # yearly_mmap_sparse
    var_year_pattern = "data/health/ccw/{var}/yearly_mmap_sparse/{var}__{year}.npz"
    denom_pattern = "data/health/denom/yearly_mmap_dense/denom__{year}.npy"  # denom always dense

extra_inputs = []
if fmt != "daily_parquet":
    extra_inputs.append("data/health/idx2zcta.txt")

# mmap formats read idx2zcta.txt at run time; make it an explicit prerequisite so
# Snakemake builds it ONCE before the parallel per-(var,year) jobs (otherwise the
# workers race to create it and some read a half-written file -> wrong n_zctas).
idx2zcta_dep = ["data/health/idx2zcta.txt"] if fmt != "daily_parquet" else []

# Rule: final output is one sentinel file per var/year
rule all:
    input:
        expand(var_year_pattern, var=vars, year=years),
        expand(denom_pattern, year=years),
        extra_inputs

rule idx2zcta:
    output:
        "data/health/idx2zcta.txt"
    shell:
        "python src/preprocessing_health.py target=idx2zcta "
        f"lego_dir={lego_dir} format={fmt} hydra.run.dir=."

# Rule: preprocess all data for given var and year
if fmt == "daily_parquet":
    rule preprocess_health:
        output:
            "data/health/ccw/{var}/daily_parquet/{var}__{year}1231.parquet"
        params:
            lego_dir = lego_dir,
        shell:
            """
            python src/preprocessing_health.py \
                hydra.run.dir=. \
                var={wildcards.var} \
                year={wildcards.year} \
                lego_dir={params.lego_dir} \
                format=daily_parquet
            """
else:
    rule preprocess_health:
        input:
            idx2zcta_dep
        output:
            var_year_pattern
        params:
            lego_dir = lego_dir,
        shell:
            f"""
            python src/preprocessing_health.py \
                hydra.run.dir=. \
                var={{wildcards.var}} \
                year={{wildcards.year}} \
                lego_dir={{params.lego_dir}} \
                format={fmt}
            """

rule preprocess_denom:
    input:
        idx2zcta_dep
    output:
        denom_pattern
    params:
        lego_dir = lego_dir,
    shell:
        f"""
        python src/preprocessing_denom.py \
            hydra.run.dir=. \
            year={{wildcards.year}} \
            lego_dir={{params.lego_dir}} \
            format={fmt}
        """
