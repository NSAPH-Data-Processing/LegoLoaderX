# snakemake file to process covariate data into per-(var, year) dense mmaps (.npy)
# see snakefile_health.smk for health data processing

import yaml

# Load config
configfile: "conf/snakemake.yaml"
min_year = config["min_year"]
max_year = config["max_year"]

wildcard_constraints:
    year=r"\d{4}"

# Build the per-(var, year) .npy output list, clamping each var_group's own year range
output_file_lst = []
var_map = {}
for vg in config["var_groups"]:
    with open(f"conf/var_group/{vg}.yaml", "r") as f:
        vg_cfg = yaml.safe_load(f)
    var_map[vg] = {
        "vars": vg_cfg["vars"],
        "temporal_res": vg_cfg["min_temporal_res"],
        "spatial_res": vg_cfg["min_spatial_res"],
    }
    y0 = max(min_year, vg_cfg["min_year"])
    y1 = min(max_year, vg_cfg["max_year"])
    output_file_lst += expand(
        "data/covars/{var_group}/{var}/{var}__{year}.npy",
        var_group=vg,
        var=vg_cfg["vars"],
        year=list(range(y0, y1 + 1)),
    )

rule all:
    input:
        "data/covars/idx2zcta.parquet",
        output_file_lst

# Build the canonical row-index -> zcta order (data/covars/idx2zcta.parquet) once
rule idx2zcta:
    output:
        "data/covars/idx2zcta.parquet"
    shell:
        "python src/preprocessing_idx2zcta.py"

# One dense .npy per (var_group, var, year), in idx2zcta order.
rule preprocess:
    input:
        "data/covars/idx2zcta.parquet"
    output:
        "data/covars/{var_group}/{var}/{var}__{year}.npy"
    run:
        spatial_res = var_map[wildcards.var_group]["spatial_res"]
        temporal_res = var_map[wildcards.var_group]["temporal_res"]
        shell(f"python src/preprocessing.py var_group={wildcards.var_group} "
              f"var={wildcards.var} "
              f"spatial_res={spatial_res} "
              f"temporal_res={temporal_res} "
              f"year={wildcards.year}")
