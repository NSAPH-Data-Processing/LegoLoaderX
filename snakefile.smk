# snakemake file to process covariate data
# see snakemake_health.smk for health data processing

# Load Hydra configs using yaml
import yaml
import pandas as pd
import math

# Load config
configfile: "conf/snakemake.yaml"
min_year = config["min_year"]
max_year = config["max_year"]
fmt = config["format"]
assert fmt in ("daily_parquet", "yearly_mmap_dense"), f"unsupported format {fmt!r} for covariates"
ext = "parquet" if fmt == "daily_parquet" else "npy"

output_file_lst = []
var_map = {}
for vg in config["var_groups"]:
    var_map[vg] = {}
    with open(f"conf/var_group/{vg}.yaml", "r") as f:
        vg_cfg = yaml.safe_load(f)
        var_map[vg]["vars"] = vg_cfg["vars"]
        var_map[vg]["temporal_res"] = vg_cfg["min_temporal_res"]
        var_map[vg]["spatial_res"] = vg_cfg["min_spatial_res"]

        # getting start/stop
        if min_year < vg_cfg["min_year"]:
            min_year_curr = vg_cfg["min_year"]
        else:
            min_year_curr = min_year

        if max_year > vg_cfg["max_year"]:
            max_year_curr = vg_cfg["max_year"]
        else:
            max_year_curr = max_year

        date_range = pd.date_range(f"{min_year_curr}-01-01", f"{max_year_curr}-12-31", freq="D")

        if fmt == "yearly_mmap_dense":
            # one per-(var, year) cell regardless of temporal_res
            var_map[vg]["date_range"] = list(range(min_year_curr, max_year_curr + 1))
            output_file_lst += expand(
                f"data/output/{{var_group}}/{{var}}/{fmt}/{{var}}__{{year}}.{ext}",
                var_group=vg,
                var=[v for v in vg_cfg["vars"]],
                year=var_map[vg]["date_range"],
            )
        elif var_map[vg]["temporal_res"] == "daily":
            var_map[vg]["date_range"] = [(d.year, d.month, d.day) for d in date_range]
            if config["max_days"]:
                var_map[vg]["date_range"] = var_map[vg]["date_range"][:int(config["max_days"])]
            # iterate through valid y/m/d combos, expand vars within each
            for y,m,d in var_map[vg]["date_range"]:
                output_file_lst += expand(
                    f"data/output/{{var_group}}/{{var}}/{fmt}/{{var}}__{{year}}{{month:02d}}{{day:02d}}.{ext}",
                    var_group=vg,
                    var=[v for v in vg_cfg["vars"]],
                    year=y,
                    month=m,
                    day=d,
                )
        elif var_map[vg]["temporal_res"] == "monthly":
            var_map[vg]["date_range"] = [(y, m) for y in range(min_year_curr, max_year_curr + 1) for m in range(1, 13)]
            output_file_lst += expand(
                f"data/output/{{var_group}}/{{var}}/{fmt}/{{var}}__{{year}}{{month:02d}}.{ext}",
                var_group=vg,
                var=[v for v in vg_cfg["vars"]],
                year=[y for y, m in var_map[vg]["date_range"]],
                month=[m for y, m in var_map[vg]["date_range"]],
            )
        else:
            var_map[vg]["date_range"] = [y for y in range(min_year_curr, max_year_curr + 1)]
            output_file_lst += expand(
                f"data/output/{{var_group}}/{{var}}/{fmt}/{{var}}__{{year}}.{ext}",
                var_group=vg,
                var=[v for v in vg_cfg["vars"]],
                year=[y for y in var_map[vg]["date_range"]],
            )
        f.close()

# yearly_mmap_dense needs idx2zcta.txt at the output root.
idx2zcta_dep = []
if fmt == "yearly_mmap_dense":
    output_file_lst.append("data/output/idx2zcta.txt")
    # Build idx2zcta.txt ONCE before the parallel per-(var,year) jobs read it,
    # else the workers race to create it and some read a half-written file.
    idx2zcta_dep = ["data/output/idx2zcta.txt"]

# Expand over all valid combinations of variable groups, variables, and dates
rule all:
    input:
        output_file_lst

rule idx2zcta:
    output:
        "data/output/idx2zcta.txt"
    shell:
        "python src/preprocessing.py target=idx2zcta "
        f"format={fmt} hydra.run.dir=."

# have excluded input entry for now
rule preprocess:
    input:
        idx2zcta_dep
    output:
        f"data/output/{{var_group}}/{{var}}/{fmt}/{{var}}__{{timestring}}.{ext}"
    params:
        script="src/preprocessing.py"
    run:
        spatial_res = var_map[wildcards.var_group]["spatial_res"]
        temporal_res = var_map[wildcards.var_group]["temporal_res"]
        timestring = str(wildcards.timestring)
        shell(f"python {params.script} var_group={wildcards.var_group} "
              f"var={wildcards.var} "
              f"spatial_res={spatial_res} "
              f"temporal_res={temporal_res} "
              f"timestr={timestring} "
              f"format={fmt}")
