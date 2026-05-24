# snakemake file to embed variable descriptions for downstream dataloader use.
# See snakefile.smk and snakefile_health.smk for covariate / health preprocessing.

import os

configfile: "conf/embeddings/snakemake.yaml"

output_dir = config["output_dir"]
var_group_dir = config["var_group_dir"]

# Invoke the python that has hydra / sentence-transformers / transformers / torch
# installed — by default, the legoloaderx conda env. Override with
# --config python_bin=/path/to/python if needed.
PYTHON = config.get("python_bin", "/n/holylabs/dominici_lab/Lab/sw/legoloaderx/bin/python")

model_to_library = {entry["model"]: entry["library"] for entry in config["models"]}
models = list(model_to_library.keys())

ARTIFACTS = ["config.json", "model.safetensors"]
PLOTS = ["latent_pca.png", "latent_tsne.png", "similarity_heatmap.png", "top_pairs.txt"]

rule all:
    input:
        expand(f"{output_dir}/{{model}}/{{artifact}}",
               model=models, artifact=ARTIFACTS),
        expand(f"{output_dir}/{{model}}/plots/{{plot}}",
               model=models, plot=PLOTS),
        f"{output_dir}/plots/model_comparison.png",

rule embed_descriptions:
    output:
        config = f"{output_dir}/{{model}}/config.json",
        matrix = f"{output_dir}/{{model}}/model.safetensors",
    params:
        library = lambda wc: model_to_library[wc.model],
        var_group_dir = var_group_dir,
        output_dir = output_dir,
        python = PYTHON,
    shell:
        """
        {params.python} src/preprocessing_embeddings.py \
            hydra.run.dir=. \
            model='{wildcards.model}' \
            library='{params.library}' \
            var_group_dir='{params.var_group_dir}' \
            output_dir='{params.output_dir}'
        """

rule plot_per_model:
    input:
        config = f"{output_dir}/{{model}}/config.json",
        matrix = f"{output_dir}/{{model}}/model.safetensors",
    output:
        pca = f"{output_dir}/{{model}}/plots/latent_pca.png",
        tsne = f"{output_dir}/{{model}}/plots/latent_tsne.png",
        heat = f"{output_dir}/{{model}}/plots/similarity_heatmap.png",
        pairs = f"{output_dir}/{{model}}/plots/top_pairs.txt",
    params:
        python = PYTHON,
        out_dir = lambda wc: f"{output_dir}/{wc.model}/plots",
        emb_dir = lambda wc: f"{output_dir}/{wc.model}",
    shell:
        """
        {params.python} src/plot_embeddings.py per-model \
            --embeddings-dir '{params.emb_dir}' \
            --out-dir '{params.out_dir}'
        """

rule plot_model_comparison:
    input:
        expand(f"{output_dir}/{{model}}/config.json", model=models),
        expand(f"{output_dir}/{{model}}/model.safetensors", model=models),
    output:
        f"{output_dir}/plots/model_comparison.png"
    params:
        python = PYTHON,
        roots = lambda wc: " ".join(f"{output_dir}/{m}" for m in models),
        out = f"{output_dir}/plots/model_comparison.png",
    shell:
        """
        mkdir -p {output_dir}/plots
        {params.python} src/plot_embeddings.py compare \
            --out '{params.out}' \
            --embeddings-dirs {params.roots}
        """
