
import hydra
import yaml
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from x_dataloader import XDataset
from utils import get_unique_ids, compute_summary

@hydra.main(config_path="../conf/dataloader", config_name="config", version_base=None)
def main(cfg: DictConfig):
    var_dict = {}
    data_out_dir = cfg.data_dir

    # iterate through variable groups and collect names of all variables
    for vg in cfg.var_groups:
        var_dict[vg] = {}
        with open(f"conf/var_group/{vg}.yaml", "r") as f:
            vg_cfg = yaml.safe_load(f)
            # add variable names
            if vg_cfg["valid_normalize"]:
                var_dict[vg]["vars"] = vg_cfg["vars"]
                # store spatial and temporal res
                var_dict[vg]["temporal_res"] = vg_cfg["min_temporal_res"]
                var_dict[vg]["spatial_res"] = vg_cfg["min_spatial_res"]
            f.close()

    # load vars from global config
    with open(f"conf/conf.yaml", "r") as f:
        cfg_all = yaml.safe_load(f)
        data_in_dir = cfg_all["input_dir"]
        zcta_uniq_dir = f"{data_in_dir}/{cfg_all['uniqid_dir']}/{cfg_all['uniqid_nm']}/zcta_yearly/"

    unique_zctas, _ = get_unique_ids(zcta_uniq_dir, cfg.min_year, cfg.max_year)

    # initialize dataset with raw vars (no normalization)
    dataset = XDataset(
        root_dir=data_out_dir,
        transform=None,
        var_dict=var_dict,
        nodes=unique_zctas,
        normalize = False,
        window=1,
        min_year = cfg.min_year, 
        max_year = cfg.max_year
    )

    # adapt to dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )

    # compute summary statistics and write to output json file
    compute_summary(dataloader, 
                    output_dir=f"{data_out_dir}/{cfg.summary_stats_dir}",
                    output_nm=cfg.sumnmary_stats_nm)


if __name__ == "__main__":
    main()
