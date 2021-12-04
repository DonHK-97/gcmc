import sys
from comet_ml import Experiment
import torch
import yaml

from dataset import MCDataset
from model import GAE
from trainer import Trainer
from utils import calc_rmse, random_init, init_kaiming, Config


def main(cfg, save=False, comet=False):
    cfg = Config(cfg)

    # comet-ml setting
    if comet:
        experiment = Experiment(
            api_key=cfg.api_key,
            project_name=cfg.project_name,
            workspace=cfg.workspace
        )
        experiment.log_parameters(cfg)
    else:
        experiment = None

    # device and dataset setting
    device = (torch.device(f'cuda:{cfg.gpu_id}')
        if torch.cuda.is_available() and cfg.gpu_id >= 0
        else torch.device('cpu'))
    dataset = MCDataset(cfg.root, cfg.dataset_name)
    data = dataset[0].to(device)

    # add some params to config
    cfg.num_nodes = dataset.num_nodes
    cfg.num_relations = dataset.num_relations
    cfg.num_users = int(data.num_users)

    # set and init model
    model = GAE(cfg, random_init).to(device)
    model.apply(init_kaiming)

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = ReduceLROnPlateau(optimizer, mode = 'min', patience= cfg.patience,
     cool_down = cfg.cool_down, min_lr= cfg.min_lr, verbose= 1
     )


    # train
    trainer = Trainer(
        model, dataset, data, calc_rmse, optimizer, scheduler, experiment,
    )
    trainer.training(cfg.epochs)


if __name__ == '__main__':
    with open('config.yml') as f:
        cfg = yaml.safe_load(f)

    if len(sys.argv) < 2:
        main(cfg)
    else:
        main(cfg, save=sys.argv[1])
    # main(cfg, comet=True)