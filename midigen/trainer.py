import os
import yaml
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np

from midigen import models
from midigen.models.gpt2.utils import GPTConfig
from midigen.data.dataset import EPianoDataset
from midigen.metrics.epiano import compute_epiano_accuracy
from midigen.optim.stepper import LrStepTracker
from midigen.utils.constants import SCHEDULER_WARMUP_STEPS, TOKEN_PAD, VOCAB_SIZE,\
    ADAM_EPSILON, ADAM_BETA_1, ADAM_BETA_2

class Trainer(yaml.YAMLObject):
    yaml_tag = '!Experiment'

    @classmethod
    def from_yaml(cls, loader, node):
        data = loader.construct_mapping(node, deep=True)
        return cls(**data)

    @classmethod
    def to_yaml(cls, dumper, obj):
        return dumper.represent_mapping("!Experiment",
                                        obj.get_parameters())

    def __init__(self, seed, save_folder, model_config, data_split_file,
                 batch_size, num_workers, tensorboard_logging, device, max_epochs,
                 max_seq, random_seq, num_files, optimizer, val_every_n_batches):
        super().__init__()
        # Initialize general / utility parameters
        self.seed = seed
        self.save_folder = save_folder
        self.num_workers = num_workers
        self.device = device
        self.tensorboard_logging = tensorboard_logging
        self.val_every_n_batches = val_every_n_batches

        self.loss = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD)
        self.max_epochs = max_epochs
        self.batch_size = batch_size

        # Initialize datasets / dataloaders
        self.max_seq = max_seq
        self.random_seq = random_seq
        self.num_files = num_files
        self.data_split_file = data_split_file
        with open(data_split_file, 'rb') as f:
            data = pickle.load(f)

        self.train_iterator = EPianoDataset(data['train'], max_seq=self.max_seq, random_seq=self.random_seq,
                                            num_files=self.num_files, type='training')
        self.train_loader = DataLoader(dataset=self.train_iterator, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.num_workers)

        self.val_iterator = EPianoDataset(data['val'], max_seq=self.max_seq, random_seq=self.random_seq,
                                            num_files=self.num_files, type='validation')
        self.val_loader = DataLoader(dataset=self.val_iterator, batch_size=self.batch_size,
                                     num_workers=self.num_workers)

        self.test_iterator = EPianoDataset(data['test'], max_seq=self.max_seq, random_seq=self.random_seq,
                                           num_files=self.num_files, type='test')
        self.test_loader = DataLoader(dataset=self.test_iterator, batch_size=self.batch_size,
                                      num_workers=self.num_workers)

        self.model_config = model_config
        self._initialize_model()
        self.optimizer = optimizer
        self.optimizer_obj = getattr(torch.optim, self.optimizer["class"])(self.model.parameters(),
                                                                           **self.optimizer["parameters"],
                                                                           betas=(ADAM_BETA_1, ADAM_BETA_2),
                                                                           eps=ADAM_EPSILON)
        # TODO: replace with model.n_embd
        lr_stepper = LrStepTracker(512, SCHEDULER_WARMUP_STEPS, init_steps=0)
        self.scheduler = LambdaLR(self.optimizer_obj, lr_stepper.step)
        self._fix_seeds()
        self._prepare_dirs()
        if self.tensorboard_logging:
            self.writer = SummaryWriter(logdir=os.path.join(self.save_folder, "TB"), flush_secs=20)
        self.best_eval_acc = 0.0
        self.best_eval_acc_epoch = -1
        self.best_eval_loss = float("inf")
        self.best_eval_loss_epoch = -1
        self.epoch = 0
        self.train_batches_seen = 0

    def train(self):
        """This method is called from train.py script to start training."""
        try:
            self.train_on_batches()
        except KeyboardInterrupt:
            print("***| Terminated training manually |***")
        finally:
            self.save(save_policy='last')

    def train_on_batches(self):
        self.model.train()
        while True:
            with tqdm(total=len(self.train_loader)) as bar_train:
                for batch_num, batch in enumerate(self.train_loader):
                    log = {}
                    self.optimizer_obj.zero_grad()
                    x = batch[0].to(self.device)
                    tgt = batch[1].to(self.device)

                    model_out = self.model(x)
                    if len(model_out) > 1:
                        y, _ = model_out
                    else:
                        y = model_out
                    y = y.reshape(y.shape[0] * y.shape[1], -1)
                    tgt = tgt.flatten()
                    loss = self.loss.forward(y, tgt)
                    loss.backward()
                    self.optimizer_obj.step()
                    self.scheduler.step()

                    lr = self.optimizer_obj.param_groups[0]['lr']
                    log['lr'] = lr
                    log['loss'] = loss.item()
                    bar_train.set_description(f'Epoch: {self.epoch} Loss: {float(loss.item()):.4} LR: {float(lr):.8}')
                    bar_train.update(1)
                    self.train_batches_seen += 1
                    self.write_tensorboard(log, mode='train')
                    if batch_num != 0 and self.train_batches_seen % self.val_every_n_batches == 0:
                        self.validate()

            self.epoch += 1
            self.train_batches_seen = 0
            self.validate()
            if 0 < self.max_epochs <= self.epoch:
                break

    def validate(self):
        self.model.eval()
        log = {}
        with torch.set_grad_enabled(False):
            n_test = len(self.val_loader)
            sum_loss = 0.0
            sum_acc = 0.0
            with tqdm(total=len(self.val_loader)) as bar_eval:
                for batch in self.val_loader:
                    x = batch[0].to(self.device)
                    tgt = batch[1].to(self.device)

                    model_out = self.model(x)
                    if len(model_out) > 1:
                        y, _ = model_out
                    else:
                        y = model_out

                    sum_acc += float(compute_epiano_accuracy(y, tgt))

                    y = y.reshape(y.shape[0] * y.shape[1], -1)
                    tgt = tgt.flatten()

                    out = self.loss.forward(y, tgt)

                    sum_loss += float(out)
                    log['curr_acc'] = sum_acc / (bar_eval.n + 1)
                    log['curr_loss'] = sum_loss / (bar_eval.n + 1)
                    self.write_tensorboard(log, mode='val')
                    bar_eval.set_description(f'Loss val: {float(out):.4}  Acc: {float(sum_acc / (bar_eval.n + 1)):.4}')
                    bar_eval.update(1)

            log['avg_loss'] = sum_loss / n_test
            log['avg_acc'] = sum_acc / n_test
        self.write_tensorboard(log, mode='val')
        self.model.train()

    def write_tensorboard(self, recieved_data, mode='train'):
        for key in recieved_data.keys():
            self.writer.add_scalar(mode + '/' + key, recieved_data[key])

    def _initialize_model(self):
        if self.model_config['name'] == 'GPT':
            config = GPTConfig(VOCAB_SIZE, self.max_seq, dim_feedforward=self.model_config['dim_feedforward'],
                               n_layer=self.model_config['n_layer'], n_head=self.model_config['n_head'],
                               n_embd=self.model_config['n_embd'], enable_rpr=self.model_config['enable_rpr'],
                               er_len=self.model_config['er_len'])
            self.model = models.GPT(config).to(self.device)
        else:
            raise NotImplementedError('MusicTransformer init not implemented yet!')

    def _fix_seeds(self):
        """Fixates random seeds for experiment reproducibility."""
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def _prepare_dirs(self):
        """Creates save folder for the experiment."""
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def get_parameters(self):
        """Gets all parameters, needed for trainer initialization"""
        params = {
                    "device": str(self.device),
                    "save_folder": self.save_folder,
                    "data_split_file": self.data_split_file,
                    "num_workers": self.num_workers,
                    "seed": self.seed,
                    "batch_size": self.batch_size,
                    "model_config": self.model_config,
                    "optimizer": self.optimizer,
                    "loss": self.loss,
                    "max_epochs": self.max_epochs,
                    "tensorboard_logging": self.tensorboard_logging,
                    "max_seq": self.max_seq,
                    "random_seq": self.random_seq,
                    "num_files": self.num_files,
                    "val_every_n_batches": self.val_every_n_batches
        }
        return params

    def save(self, save_policy='best'):
        """Saves trainer, scores and model in experiment folder."""
        print(f"Saving trainer to {self.save_folder}.")
        if len(self.save_folder) > 0 and not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        if save_policy == 'best':
            self.model.save(os.path.join(self.save_folder, "model"))
        elif save_policy == 'last':
            self.model.save(os.path.join(self.save_folder, "model_last"))

        torch.save({
            "parameters": self.get_parameters()
        }, os.path.join(self.save_folder, "trainer"))
        print("Trainer is saved.")

    @classmethod
    def load(cls, load_folder, device="cpu", load_policy='last'):

        checkpoint = torch.load(os.path.join(load_folder, "trainer"), map_location=device)
        parameters = checkpoint["parameters"]
        parameters.pop("device", None)
        trainer = cls(device=device, **parameters)

        if load_policy == 'best':
            trainer.model = trainer.model.load(os.path.join(load_folder, "model"), device)
        elif load_policy == 'last':
            trainer.model = trainer.model.load(os.path.join(load_folder, "model_last"), device)
        return trainer
