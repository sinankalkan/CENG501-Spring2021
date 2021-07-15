import logging
import datetime
from typing import Optional
import statistics
import torch
from torch import nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from torch.utils.data import DataLoader, ConcatDataset, random_split
import torch.nn.functional as F
import torchvision.transforms as T
from torchdiffeq import odeint_adjoint as odeint
import foolbox
torch.backends.cudnn.deterministic = True

# Turn off PyTorch info logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

logging.basicConfig(format='%(message)s')
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class AddGaussianNoise():
  """ Transform to add Gaussian noise to a training or test sample. """
  def __init__(self, mean=0., std=1., min=0, max=1.):
    """ Set min and/or max to None to not apply clamping of output values. """
    self.std = std
    self.mean = mean
    self.min = min
    self.max = max
      
  def __call__(self, tensor):
    y = tensor + torch.randn_like(tensor) * self.std + self.mean
    if self.min is not None and self.max is not None:
      y = torch.clamp(y, self.min, self.max)
    elif self.min is not None:
      y = torch.clamp(y, min=self.min)
    elif self.max is not None:
      y = torch.clamp(y, max=self.max)
    return y
  
  def __repr__(self):
    return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def train_and_save(model_class, dataset, seeds, **hp):
  """
  Trains models of the given class on the given dataset. Each model
  will be initialized with weights after setting the different seed values.
  Hyperparameters hp will be passed to the model class. Hyperparameters
  of training are also passed in hp. Currently, this is only max_epochs.
  """
  systems = []
  trainers = []
  model_name = model_class.__name__

  for seed in seeds:
    log.info('=' * 80)
    log.info(f'Training {dataset.name} {model_name} with seed={seed} ' \
             f'and {"augmented" if dataset.augment_training else "vanilla"} ' \
             f'training dataset.')

    pl.seed_everything(seed, workers=True)
    system = model_class(**hp)
    log.info('model hparams = \n' + str(system.hparams))

    # Monkey patch some extra data on to the model object
    system.training_dataset = dataset
    system.seed = seed

    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    log_dir = f'{dataset.name}/runs/{model_name}'
    log_dir += f'{"_aug" if dataset.augment_training else ""}'
    log_dir += f'/{seed}/{current_time}'
    log_dir += f" weight_decay={hp['weight_decay']}" \
               f" learning_rate={hp['learning_rate']}" \
               f" optimizer={hp['optimizer']}" \
               f" epochs={hp['epochs']}"
    if hp['optimizer'] == 'SGD' and 'momentum' in hp.keys():
      log_dir += f" momentum={hp['momentum']}"
    log_dir += f" augmented_training={dataset.augment_training}"

    model_save = f'{dataset.name}/models/{model_name}' \
                 f'{"_aug" if dataset.augment_training else ""}' \
                 f'_seed_{seed}'

    checkpoint_callback = ModelCheckpoint(
      monitor='val_loss',
      dirpath=model_save,
      filename=f'{current_time}_{{epoch:02d}}_{{val_loss:.3f}}',
      save_top_k=3,
      mode='min',
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    early_stopping = EarlyStopping('val_loss', min_delta=system.hparams.early_stopping_min_delta,
      patience=system.hparams.early_stopping_patience, verbose=True)

    trainer = pl.Trainer(gpus=-1 if device == 'cuda' else 0,
                         max_epochs=hp['epochs'],
                         check_val_every_n_epoch=1,
                         callbacks=[checkpoint_callback,
                                    lr_monitor,
                                    early_stopping],
                         default_root_dir=log_dir)
    trainer.fit(system, dataset)

    # Load best model
    system = system.load_from_checkpoint(checkpoint_path=checkpoint_callback.best_model_path, verbose=True)

    # Test on the vanilla test data from the datamodule
    test_results = trainer.test(verbose=False)
    log.info(test_results)
    trainer.save_checkpoint(model_save + '.pt')
    systems.append(system)
    trainers.append(trainer)

  return systems, trainers


def load_models(dataset, model_class, seeds):
  models = []
  for seed in seeds:
    model_name = model_class.__name__
    log.debug(f'Loading trained parameters from disk for {model_name} with seed {seed}')
    model_save = f'{dataset.name}/models/{model_name}' \
                 f'{"_aug" if dataset.augment_training else ""}' \
                 f'_seed_{seed}.pt'
    model = model_class.load_from_checkpoint(checkpoint_path=model_save, verbose=False)
    # Monkey patch some extra data on to the model object
    model.training_dataset = dataset
    model.seed = seed
    models.append(model)
  return models


def foolbox_test(fb_model, test_loader, attack, epsilons):
  attack_success = torch.zeros(len(epsilons)).to(device)
  attack_samples = 0

  for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)
    raw_advs, clipped_advs, success = attack(fb_model, images, labels, epsilons=epsilons)
    attack_success += success.sum(axis=-1)
    attack_samples += images.shape[0]
    
  robust_accuracy = 1 - attack_success/attack_samples
  return robust_accuracy


def test_models(models, vanilla_dm, gauss_test_sets, fgsm_epsilons=None, pgd_epsilons=None):
  results = {f'gaussian={gauss_level}': {} for gauss_level in gauss_test_sets.keys()}
  if fgsm_epsilons is not None:
    results.update({f'fgsm={epsilon}': {} for epsilon in fgsm_epsilons})
  if pgd_epsilons is not None:
    results.update({f'pgd={epsilon}': {} for epsilon in pgd_epsilons})

  model_name = type(models[0]).__name__
  model_key = f'{model_name}' \
              f'{"_aug" if models[0].training_dataset.augment_training else ""}'

  fgsm = foolbox.attacks.FGSM()
  pgd = foolbox.attacks.PGD()
  test_loader = DataLoader(vanilla_dm.get_test(transform=T.ToTensor()), batch_size=vanilla_dm.batch_size)

  for model in models:
    # Gaussian noise
    for gauss_level, gauss_dm in gauss_test_sets.items():
      pl.seed_everything(model.seed)  # For repeatability
      
      trainer = pl.Trainer(gpus=-1 if device == 'cuda' else 0, logger=False, progress_bar_refresh_rate=0)
      [test_out] = trainer.test(model,
                                test_dataloaders=gauss_dm.test_dataloader(),
                                verbose=False)
      log.debug(
        f"{model_key} seed={model.seed}, gaussian={gauss_level}, accuracy={test_out['test_acc'] * 100:.1f}%")
      results[f'gaussian={gauss_level}'][f'seed={model.seed}'] = test_out['test_acc']
    # Foolbox for FGSM and PGD
    preprocessing = dict(mean=vanilla_dm.mean, std=vanilla_dm.stddev, axis=-3)
    fb_model = foolbox.PyTorchModel(model.eval(), bounds=(0, 1), preprocessing=preprocessing)
    if fgsm_epsilons is not None:
      test_out = foolbox_test(fb_model, test_loader, fgsm, fgsm_epsilons)
      for epsilon, test_acc in zip(fgsm_epsilons, test_out):
        results[f'fgsm={epsilon}'][f'seed={model.seed}'] = test_acc.item()
    if pgd_epsilons is not None:
      test_out = foolbox_test(fb_model, test_loader, pgd, pgd_epsilons)
      for epsilon, test_acc in zip(pgd_epsilons, test_out):
        results[f'pgd={epsilon}'][f'seed={model.seed}'] = test_acc.item()

  
  # Aggregate results
  for gauss_level in gauss_test_sets.keys():
    acc_list = [results[f'gaussian={gauss_level}'][f'seed={m.seed}'] for m in models]
    results[f'gaussian={gauss_level}']['mean'] = statistics.mean(acc_list)
    results[f'gaussian={gauss_level}']['std'] = statistics.stdev(acc_list)
    log.info(f"{model_key} gaussian={gauss_level}: accuracy={results[f'gaussian={gauss_level}']['mean'] * 100:.1f} " \
             f"± {results[f'gaussian={gauss_level}']['std'] * 100:.1f}")
  if fgsm_epsilons is not None:
    for epsilon in fgsm_epsilons:
      acc_list = [results[f'fgsm={epsilon}'][f'seed={m.seed}'] for m in models]
      results[f'fgsm={epsilon}']['mean'] = statistics.mean(acc_list)
      results[f'fgsm={epsilon}']['std'] = statistics.stdev(acc_list)
      log.info(f"{model_key} fgsm={epsilon}: accuracy={results[f'fgsm={epsilon}']['mean'] * 100:.1f} " \
              f"± {results[f'fgsm={epsilon}']['std'] * 100:.1f}")
  if pgd_epsilons is not None:
    for epsilon in pgd_epsilons:
      acc_list = [results[f'pgd={epsilon}'][f'seed={m.seed}'] for m in models]
      results[f'pgd={epsilon}']['mean'] = statistics.mean(acc_list)
      results[f'pgd={epsilon}']['std'] = statistics.stdev(acc_list)
      log.info(f"{model_key} pdg={epsilon}: accuracy={results[f'pgd={epsilon}']['mean'] * 100:.1f} " \
              f"± {results[f'pgd={epsilon}']['std'] * 100:.1f}")

  return results


class BasePLDataModule(pl.LightningDataModule):
  def __init__(self, data_dir: str = '/dev/shm/', batch_size: int = 128,
               augment_training=None,
               test_gaussian_noise: Optional[float] = None):
    super().__init__()
    self.augment_training = augment_training
    self.data_dir = data_dir
    self.batch_size = batch_size
    self.test_gaussian_noise = test_gaussian_noise

  def prepare_data(self):
    # Ensure the data is downloaded
    self.get_train(download=True)
    self.get_test(download=True)

  def setup(self, stage: Optional[str] = None):
    if stage in (None, 'fit'):
      training_full = self.get_train(transform=T.Compose([T.ToTensor(), self.normalize]))
      self.train_set, self.val_set = random_split(training_full, self.train_val_split)

    if self.augment_training:
      add_gaussian_transform = T.Compose([
        T.ToTensor(),
        T.RandomApply([T.RandomChoice([AddGaussianNoise(std=std / 255) for std in self.augment_training])], 0.5),
        self.normalize])
      self.aug_training_data = self.get_train(transform=add_gaussian_transform)

    if stage in (None, 'test'):
      if self.test_gaussian_noise is not None:
        self.test_transform = T.Compose([
          T.ToTensor(),
          AddGaussianNoise(std=self.test_gaussian_noise / 255),
          self.normalize
        ])
      else:
        self.test_transform = T.Compose([T.ToTensor(), self.normalize])
      self.test_set = self.get_test(transform=self.test_transform)

  def train_dataloader(self):
    if self.augment_training:
      return DataLoader(self.aug_training_data, batch_size=self.batch_size, shuffle=True, num_workers=4)
    else:
      return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=4)

  def val_dataloader(self):
    return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=4)

  def test_dataloader(self):
    return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=4)

  def teardown(self, stage: Optional[str] = None):
    # Used to clean-up when the run is finished
    pass


class BasePLSystem(pl.LightningModule):
  def __init__(self, optimizer='Adam', learning_rate=0.001,
               weight_decay=0.0005, plateau_min_delta=0, plateau_patience=3, plateau_cooldown=0, 
               early_stopping_min_delta=0.001, early_stopping_patience=10, **kwargs):
    super().__init__()
    self.save_hyperparameters()
    self.train_acc = pl.metrics.Accuracy()
    self.val_acc = pl.metrics.Accuracy()
    self.test_acc = pl.metrics.Accuracy()

  def forward(self, x):
    return self.model(x)

  def configure_optimizers(self):
    if self.hparams.optimizer == 'SGD':
      optimizer = optim.SGD(self.parameters(), lr=self.hparams.learning_rate,
                            weight_decay=self.hparams.weight_decay,
                            momentum=self.hparams.momentum)
    elif self.hparams.optimizer == 'Adam':
      optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate,
                             weight_decay=self.hparams.weight_decay)
    else:
      raise NotImplemented(f'optimizer={self.hparams.optimizer}')

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=self.hparams.plateau_patience,
      threshold=self.hparams.plateau_min_delta, cooldown=self.hparams.plateau_cooldown, factor=0.25, verbose=True)
    return {'optimizer': optimizer,
      'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}}

  def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self(x)
    loss = F.cross_entropy(y_hat, y)
    self.log('train_loss', loss)
    self.log('train_acc_step', self.train_acc(F.softmax(y_hat, dim=-1), y))
    return loss

  def training_epoch_end(self, outs):
    self.log('train_acc_epoch', self.train_acc.compute())

  def validation_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self(x)
    loss = F.cross_entropy(y_hat, y)
    self.log('val_loss', loss)
    self.log('val_acc_step', self.val_acc(F.softmax(y_hat, dim=-1), y))
    return loss

  def validation_epoch_end(self, outs):
    self.log('val_acc_epoch', self.val_acc.compute())

  def test_step(self, batch, batch_idx, test_idx=0):
    x, y = batch
    y_hat = self(x)
    loss = F.cross_entropy(y_hat, y)

    self.log(f'test_loss', loss)
    self.log(f'test_acc', self.test_acc(F.softmax(y_hat, dim=-1), y))
    return loss


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class FeatureDenoising(nn.Module):
  def __init__(self, dim=64):
    super().__init__()
    self.conv = nn.Conv2d(dim, dim, 1, 1)
    self.norm = norm(dim)

  def forward(self, x):
    H, W = x.shape[-2:]
    f = torch.einsum('nihw,njhw->nij', x, x)
    out = torch.einsum('nij,nihw->njhw', f, x)
    out = out / (H * W)
    out = self.conv(out)
    out = self.norm(out)
    return F.relu(out + x)


class InputRandomization(nn.Module):
  def __init__(self, output_size):
    super().__init__()
    self.output_size = output_size

  def forward(self, x):
    N, C, H, W = x.shape
    
    # It appears the implementation in the original paper [xie2017] uses the
    # same resize and pad parameters for the whole batch rather than different
    # random parameters for each image.
    
    # First step: Random resizing
    delta_h = torch.randint(self.output_size - H, (1,)).item()
    delta_w = torch.randint(self.output_size - W, (1,)).item()
    x = T.functional.resize(x, [H + delta_h, W + delta_w])
    
    # Second step: Random padding
    pad_h1 = torch.randint(self.output_size - H - delta_h + 1, (1,)).item()
    pad_h2 = self.output_size - H - delta_h - pad_h1
    pad_w1 = torch.randint(self.output_size - W - delta_w + 1, (1,)).item()
    pad_w2 = self.output_size - W - delta_w - pad_w1
    x = T.functional.pad(x, [pad_w1, pad_h1, pad_w2, pad_h2])

    return x



class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ODEBlock(nn.Module):

    def __init__(self, odefunc, step_size=0.1, integration_time=(0, 1)):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.step_size = step_size
        self.integration_time = torch.tensor(integration_time).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, method='euler', options={'step_size': self.step_size})
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class TisODEBase(BasePLSystem):
  def __init__(self, robustness=None, step_size=0.1, L_ss_weight=0.01,
               start_regularization=5, **kwargs):
    super().__init__(robustness=robustness, step_size=step_size,
                     L_ss_weight=L_ss_weight,
                     start_regularization=start_regularization, **kwargs)
    self.robustness = robustness

    self.step_size = step_size
    self.L_ss_weight = L_ss_weight
    self.start_regularization = start_regularization

    self.L_ss_integration_time = torch.arange(1, 2 + step_size/2, step_size)

    # Define the following in child classes:
    self.model1 = None   # Before ODE
    self.odeblock = None # ODE function block (of class ODEBlock) to integrate
    self.model2 = None   # After ODE

  def forward(self, x, return_L_ss=False):
    ode_start = self.model1(x)
    ode_out = self.odeblock(ode_start)
    y_hat = self.model2(ode_out)
    if not return_L_ss:
      return y_hat
    # Add additional L_ss loss term for TisODE model
    # Implement abs be calculating all f(z) terms, using that to calculate Δf,
    # Then sum(abs(Δf)) to get the L_ss value.
    ode_out2 = odeint(self.odeblock.odefunc,
                      ode_out, self.L_ss_integration_time,
                      method='euler', options={'step_size': self.step_size})
    Δf = ode_out2[1:] - ode_out2[:-1]
    integral = Δf.abs().sum(dim=0)
    L_ss = torch.linalg.norm(integral).mean()
    return y_hat, L_ss

  def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat, L_ss = self(x, return_L_ss=True)
    loss = F.cross_entropy(y_hat, y)
    if self.current_epoch < self.start_regularization:
      # Fix an arbitrary constant so that it doesn't impact
      # the gradient and backpropagation but keeps these
      # models from looking good compared to when regularization
      # kicks in.
      L_ss = 100/self.L_ss_weight
    # Add in regularization gently in two steps of two epochs
    elif self.current_epoch < self.start_regularization + 2:
      L_ss = 0.01*L_ss + 40/self.L_ss_weight
    elif self.current_epoch < self.start_regularization + 4:
      L_ss = 0.1*L_ss + 20/self.L_ss_weight
    self.log('train_output_loss', loss)
    self.log('train_acc_step', self.train_acc(F.softmax(y_hat, dim=-1), y))
    L_ss = self.L_ss_weight*L_ss
    self.log('train_L_ss_loss', L_ss)
    loss = loss + L_ss
    self.log('train_loss', loss)
    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    y_hat, L_ss = self(x, return_L_ss=True) 
    if self.current_epoch < self.start_regularization:
      # Fix an arbitrary constant so that it doesn't impact
      # the gradient and backpropagation but keeps these
      # models from looking good compared to when regularization
      # kicks in.
      L_ss = 100/self.L_ss_weight
    # Add in regularization gently in two steps of two epochs
    elif self.current_epoch < self.start_regularization + 2:
      L_ss = 0.01*L_ss + 40/self.L_ss_weight
    elif self.current_epoch < self.start_regularization + 4:
      L_ss = 0.1*L_ss + 20/self.L_ss_weight
    loss = F.cross_entropy(y_hat, y)
    self.log('val_output_loss', loss)
    self.log('val_acc_step', self.val_acc(F.softmax(y_hat, dim=-1), y))
    L_ss = self.L_ss_weight*L_ss
    self.log('val_L_ss_loss', L_ss)
    loss = loss + L_ss
    self.log('val_loss', loss)
    return loss

  def test_step(self, batch, batch_idx):
    x, y = batch
    y_hat, L_ss = self(x, return_L_ss=True)
    if self.current_epoch < self.start_regularization:
      # Fix an arbitrary constant so that it doesn't impact
      # the gradient and backpropagation but keeps these
      # models from looking good compared to when regularization
      # kicks in.
      L_ss = 100/self.L_ss_weight
    # Add in regularization gently in two steps of two epochs
    elif self.current_epoch < self.start_regularization + 2:
      L_ss = 0.01*L_ss + 40/self.L_ss_weight
    elif self.current_epoch < self.start_regularization + 4:
      L_ss = 0.1*L_ss + 20/self.L_ss_weight
    loss = F.cross_entropy(y_hat, y)
    self.log('test_output_loss', loss)
    self.log('test_acc', self.test_acc(F.softmax(y_hat, dim=-1), y))
    L_ss = self.L_ss_weight*L_ss
    self.log('test_L_ss_loss', L_ss)
    loss = loss + L_ss
    self.log('test_loss', loss)
    return loss