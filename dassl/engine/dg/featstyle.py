import torch
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import count_num_param, load_pretrained_weights
from dassl.engine.trainer import FeatureStylizationNet
from dassl.modeling.ops.utils import sharpen_prob

@TRAINER_REGISTRY.register()
class FeatureStylization(TrainerX):
    """Vanilla baseline."""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.n_domain = cfg.DATALOADER.TRAIN_X.N_DOMAIN

        self.lmda_contra = cfg.TRAINER.FEATSTYLE.LMDA_CONTRA
        self.tau_contra = cfg.TRAINER.FEATSTYLE.TAU_CONTRA
        self.lmda_consistency = cfg.TRAINER.FEATSTYLE.LMDA_CONSISTENCY
        self.tau_consistency = cfg.TRAINER.FEATSTYLE.TAU_CONSISTENCY
        self.measure_consistency = cfg.TRAINER.FEATSTYLE.MEASURE_CONSISTENCY
        self.stylization_layer_idx = cfg.TRAINER.FEATSTYLE.LAYER_IDX

    def build_model(self):
        cfg = self.cfg

        # self.stylize_feature = cfg.TRAINER.FEATSTYLE.TRANSFORM_LL
        self.scaling_factor = cfg.TRAINER.FEATSTYLE.SCALING_FACTOR
        self.encode_mode = cfg.TRAINER.FEATSTYLE.ENCODE_MODE

        print('Building model')

        self.model = FeatureStylizationNet(cfg, cfg.MODEL, self.num_classes, scaling_factor = self.scaling_factor, encode_mode=self.encode_mode)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.model)))
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model('model', self.model, self.optim, self.sched)

    @staticmethod
    def domain_aware_contrastive_loss(f, f_tr, label, domain_label, tau=0.05):
        N, T = f.shape

        # Concat N originals and N stylized ones -> 2N samples
        f_list = torch.cat((f, f_tr), dim=0).clone()

        # Reshape labels N -> N * 1
        cls_label = label.contiguous().view(-1, 1)
        # N * N matrix where samples of same class label with an anchor are marked as positive
        cls_mask = torch.eq(cls_label, cls_label.T).float().cuda()

        # Reshape domain labels N -> N * 1
        domain_label = domain_label.contiguous().view(-1, 1)
        # N * N matrix where samples of same domain label with an anchor are marked as positive
        domain_mask = torch.eq(domain_label, domain_label.T).float().cuda()

        cls_mask = cls_mask.repeat(2, 2)
        domain_mask = domain_mask.repeat(2, 2)

        logits_mask = torch.scatter(
            torch.ones_like(cls_mask),
            1,
            torch.arange(N * 2).view(-1, 1).cuda(),
            0
        )

        label_list = logits_mask * cls_mask

        # Normalize feature vectors
        f_list = f_list / torch.norm(f_list, p=2, dim=1).unsqueeze(1)

        # 2N X 2N similarity matrix
        sim_matrix = torch.matmul(f_list, torch.transpose(f_list, 0, 1)) / tau

        # exp
        sim_matrix = torch.exp(sim_matrix)

        # set diagonal elements to zero
        sim_matrix = sim_matrix.clone().fill_diagonal_(0)

        # exclude cases ( different class, different domain )
        logits_mask = logits_mask * (1 - ((1 - cls_mask) * (1 - domain_mask)))

        # / sum()
        scores = (sim_matrix * label_list).sum(dim=0) / (sim_matrix * logits_mask).sum(dim=0)

        loss_contrast = - torch.log(scores).mean()

        return loss_contrast

    def forward_backward(self, batch):
        input, label, domain = self.parse_batch_train(batch)

        (output, f), (output_tr, f_tr) = self.model(input, return_feature=True, stylization_layer_idx = self.stylization_layer_idx)

        output_tr_mean = output_tr

        loss_ce = F.cross_entropy(output, label)

        loss_consistency = self.lmda_consistency * F.kl_div(F.log_softmax(output_tr_mean, dim=1),
                                                            sharpen_prob(F.softmax(output, dim=1),
                                                                         temperature=self.tau_consistency))

        loss_contrastive = self.lmda_contra * self.domain_aware_contrastive_loss(f, f_tr, label, domain_label=domain, tau=self.tau_contra)

        loss = loss_ce + loss_consistency + loss_contrastive

        self.model_backward_and_update(loss)

        loss_summary = {
            'loss': loss.item(),
            'loss_ce': loss_ce.item(),
            'loss_consistency':loss_consistency.item(),
            'loss_contrastive': loss_contrastive.item(),
            'acc': compute_accuracy(output, label)[0].item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch['img']
        label = batch['label']
        domain = batch['domain']
        input = input.to(self.device)
        label = label.to(self.device)
        domain = domain.to(self.device)
        return input, label, domain

    def model_inference(self, input):
        output, output_tr = self.model(input)
        return output
