import torch
from torch import nn
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import count_num_param, load_pretrained_weights
from dassl.engine.trainer import TextureSimpleNet
from dassl.modeling.ops.utils import (
    sharpen_prob, linear_rampup, sigmoid_rampup
)
import copy


@TRAINER_REGISTRY.register()
class TextureTransform(TrainerX):
    """Vanilla baseline."""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.lmda_contra = cfg.TRAINER.TEXTRANS.LMDA_CONTRA
        self.tau_contra = cfg.TRAINER.TEXTRANS.TAU_CONTRA
        self.lmda_consistency = cfg.TRAINER.TEXTRANS.LMDA_CONSISTENCY
        self.temp_consistency = cfg.TRAINER.TEXTRANS.TEMP_CONSISTENCY
        self.measure_consistency = cfg.TRAINER.TEXTRANS.MEASURE_CONSISTENCY
        self.ema_alpha = cfg.TRAINER.TEXTRANS.EMA_ALPHA

        self.contrastive_loss = self.sup_contrastive_loss
        # self.contrastive_loss = SupConLoss()
        self.n_domain = cfg.DATALOADER.TRAIN_X.N_DOMAIN
        # self.build_model()

    def build_model(self):
        cfg = self.cfg

        self.transform_LL = cfg.TRAINER.TEXTRANS.TRANSFORM_LL
        self.transform_HH = cfg.TRAINER.TEXTRANS.TRANSFORM_HH
        self.LL_permute_scale = cfg.TRAINER.TEXTRANS.LL_PERMUTE_SCALE
        self.HH_kernel_size = cfg.TRAINER.TEXTRANS.HH_KERNEL_SIZE

        self.LL_transform_ratio = cfg.TRAINER.TEXTRANS.LL_TRANSFORM_RATIO
        self.HH_transform_ratio = cfg.TRAINER.TEXTRANS.HH_TRANSFORM_RATIO
        self.BOTH_transform_ratio = cfg.TRAINER.TEXTRANS.BOTH_TRANSFORM_RATIO
        self.encode_mode = cfg.TRAINER.TEXTRANS.ENCODE_MODE

        self.use_growing = cfg.TRAINER.TEXTRANS.USE_GROWING

        print('Building model')

        self.model = TextureSimpleNet(cfg, cfg.MODEL, self.num_classes,
                                      transform_LL = self.transform_LL, transform_HH = self.transform_HH,
                                      LL_permute_scale = self.LL_permute_scale, HH_kernel_size = self.HH_kernel_size,
                                      LL_transform_ratio = self.LL_transform_ratio, HH_transform_ratio = self.HH_transform_ratio, BOTH_transform_ratio = self.BOTH_transform_ratio, encode_mode = self.encode_mode,
                                      use_growing = self.use_growing)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.model)))
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model('model', self.model, self.optim, self.sched)

    @staticmethod
    def sup_contrastive_loss(f, f_tr, label, domain_label, tau=0.05):
        N, T = f.shape
        f_list = torch.cat((f, f_tr), dim=0).clone()

        #label -> N * N
        # N * 1 --> same class is positive
        # import pdb; pdb.set_trace()
        cls_label = label.contiguous().view(-1, 1)
        cls_mask = torch.eq(cls_label, cls_label.T).float().cuda()

        domain_label = domain_label.contiguous().view(-1, 1)
        domain_mask = torch.eq(domain_label, domain_label.T).float().cuda()

        cls_mask = cls_mask.repeat(2, 2)
        domain_mask = domain_mask.repeat(2, 2)

        #make label based on augment
        # aug_label = torch.eye(n=N, m=N).cuda().repeat(2, 2)

        # label_list = torch.logical_or(cls_label, aug_label)
        # label_list = cls_label
        logits_mask = torch.scatter(
            torch.ones_like(cls_mask),
            1,
            torch.arange(N * 2).view(-1, 1).cuda(),
            0
        )

        label_list = logits_mask * cls_mask

        #normalize
        f_list = f_list / torch.norm(f_list, p=2, dim=1).unsqueeze(1)

        # 2N X 2N
        sim_matrix = torch.matmul(f_list, torch.transpose(f_list, 0, 1)) / tau

        # exp
        sim_matrix = torch.exp(sim_matrix)

        # set diagonal elements to zero
        sim_matrix = sim_matrix.clone().fill_diagonal_(0)

        # exclude cases ( different class, different domain ) --> to reduce effect of pulling on different domain
        logits_mask = logits_mask * (1 - ((1 - cls_mask) * (1 - domain_mask)))

        # / sum()
        scores = (sim_matrix * label_list).sum(dim=0) / (sim_matrix * logits_mask).sum(dim=0)

        # import pdb; pdb.set_trace()
        loss_contrast = - torch.log(scores).mean()

        return loss_contrast

    def coral_loss(self, x, y):
        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x
        cent_y = y - mean_y
        cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
        cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()

        return mean_diff + cova_diff

    def coral_loss_split_input_by_domain(self, x_feat, x_feat_tr, domain):

        domain_idxs = [torch.nonzero((domain == i), as_tuple=False).squeeze() for i in range(self.n_domain)]
        features = [x_feat[domain_idx] for domain_idx in domain_idxs]
        features_tr = [x_feat_tr[domain_idx] for domain_idx in domain_idxs]
        features = features + features_tr

        penalty = 0.

        nmb = len(features)
        for i in range(nmb):
            for j in range(i + 1, nmb):
                penalty += self.coral_loss(features[i], features[j])

        if nmb > 1:
            penalty /= (nmb * (nmb - 1) / 2)

        return penalty


    def forward_backward(self, batch):
        input, label, domain = self.parse_batch_train(batch)

        (output, f), (output_tr, f_tr) = self.model(input, return_feature=True, cur_epoch_ratio=((self.epoch+1)/float(self.max_epoch)))
        # (_, _), (output_tr2, _) = self.model(input, return_feature=True)
        # (_, _), (output_tr3, _) = self.model(input, return_feature=True)

        # output_tr_mean = (output_tr + output_tr2 + output_tr3)/3
        output_tr_mean = output_tr

        # output, output_tr, loss_sim = self.model(input, return_loss=True)
        loss = F.cross_entropy(output, label)
        # loss += F.cross_entropy(output_tr, label.clone())
        # loss = F.cross_entropy(output_tr, label)

        # loss_consistency = F.mse_loss(output, output_tr)

        # loss_sim = 1 - torch.mean(torch.cosine_similarity(f, f_tr))
        # loss_sim = self.lmda_contra * self.coral_loss_split_input_by_domain(f, f_tr, domain)
        loss_sim = self.lmda_contra * self.contrastive_loss(f, f_tr, label, domain_label=domain, tau=self.tau_contra)
        # loss_sim = self.lmda_contra * self.contrastive_loss(torch.stack((f, f_tr), dim=1), label, domain_label = domain)

        # loss_consistency = self.lmda_consistency * F.kl_div(F.log_softmax(output_tr, dim=1), sharpen_prob(F.softmax(output, dim=1), temperature=self.temp_consistency))

        # collect samples which surpass the confidence threshold
        # confidence_idx = (torch.max(torch.softmax(output, dim=1), dim=1)[0] > 0.5).nonzero(as_tuple=True)[0]
        # correct_idx = (torch.max(output, dim=1)[1] == label).nonzero(as_tuple=True)[0]

        loss_consistency = self.lmda_consistency * F.kl_div(F.log_softmax(output_tr_mean, dim=1),
                                                            sharpen_prob(F.softmax(output, dim=1),
                                                                         temperature=self.temp_consistency))
        # loss_consistency = F.kl_div(F.log_softmax(output_tr, dim=1), F.softmax(output_tr_mean, dim=1)) + F.kl_div(
        #     F.log_softmax(output_tr2, dim=1), F.softmax(output_tr_mean, dim=1)) + F.kl_div(
        #     F.log_softmax(output_tr3, dim=1), F.softmax(output_tr_mean, dim=1))
        loss = loss + loss_sim + loss_consistency
        # loss = loss + loss_sim + loss_consistency
        self.model_backward_and_update(loss)

        loss_summary = {
            'loss': loss.item(),
            'loss_ce': (loss - loss_sim - loss_consistency).item(),
            'loss_sim': loss_sim.item(),
            'loss_consistency':loss_consistency.item(),
            'mean_LL_mean':self.model.backbone.transform_layer.mean_LL.mean().item(),
            'mean_LL_std': self.model.backbone.transform_layer.mean_LL.std().item(),
            'std_LL_mean': self.model.backbone.transform_layer.std_LL.mean().item(),
            'std_LL_std': self.model.backbone.transform_layer.std_LL.std().item(),
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

    @torch.no_grad()
    def test(self):
        """A generic testing pipeline."""
        self.set_model_mode('eval')
        self.evaluator.reset()

        split = self.cfg.TEST.SPLIT
        print('Do evaluation on {} set'.format(split))
        data_loader = self.val_loader if split == 'val' else self.test_loader
        assert data_loader is not None

        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            output, output_tr = self.model_inference(input)

            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = '{}/{}'.format(split, k)
            self.write_scalar(tag, v, self.epoch)

        return results['accuracy']

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, domain_label=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        qn = torch.norm(features, p=2, dim=2, keepdim=True).detach()
        features = features.div(qn.expand_as(features))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
            # for each sample, mark positive samples when classes match
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits # eq2/eq3 lower part
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        if domain_label is not None:
            domain_label = domain_label.contiguous().view(-1, 1)
            domain_mask = torch.eq(domain_label, domain_label.T).float().to(device)


        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        domain_mask = domain_mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # find (different class, same domain)
        logits_mask = logits_mask * (1-mask) * domain_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss