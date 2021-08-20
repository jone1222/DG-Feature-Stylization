import torch
from torch.nn import functional as F
from torch import autograd
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
import numpy as np
import os

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy


@TRAINER_REGISTRY.register()
class RSC(TrainerX):
    """Vanilla baseline."""

    def __init__(self, cfg):
        super().__init__(cfg)

        # drop percentile random uniform btw (0, 0.5)
        self.drop_f = (1 - 1/3) * 100
        self.drop_b = (1 - 1/3) * 100

    def forward_backward(self, batch):
        import pdb; pdb.set_trace()
        # inputs
        all_x, all_y = self.parse_batch_train(batch)
        # one-hot labels
        all_o = torch.nn.functional.one_hot(all_y, self.num_classes)
        # features
        all_f = self.model.get_feature(all_x)
        # predictions
        all_p = self.model.classifier(all_f)

        # Equation (1): compute gradients with respect to representation
        all_g = autograd.grad((all_p * all_o).sum(), all_f)[0]

        # Equation (2): compute top-gradient-percentile mask
        percentiles = np.percentile(all_g.cpu(), self.drop_f, axis=1)
        percentiles = torch.Tensor(percentiles)
        percentiles = percentiles.unsqueeze(1).repeat(1, all_g.size(1))
        mask_f = all_g.lt(percentiles.cuda()).float()

        # Equation (3): mute top-gradient-percentile activations
        all_f_muted = all_f * mask_f

        # Equation (4): compute muted predictions
        all_p_muted = self.model.classifier(all_f_muted)

        # Section 3.3: Batch Percentage
        all_s = F.softmax(all_p, dim=1)
        all_s_muted = F.softmax(all_p_muted, dim=1)
        changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)
        percentile = np.percentile(changes.detach().cpu(), self.drop_b)
        mask_b = changes.lt(percentile).float().view(-1, 1)
        mask = torch.logical_or(mask_f, mask_b).float()

        # Equations (3) and (4) again, this time mutting over examples
        all_p_muted_again = self.model.classifier(all_f * mask)

        # Equation (5): update
        loss = F.cross_entropy(all_p_muted_again, all_y)
        self.model_backward_and_update(loss)

        loss_summary = {
            'loss': loss.item(),
            'acc': compute_accuracy(all_p, all_y)[0].item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch['img']
        label = batch['label']
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

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
            output = self.model_inference(input)

            # if batch_idx == 0:
            #     # draw cam
            #     fc_weight = self.model.classifier.weight # 7 * 512
            #     feature_map = self.model.backbone.featuremaps(input) # 128 * 512 * 7 * 7
            #     b, c, h, w = feature_map.shape
            #     output_argmax = output.argmax(dim=1) # 128
            #     w_output = fc_weight[output_argmax].view(b, c, 1, 1).repeat(1, 1, 7, 7) # 128 * 512 * 7 * 7
            #     weighted_feat_map = w_output * feature_map
            #     weighted_feat_map = weighted_feat_map.view(b, c, -1).sum(dim=1).view(b, h * w)
            #
            #     weighted_feat_map = weighted_feat_map - weighted_feat_map.min(dim=1)[0].unsqueeze(1)
            #     weighted_feat_map = weighted_feat_map / weighted_feat_map.max(dim=1)[0].unsqueeze(1)
            #     alpha = weighted_feat_map.view(b, 1, h, w)
            #     alpha = torch.nn.functional.interpolate(alpha, size=(input.size(2), input.size(3)))
            #
            #     img_dir = self.cfg.OUTPUT_DIR + "/cam_img"
            #     if not os.path.exists(img_dir):
            #         os.makedirs(img_dir)
            #
            #     alpha_list = torch.split(alpha, 1, dim=0)
            #     cm_jet = cm.get_cmap('jet')
            #     heatmap_list = []
            #     for idx in range(len(alpha_list)):
            #         hmap = np.delete(cm_jet(alpha_list[idx].squeeze().cpu().numpy()), 3, 2)
            #         heatmap_list.append(T.ToTensor()(hmap).unsqueeze(0))
            #     heatmap = torch.cat(heatmap_list, dim=0).cuda()
            #     img_out = torch.cat((input, heatmap * 0.5 + input * 0.5), dim=2)
            #     save_image(make_grid(img_out, nrow=8), img_dir + "/cam_epoch_{}_batch_{}.jpg".format(self.epoch, batch_idx))

            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()
        # if self.cfg.EVAL_ONLY:
        #     for k, v in self.model.bn_statistics_dict.items():
        #         print("Layer : {}, mean_difference : {}, var_difference : {}".format(k, torch.mean(torch.stack(v['mean_diff'])), torch.mean(torch.stack(v['var_diff']))))

        for k, v in results.items():
            tag = '{}/{}'.format(split, k)
            self.write_scalar(tag, v, self.epoch)

        return results
