import torch
from torch import nn

from boxlist import BoxList, boxlist_nms, remove_small_box, cat_boxlist


class FCOSPostprocessor(nn.Module):
    def __init__(self, threshold, top_n, nms_threshold, post_top_n, min_size, n_class):
        super().__init__()

        self.threshold = threshold
        self.top_n = top_n
        self.nms_threshold = nms_threshold
        self.post_top_n = post_top_n
        self.min_size = min_size
        self.n_class = n_class

    def forward_single_feature_map(
        self, location, cls_pred, box_pred, center_pred, image_sizes
    ):
        batch, channel, height, width = cls_pred.shape

        cls_pred = cls_pred.view(batch, channel, height, width).permute(0, 2, 3, 1)
        cls_pred = cls_pred.reshape(batch, -1, channel).sigmoid()

        box_pred = box_pred.view(batch, 4, height, width).permute(0, 2, 3, 1)
        box_pred = box_pred.reshape(batch, -1, 4)

        center_pred = center_pred.view(batch, 1, height, width).permute(0, 2, 3, 1)
        center_pred = center_pred.reshape(batch, -1).sigmoid()

        candid_ids = cls_pred > self.threshold
        top_ns = candid_ids.view(batch, -1).sum(1)
        top_ns = top_ns.clamp(max=self.top_n)

        cls_pred = cls_pred * center_pred[:, :, None]

        results = []

        for i in range(batch):
            cls_p = cls_pred[i]
            candid_id = candid_ids[i]
            cls_p = cls_p[candid_id]
            candid_nonzero = candid_id.nonzero()
            box_loc = candid_nonzero[:, 0]
            class_id = candid_nonzero[:, 1] + 1

            box_p = box_pred[i]
            box_p = box_p[box_loc]
            loc = location[box_loc]

            top_n = top_ns[i]

            if candid_id.sum().item() > top_n.item():
                cls_p, top_k_id = cls_p.topk(top_n, sorted=False)
                class_id = class_id[top_k_id]
                box_p = box_p[top_k_id]
                loc = loc[top_k_id]

            detections = torch.stack(
                [
                    loc[:, 0] - box_p[:, 0],
                    loc[:, 1] - box_p[:, 1],
                    loc[:, 0] + box_p[:, 2],
                    loc[:, 1] + box_p[:, 3],
                ],
                1,
            )

            height, width = image_sizes[i]

            boxlist = BoxList(detections, (int(width), int(height)), mode='xyxy')
            boxlist.fields['labels'] = class_id
            boxlist.fields['scores'] = torch.sqrt(cls_p)
            boxlist = boxlist.clip(remove_empty=False)
            boxlist = remove_small_box(boxlist, self.min_size)

            results.append(boxlist)

        return results

    def forward(self, location, cls_pred, box_pred, center_pred, image_sizes):
        boxes = []

        for loc, cls_p, box_p, center_p in zip(
            location, cls_pred, box_pred, center_pred
        ):
            boxes.append(
                self.forward_single_feature_map(
                    loc, cls_p, box_p, center_p, image_sizes
                )
            )

        boxlists = list(zip(*boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_scales(boxlists)

        return boxlists

    def select_over_scales(self, boxlists):
        results = []

        for boxlist in boxlists:
            scores = boxlist.fields['scores']
            labels = boxlist.fields['labels']
            box = boxlist.box

            result = []

            for j in range(1, self.n_class):
                id = (labels == j).nonzero().view(-1)
                score_j = scores[id]
                box_j = box[id, :].view(-1, 4)
                box_by_class = BoxList(box_j, boxlist.size, mode='xyxy')
                box_by_class.fields['scores'] = score_j
                box_by_class = boxlist_nms(box_by_class, score_j, self.nms_threshold)
                n_label = len(box_by_class)
                box_by_class.fields['labels'] = torch.full(
                    (n_label,), j, dtype=torch.int64, device=scores.device
                )
                result.append(box_by_class)

            result = cat_boxlist(result)
            n_detection = len(result)

            if n_detection > self.post_top_n > 0:
                scores = result.fields['scores']
                img_threshold, _ = torch.kthvalue(
                    scores.cpu(), n_detection - self.post_top_n + 1
                )
                keep = scores >= img_threshold.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]

            results.append(result)

        return results
