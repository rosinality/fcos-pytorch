import torch
from torchvision import ops


FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class BoxList:
    def __init__(self, box, image_size, mode='xyxy'):
        device = box.device if hasattr(box, 'device') else 'cpu'
        box = torch.as_tensor(box, dtype=torch.float32, device=device)

        self.box = box
        self.size = image_size
        self.mode = mode

        self.fields = {}

    def convert(self, mode):
        if mode == self.mode:
            return self

        x_min, y_min, x_max, y_max = self.split_to_xyxy()

        if mode == 'xyxy':
            box = torch.cat([x_min, y_min, x_max, y_max], -1)
            box = BoxList(box, self.size, mode=mode)

        elif mode == 'xywh':
            remove = 1
            box = torch.cat(
                [x_min, y_min, x_max - x_min + remove, y_max - y_min + remove], -1
            )
            box = BoxList(box, self.size, mode=mode)

        box.copy_field(self)

        return box

    def copy_field(self, box):
        for k, v in box.fields.items():
            self.fields[k] = v

    def area(self):
        box = self.box

        if self.mode == 'xyxy':
            remove = 1

            area = (box[:, 2] - box[:, 0] + remove) * (box[:, 3] - box[:, 1] + remove)

        elif self.mode == 'xywh':
            area = box[:, 2] * box[:, 3]

        return area

    def split_to_xyxy(self):
        if self.mode == 'xyxy':
            x_min, y_min, x_max, y_max = self.box.split(1, dim=-1)

            return x_min, y_min, x_max, y_max

        elif self.mode == 'xywh':
            remove = 1
            x_min, y_min, w, h = self.box.split(1, dim=-1)

            return (
                x_min,
                y_min,
                x_min + (w - remove).clamp(min=0),
                y_min + (h - remove).clamp(min=0),
            )

    def __len__(self):
        return self.box.shape[0]

    def __getitem__(self, index):
        box = BoxList(self.box[index], self.size, self.mode)

        for k, v in self.fields.items():
            box.fields[k] = v[index]

        return box

    def resize(self, size, *args, **kwargs):
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))

        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            scaled = self.box * ratio
            box = BoxList(scaled, size, mode=self.mode)

            for k, v in self.fields.items():
                if not isinstance(v, torch.Tensor):
                    v = v.resize(size, *args, **kwargs)

                box.fields[k] = v

            return box

        ratio_w, ratio_h = ratios
        x_min, y_min, x_max, y_max = self.split_to_xyxy()
        scaled_x_min = x_min * ratio_w
        scaled_x_max = x_max * ratio_w
        scaled_y_min = y_min * ratio_h
        scaled_y_max = y_max * ratio_h
        scaled = torch.cat([scaled_x_min, scaled_y_min, scaled_x_max, scaled_y_max], -1)
        box = BoxList(scaled, size, mode='xyxy')

        for k, v in self.fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.resize(size, *args, **kwargs)

            box.fields[k] = v

        return box.convert(self.mode)

    def transpose(self, method):
        width, height = self.size
        x_min, y_min, x_max, y_max = self.split_to_xyxy()

        if method == FLIP_LEFT_RIGHT:
            remove = 1

            transpose_x_min = width - x_max - remove
            transpose_x_max = width - x_min - remove
            transpose_y_min = y_min
            transpose_y_max = y_max

        elif method == FLIP_TOP_BOTTOM:
            transpose_x_min = x_min
            transpose_x_max = x_max
            transpose_y_min = height - y_max
            transpose_y_max = height - y_min

        transpose_box = torch.cat(
            [transpose_x_min, transpose_y_min, transpose_x_max, transpose_y_max], -1
        )
        box = BoxList(transpose_box, self.size, mode='xyxy')

        for k, v in self.fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.transpose(method)

            box.fields[k] = v

        return box.convert(self.mode)

    def clip(self, remove_empty=True):
        remove = 1

        max_width = self.size[0] - remove
        max_height = self.size[1] - remove

        self.box[:, 0].clamp_(min=0, max=max_width)
        self.box[:, 1].clamp_(min=0, max=max_height)
        self.box[:, 2].clamp_(min=0, max=max_width)
        self.box[:, 3].clamp_(min=0, max=max_height)

        if remove_empty:
            box = self.box
            keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])

            return self[keep]

        else:
            return self

    def to(self, device):
        box = BoxList(self.box.to(device), self.size, self.mode)

        for k, v in self.fields.items():
            if hasattr(v, 'to'):
                v = v.to(device)

            box.fields[k] = v

        return box


def remove_small_box(boxlist, min_size):
    box = boxlist.convert('xywh').box
    _, _, w, h = box.unbind(dim=1)
    keep = (w >= min_size) & (h >= min_size)
    keep = keep.nonzero().squeeze(1)

    return boxlist[keep]


def cat_boxlist(boxlists):
    size = boxlists[0].size
    mode = boxlists[0].mode
    field_keys = boxlists[0].fields.keys()

    box_cat = torch.cat([boxlist.box for boxlist in boxlists], 0)
    new_boxlist = BoxList(box_cat, size, mode)

    for field in field_keys:
        data = torch.cat([boxlist.fields[field] for boxlist in boxlists], 0)
        new_boxlist.fields[field] = data

    return new_boxlist


def boxlist_nms(boxlist, scores, threshold, max_proposal=-1):
    if threshold <= 0:
        return boxlist

    mode = boxlist.mode
    boxlist = boxlist.convert('xyxy')
    box = boxlist.box
    keep = ops.nms(box, scores, threshold)

    if max_proposal > 0:
        keep = keep[:max_proposal]

    boxlist = boxlist[keep]

    return boxlist.convert(mode)
