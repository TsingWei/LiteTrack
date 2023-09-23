import numpy as np
import torch


def get_2d_gaussian_map(boxes, heatmap_size=16, alpha=1.0):
    """
    Generate a 2D gaussian map based on the given box.

    Args:
        box (tensor): 4D tensor of shape (bs, 4) for [cx, cy, w, h].
        S (int): Map size.
        alpha (float): Gaussian distribution factor.

    Returns:
        tensor: 2D gaussian map tensor of shape (S, S).
    """
    bs = boxes.shape[0]
    device = boxes.device
    boxes_sized = boxes*heatmap_size
    cx, cy, w, h = boxes_sized[:, 1], boxes_sized[:, 0], boxes_sized[:, 3], boxes_sized[:, 2]

    # create coordinate grid
    x, y = torch.meshgrid(torch.arange(heatmap_size, dtype=torch.float32, device=device),
                          torch.arange(heatmap_size, dtype=torch.float32, device=device))

    # calculate gaussian distribution
    gauss_map = torch.exp(-((x.unsqueeze(0) - cx.unsqueeze(-1).unsqueeze(-1)) ** 2 / (alpha * w.unsqueeze(-1).unsqueeze(-1)) +
                            (y.unsqueeze(0) - cy.unsqueeze(-1).unsqueeze(-1)) ** 2 / (alpha * h.unsqueeze(-1).unsqueeze(-1))))

    # normalize gauss_map
    # gauss_map_max, _= gauss_map.flatten(1).max(dim=-1)
    # gauss_map_sum = gauss_map.max(dim=(1, 2))
    # gauss_map /= gauss_map_max.view(bs, 1, 1)

    return gauss_map.squeeze()

def generate_cls_map(bboxes, heatmap_size=16):
    # gaussian_maps = []
    # heatmap_size = patch_size // stride
    bs = bboxes.shape[0]
    gt_scoremap = torch.zeros(bs, heatmap_size, heatmap_size).to(bboxes.device)
    bbox = (bboxes * heatmap_size + 0.5).long()
    for i in range(bs):
        gt_scoremap[i, bbox[i,1]:bbox[i,3], bbox[i,0]:bbox[i,2]] = 1
    return gt_scoremap

def generate_heatmap(bboxes, heatmap_size=16):
    """
    Generate ground truth heatmap same as CenterNet
    Args:
        bboxes (torch.Tensor): shape of [num_search, bs, 4]

    Returns:
        gaussian_maps: list of generated heatmap

    """
    gaussian_maps = []
    # heatmap_size = patch_size // stride
    for single_patch_bboxes in bboxes:
        bs = single_patch_bboxes.shape[0]
        gt_scoremap = torch.zeros(bs, heatmap_size, heatmap_size)
        classes = torch.arange(bs).to(torch.long)
        bbox = single_patch_bboxes * heatmap_size
        wh = bbox[:, 2:]
        centers_int = (bbox[:, :2] + wh / 2 - 0.5).round()
        CenterNetHeatMap.generate_score_map(gt_scoremap, classes, wh, centers_int, 0.7)
        gaussian_maps.append(gt_scoremap.to(bbox.device))
    return gaussian_maps

def generate_distribution_heatmap(bboxes, map_size=16):
    """
    Generate ground truth ditribution heatmap
    Args:
        bboxes_xyxy (torch.Tensor): shape of [num_search, bs, 4]

    Returns:
        gaussian_maps: list of generated heatmap

    """
    gaussian_maps = []
    heatmap_size = int(map_size+1)
    for single_patch_bboxes in bboxes:
        bs = single_patch_bboxes.shape[0]
        bbox = single_patch_bboxes * map_size
        tl, dr = bbox[:, :2], bbox[:, 2:]
        wh = tl - dr
        tl_scoremap = torch.zeros(bs, heatmap_size, heatmap_size)
        br_scoremap = torch.zeros(bs, heatmap_size, heatmap_size)
        classes = torch.arange(bs).to(torch.long)
        
        # centers= (bbox[:, :2] + wh / 2).round()
        two_IntegrationHeatMap.generate_score_map(tl_scoremap, classes, wh, tl, 0.6)
        two_IntegrationHeatMap.generate_score_map(br_scoremap, classes, wh, dr, 0.6)
        gaussian_maps.append(torch.stack((tl_scoremap.to(bbox.device), br_scoremap.to(bbox.device)), dim=1))
        # gaussian_maps.append()
    return gaussian_maps

class CenterNetHeatMap(object):
    @staticmethod
    def generate_score_map(fmap, gt_class, gt_wh, centers_int, min_overlap):
        radius = CenterNetHeatMap.get_gaussian_radius(gt_wh, min_overlap)
        radius = torch.clamp_min(radius, 0)
        radius = radius.type(torch.int).cpu().numpy()
        for i in range(gt_class.shape[0]):
            channel_index = gt_class[i]
            CenterNetHeatMap.draw_gaussian(fmap[channel_index], centers_int[i], radius[i])

    @staticmethod
    def get_gaussian_radius(box_size, min_overlap):
        """
        copyed from CornerNet
        box_size (w, h), it could be a torch.Tensor, numpy.ndarray, list or tuple
        notice: we are using a bug-version, please refer to fix bug version in CornerNet
        """
        # box_tensor = torch.Tensor(box_size)
        box_tensor = box_size
        width, height = box_tensor[..., 0], box_tensor[..., 1]

        a1 = 1
        b1 = height + width
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = torch.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = torch.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = torch.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2

        return torch.min(r1, torch.min(r2, r3))

    @staticmethod
    def gaussian2D(radius, sigma=1):
        # m, n = [(s - 1.) / 2. for s in shape]
        m, n = radius
        y, x = np.ogrid[-m: m + 1, -n: n + 1]

        gauss = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        gauss[gauss < np.finfo(gauss.dtype).eps * gauss.max()] = 0
        return gauss

    @staticmethod
    def draw_gaussian(fmap, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = CenterNetHeatMap.gaussian2D((radius, radius), sigma=diameter / 6)
        gaussian = torch.Tensor(gaussian)
        x, y = int(center[0]), int(center[1])
        height, width = fmap.shape[:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_fmap = fmap[y - top: y + bottom, x - left: x + right]
        masked_gaussian = gaussian[radius - top: radius + bottom, radius - left: radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_fmap.shape) > 0:
            masked_fmap = torch.max(masked_fmap, masked_gaussian * k)
            fmap[y - top: y + bottom, x - left: x + right] = masked_fmap
        # return fmap

class IntegrationHeatMap(object):
    @staticmethod
    def generate_score_map(fmap, gt_class, gt_wh, centers, min_overlap):
        radius = IntegrationHeatMap.get_gaussian_radius(gt_wh, min_overlap)
        radius = torch.clamp_min(radius, 2) # min to neibghour-like
        radius = radius.type(torch.int).cpu().numpy()
        for i in range(gt_class.shape[0]):
            channel_index = gt_class[i]
            IntegrationHeatMap.draw_gaussian(fmap[channel_index], centers[i], radius[i])

    @staticmethod
    def get_gaussian_radius(box_size, min_overlap):
        """
        copyed from CornerNet
        box_size (w, h), it could be a torch.Tensor, numpy.ndarray, list or tuple
        notice: we are using a bug-version, please refer to fix bug version in CornerNet
        """
        # box_tensor = torch.Tensor(box_size)
        box_tensor = box_size
        width, height = box_tensor[..., 0], box_tensor[..., 1]
        r_X = width * min_overlap
        r_Y = height * min_overlap
        return torch.stack((r_X, r_Y), dim=1)

    @staticmethod
    def gaussian2D(radius, bias=(0,0), sigma=(1,1)):
        # m, n = [(s - 1.) / 2. for s in shape]
        n, m = radius[0], radius[1]
        # y, x = np.ogrid[-m: m + 1, -n: n + 1]
        # x, y = x-bias[0], y-bias[1]
        sx, sy = sigma[0], sigma[1]
        # normal = 1/(2. * np.pi * sx * sy)
        mx, my = bias[0], bias[1]
        x = np.linspace(-m, m)
        y = np.linspace(-n, n)  
        x, y = np.meshgrid(x, y)
        gauss = 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))
        # gauss = np.exp(-(x * x / (2. * sx**2.) + y * y / (2. * sy**2.)) / (2. * sx * sy)) * normal
        gauss[gauss < np.finfo(gauss.dtype).eps * gauss.max()] = 0
        return gauss

    @staticmethod
    def draw_gaussian(fmap, center, radius, k=1):
        height, width = fmap.shape[:2] #(17,17)
        x, y = center[0], center[1]
        x_bottom, y_bottom = x.long().clamp(max=width-1), y.long()
        # x_ceiling, y_ceiling = x_bottom+1, y_bottom+1
        bias_x = x - x_bottom.float()
        bias_y = y - y_bottom.float()

        diameter = 2 * radius + 1
        gaussian = IntegrationHeatMap.gaussian2D(radius, (bias_x.item(),bias_y.item()), sigma=diameter / 18)
        gaussian = torch.Tensor(gaussian)
        x, y = int(center[0]), int(center[1])
        height, width = fmap.shape[:2]

        left, right = min(x, radius[0]), min(width - x, radius[0] + 1)
        top, bottom = min(y, radius[1]), min(height - y, radius[1] + 1)

        masked_fmap = fmap[y - top: y + bottom, x - left: x + right]
        masked_gaussian = gaussian[radius[1] - top: radius[1] + bottom, radius[0] - left: radius[0] + right]
        if min(masked_gaussian.shape) > 0 and min(masked_fmap.shape) > 0:
            masked_fmap = torch.max(masked_fmap, masked_gaussian * k)
            fmap[y - top: y + bottom, x - left: x + right] = masked_fmap
        # return fmap

            
class two_IntegrationHeatMap(object):
    @staticmethod
    def generate_score_map(fmap, gt_class, wh, centers, overlap):
        # gt_class = batchs
        for i in range(gt_class.shape[0]):
            channel_index = gt_class[i]
            two_IntegrationHeatMap.draw_neighbour(fmap[channel_index], centers[i],)

    @staticmethod
    def draw_neighbour(fmap, center):
        # diameter = 2 * radius + 1
        # gaussian = CenterNetHeatMap.neibhour2D((radius, radius), sigma=diameter / 6)
        # gaussian = torch.Tensor(gaussian)

        height, width = fmap.shape[:2] #(17,17)
        x, y = center[0], center[1]
        x_bottom, y_bottom = x.long().clamp(max=width-2), y.long().clamp(max=width-2)
        x_ceiling, y_ceiling = x_bottom+1, y_bottom+1
        weight_x_bottom = x_ceiling.float() - x
        weight_x_ceiling = x - x_bottom.float()
        weight_y_bottom = y_ceiling.float() - y
        weight_y_ceiling = y - y_bottom.float()


        fmap[y_bottom, x_bottom] = weight_x_bottom * weight_y_bottom
        fmap[y_ceiling, x_bottom] = weight_x_bottom * weight_y_ceiling
        fmap[y_bottom, x_ceiling] = weight_x_ceiling * weight_y_bottom
        fmap[y_ceiling, x_ceiling] = weight_x_ceiling * weight_y_ceiling
        # return fmap



def compute_grids(features, strides):
    """
    grids regret to the input image size
    """
    grids = []
    for level, feature in enumerate(features):
        h, w = feature.size()[-2:]
        shifts_x = torch.arange(
            0, w * strides[level],
            step=strides[level],
            dtype=torch.float32, device=feature.device)
        shifts_y = torch.arange(
            0, h * strides[level],
            step=strides[level],
            dtype=torch.float32, device=feature.device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        grids_per_level = torch.stack((shift_x, shift_y), dim=1) + \
                          strides[level] // 2
        grids.append(grids_per_level)
    return grids


def get_center3x3(locations, centers, strides, range=3):
    '''
    Inputs:
        locations: M x 2
        centers: N x 2
        strides: M
    '''
    range = (range - 1) / 2
    M, N = locations.shape[0], centers.shape[0]
    locations_expanded = locations.view(M, 1, 2).expand(M, N, 2)  # M x N x 2
    centers_expanded = centers.view(1, N, 2).expand(M, N, 2)  # M x N x 2
    strides_expanded = strides.view(M, 1, 1).expand(M, N, 2)  # M x N
    centers_discret = ((centers_expanded / strides_expanded).int() * strides_expanded).float() + \
                      strides_expanded / 2  # M x N x 2
    dist_x = (locations_expanded[:, :, 0] - centers_discret[:, :, 0]).abs()
    dist_y = (locations_expanded[:, :, 1] - centers_discret[:, :, 1]).abs()
    return (dist_x <= strides_expanded[:, :, 0] * range) & \
           (dist_y <= strides_expanded[:, :, 0] * range)


def get_pred(score_map_ctr, size_map, offset_map, feat_size):
    max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)

    idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
    size = size_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)
    offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

    return size * feat_size, offset


def bbox2distance(points, bbox, max_dis=None, dt=0.1):
    """Decode bounding box based to distances.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        bbox (Tensor): Shape (n, 4), "xyxy" format
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded distances.
    """
    l = points[:, 0] - bbox[:, 0]
    t = points[:, 1] - bbox[:, 1]
    r = bbox[:, 2] - points[:, 0]
    b = bbox[:, 3] - points[:, 1]
    if max_dis is not None:
        l = l.clamp(min=0, max=max_dis - dt)
        t = t.clamp(min=0, max=max_dis - dt)
        r = r.clamp(min=0, max=max_dis - dt)
        b = b.clamp(min=0, max=max_dis - dt)
    return torch.stack([l, t, r, b], -1)

# Not norm to 1
def grid_center_flattened(gt_class_map, feat_size=16):
    '''
    gt_class_map: flattened map (bs, n*n)
    '''
    # feat_size = img_size // stride
    pos_ind = torch.nonzero(gt_class_map)
    idx_y = torch.div(pos_ind[:,1], feat_size , rounding_mode='trunc').to(torch.float) + 0.5
    idx_x = (pos_ind[:,1] % feat_size).to(torch.float) + 0.5
    return torch.stack([idx_x, idx_y], dim=1)

def grid_center_2d(gt_class_map, feat_size=16):
    '''
    gt_class_map: 2d map (bs, n, n)
    '''
    # feat_size = img_size // stride
    pos_ind = torch.nonzero(gt_class_map)
    idx_y = pos_ind[:,1].to(torch.float) + 0.5
    idx_x = pos_ind[:,2].to(torch.float) + 0.5
    return torch.stack([idx_x, idx_y], dim=1)