import numpy as np


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma_x=1, sigma_y=1):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m: m + 1, -n: n + 1]
    h = np.exp(-x * x / (2 * sigma_x ** 2) - y * y / (2 * sigma_y ** 2))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius_x, radius_y):
    diameter_x = 2 * radius_x + 1
    diameter_y = 2 * radius_y + 1
    gaussian = gaussian2D(
        (diameter_y, diameter_x), sigma_y=diameter_y / 6, sigma_x=diameter_x / 6
    )

    x, y = int(np.round(center[0])), int(np.round(center[1]))

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius_x), min(width - x, radius_x + 1)
    top, bottom = min(y, radius_y), min(height - y, radius_y + 1)

    masked_heatmap = heatmap[y - top: y + bottom, x - left: x + right]
    masked_gaussian = gaussian[
                      radius_y - top: radius_y + bottom, radius_x - left: radius_x + right
                      ]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)
    return heatmap


class ImageNetEncoder:
    def __init__(self, img_height=320, img_width=320, down_ratio=4, n_classes=20):
        self._img_height = img_height
        self._img_width = img_width
        self._out_height = self._img_height // down_ratio
        self._out_width = self._img_width // down_ratio
        self._n_classes = n_classes
        self._down_ratio = down_ratio
        print(f"ImageNetEncoder initialized with down_ratio = {self._down_ratio}")
        print(f"Output dimensions: {self._out_height}x{self._out_width}")

    def __call__(self, bboxes, labels):
        # Initialize heatmaps for each class (64x64 output)
        hm = np.zeros(
            (64, 64, self._n_classes), dtype=np.float32
        )
        # Initialize coordinate maps
        coors = np.zeros((64, 64, 4), dtype=np.float32)

        # Calculate scaling factors for input coordinates
        scale_x = 64.0 / self._img_width
        scale_y = 64.0 / self._img_height

        for cls_id, bbox in zip(labels.data.numpy(), bboxes.data.numpy()):
            # Scale box coordinates to 64x64 grid
            box_s = bbox * np.array([scale_x, scale_y, scale_x, scale_y])

            # Calculate height and width
            h, w = box_s[3] - box_s[1], box_s[2] - box_s[0]

            # Calculate Gaussian radius based on box size
            rad_w = int(np.round(gaussian_radius([h, w])))
            rad_h = rad_w

            if h > 0 and w > 0:
                # Calculate center point
                center = np.array(
                    [(box_s[0] + box_s[2]) / 2, (box_s[1] + box_s[3]) / 2],
                    dtype=np.float32,
                )
                center = np.round(center)
                # Clip center to ensure it's within bounds
                center = np.clip(
                    center, [0, 0], [63, 63]  # 64x64 grid
                )
                center_int = center.astype(np.int32)

                # Draw Gaussian for current class
                draw_gaussian(hm[..., cls_id - 1], center_int, rad_w, rad_h)

                # Store original bbox coordinates at center point
                coors[center_int[1], center_int[0]] = bbox

        # Concatenate heatmaps and coordinate maps
        return np.concatenate((hm, coors), axis=-1)