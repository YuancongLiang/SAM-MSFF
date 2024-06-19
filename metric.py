import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import surface_distance as surfdist
from skimage.morphology import skeletonize, skeletonize_3d
from skimage.measure import label
from skimage.measure import regionprops
from skimage.morphology import skeletonize
from scipy.spatial.distance import cdist
from scipy.ndimage.morphology import distance_transform_edt as edt
# import ext_libs.Gudhi as gdh
import gudhi

# def betti_numbers(imagely):
#     # Convert the input image from a PyTorch tensor to a numpy array
#     # Get the width and height of the image
#     width, height = imagely.shape
#     # Set the edges of the image to 0
#     imagely[width - 1, :] = 0
#     imagely[:, height - 1] = 0
#     imagely[0, :] = 0
#     imagely[:, 0] = 0
#     # Compute the 0th and 1st Betti numbers
#     betti_0 = len(gdh.compute_persistence_diagram(imagely, i=0))
#     betti_1 = len(gdh.compute_persistence_diagram(imagely, i=1))
#     return betti_0, betti_1

def gudhi_betti_numbers(image):
    img_vector = image.flatten()
    # Create a cubical complex from the image
    cubical_complex = gudhi.CubicalComplex(dimensions=image.shape, top_dimensional_cells=img_vector)
    cubical_complex.persistence()
    betti_numbers = cubical_complex.betti_numbers()
    return betti_numbers[0], betti_numbers[1]

def calculate_betti_numbers(ground_truth, prediction, patch_size=224) -> np.ndarray:
    ground_truth_patch, mask_patch=_extract_same_region(ground_truth, prediction, patch_size=patch_size)
    beta0_error, beta1_error = _calculate_patch_betti_numbers(ground_truth_patch, mask_patch)
    # beta0_mask , beta1_mask = betti_numbers(mask_patch)
    # beta0_ground, beta1_ground = betti_numbers(ground_truth_patch)
    # beta0_mask , beta1_mask = gudhi_betti_numbers(mask_patch)
    # beta0_ground, beta1_ground = gudhi_betti_numbers(ground_truth_patch)
    # beta0_error = abs(beta0_mask - beta0_ground)
    # beta1_error = abs(beta1_mask - beta1_ground)
    return beta0_error, beta1_error

def _extract_same_region(image1, image2, patch_size) -> np.ndarray:
    assert image1.shape == image2.shape, "两张图片的形状必须相同"
    h, w = image1.shape  # 获取图片的高度和宽度
    max_top = h - patch_size  # 允许的顶部最大位置
    max_left = w - patch_size  # 允许的左侧最大位置
    # 生成随机顶部和左侧位置
    top = np.random.randint(0, max_top + 1)
    left = np.random.randint(0, max_left + 1)

    # 从两张图片中提取相同区域
    patch1 = image1[top:top+patch_size, left:left+patch_size]
    patch2 = image2[top:top+patch_size, left:left+patch_size]

    return patch1, patch2

def _calculate_patch_betti_numbers(ground_truth, prediction) -> np.ndarray:
    # 二值化掩码
    prediction_binary = prediction > 0
    ground_truth_binary = ground_truth > 0

    # 计算骨架
    skeleton_prediction = skeletonize(prediction_binary)
    skeleton_ground_truth = skeletonize(ground_truth_binary)

    # 对骨架进行标签化，为每个连通组件分配唯一的标签
    pred_label = label(skeleton_prediction)
    gt_label = label(skeleton_ground_truth)

    # 计算Betti数
    pred_props = regionprops(pred_label)
    gt_props = regionprops(gt_label)

    beta0 = len(pred_props)
    beta1 = 0
    beta0_gt = len(gt_props)
    beta1_gt = 0
    beta0_error = abs(beta0 - beta0_gt)
    for pred_prop in pred_props:
        beta1 += pred_prop.euler_number
    for gt_prop in gt_props:
        beta1_gt += gt_prop.euler_number
    beta1_error = abs(beta1 - beta1_gt)
    return beta0_error, beta1_error

def _cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)

def calculate_cldice(v_l, v_p) -> np.ndarray:
    """[this function computes the cldice metric]

    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    if len(v_p.shape)==2:
        tprec = _cl_score(v_p,skeletonize(v_l))
        tsens = _cl_score(v_l,skeletonize(v_p))
    elif len(v_p.shape)==3:
        tprec = _cl_score(v_p,skeletonize_3d(v_l))
        tsens = _cl_score(v_l,skeletonize_3d(v_p))
    return 2*tprec*tsens/(tprec+tsens)

def calculate_iou(true_mask, predicted_mask) -> np.ndarray:
    intersection = np.logical_and(true_mask, predicted_mask)
    union = np.logical_or(true_mask, predicted_mask)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def calculate_dice_score(true_mask, predicted_mask) -> np.ndarray:
    intersection = np.logical_and(true_mask, predicted_mask)
    dice_score = 2 * np.sum(intersection) / (np.sum(true_mask) + np.sum(predicted_mask))
    return dice_score

def calculate_acc(true_mask, predicted_mask) -> np.ndarray:
    # 将掩码转换为布尔型数组
    gt_mask = np.asarray(true_mask).astype(bool)
    pred_mask = np.asarray(predicted_mask).astype(bool)
    # 计算掩码相等的像素数量
    num_correct_pixels = np.sum(gt_mask == pred_mask)
    # 计算总像素数量
    total_pixels = np.prod(gt_mask.shape)
    # 计算准确率
    accuracy = num_correct_pixels / total_pixels
    return accuracy

def calculate_auc(true_mask, predicted_mask) -> np.ndarray:
    true_mask_flattened = true_mask.flatten()
    predicted_mask_flattened = predicted_mask.flatten()
    auc = roc_auc_score(true_mask_flattened,predicted_mask_flattened)
    return auc


def calculate_hausdorff(true_mask, predicted_mask) -> np.ndarray:
    true_mask_bool, predicted_mask_bool = true_mask.copy().astype(bool), predicted_mask.copy().astype(bool)
    surface_distances = surfdist.compute_surface_distances(true_mask_bool, predicted_mask_bool, spacing_mm=(1.0, 1.0))
    hd_dist_95 = surfdist.compute_robust_hausdorff(surface_distances, 95)
    return hd_dist_95

def _hd_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:

    if np.count_nonzero(x) == 0 or np.count_nonzero(y) == 0:
        return np.array([np.Inf])

    indexes = np.nonzero(x)
    distances = edt(np.logical_not(y))

    return np.array(np.max(distances[indexes]))

# 已弃用
def _calculate_hausdorff(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

    pred = (pred > 0.5).astype(bool)
    target = (target > 0.5).astype(bool)

    right_hd = torch.from_numpy(
        _hd_distance(pred, target)
    ).float()

    left_hd = torch.from_numpy(
        _hd_distance(target, pred)
    ).float()

    return torch.max(right_hd, left_hd).numpy()