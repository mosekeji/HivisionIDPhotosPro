from functools import lru_cache
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import onnxruntime

from creator.face.face_detector import detect_faces_yunet
from tools.exceptions import ModelNotFoundError
from tools.model_files import get_model_files


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = PROJECT_ROOT / "tools" / "clothes" / "assets"
CLOTHES_COUNTS = {
    1: 17,
    2: 17,
    3: 9,
}
# 换装使用YuNet的人脸框和关键点坐标
CLOTHES_FACE_DETECTORS = {
    "yunet": detect_faces_yunet,
}
CLOTHES_PARSING_MODELS = {
    "selfieMulticlass",
}
CLOTHES_PARSING_SESSION = None
CLOTHES_PARSING_MODEL_OPTION = None
CLOTHES_PARSING_LOCK = Lock()
MIN_PERSON_SCALE = 0.88
MAX_PERSON_SCALE = 1.15
MIN_CLOTHES_DETAIL_WIDTH = 512


def _get_clothes_path(category: int, clothes_id: int) -> Path:
    """根据服装分类和编号返回项目内置透明服装模板"""
    if category not in CLOTHES_COUNTS:
        raise ValueError("服装分类无效")

    if clothes_id < 1 or clothes_id > CLOTHES_COUNTS[category]:
        raise ValueError("服装不存在")

    clothes_path = ASSETS_DIR / str(category) / f"{clothes_id:02d}.png"
    if not clothes_path.is_file():
        raise ValueError("服装不存在")
    return clothes_path


def _alpha_composite(base_image: np.ndarray, overlay_image: np.ndarray) -> np.ndarray:
    """把同尺寸的两个BGRA图层按照Alpha通道合成"""
    base = base_image.astype(np.float32) / 255.0
    overlay = overlay_image.astype(np.float32) / 255.0

    base_alpha = base[:, :, 3:4]
    overlay_alpha = overlay[:, :, 3:4]
    output_alpha = overlay_alpha + base_alpha * (1.0 - overlay_alpha)
    premultiplied_color = (
        overlay[:, :, :3] * overlay_alpha
        + base[:, :, :3] * base_alpha * (1.0 - overlay_alpha)
    )
    output_color = np.divide(
        premultiplied_color,
        output_alpha,
        out=np.zeros_like(premultiplied_color),
        where=output_alpha > 0,
    )
    return np.uint8(
        np.clip(np.concatenate((output_color, output_alpha), axis=2) * 255.0, 0, 255)
    )


def _resize_bgra(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """使用预乘Alpha缩放透明图层"""
    normalized = image.astype(np.float32) / 255.0
    alpha = normalized[:, :, 3:4]
    premultiplied_color = normalized[:, :, :3] * alpha
    resized_color = cv2.resize(
        premultiplied_color,
        (width, height),
        interpolation=cv2.INTER_LANCZOS4,
    )
    resized_alpha = cv2.resize(
        alpha,
        (width, height),
        interpolation=cv2.INTER_LANCZOS4,
    )
    if resized_alpha.ndim == 2:
        resized_alpha = resized_alpha[:, :, None]

    output_color = np.divide(
        resized_color,
        resized_alpha,
        out=np.zeros_like(resized_color),
        where=resized_alpha > 0,
    )
    return np.uint8(
        np.clip(
            np.concatenate((output_color, resized_alpha), axis=2) * 255.0,
            0,
            255,
        )
    )


def _scale_person_to_target(
    input_image: np.ndarray,
    face: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """根据证件照规格统一双眼距离和眼线位置"""
    image_height, image_width = input_image.shape[:2]
    aspect_ratio = image_height / image_width
    tall_ratio = float(np.clip((aspect_ratio - 1.30) / 0.08, 0.0, 1.0))
    tall_ratio = tall_ratio * tall_ratio * (3.0 - 2.0 * tall_ratio)
    target_eye_distance_ratio = 0.21 + tall_ratio * 0.022

    target_eye_y_ratio = 0.43521107080196353

    right_eye_x, right_eye_y = float(face[4]), float(face[5])
    left_eye_x, left_eye_y = float(face[6]), float(face[7])
    eye_center_x = (right_eye_x + left_eye_x) / 2.0
    eye_center_y = (right_eye_y + left_eye_y) / 2.0
    current_eye_distance = abs(left_eye_x - right_eye_x)
    scale = float(
        np.clip(
            image_width * target_eye_distance_ratio / current_eye_distance,
            MIN_PERSON_SCALE,
            MAX_PERSON_SCALE,
        )
    )

    # 根据主体头顶位置保留留白
    _, _, components, _ = cv2.connectedComponentsWithStats(
        np.uint8(input_image[:, :, 3] > 127),
        connectivity=8,
    )
    person_component = 1 + np.argmax(components[1:, cv2.CC_STAT_AREA])
    head_top = float(components[person_component, cv2.CC_STAT_TOP])
    head_margin = min(head_top, image_height * 0.10)
    scale = min(
        scale,
        (image_height * target_eye_y_ratio - head_margin) / (eye_center_y - head_top),
    )
    offset_x = image_width / 2.0 - eye_center_x * scale
    offset_y = image_height * target_eye_y_ratio - eye_center_y * scale
    if (
        abs(scale - 1.0) < 0.005
        and abs(offset_x) < 0.5
        and abs(offset_y) < 0.5
    ):
        return input_image, face

    transform = np.array(
        [
            [scale, 0.0, offset_x],
            [0.0, scale, offset_y],
        ],
        dtype=np.float32,
    )

    normalized = input_image.astype(np.float32) / 255.0
    alpha = normalized[:, :, 3:4]
    premultiplied = np.concatenate((normalized[:, :, :3] * alpha, alpha), axis=2)
    scaled = cv2.warpAffine(
        premultiplied,
        transform,
        (image_width, image_height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    scaled_alpha = scaled[:, :, 3:4]
    scaled_color = np.divide(
        scaled[:, :, :3],
        scaled_alpha,
        out=np.zeros_like(scaled[:, :, :3]),
        where=scaled_alpha > 0,
    )
    scaled_image = np.uint8(
        np.clip(np.concatenate((scaled_color, scaled_alpha), axis=2) * 255.0, 0, 255)
    )

    scaled_face = face.copy().astype(np.float32)
    scaled_face[0] = scaled_face[0] * scale + offset_x
    scaled_face[1] = scaled_face[1] * scale + offset_y
    scaled_face[2] *= scale
    scaled_face[3] *= scale
    for index in range(4, 14, 2):
        scaled_face[index] = scaled_face[index] * scale + offset_x
        scaled_face[index + 1] = scaled_face[index + 1] * scale + offset_y
    return scaled_image, scaled_face


def _get_parsing_session(parsing_model: str):
    """加载纯CPU服装人体解析模型并复用推理会话"""
    global CLOTHES_PARSING_SESSION, CLOTHES_PARSING_MODEL_OPTION

    with CLOTHES_PARSING_LOCK:
        if (
            CLOTHES_PARSING_SESSION is None
            or CLOTHES_PARSING_MODEL_OPTION != parsing_model
        ):
            model_path, = get_model_files(parsing_model, "selfie_multiclass.onnx")
            session_options = onnxruntime.SessionOptions()
            session_options.log_severity_level = 3
            CLOTHES_PARSING_SESSION = onnxruntime.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
            CLOTHES_PARSING_MODEL_OPTION = parsing_model
    return CLOTHES_PARSING_SESSION


def _create_parsing_probabilities(
    input_image: np.ndarray,
    parsing_model: str,
) -> np.ndarray:
    """返回与输入图片同尺寸的六分类人体解析概率"""
    session = _get_parsing_session(parsing_model)
    rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGRA2RGB)
    model_image = cv2.resize(rgb_image, (256, 256), interpolation=cv2.INTER_AREA)
    model_input = model_image.astype(np.float32)[None, :, :, :] / 255.0
    logits = session.run(
        None,
        {session.get_inputs()[0].name: model_input},
    )[0][0]
    logits = logits - np.max(logits, axis=2, keepdims=True)
    probabilities = np.exp(logits)
    probabilities = probabilities / np.sum(probabilities, axis=2, keepdims=True)

    image_height, image_width = input_image.shape[:2]
    return cv2.resize(
        probabilities,
        (image_width, image_height),
        interpolation=cv2.INTER_LINEAR,
    )


def _find_center_span(row_mask: np.ndarray, center_x: int, search_radius: int):
    """返回当前行最靠近人物中心的连续有效区间"""
    image_width = row_mask.shape[0]
    center_x = int(np.clip(center_x, 0, image_width - 1))
    if row_mask[center_x]:
        seed_x = center_x
    else:
        left = max(0, center_x - search_radius)
        right = min(image_width, center_x + search_radius + 1)
        candidates = np.flatnonzero(row_mask[left:right]) + left
        if candidates.size == 0:
            return None
        seed_x = int(candidates[np.argmin(np.abs(candidates - center_x))])

    span_left = seed_x
    span_right = seed_x
    while span_left > 0 and row_mask[span_left - 1]:
        span_left -= 1
    while span_right + 1 < image_width and row_mask[span_right + 1]:
        span_right += 1
    return span_left, span_right


def _measure_neck_geometry(
    input_image: np.ndarray,
    face: np.ndarray,
    parsing_probabilities: np.ndarray,
) -> dict:
    """以嘴部关键点约束下巴位置，测量连续可用颈部和皮肤下沿"""
    image_height, image_width = input_image.shape[:2]
    face_x, face_y, face_width, face_height = [float(value) for value in face[:4]]
    center_x = face_x + face_width / 2.0
    face_bottom = face_y + face_height
    mouth_y = float((face[11] + face[13]) / 2.0)
    landmark_chin_y = float(
        np.clip(
            mouth_y + face_height * 0.21,
            face_y + face_height * 0.78,
            face_bottom,
        )
    )

    band_left = max(0, round(center_x - face_width * 0.14))
    band_right = min(image_width, round(center_x + face_width * 0.14) + 1)
    face_probability = parsing_probabilities[:, :, 3]
    body_probability = parsing_probabilities[:, :, 2]
    skin_probability = np.clip(face_probability + body_probability, 0.0, 1.0)
    original_alpha = input_image[:, :, 3].astype(np.float32) / 255.0

    transition_start = int(
        np.clip(round(face_y + face_height * 0.75), 0, image_height - 1)
    )
    transition_end = int(
        np.clip(
            round(face_bottom + face_height * 0.08),
            transition_start,
            image_height - 1,
        )
    )
    parsed_chin_y = landmark_chin_y
    transition_rows = 0
    for row in range(transition_start, transition_end + 1):
        row_face = np.mean(face_probability[row, band_left:band_right])
        row_body = np.mean(body_probability[row, band_left:band_right])
        row_alpha = np.mean(original_alpha[row, band_left:band_right])
        if row_face < 0.15 and row_body > 0.60 and row_alpha > 0.50:
            transition_rows += 1
            if transition_rows >= 2:
                parsed_chin_y = float(row - transition_rows + 1)
                break
        else:
            transition_rows = 0

    chin_y = float(
        np.clip(
            max(landmark_chin_y, min(parsed_chin_y, landmark_chin_y + face_height * 0.04)),
            face_y + face_height * 0.78,
            face_bottom + face_height * 0.04,
        )
    )

    scan_start = int(np.clip(round(chin_y), 0, image_height - 1))
    scan_end = int(
        np.clip(round(chin_y + face_height * 0.60), scan_start, image_height - 1)
    )
    skin_bottom = scan_start
    missing_rows = 0
    minimum_rows = max(2, round(face_height * 0.008))
    for row in range(scan_start, scan_end + 1):
        row_skin = skin_probability[row, band_left:band_right]
        row_clothes = parsing_probabilities[row, band_left:band_right, 4]
        row_alpha = original_alpha[row, band_left:band_right]
        if np.mean((row_skin > 0.45) & (row_skin > row_clothes) & (row_alpha > 0.50)) >= 0.55:
            skin_bottom = row
            missing_rows = 0
        else:
            missing_rows += 1
            if missing_rows >= minimum_rows:
                break

    neck_length = 0
    continuous_rows = 0
    for row in range(scan_start, skin_bottom + 1):
        row_body = body_probability[row, band_left:band_right]
        row_clothes = parsing_probabilities[row, band_left:band_right, 4]
        row_alpha = original_alpha[row, band_left:band_right]
        if np.mean((row_body > 0.45) & (row_body > row_clothes) & (row_alpha > 0.50)) >= 0.55:
            continuous_rows += 1
            neck_length = max(neck_length, continuous_rows)
        else:
            continuous_rows = 0

    skin_outline = np.uint8((skin_probability > 0.45) & (original_alpha > 0.50))
    radius = max(1, round(face_width * 0.015))
    skin_outline = cv2.morphologyEx(
        skin_outline,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)),
    )
    neck_spans = []
    for row in range(
        max(0, round(chin_y - face_height * 0.28)),
        min(image_height, round(chin_y - face_height * 0.06)),
    ):
        span = _find_center_span(skin_outline[row], round(center_x), round(face_width * 0.12))
        if span is not None and span[1] - span[0] > face_width * 0.35:
            neck_spans.append((span[1] - span[0], row, span[0], span[1]))
    neck_left = None
    neck_right = None
    if neck_spans:
        _, _, neck_left, neck_right = min(neck_spans)
        neck_left = float(neck_left)
        neck_right = float(neck_right)

    return {
        "chin_y": chin_y,
        "skin_bottom": float(skin_bottom),
        "neck_length": float(neck_length),
        "face_height": face_height,
        "retained_probability": (
            np.sum(parsing_probabilities[:, :, [1, 2, 3, 5]], axis=2)
        ),
        "neck_left": neck_left,
        "neck_right": neck_right,
        "skin_probability": skin_probability,
        "face_probability": face_probability,
        "clothes_probability": parsing_probabilities[:, :, 4],
        "hair_probability": parsing_probabilities[:, :, 1],
    }


def _bounded_opening_span(alpha_row, center_x, previous_span):
    """只追踪两侧均有真实衣料包围的中央透明区"""
    if alpha_row[center_x] > 32:
        if previous_span is None:
            return None
        candidates = np.flatnonzero(alpha_row[previous_span[0]:previous_span[1] + 1] <= 32)
        if candidates.size == 0:
            return None
        candidates += previous_span[0]
        center_x = int(candidates[np.argmin(np.abs(candidates - center_x))])
    left = center_x
    right = center_x
    while left > 0 and alpha_row[left - 1] <= 32:
        left -= 1
    while right + 1 < len(alpha_row) and alpha_row[right + 1] <= 32:
        right += 1
    if left == 0 or right == len(alpha_row) - 1:
        return None
    return left, right


def _sample_mask(mask, count):
    """按规则网格采样区域内部，返回素材像素坐标x和y"""
    area = int(np.count_nonzero(mask))
    step = max(1, int(np.sqrt(area / count)))
    rows, columns = np.mgrid[0:mask.shape[0]:step, 0:mask.shape[1]:step]
    selected = mask[rows, columns] > 0
    points = np.column_stack((columns[selected], rows[selected])).astype(np.float32)
    if len(points) > count:
        points = points[np.linspace(0, len(points) - 1, count).astype(np.int32)]
    return points


def _sample_boundary(mask, count):
    """均匀采样真实区域边界"""
    contours, _ = cv2.findContours(np.uint8(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.empty((0, 2), dtype=np.float32)
    points = np.concatenate([contour[:, 0] for contour in contours]).astype(np.float32)
    if len(points) > count:
        points = points[np.linspace(0, len(points) - 1, count).astype(np.int32)]
    return points


def _analyze_clothes(clothes):
    """抽取领座以下的完整中央开口及衣料碰撞样本"""
    height, width = clothes.shape[:2]
    alpha = clothes[:, :, 3]
    _, opaque_columns = np.where(alpha > 32)
    if opaque_columns.size == 0:
        raise ValueError("服装图片无效")
    center_x = int(round((opaque_columns.min() + opaque_columns.max()) / 2.0))
    opening_mask = np.zeros((height, width), dtype=np.uint8)
    spans = []
    previous = None
    closed_rows = 0
    for row in range(height):
        span = _bounded_opening_span(alpha[row], center_x, previous)
        if span is None:
            if spans:
                closed_rows += 1
                if closed_rows >= 2:
                    break
            continue
        if not spans and not width * 0.08 <= span[1] - span[0] + 1 <= width * 0.45:
            continue
        previous = span
        closed_rows = 0
        spans.append((row, span[0], span[1]))
        opening_mask[row, span[0]:span[1] + 1] = 1
    if len(spans) < 3:
        raise ValueError("服装领口识别失败")
    top_spans = spans[:min(4, len(spans))]
    opaque_mask = np.uint8(alpha > 200)
    upper_mask = opaque_mask.copy()
    upper_mask[min(height, max(spans[-1][0] + 1, round(height * 0.45))):] = 0
    return {
        "opening_mask": opening_mask,
        "opening_points": _sample_mask(opening_mask, 384),
        "opening_boundary_points": _sample_boundary(opening_mask, 256),
        "garment_points": _sample_mask(opaque_mask, 256),
        "garment_boundary_points": _sample_boundary(upper_mask, 256),
        "opaque_mask": opaque_mask,
        "center_x": float(center_x),
        "opening_left": float(np.median([span[1] for span in top_spans])),
        "opening_right": float(np.median([span[2] for span in top_spans])),
        "opening_top": float(spans[0][0]),
        "opening_bottom": float(spans[-1][0] + 1),
        "spans": np.asarray(spans, dtype=np.int32),
    }


def _render_garment(clothes, input_shape, scale, left, top, protect_y):
    """领部按比例放置，保护线以下以C1连续方式纵向延展衣身"""
    height, width = input_shape[:2]
    source_height, _ = clothes.shape[:2]
    if scale <= 0 or not 0 <= protect_y < source_height - 1:
        raise ValueError("服装放置参数无效")
    extra = max(0.0, height + 2.0 - (top + scale * (source_height - 1)))
    remaining = source_height - 1.0 - protect_y
    rows = np.arange(height, dtype=np.float64)
    relative = rows - top - scale * protect_y
    source_rows = (rows - top) / scale
    if extra > 0:
        coefficient = extra / (remaining * remaining)
        below = relative > 0
        # 用有理化正根计算纵向映射，保持小延展量下的数值稳定性
        source_rows[below] = protect_y + (
            2.0 * relative[below]
            / (scale + np.sqrt(scale * scale + 4.0 * coefficient * relative[below]))
        )
    source_columns = (np.arange(width, dtype=np.float32) - left) / scale
    map_x = np.broadcast_to(source_columns[None, :], (height, width)).copy()
    map_y = np.broadcast_to(source_rows.astype(np.float32)[:, None], (height, width)).copy()
    normalized = clothes.astype(np.float32) / 255.0
    normalized[:, :, :3] *= normalized[:, :, 3:4]
    remapped = cv2.remap(
        normalized, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    remapped[:, :, :3] = np.divide(
        remapped[:, :, :3], remapped[:, :, 3:4],
        out=np.zeros_like(remapped[:, :, :3]),
        where=remapped[:, :, 3:4] > 0,
    )
    return np.uint8(np.clip(np.rint(remapped * 255.0), 0, 255))


@lru_cache(maxsize=sum(CLOTHES_COUNTS.values()))
def _load_clothes_asset(category: int, clothes_id: int) -> tuple[np.ndarray, dict]:
    """读取并缓存内置服装素材及几何信息"""
    clothes_path = _get_clothes_path(category, clothes_id)
    clothes_image = cv2.imread(str(clothes_path), cv2.IMREAD_UNCHANGED)
    if clothes_image is None or clothes_image.ndim != 3 or clothes_image.shape[2] != 4:
        raise ValueError("服装图片无效")
    return clothes_image, _analyze_clothes(clothes_image)


def _create_face_protection_mask(
    input_image: np.ndarray,
    face: np.ndarray,
    parsing_probabilities: np.ndarray,
    chin_y: float,
) -> np.ndarray:
    """生成下巴和脸部皮肤的保护蒙版"""
    image_height, image_width = input_image.shape[:2]
    face_x, face_y, face_width, face_height = [float(value) for value in face[:4]]
    face_probability = parsing_probabilities[:, :, 3]
    mask = np.clip((face_probability - 0.08) / 0.72, 0.0, 1.0)

    geometry_mask = np.zeros((image_height, image_width), dtype=np.float32)
    geometry_left = int(np.clip(face_x - face_width * 0.08, 0, image_width))
    geometry_right = int(np.clip(face_x + face_width * 1.08, 0, image_width))
    geometry_top = int(np.clip(face_y - face_height * 0.05, 0, image_height))
    geometry_bottom = int(np.clip(chin_y + 2, 0, image_height))
    geometry_mask[
        geometry_top:geometry_bottom + 1,
        geometry_left:geometry_right,
    ] = 1.0
    mask *= geometry_mask
    mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8))
    return cv2.GaussianBlur(mask, (0, 0), 0.45)


def _match_clothes_detail(
    clothes_layer: np.ndarray,
    input_image: np.ndarray,
    face: np.ndarray,
) -> np.ndarray:
    """按柔焦人像的清晰度减弱服装细节"""
    face_x, face_y, face_width, face_height = [float(value) for value in face[:4]]
    face_image = input_image[
        max(0, int(face_y)):min(input_image.shape[0], int(face_y + face_height)),
        max(0, int(face_x)):min(input_image.shape[1], int(face_x + face_width)),
        :3,
    ]
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.resize(
        gray,
        (128, round(128 * gray.shape[0] / gray.shape[1])),
        interpolation=cv2.INTER_AREA if gray.shape[1] > 128 else cv2.INTER_CUBIC,
    )
    gray = cv2.GaussianBlur(gray, (0, 0), 0.8)
    blurred_gray = cv2.GaussianBlur(gray, (0, 0), 1.2)
    gradients = [
        cv2.magnitude(
            cv2.Scharr(image, cv2.CV_32F, 1, 0),
            cv2.Scharr(image, cv2.CV_32F, 0, 1),
        )
        for image in (gray, blurred_gray)
    ]
    interior = np.zeros(gray.shape, dtype=bool)
    interior[int(gray.shape[0] * 0.12):int(gray.shape[0] * 0.90), 20:108] = True
    strong_edges = interior & (gradients[0] >= np.percentile(gradients[0][interior], 80))
    edge_strength = float(gradients[0][strong_edges].sum())
    if edge_strength <= strong_edges.sum():
        return clothes_layer
    detail = 1.0 - float(gradients[1][strong_edges].sum()) / edge_strength
    if detail >= 0.16:
        return clothes_layer
    sigma = min(2.0, np.sqrt(4.0 * (0.16 / max(detail, 0.01) - 1.0)))
    sigma *= face_width / 128.0
    if sigma < 0.25:
        return clothes_layer
    normalized = clothes_layer.astype(np.float32) / 255.0
    normalized[:, :, :3] *= normalized[:, :, 3:4]
    blurred = cv2.GaussianBlur(normalized, (0, 0), sigma)
    blurred[:, :, :3] = np.divide(
        blurred[:, :, :3],
        blurred[:, :, 3:4],
        out=np.zeros_like(blurred[:, :, :3]),
        where=blurred[:, :, 3:4] > 0,
    )
    return np.uint8(np.clip(blurred * 255.0, 0, 255))


def _create_head_layer(
    input_image: np.ndarray,
    face: np.ndarray,
    parsing_probabilities: np.ndarray,
    chin_y: float,
) -> np.ndarray:
    """提取原始头部、头发和饰品供前后景分层"""
    image_height = input_image.shape[0]
    _, _, _, face_height = [float(value) for value in face[:4]]
    rows = np.arange(image_height, dtype=np.float32)
    face_fade_start = chin_y - face_height * 0.01
    face_fade_end = chin_y + face_height * 0.025
    face_keep = 1.0 - np.clip(
        (rows - face_fade_start) / max(face_fade_end - face_fade_start, 1.0),
        0.0,
        1.0,
    )
    face_keep = face_keep * face_keep * (3.0 - 2.0 * face_keep)
    head_probability = (
        parsing_probabilities[:, :, 1]
        + parsing_probabilities[:, :, 3] * face_keep[:, None]
        + parsing_probabilities[:, :, 5]
    )
    head_support = head_probability > 0.50
    lower_head_share = head_probability / (
        head_probability + parsing_probabilities[:, :, 2] + parsing_probabilities[:, :, 4]
    )
    # 按衣物类别概率清理颈肩区域
    clothes_rejection = np.clip(
        (parsing_probabilities[:, :, 4] - head_probability) / 0.20,
        0.0,
        1.0,
    )
    clothes_rejection[rows < chin_y - face_height * 0.10] = 0.0
    # 在人物内部类别中归一化头部概率
    head_probability /= np.sum(parsing_probabilities[:, :, 1:], axis=2)
    mask = np.clip((head_probability - 0.08) / 0.72, 0.0, 1.0)
    mask *= 1.0 - clothes_rejection
    # 下巴以下按人物内部类别保留头发
    mask[rows >= chin_y] *= np.clip((lower_head_share[rows >= chin_y] - 0.25) / 0.25, 0.0, 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), 0.6)

    head_layer = input_image.copy()
    original_alpha = input_image[:, :, 3].astype(np.float32) / 255.0
    head_layer[:, :, 3] = np.uint8(np.clip(original_alpha * mask * 255.0, 0, 255))
    # 移除缺少头部类别支撑的独立碎片
    _, components = cv2.connectedComponents(np.uint8(head_layer[:, :, 3] > 0))
    head_components = np.unique(components[head_support & (head_layer[:, :, 3] > 0)])
    head_layer[:, :, 3][~np.isin(components, head_components)] = 0
    return head_layer


def _create_face_protection_layer(
    input_image: np.ndarray,
    face: np.ndarray,
    parsing_probabilities: np.ndarray,
    chin_y: float,
) -> np.ndarray:
    """恢复脸部保护区域内的原始像素"""
    mask = _create_face_protection_mask(
        input_image,
        face,
        parsing_probabilities,
        chin_y,
    )
    image_height, image_width = input_image.shape[:2]
    face_x, face_y, face_width, face_height = [float(value) for value in face[:4]]
    center_x = face_x + face_width / 2.0
    jaw_mask = np.zeros((image_height, image_width), dtype=np.float32)
    jaw_polygon = np.array(
        [
            [center_x - face_width * 0.48, face_y + face_height * 0.40],
            [center_x + face_width * 0.48, face_y + face_height * 0.40],
            [center_x + face_width * 0.44, face_y + face_height * 0.58],
            [center_x + face_width * 0.34, face_y + face_height * 0.76],
            [center_x + face_width * 0.18, chin_y - face_height * 0.03],
            [center_x, chin_y + 2],
            [center_x - face_width * 0.18, chin_y - face_height * 0.03],
            [center_x - face_width * 0.34, face_y + face_height * 0.76],
            [center_x - face_width * 0.44, face_y + face_height * 0.58],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(jaw_mask, jaw_polygon, 1.0)
    jaw_mask = cv2.GaussianBlur(jaw_mask, (0, 0), 0.55)
    mask = np.maximum(mask, jaw_mask)
    face_layer = input_image.copy()
    original_alpha = input_image[:, :, 3].astype(np.float32) / 255.0
    face_layer[:, :, 3] = np.uint8(
        np.clip(original_alpha * mask * 255.0, 0, 255)
    )
    return face_layer


def _create_clothes_layer(
    input_image: np.ndarray,
    clothes_image: np.ndarray,
    clothes_geometry: dict,
    face: np.ndarray,
    neck_geometry: dict,
) -> tuple[np.ndarray, dict]:
    """在可见皮肤范围内搜索完整领口的放置位置"""
    height, width = input_image.shape[:2]
    face_width, face_height = float(face[2]), float(face[3])
    chin_y = neck_geometry["chin_y"]
    neck_width = neck_geometry["neck_right"] - neck_geometry["neck_left"]
    center_x = (neck_geometry["neck_right"] + neck_geometry["neck_left"]) / 2.0
    opening_width = clothes_geometry["opening_right"] - clothes_geometry["opening_left"]
    opening_center = (clothes_geometry["opening_right"] + clothes_geometry["opening_left"]) / 2.0
    opening_depth = clothes_geometry["opening_bottom"] - clothes_geometry["opening_top"]
    support = np.uint8(neck_geometry["opening_support"] > 0)
    outside = cv2.distanceTransform(1 - support, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    points = np.concatenate((clothes_geometry["opening_points"], clothes_geometry["opening_boundary_points"]))
    scales = neck_width / opening_width * np.linspace(0.78, 1.30, 27)
    centers = center_x + np.linspace(-0.08, 0.08, 17) * face_width
    tops = chin_y + np.linspace(-0.16, 0.08, 25) * face_height
    scale, center, opening_top = np.meshgrid(scales, centers, tops, indexing="ij")
    scale, center, opening_top = scale.ravel(), center.ravel(), opening_top.ravel()
    left = center - opening_center * scale
    top = opening_top - clothes_geometry["opening_top"] * scale
    bottom = top + clothes_geometry["opening_bottom"] * scale
    x = np.rint(left[:, None] + scale[:, None] * points[None, :, 0]).astype(np.int32)
    y = np.rint(top[:, None] + scale[:, None] * points[None, :, 1]).astype(np.int32)
    distances = outside[np.clip(y, 0, height - 1), np.clip(x, 0, width - 1)]
    distances[(x < 0) | (x >= width) | (y < 0) | (y >= height)] = max(height, width)
    worst_gap = distances.max(axis=1)
    mean_gap = distances.mean(axis=1)
    collar_depth = opening_depth * scale
    target_bottom = chin_y + np.maximum(face_height * 0.18, collar_depth - face_height * 0.04)
    target_bottom = np.minimum(target_bottom, neck_geometry["skin_bottom"] - face_height * 0.04)
    preferred_scale = neck_width / opening_width
    score = (
        3.0 * ((scale / preferred_scale - 1.0) ** 2)
        + ((bottom - target_bottom) / (face_height * 0.12)) ** 2
        + ((center - center_x) / (face_width * 0.12)) ** 2
        + 6.0 * (mean_gap / max(1.0, face_height * 0.01)) ** 2
    )
    # 按实际皮肤边界和前景头发约束领座可见端
    skin_support = np.uint8(neck_geometry["skin_mask"] > 0.5)
    left_boundary = np.zeros(height, dtype=np.float32)
    right_boundary = np.zeros(height, dtype=np.float32)
    valid_rows = np.zeros(height, dtype=bool)
    for row in range(height):
        span = _find_center_span(skin_support[row], round(center_x), 0)
        if span is not None:
            left_boundary[row], right_boundary[row] = span
            valid_rows[row] = True
    seam_x = left[:, None] + scale[:, None] * np.array([
        clothes_geometry["opening_left"], clothes_geometry["opening_right"],
    ])[None, :]
    seam_y = np.rint(opening_top).astype(np.int32)
    seam_inside = np.all((seam_x >= 0) & (seam_x < width), axis=1) & (seam_y >= 0) & (seam_y < height)
    seam_columns = np.clip(np.rint(seam_x).astype(np.int32), 0, width - 1)
    seam_rows = np.clip(seam_y, 0, height - 1)
    seam_hair = (
        (neck_geometry["hair_probability"][seam_rows[:, None], seam_columns] > 0.5)
        & (neck_geometry["head_alpha"][seam_rows[:, None], seam_columns] > 0.90)
    )
    seam_gap = np.abs(seam_x - np.column_stack((left_boundary[seam_rows], right_boundary[seam_rows])))
    seam_gap[seam_hair] = 0.0
    attached = seam_inside & valid_rows[seam_rows] & np.all(seam_gap <= max(2.0, face_width * 0.015), axis=1)
    # 校验中央脸部与衣料的重叠范围
    rows, columns = np.ogrid[:height, :width]
    face_core = (
        (neck_geometry["face_probability"] > 0.75)
        & (rows < chin_y - face_height * 0.08)
        & (columns > float(face[0]) + face_width * 0.22)
        & (columns < float(face[0]) + face_width * 0.78)
    )
    garment_points = np.concatenate((clothes_geometry["garment_points"], clothes_geometry["garment_boundary_points"]))
    garment_x = np.rint(left[:, None] + scale[:, None] * garment_points[None, :, 0]).astype(np.int32)
    garment_y = np.rint(top[:, None] + scale[:, None] * garment_points[None, :, 1]).astype(np.int32)
    garment_in_frame = (garment_x >= 0) & (garment_x < width) & (garment_y >= 0) & (garment_y < height)
    collision = np.any(
        garment_in_frame & face_core[np.clip(garment_y, 0, height - 1), np.clip(garment_x, 0, width - 1)],
        axis=1,
    )
    feasible = (
        attached & ~collision
        & (worst_gap <= max(1.0, face_width * 0.006))
        & (bottom >= chin_y + face_height * 0.12)
        & (bottom <= neck_geometry["skin_bottom"] - face_height * 0.015)
    )
    score[~feasible] = np.inf
    chosen = int(np.argmin(score))
    if not np.isfinite(score[chosen]):
        raise ValueError("服装领口超过可见皮肤范围，请选择其它服装")
    source_height, source_width = clothes_image.shape[:2]
    protect_y = max(source_height * 0.66, clothes_geometry["opening_bottom"] + source_height * 0.10)
    clothes_layer = _render_garment(
        clothes_image, input_image.shape, float(scale[chosen]), float(left[chosen]),
        float(top[chosen]), min(protect_y, source_height - 2.0),
    )
    if np.any(face_core & (clothes_layer[:, :, 3] > 200)):
        raise ValueError("服装结构与当前人物比例不匹配")
    if source_width >= MIN_CLOTHES_DETAIL_WIDTH:
        clothes_layer = _match_clothes_detail(clothes_layer, input_image, face)
    transform = np.float32([[scale[chosen], 0.0, left[chosen]], [0.0, scale[chosen], top[chosen]]])
    opening_mask = cv2.warpAffine(
        clothes_geometry["opening_mask"].astype(np.float32), transform, (width, height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
    )
    neck_geometry["opening_mask"] = opening_mask
    neck_geometry["opening_top_y"] = float(opening_top[chosen])
    neck_geometry["opening_left_x"] = float(left[chosen] + clothes_geometry["opening_left"] * scale[chosen])
    neck_geometry["opening_right_x"] = float(left[chosen] + clothes_geometry["opening_right"] * scale[chosen])
    return clothes_layer, {
        "body_extension": max(0.0, height + 2.0 - (top[chosen] + scale[chosen] * (source_height - 1))),
        "protected_source_y": float(protect_y),
        "opening_mask": opening_mask,
        "opening_left_x": neck_geometry["opening_left_x"],
        "opening_right_x": neck_geometry["opening_right_x"],
        "opening_top_y": float(opening_top[chosen]),
        "collar_y": float(bottom[chosen]),
        "scale": float(scale[chosen]),
        "worst_gap": float(worst_gap[chosen]),
        "seam_gap": float(seam_gap[chosen].max()),
        "score": float(score[chosen]),
        "feasible_candidates": int(feasible.sum()),
    }


def _create_neck_mask(input_image: np.ndarray, neck_geometry: dict) -> np.ndarray:
    """按类别概率和中心连通截面建立搜索与合成共用的颈部保留域"""
    height = input_image.shape[0]
    skin = neck_geometry["skin_probability"]
    cloth = neck_geometry["clothes_probability"]
    original = input_image[:, :, 3].astype(np.float32) / 255.0
    region = np.uint8((skin > cloth) & (skin > neck_geometry["hair_probability"]) & (original > 0.5))
    _, labels = cv2.connectedComponents(region)
    center = round((neck_geometry["neck_left"] + neck_geometry["neck_right"]) / 2.0)
    seed_row = int(np.clip(round(neck_geometry["chin_y"]), 0, height - 1))
    seed = labels[seed_row, center]
    region = np.float32((labels == seed) & (seed != 0))
    # 只保留穿过颈部中心的横截面
    for row in range(max(0, round(neck_geometry["chin_y"] - neck_geometry["face_height"] * 0.25)), height):
        span = _find_center_span(region[row] > 0, center, 0)
        if span is None:
            region[row] = 0.0
        else:
            region[row, :span[0]] = 0.0
            region[row, span[1] + 1:] = 0.0
    region = cv2.GaussianBlur(region, (0, 0), 0.55)
    return region


def _refine_neck_boundary(
    input_image: np.ndarray,
    neck_layer: np.ndarray,
    neck_geometry: dict,
    clothes_layer: np.ndarray,
) -> np.ndarray:
    """用同一行真实皮肤和旧衣颜色精修有歧义的颈部窄边"""
    result = neck_layer.copy()
    height, width = input_image.shape[:2]
    skin = neck_geometry["skin_probability"]
    cloth = neck_geometry["clothes_probability"]
    hair = neck_geometry["hair_probability"]
    radius = max(1, round(neck_geometry["face_height"] * 0.005))
    center = round((neck_geometry["neck_left"] + neck_geometry["neck_right"]) / 2.0)
    top = max(0, round(neck_geometry["chin_y"] - neck_geometry["face_height"] * 0.25))
    bottom = min(height, round(neck_geometry["skin_bottom"]) + 1)
    for row in range(top, bottom):
        span = _find_center_span(neck_geometry["skin_mask"][row] > 0.5, center, 0)
        if span is None:
            continue
        for boundary, direction in ((span[0], 1), (span[1], -1)):
            for column in range(max(0, boundary - radius), min(width, boundary + radius + 1)):
                if (
                    neck_layer[row, column, 3] == 0
                    or clothes_layer[row, column, 3] >= 8
                    or neck_geometry["opening_mask"][row, column] > 0.01
                ):
                    continue
                if (
                    skin[row, column] >= 0.75
                    or cloth[row, column] <= hair[row, column]
                    or cloth[row, column] <= skin[row, column] * 0.5
                ):
                    continue
                inside = np.arange(column + direction * radius * 2, column + direction * (radius * 4 + 1), direction)
                outside = np.arange(column - direction * radius, column - direction * (radius * 4 + 1), -direction)
                inside = inside[(inside >= 0) & (inside < width)]
                outside = outside[(outside >= 0) & (outside < width)]
                inside = inside[(skin[row, inside] > 0.75) & (input_image[row, inside, 3] > 240)]
                outside = outside[
                    (cloth[row, outside] > skin[row, outside])
                    & (cloth[row, outside] > hair[row, outside])
                    & (input_image[row, outside, 3] > 240)
                ]
                if inside.size < 2 or outside.size < 2:
                    continue
                foreground = np.median(input_image[row, inside, :3], axis=0)
                background = np.median(input_image[row, outside, :3], axis=0)
                difference = foreground - background
                magnitude = np.dot(difference, difference)
                if magnitude < 64:
                    continue
                alpha = np.clip(np.dot(input_image[row, column, :3] - background, difference) / magnitude, 0.0, 1.0)
                estimated = min(int(neck_layer[row, column, 3]), round(alpha * int(input_image[row, column, 3])))
                if estimated == int(neck_layer[row, column, 3]):
                    continue
                # 将更接近旧衣颜色的边缘像素交还后方头发
                foreground_distance = np.sum((input_image[row, column, :3] - foreground) ** 2)
                background_distance = np.sum((input_image[row, column, :3] - background) ** 2)
                result[row, column, 3] = 0 if background_distance < foreground_distance else estimated
    return result


def _create_neck_layer(
    input_image: np.ndarray,
    neck_geometry: dict,
    clothes_layer: np.ndarray,
) -> np.ndarray:
    """根据已配准领口保留原位置皮肤"""
    height, width = input_image.shape[:2]
    original = input_image[:, :, 3].astype(np.float32) / 255.0
    region = np.maximum(neck_geometry["skin_mask"], neck_geometry["opening_mask"] * neck_geometry["opening_support"])
    # 领口以下仅保留实际开口及衣料后方，领口上方保留完整原始颈部轮廓
    aperture = neck_geometry["opening_mask"]
    keep = np.maximum(aperture, clothes_layer[:, :, 3].astype(np.float32) / 255.0)
    rows = np.arange(height)[:, None]
    keep[rows[:, 0] < neck_geometry["opening_top_y"]] = 1.0
    neck_layer = input_image.copy()
    neck_layer[:, :, 3] = np.uint8(np.clip(original * region * keep * 255.0, 0, 255))
    return _refine_neck_boundary(input_image, neck_layer, neck_geometry, clothes_layer)


def _create_back_head_layer(
    head_layer: np.ndarray,
    probabilities: np.ndarray,
    clothes_layer: np.ndarray,
    face_height: float,
) -> np.ndarray:
    """生成衣服后方的原发图层并清理肩线附近残片"""
    height, width = head_layer.shape[:2]
    clothes_mask = clothes_layer[:, :, 3] > 127
    columns = np.flatnonzero(clothes_mask.any(axis=0))
    shoulder = np.interp(np.arange(width), columns, np.argmax(clothes_mask[:, columns], axis=0))
    # 用肩线中位高度约束外侧袖边的清理范围
    shoulder = np.minimum(shoulder, np.median(shoulder))
    transition_width = max(2.0, face_height * 0.04)
    transition = np.clip(
        (np.arange(height)[:, None] - shoulder[None, :] + transition_width)
        / (2.0 * transition_width),
        0.0,
        1.0,
    )
    transition = transition * transition * (3.0 - 2.0 * transition)
    head_probability = np.maximum.reduce((
        probabilities[:, :, 1], probabilities[:, :, 3], probabilities[:, :, 5],
    ))
    other_probability = np.maximum.reduce((
        probabilities[:, :, 0], probabilities[:, :, 2], probabilities[:, :, 4],
    ))
    # 结合头部类别概率与原发透明度确定保留区域
    keep = cv2.GaussianBlur(np.float32(
        (head_probability > other_probability) | (head_layer[:, :, 3] > 240),
    ), (0, 0), 0.6)
    # 按肩线过渡清理后景残片
    back_head = head_layer.copy()
    back_head[:, :, 3] = np.uint8(np.clip(
        head_layer[:, :, 3] * (1.0 - transition * (1.0 - keep)), 0, 255,
    ))
    return back_head


def _create_back_hair_fill(
    input_image: np.ndarray,
    head_layer: np.ndarray,
    probabilities: np.ndarray,
    face: np.ndarray,
    neck_geometry: dict,
    clothes_layer: np.ndarray,
) -> np.ndarray:
    """把同侧可靠原发纹理延续到新肩线后方"""
    height, width = input_image.shape[:2]
    face_width, face_height = float(face[2]), float(face[3])
    chin_y = neck_geometry["chin_y"]
    start = int(max(0, chin_y - face_height * 0.15))
    donor_end = int(min(height - 1, chin_y + face_height * 0.14))
    end = int(min(height - 1, neck_geometry["skin_bottom"] + face_height * 0.10))
    hair = (probabilities[:, :, 1] > 0.5) & (head_layer[:, :, 3] > 240)
    hair_edge = (head_layer[:, :, 3] > 32) & (
        probabilities[:, :, 1] > np.maximum.reduce((
            probabilities[:, :, 2], probabilities[:, :, 3], probabilities[:, :, 4], probabilities[:, :, 5],
        ))
    )
    clothes_mask = clothes_layer[:, :, 3] > 127
    columns = np.flatnonzero(clothes_mask.any(axis=0))
    shoulder = np.interp(np.arange(width), columns, np.argmax(clothes_mask[:, columns], axis=0))
    rows, columns = np.mgrid[:height, :width].astype(np.float32)
    result = np.zeros_like(input_image)
    for left_side in (True, False):
        boundary = neck_geometry["neck_left"] if left_side else neck_geometry["neck_right"]
        side = np.arange(width) < boundary if left_side else np.arange(width) > boundary
        count, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(hair & side[None, :]))
        if count <= 1:
            continue
        side_hair = labels == np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        samples = []
        for row in range(start, min(height, int(chin_y + face_height * 0.25))):
            positions = np.flatnonzero(side_hair[row])
            if positions.size >= face_width * 0.08:
                # 发芯提供纹理，原始发边提供外轮廓
                edge = np.flatnonzero(
                    hair_edge[row] & side
                    & (np.arange(width) >= positions[0] - face_width * 0.06)
                    & (np.arange(width) <= positions[-1] + face_width * 0.06)
                )
                samples.append((row, positions[0], positions[-1], edge[0] if left_side else edge[-1]))
        if len(samples) < face_height * 0.15:
            continue
        samples = np.asarray(samples, dtype=np.float32)
        # 仅延续垂到下巴以下的同侧发束
        if samples[-1, 0] < chin_y + face_height * 0.10:
            continue
        donor_rows = samples[samples[:, 0] <= donor_end]
        if len(donor_rows) < face_height * 0.12:
            continue
        source_y = start + np.clip((rows - start) / max(end - start, 1), 0.0, 1.0) * (donor_rows[-1, 0] - start)
        source_left = np.interp(source_y[:, 0], donor_rows[:, 0], donor_rows[:, 1] + 2)[:, None]
        source_right = np.interp(source_y[:, 0], donor_rows[:, 0], donor_rows[:, 2] - 2)[:, None]
        outer_column = 3
        tail = samples[-max(8, int(face_height * 0.10)):]
        slope, intercept = np.polyfit(tail[:, 0], tail[:, outer_column], 1)
        outer = np.interp(rows[:, 0], samples[:, 0], samples[:, outer_column])
        beyond = rows[:, 0] > samples[-1, 0]
        outer[beyond] = np.clip(
            slope * rows[beyond, 0] + intercept,
            samples[-1, outer_column] - face_width * 0.05,
            samples[-1, outer_column] + face_width * 0.05,
        )
        inner = boundary + (1 if left_side else -1) * face_width * 0.03
        target_left = outer[:, None] if left_side else inner
        target_right = inner if left_side else outer[:, None]
        fraction = np.clip((columns - target_left) / np.maximum(target_right - target_left, 1.0), 0.0, 1.0)
        source_x = np.float32(source_left + fraction * (source_right - source_left))
        alpha = input_image[:, :, 3].astype(np.float32) / 255.0 * side_hair
        sampled_alpha = cv2.remap(alpha, source_x, source_y, cv2.INTER_LINEAR)
        sampled_rgb = cv2.remap(input_image[:, :, :3].astype(np.float32) * alpha[:, :, None], source_x, source_y, cv2.INTER_LINEAR)
        region = np.minimum(np.clip((columns - target_left) / 2.0, 0.0, 1.0), np.clip((target_right - columns) / 2.0, 0.0, 1.0))
        region *= np.clip((rows - start) / 3.0, 0.0, 1.0)
        region *= (rows <= end) & (rows <= shoulder[None, :] + 3)
        layer = np.zeros_like(input_image)
        layer[:, :, :3] = np.uint8(np.clip(sampled_rgb / np.maximum(sampled_alpha[:, :, None], 1e-6), 0, 255))
        layer[:, :, 3] = np.uint8(np.clip(sampled_alpha * region * 255.0, 0, 255))
        result = _alpha_composite(result, layer)
    return result


def _refine_hair_tail_contour(
    head: np.ndarray,
    fill: np.ndarray,
    exposed: np.ndarray,
    source_end: int,
    radius: int,
    left: int,
    right: int,
) -> tuple[np.ndarray, np.ndarray, int | None]:
    """收拢右侧发尾末端的短小外凸"""
    y1 = max(0, source_end - radius * 3)
    y2 = min(head.shape[0], source_end + radius * 3 + 1)
    region = np.s_[y1:y2, left:right]
    alpha = 1.0 - (1.0 - head[region][:, :, 3] / 255.0) * (1.0 - fill[region][:, :, 3] / 255.0)
    valid = np.flatnonzero((alpha > 0.5).any(axis=1))
    if len(valid) < radius * 2 + 1:
        return head, fill, None
    edge = np.array([np.flatnonzero(alpha[row] > 0.5).max() for row in valid], dtype=np.float32)
    edge = np.interp(np.arange(y2 - y1), valid, edge).astype(np.float32)
    opened = cv2.morphologyEx(edge[:, None], cv2.MORPH_OPEN, np.ones((radius * 2 + 1, 1), np.uint8), borderType=cv2.BORDER_REPLICATE)[:, 0]
    opened = cv2.GaussianBlur(opened[:, None], (1, 0), max(1, radius * 0.25))[:, 0]
    excess = np.maximum(edge - opened, 0.0)
    seed = (excess > 0.5) & (np.arange(y1, y2) >= source_end - radius * 2) & (np.arange(y1, y2) <= source_end + radius * 2)
    count, labels = cv2.connectedComponents(seed.astype(np.uint8)[:, None])
    rows, columns = np.mgrid[y1:y2, left:right].astype(np.float32)
    weight = np.clip((columns - left - edge[:, None] + radius) / radius, 0.0, 1.0)
    visible = exposed[region] & (alpha > 0.03) & (weight > 0.0)
    shift = np.zeros_like(edge)
    source_start = None
    for index in range(1, count):
        positions = np.flatnonzero(labels[:, 0] == index)
        if len(positions) > radius * 3 or excess[positions].max() < 1.0:
            continue
        start = max(0, int(positions[0]) - radius // 2)
        end = min(len(edge) - 1, int(positions[-1]) + radius // 2)
        target = np.interp(np.arange(start, end + 1), [start, end], [edge[start], edge[end]])
        displacement = np.maximum(edge[start:end + 1] - target, 0.0)
        if not np.any(visible[start:end + 1] & (displacement[:, None] > 0.0)):
            continue
        source_start = y1 + start if source_start is None else min(source_start, y1 + start)
        shift[start:end + 1] = displacement
    map_x = columns + shift[:, None] * weight
    domain = exposed[region] & (shift[:, None] > 0.0) & (weight > 0.0)
    for layer in (head, fill):
        sampled = cv2.remap(layer[:, :, 3], map_x.astype(np.float32), rows, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        original = layer[region][:, :, 3]
        original[domain] = np.minimum(original[domain], sampled[domain])
    return head, fill, source_start


def _extend_back_hair_tail(
    back_head: np.ndarray,
    back_fill: np.ndarray,
    input_image: np.ndarray,
    probabilities: np.ndarray,
    neck_layer: np.ndarray,
    face_layer: np.ndarray,
    clothes_layer: np.ndarray,
    chin_y: float,
    face_height: float,
) -> tuple[np.ndarray, np.ndarray]:
    """收拢发尾短小外凸并将原发纹理连续延展到衣肩后方"""
    head = back_head.copy()
    fill = back_fill.copy()
    height, width = input_image.shape[:2]
    radius = max(2, round(face_height * 0.015))
    rows, columns = np.mgrid[:height, :width].astype(np.float32)
    exposed = (neck_layer[:, :, 3] < 8) & (face_layer[:, :, 3] < 8) & (clothes_layer[:, :, 3] < 8)
    below = cv2.erode(back_head[:, :, 3], np.ones((4, 1), np.uint8), anchor=(0, 0), borderType=cv2.BORDER_REPLICATE)
    seam = (
        (rows > chin_y + face_height * 0.10) & exposed & (back_fill[:, :, 3] > 8)
        & (back_head[:, :, 3] > 32) & (back_head[:, :, 3].astype(np.int16) - below > 32)
    )
    reliable = (input_image[:, :, 3] > 240) & (back_head[:, :, 3] > 240) & (
        probabilities[:, :, 1] > np.max(probabilities[:, :, 2:], axis=2)
    )
    combined_alpha = 1.0 - (1.0 - back_head[:, :, 3] / 255.0) * (1.0 - back_fill[:, :, 3] / 255.0)
    clothes_mask = clothes_layer[:, :, 3] > 127
    clothes_columns = np.flatnonzero(clothes_mask.any(axis=0))
    shoulder = np.interp(np.arange(width), clothes_columns, np.argmax(clothes_mask[:, clothes_columns], axis=0))
    for name, left, right, sign in (("left", 0, width // 2, 1), ("right", width // 2, width, -1)):
        side = (columns >= left) & (columns < right)
        seam_y, seam_x = np.where(seam & side)
        if not len(seam_y):
            continue
        terminal = seam_y >= seam_y.max() - radius
        seam_y, seam_x = seam_y[terminal], seam_x[terminal]
        top = int(seam_y.min()) - round(face_height * 0.15)
        source_end = int(seam_y.min()) - radius
        inner = int(seam_x.max()) + radius * 3 if name == "left" else int(seam_x.min()) - radius * 3
        outer = int(seam_x.min()) - radius * 3 if name == "left" else int(seam_x.max()) + radius * 3
        if name == "right":
            head, fill, contour_start = _refine_hair_tail_contour(
                head, fill, exposed, source_end, radius, left, right,
            )
            if contour_start is not None:
                # 从交界上方采样原发纹理并延续到肩后
                source_end = contour_start - radius * 2
                top = source_end - round(face_height * 0.15)
                inner -= radius * 6
        x1, x2 = max(left, min(inner, outer)), min(right, max(inner, outer) + 1)
        y1, y2 = max(0, top), min(height, int(shoulder[x1:x2].max()) + 3)
        region = np.s_[y1:y2, x1:x2]
        target_y, target_x = rows[region], columns[region]
        bottom = np.maximum(shoulder[x1:x2][None, :] + 2.0, float(source_end))
        t = np.clip((target_y - top) / np.maximum(bottom - top, 1.0), 0.0, 1.0)
        smooth = t * t * (3.0 - 2.0 * t)
        # 使用单调映射将原发纹理连续延伸到衣肩
        span = bottom - top
        source_span = float(source_end - top)
        offset = np.clip(target_y - top, 0.0, span)
        map_y = top + offset / (1.0 + (span / source_span - 1.0) * offset / span)
        source_contour = []
        target_contour = []
        valid_rows = []
        target_rows = []
        for y in range(y1, min(height, source_end + 1)):
            xs = np.flatnonzero(reliable[y, left:right]) + left
            if len(xs):
                valid_rows.append(y)
                source_contour.append(xs.min() if name == "left" else xs.max())
        for y in range(y1, y2):
            xs = np.flatnonzero(combined_alpha[y, left:right] > 0.95) + left
            if len(xs):
                target_rows.append(y)
                target_contour.append(xs.min() if name == "left" else xs.max())
        if len(valid_rows) < 2 or len(target_rows) < 2:
            continue
        source_edge = np.interp(map_y, valid_rows, source_contour)
        target_edge = np.interp(target_y, target_rows, target_contour)
        map_x = target_x + smooth * (source_edge - target_edge)
        # 外侧半透明轮廓采样同一发束的最外层可靠纹理
        map_x = np.maximum(map_x, source_edge) if name == "left" else np.minimum(map_x, source_edge)
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)
        sample = cv2.remap(input_image[:, :, :3], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        sample_alpha = cv2.remap(input_image[:, :, 3], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        sampled_reliable = cv2.remap(reliable.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
        horizontal = np.clip((inner - target_x) * sign / float(radius * 3), 0.0, 1.0)
        horizontal = horizontal * horizontal * (3.0 - 2.0 * horizontal)
        vertical = np.clip((target_y - top) / float(radius * 2), 0.0, 1.0)
        weight = horizontal * vertical
        domain = exposed[region] & (combined_alpha[region] > 0.03) & (target_y < bottom) & (weight > 0)
        domain &= (sample_alpha > 240) & (sampled_reliable > 0)
        for original, result in ((back_head, head), (back_fill, fill)):
            previous = original[region][:, :, :3].astype(np.float32)
            updated = np.uint8(np.clip(np.rint(previous + weight[:, :, None] * (sample.astype(np.float32) - previous)), 0, 255))
            result[region][:, :, :3][domain] = updated[domain]
    return head, fill


def change_clothes(
    input_image: np.ndarray,
    category: int,
    clothes_id: int,
    face_detect_model: str,
    parsing_model: str,
) -> np.ndarray:
    """统一人物比例并按真实下巴和服装开口完成换装"""
    if input_image.ndim != 3 or input_image.shape[2] != 4:
        raise ValueError("换装图片必须是透明证件照")
    face_detector = CLOTHES_FACE_DETECTORS.get(face_detect_model)
    if face_detector is None or parsing_model not in CLOTHES_PARSING_MODELS:
        raise ModelNotFoundError()

    clothes_image, clothes_geometry = _load_clothes_asset(category, clothes_id)

    faces = face_detector(input_image[:, :, :3])
    faces_num = 0 if faces is None else len(faces)
    if faces_num != 1:
        raise ValueError("换装图片人脸识别失败")
    input_image, face = _scale_person_to_target(input_image, faces[0])
    parsing_probabilities = _create_parsing_probabilities(input_image, parsing_model)
    neck_geometry = _measure_neck_geometry(
        input_image,
        face,
        parsing_probabilities,
    )
    if (
        neck_geometry["neck_left"] is None
        or neck_geometry["neck_length"] < float(face[3]) * 0.04
        or neck_geometry["skin_bottom"] - neck_geometry["chin_y"] < float(face[3]) * 0.12
    ):
        raise ValueError("未检测到脖子，请露出脖子")
    head_layer = _create_head_layer(
        input_image,
        face,
        parsing_probabilities,
        neck_geometry["chin_y"],
    )
    neck_geometry["skin_mask"] = _create_neck_mask(input_image, neck_geometry)
    neck_geometry["head_alpha"] = head_layer[:, :, 3].astype(np.float32) / 255.0
    # 用皮肤和头发类别概率确定开口支撑范围
    retained_class = np.maximum(
        neck_geometry["skin_probability"], neck_geometry["hair_probability"],
    ) > neck_geometry["clothes_probability"]
    neck_geometry["opening_support"] = np.uint8(retained_class & (input_image[:, :, 3] > 240))
    clothes_layer, garment_geometry = _create_clothes_layer(
        input_image,
        clothes_image,
        clothes_geometry,
        face,
        neck_geometry,
    )
    neck_layer = _create_neck_layer(
        input_image,
        neck_geometry,
        clothes_layer,
    )
    face_layer = _create_face_protection_layer(
        input_image,
        face,
        parsing_probabilities,
        neck_geometry["chin_y"],
    )

    back_head = _create_back_head_layer(
        head_layer, parsing_probabilities, clothes_layer, float(face[3]),
    )
    # 按肩线和领口轮廓将长发置于服装后方
    back_fill = _create_back_hair_fill(
        input_image, head_layer, parsing_probabilities, face, neck_geometry, clothes_layer,
    )
    # 平滑补发覆盖处的原发末端边界
    tail_region = (
        (back_fill[:, :, 3] > 240)
        & (np.arange(input_image.shape[0])[:, None] > neck_geometry["chin_y"] + float(face[3]) * 0.10)
        & (neck_layer[:, :, 3] < 8) & (face_layer[:, :, 3] < 8) & (clothes_layer[:, :, 3] < 8)
    )
    tail_alpha = cv2.boxFilter(back_head[:, :, 3], -1, (1, 4), anchor=(0, 0), borderType=cv2.BORDER_REPLICATE)
    back_head[:, :, 3][tail_region] = np.minimum(back_head[:, :, 3], tail_alpha)[tail_region]
    back_head, back_fill = _extend_back_hair_tail(
        back_head, back_fill, input_image, parsing_probabilities,
        neck_layer, face_layer, clothes_layer, neck_geometry["chin_y"], float(face[3]),
    )
    dressed_image = _alpha_composite(back_fill, neck_layer)
    dressed_image = _alpha_composite(dressed_image, back_head)
    dressed_image = _alpha_composite(dressed_image, clothes_layer)
    dressed_image = _alpha_composite(dressed_image, face_layer)
    # 用最终合成Alpha复验整个领口开口
    opening_interior = cv2.erode(
        np.uint8(garment_geometry["opening_mask"] > 0.99),
        np.ones((3, 3), dtype=np.uint8),
    )
    if np.any((opening_interior > 0) & (dressed_image[:, :, 3] < 240)):
        raise ValueError("服装领口超过可见皮肤范围，请选择其它服装")
    return dressed_image
