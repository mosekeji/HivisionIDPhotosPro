import cv2
import numpy as np
from ..context import Context, ContextHandler
from tools.exceptions import FaceError
from tools.rotation import rotate_bound


def align_face(ctx: Context, detection_handler: ContextHandler):
    """使用当前检测器复测人脸角度，只对稳定的倾斜执行旋转"""
    if abs(ctx.face["roll_angle"]) <= 2:
        return False

    angles = [ctx.face["roll_angle"]]
    height, width = ctx.origin_image.shape[:2]
    x, y, face_width, face_height = ctx.face["rectangle"]
    x1 = max(0, int(x - face_width * 0.5))
    y1 = max(0, int(y - face_height * 0.5))
    x2 = min(width, int(x + face_width * 1.5))
    y2 = min(height, int(y + face_height * 1.5))
    face_image = ctx.origin_image[y1:y2, x1:x2]

    # 按人脸大小复测，避免整图缩小后小脸丢失细节
    for target_width in (192, 256):
        scale = target_width / face_width
        probe = Context(ctx.params)
        probe.origin_image = cv2.resize(
            face_image,
            (round(face_image.shape[1] * scale), round(face_image.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
        try:
            detection_handler(probe)
        except FaceError:
            return False
        angles.append(probe.face["roll_angle"])

    angle = float(np.median(angles))
    if min(abs(value) for value in angles) <= 2:
        return False
    if max(angles) - min(angles) > min(5, abs(angle) * 0.5):
        return False

    # 复检始终使用原图，确认候选后再更新处理上下文
    rotated = Context(ctx.params)
    rotated.origin_image = rotate_bound(ctx.origin_image, -angle)[0]
    try:
        detection_handler(rotated)
    except FaceError:
        return False
    if abs(rotated.face["roll_angle"]) >= abs(angle):
        return False
    rotated.matting_image = rotate_bound(ctx.matting_image, -angle)[0]

    ctx.origin_image = rotated.origin_image
    ctx.matting_image = rotated.matting_image
    ctx.face = rotated.face
    return True
