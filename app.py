# -*- coding: utf-8 -*-
"""
街景盆栽圖像批次遮罩與灰階化標註工具 (Street Potted-Plant Masking & Selective Color App)
=====================================================================================

用途：
    針對街景照片中的「盆栽」（植物 + 花盆/支架）進行語意分割，保留盆栽原始色彩、
    其餘背景轉為亮度加權灰階，供環境心理學視覺刺激物 (visual stimuli) 製作使用。

架構：
    Python + Streamlit 單機應用。分割引擎採 Meta SAM (Segment Anything)，
    輔以 OpenCV GrabCut 作為免下載模型的輕量備援；互動遮罩編輯（點選提示 /
    畫筆 / 橡皮擦）透過 streamlit-drawable-canvas 實作。

執行：
    streamlit run app.py

詳見同目錄 README.md 的安裝與使用說明。
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

# HEIC/HEIF 支援 (選用；未安裝 pillow-heif 時仍可使用 JPG/PNG/TIFF)
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIC_SUPPORTED = True
except ImportError:
    HEIC_SUPPORTED = False

from streamlit_drawable_canvas import st_canvas

# =====================================================================================
# 全域設定
# =====================================================================================

st.set_page_config(page_title="盆栽遮罩批次標註工具", layout="wide")

MAX_DISPLAY_DIM = 900          # 互動畫布最大顯示邊長 (px)；不影響最終輸出解析度
HISTORY_LIMIT = 25             # Undo/Redo 堆疊上限
ACCEPTED_TYPES = ["jpg", "jpeg", "png", "tif", "tiff"]
if HEIC_SUPPORTED:
    ACCEPTED_TYPES += ["heic", "heif"]

STATUS_LABELS = {
    "untouched": "未處理",
    "in_progress": "待確認",
    "done": "已校正完成",
}
STATUS_COLORS = {
    "untouched": "#999999",
    "in_progress": "#d9a441",
    "done": "#2e9e5b",
}


# =====================================================================================
# 資料結構
# =====================================================================================

@dataclass
class ImageRecord:
    """單張影像在工作階段中的完整狀態。"""

    filename: str
    orig: Image.Image                       # 原始解析度 PIL Image (RGB)
    np_orig: np.ndarray                     # 對應的 numpy array，快取避免重複轉換
    mask: np.ndarray                        # bool array，形狀同 (H, W)，True = 保留彩色(盆栽)
    history: list = field(default_factory=list)   # Undo 堆疊 (mask 的深拷貝)
    redo: list = field(default_factory=list)       # Redo 堆疊
    status: str = "untouched"
    pending_points: list = field(default_factory=list)   # 目前 SAM 提示點 [(x, y, label), ...]
    pending_logits: Optional[np.ndarray] = None           # SAM 上一輪 low-res logits，供疊代精修
    stroke_session: int = 0                                 # 畫筆/橡皮擦 canvas 版本號，用於清空筆畫


def push_history(rec: ImageRecord) -> None:
    """在任何會修改 mask 的操作前呼叫，保存目前狀態供 Undo。"""
    rec.history.append(rec.mask.copy())
    if len(rec.history) > HISTORY_LIMIT:
        rec.history.pop(0)
    rec.redo.clear()


def undo(rec: ImageRecord) -> None:
    if rec.history:
        rec.redo.append(rec.mask.copy())
        rec.mask = rec.history.pop()


def redo_action(rec: ImageRecord) -> None:
    if rec.redo:
        rec.history.append(rec.mask.copy())
        rec.mask = rec.redo.pop()


# =====================================================================================
# 影像讀取 (維持原始解析度 / 校正 EXIF 方向)
# =====================================================================================

def load_image_from_upload(uploaded_file) -> Image.Image:
    """讀取上傳檔案為 RGB PIL Image，校正 EXIF 旋轉，不做任何縮放。"""
    img = Image.open(io.BytesIO(uploaded_file.getvalue()))
    img = ImageOps.exif_transpose(img)  # 修正手機/相機的 EXIF 方向資訊
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def get_display_image(pil_img: Image.Image) -> tuple[Image.Image, float]:
    """回傳供畫布顯示用的縮小版影像，以及對應的縮放比例 (display = orig * scale)。
    真正的遮罩運算與匯出永遠使用原始解析度，畫布縮放僅為了互動流暢度。"""
    w, h = pil_img.size
    scale = min(1.0, MAX_DISPLAY_DIM / max(w, h))
    if scale < 1.0:
        disp = pil_img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    else:
        disp = pil_img
        scale = 1.0
    return disp, scale


# =====================================================================================
# 分割引擎 A：SAM (Segment Anything) — 高精準度，需下載模型檔
# =====================================================================================

@st.cache_resource(show_spinner="正在載入 SAM 模型（首次載入較久）…")
def load_sam_predictor(checkpoint_path: str, model_type: str, device: str):
    """載入並快取 SAM predictor。checkpoint_path / model_type / device 任一改變都會觸發重新載入。"""
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as e:
        raise RuntimeError(
            "未安裝 SAM 所需套件（torch / segment-anything）。"
            "本機使用請改安裝 requirements-sam.txt；雲端部署預設不含此套件，"
            "請改用 GrabCut 引擎。"
        ) from e

    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    return SamPredictor(sam)


def sam_predict(predictor, np_image: np.ndarray, points: list,
                 prev_logits: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """以目前累積的正/負樣本點呼叫 SAM，回傳 (mask_bool, low_res_logits)。"""
    predictor.set_image(np_image)
    coords = np.array([[p[0], p[1]] for p in points], dtype=np.float32)
    labels = np.array([p[2] for p in points], dtype=np.int32)
    mask_input = prev_logits[None, :, :] if prev_logits is not None else None
    masks, scores, logits = predictor.predict(
        point_coords=coords,
        point_labels=labels,
        mask_input=mask_input,
        multimask_output=False,
    )
    return masks[0].astype(bool), logits[0]


# =====================================================================================
# 分割引擎 B：GrabCut — 內建於 OpenCV，免下載模型，適合快速草稿或無 GPU 環境
# =====================================================================================

def grabcut_predict(np_image: np.ndarray, points: list) -> np.ndarray:
    """以正/負樣本點做種子的簡易 GrabCut。點的外接框 (加上邊距) 作為初始 ROI。"""
    h, w = np_image.shape[:2]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    pad = max(20, int(0.08 * max(w, h)))
    x0, x1 = max(0, min(xs) - pad), min(w, max(xs) + pad)
    y0, y1 = max(0, min(ys) - pad), min(h, max(ys) + pad)

    mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    mask[int(y0):int(y1), int(x0):int(x1)] = cv2.GC_PR_FGD
    for x, y, label in points:
        r = max(4, int(0.01 * max(w, h)))
        cv2.circle(mask, (int(x), int(y)), r,
                   cv2.GC_FGD if label == 1 else cv2.GC_BGD, -1)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    bgr = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    try:
        cv2.grabCut(bgr, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        # 種子點過少或區域太小時 GrabCut 可能失敗，回傳空遮罩讓使用者改用畫筆手動繪製
        return np.zeros((h, w), dtype=bool)
    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), True, False)


# =====================================================================================
# 影像處理：灰階化 / 邊緣羽化 / 選擇性上色合成
# =====================================================================================

def rgb_to_luminance_gray3(rgb: np.ndarray) -> np.ndarray:
    """Y = 0.299R + 0.587G + 0.114B，輸出仍為 3 channel 以利與彩色合成。"""
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    gray3 = np.stack([y, y, y], axis=-1)
    return np.clip(gray3, 0, 255)


def feather_alpha(mask_bool: np.ndarray, radius_px: float) -> np.ndarray:
    """將二值遮罩轉為 0~1 的柔化 alpha，避免邊緣鋸齒/硬邊。radius_px 建議 1~3。"""
    alpha = mask_bool.astype(np.float32)
    if radius_px and radius_px > 0:
        k = int(round(radius_px)) * 2 + 1  # 確保為奇數核心
        alpha = cv2.GaussianBlur(alpha, (k, k), sigmaX=radius_px)
    return np.clip(alpha, 0.0, 1.0)


def composite_selective_color(orig_rgb: np.ndarray, mask_bool: np.ndarray,
                               feather_radius: float) -> np.ndarray:
    """主體維持原始 sRGB 色彩，背景轉亮度加權灰階，邊緣以羽化 alpha 混合。"""
    alpha = feather_alpha(mask_bool, feather_radius)[..., None]
    gray3 = rgb_to_luminance_gray3(orig_rgb)
    color = orig_rgb.astype(np.float32)
    result = alpha * color + (1 - alpha) * gray3
    return np.clip(result, 0, 255).astype(np.uint8)


def overlay_preview(orig_rgb: np.ndarray, mask_bool: np.ndarray, opacity: float) -> np.ndarray:
    """原圖 + 半透明色塊標示遮罩區域，供檢查用。"""
    overlay = orig_rgb.copy().astype(np.float32)
    tint = np.array([46, 204, 113], dtype=np.float32)  # 綠色
    m = mask_bool[..., None].astype(np.float32)
    overlay = overlay * (1 - m * opacity) + tint * (m * opacity)
    return np.clip(overlay, 0, 255).astype(np.uint8)


# =====================================================================================
# 畫布座標 <-> 原始解析度 對應
# =====================================================================================

def canvas_point_to_original(obj: dict, scale: float) -> tuple[float, float]:
    """streamlit-drawable-canvas 的 point 物件是以左上角 (left, top) + radius 描述的圓，
    需換算回圓心座標，再除以縮放比例還原至原始影像座標。"""
    r = obj.get("radius", 3) * obj.get("scaleX", 1)
    cx = obj.get("left", 0) + r
    cy = obj.get("top", 0) + r
    return cx / scale, cy / scale


def stroke_layer_to_mask(image_data: np.ndarray, scale: float, target_hw: tuple[int, int]) -> np.ndarray:
    """把畫筆/橡皮擦 canvas 的 RGBA 繪製層轉成與原始影像同尺寸的 bool 遮罩。"""
    alpha = image_data[..., 3]
    stroke_bool = (alpha > 10)
    h, w = target_hw
    stroke_resized = cv2.resize(
        stroke_bool.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
    )
    return stroke_resized.astype(bool)


# =====================================================================================
# 匯出
# =====================================================================================

def export_one(rec: ImageRecord, feather_radius: float, export_mask: bool) -> tuple[dict, dict]:
    """回傳 (紀錄 dict, {檔名: bytes} 用於打包下載)。"""
    result_rgb = composite_selective_color(rec.np_orig, rec.mask, feather_radius)
    stem = Path(rec.filename).stem

    files: dict[str, bytes] = {}
    buf = io.BytesIO()
    Image.fromarray(result_rgb).save(buf, format="PNG")
    out_name = f"{stem}_selective_color.png"
    files[out_name] = buf.getvalue()

    record = {
        "filename": rec.filename,
        "output_file": out_name,
        "image_width": rec.np_orig.shape[1],
        "image_height": rec.np_orig.shape[0],
        "mask_pixel_percent": round(100.0 * rec.mask.sum() / rec.mask.size, 4),
        "feather_radius_px": feather_radius,
        "status": rec.status,
        "processed_at": datetime.now().isoformat(timespec="seconds"),
    }

    if export_mask:
        mask_buf = io.BytesIO()
        Image.fromarray((rec.mask * 255).astype(np.uint8)).save(mask_buf, format="PNG")
        mask_name = f"{stem}_mask.png"
        files[mask_name] = mask_buf.getvalue()
        record["mask_file"] = mask_name

    return record, files


def build_export_zip(records_and_files: list[tuple[dict, dict]]) -> bytes:
    """把所有輸出檔案與一份彙整 JSON 記錄打包成單一 ZIP，供下載按鈕使用。"""
    buf = io.BytesIO()
    all_records = []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for record, files in records_and_files:
            all_records.append(record)
            for name, data in files.items():
                zf.writestr(name, data)
        zf.writestr(
            "processing_log.json",
            json.dumps({"generated_at": datetime.now().isoformat(timespec="seconds"),
                        "records": all_records}, ensure_ascii=False, indent=2),
        )
    return buf.getvalue()


def save_to_local_folder(records_and_files: list[tuple[dict, dict]], out_dir: str) -> str:
    """直接寫入本機資料夾（此 app 以 `streamlit run` 在使用者本機執行，故可直接存取檔案系統）。"""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    all_records = []
    for record, files in records_and_files:
        all_records.append(record)
        for name, data in files.items():
            (out_path / name).write_bytes(data)
    (out_path / "processing_log.json").write_text(
        json.dumps({"generated_at": datetime.now().isoformat(timespec="seconds"),
                    "records": all_records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(out_path.resolve())


# =====================================================================================
# Session State 初始化
# =====================================================================================

if "images" not in st.session_state:
    st.session_state.images: dict[str, ImageRecord] = {}
if "current" not in st.session_state:
    st.session_state.current: Optional[str] = None
if "point_label_mode" not in st.session_state:
    st.session_state.point_label_mode = "positive"


# =====================================================================================
# 側邊欄：分割引擎設定
# =====================================================================================

st.sidebar.header("⚙️ 分割引擎設定")

engine = st.sidebar.radio(
    "分割引擎",
    ["SAM（高精準度，需下載模型檔）", "GrabCut（內建，免安裝模型，適合快速草稿）"],
    index=1,
    help="雲端部署（如 Streamlit Cloud）預設只裝了 GrabCut 所需的套件；"
         "要用 SAM 請在本機安裝 requirements-sam.txt，詳見 README。",
)

sam_predictor = None
if engine.startswith("SAM"):
    with st.sidebar.expander("SAM 模型設定", expanded=True):
        model_type = st.selectbox("模型類型", ["vit_b", "vit_l", "vit_h"], index=0,
                                   help="vit_b 最小最快 (~375MB)，vit_h 最精準但較慢、較大 (~2.4GB)")
        checkpoint_path = st.text_input(
            "checkpoint 檔案路徑 (.pth)",
            value="./checkpoints/sam_vit_b_01ec64.pth",
            help="請先至 Meta SAM 官方 repo 下載對應權重檔，詳見 README",
        )
        device = st.selectbox("運算裝置", ["cpu", "cuda"], index=0,
                               help="若本機有 NVIDIA GPU 且已安裝對應 CUDA 版 PyTorch，選 cuda 可大幅加速")
        if Path(checkpoint_path).exists():
            try:
                sam_predictor = load_sam_predictor(checkpoint_path, model_type, device)
                st.success("模型已載入")
            except RuntimeError as e:
                st.error(str(e))
        else:
            st.warning("尚未找到 checkpoint 檔案，請確認路徑，或改用 GrabCut 引擎")

# =====================================================================================
# 側邊欄：批次上傳與縮圖佇列
# =====================================================================================

st.sidebar.header("📁 影像批次管理")
uploaded = st.sidebar.file_uploader(
    "選取或拖曳多張影像（可於系統對話框中框選整個資料夾內的檔案）",
    type=ACCEPTED_TYPES,
    accept_multiple_files=True,
)
if not HEIC_SUPPORTED:
    st.sidebar.caption("⚠️ 未安裝 pillow-heif，暫不支援 HEIC/HEIF；詳見 README。")

if uploaded:
    for f in uploaded:
        if f.name not in st.session_state.images:
            pil_img = load_image_from_upload(f)
            np_img = np.array(pil_img)
            st.session_state.images[f.name] = ImageRecord(
                filename=f.name,
                orig=pil_img,
                np_orig=np_img,
                mask=np.zeros(np_img.shape[:2], dtype=bool),
            )
    if st.session_state.current is None:
        st.session_state.current = uploaded[0].name

st.sidebar.markdown("---")
st.sidebar.subheader("縮圖佇列")
for fname, rec in st.session_state.images.items():
    cols = st.sidebar.columns([1, 2])
    with cols[0]:
        thumb, _ = get_display_image(rec.orig)
        thumb.thumbnail((80, 80))
        st.image(thumb)
    with cols[1]:
        badge = f":{'green' if rec.status=='done' else 'orange' if rec.status=='in_progress' else 'gray'}[{STATUS_LABELS[rec.status]}]"
        if st.button(f"{fname}\n{badge}", key=f"select_{fname}", width="stretch"):
            st.session_state.current = fname

# =====================================================================================
# 主畫面
# =====================================================================================

st.title("🌿 街景盆栽批次遮罩 / 選擇性上色工具")

if not st.session_state.images:
    st.info("請先於左側側邊欄上傳影像以開始標註。")
    st.stop()

cur_name = st.session_state.current
rec = st.session_state.images[cur_name]
h, w = rec.np_orig.shape[:2]
disp_img, scale = get_display_image(rec.orig)

st.caption(f"目前影像：**{cur_name}** ｜ 原始解析度 {w}×{h}px ｜ 狀態：{STATUS_LABELS[rec.status]}")

# ---- 工具列 ----
tool_col, view_col, param_col = st.columns([1.2, 1, 1])

with tool_col:
    tool = st.radio("編輯工具", ["SAM／GrabCut 點選提示", "畫筆（新增）", "橡皮擦（移除）"], horizontal=False)

with view_col:
    view_mode = st.radio("預覽模式", ["原圖 + 遮罩疊圖", "最終選擇性上色預覽", "純黑白二值遮罩"], horizontal=False)
    overlay_opacity = st.slider("疊圖透明度", 0.1, 1.0, 0.45, 0.05) if view_mode == "原圖 + 遮罩疊圖" else 0.45

with param_col:
    feather_radius = st.slider("邊緣羽化半徑 (px)", 0.0, 3.0, 1.5, 0.5)
    brush_size = st.slider("筆刷/橡皮擦大小 (顯示座標 px)", 3, 80, 20, 1)

# ---- 點選提示的正/負樣本切換 ----
if tool.startswith("SAM／GrabCut"):
    st.session_state.point_label_mode = st.radio(
        "樣本類型", ["positive", "negative"],
        format_func=lambda x: "➕ 正樣本（點在盆栽上）" if x == "positive" else "➖ 負樣本（點在背景上）",
        horizontal=True,
    )

# ---- 畫布 ----
drawing_mode = "point" if tool.startswith("SAM／GrabCut") else "freedraw"
stroke_color = "#1E90FF" if st.session_state.get("point_label_mode") == "positive" else "#FF4136"
if tool == "畫筆（新增）":
    stroke_color = "rgba(46, 204, 113, 0.9)"
elif tool == "橡皮擦（移除）":
    stroke_color = "rgba(255, 65, 54, 0.9)"

canvas_key = f"canvas_{cur_name}_{drawing_mode}_{rec.stroke_session}"

canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",
    stroke_width=brush_size if drawing_mode == "freedraw" else 3,
    stroke_color=stroke_color,
    background_image=disp_img,
    update_streamlit=True,
    height=disp_img.height,
    width=disp_img.width,
    drawing_mode=drawing_mode,
    point_display_radius=6,
    return_image_data=True,  # streamlit-drawable-canvas>=0.10 需明確要求才會回傳 image_data
    key=canvas_key,
)

btn_col1, btn_col2, btn_col3, btn_col4, btn_col5 = st.columns(5)

# ---- 點選提示邏輯：SAM / GrabCut ----
if drawing_mode == "point" and canvas_result.json_data is not None:
    objs = canvas_result.json_data.get("objects", [])
    if len(objs) > len(rec.pending_points):
        # 只處理新增的點，換算回原始解析度座標
        for obj in objs[len(rec.pending_points):]:
            ox, oy = canvas_point_to_original(obj, scale)
            label = 1 if st.session_state.point_label_mode == "positive" else 0
            rec.pending_points.append((ox, oy, label))

        if engine.startswith("SAM") and sam_predictor is not None:
            candidate_mask, rec.pending_logits = sam_predict(
                sam_predictor, rec.np_orig, rec.pending_points, rec.pending_logits
            )
        else:
            candidate_mask = grabcut_predict(rec.np_orig, rec.pending_points)
            rec.pending_logits = None

        rec.status = "in_progress"
        with btn_col1:
            if st.button("✅ 加入目前物件至遮罩", width="stretch"):
                push_history(rec)
                rec.mask = rec.mask | candidate_mask
                rec.pending_points, rec.pending_logits = [], None
                rec.stroke_session += 1
                st.rerun()
        # 暫存候選遮罩供下方預覽使用
        rec_candidate = candidate_mask
    else:
        rec_candidate = None
else:
    rec_candidate = None

with btn_col2:
    if st.button("↩️ 復原 (Undo)", width="stretch", disabled=not rec.history):
        undo(rec)
        st.rerun()
with btn_col3:
    if st.button("↪️ 重做 (Redo)", width="stretch", disabled=not rec.redo):
        redo_action(rec)
        st.rerun()
with btn_col4:
    if st.button("🧹 清空目前點選提示", width="stretch"):
        rec.pending_points, rec.pending_logits = [], None
        rec.stroke_session += 1
        st.rerun()

# ---- 畫筆 / 橡皮擦邏輯 ----
if drawing_mode == "freedraw" and canvas_result.image_data is not None:
    with btn_col5:
        if st.button("🖌️ 套用目前筆畫", width="stretch"):
            stroke_mask = stroke_layer_to_mask(canvas_result.image_data, scale, (h, w))
            if stroke_mask.any():
                push_history(rec)
                if tool == "畫筆（新增）":
                    rec.mask = rec.mask | stroke_mask
                else:
                    rec.mask = rec.mask & ~stroke_mask
                rec.status = "in_progress"
            rec.stroke_session += 1  # 清空畫布，開始下一段筆畫
            st.rerun()

st.markdown("---")

# ---- 預覽 ----
preview_source = rec.mask
if rec_candidate is not None:
    preview_source = rec.mask | rec_candidate  # 顯示「已確認遮罩 + 目前候選」的合併結果供檢查

if view_mode == "原圖 + 遮罩疊圖":
    preview = overlay_preview(rec.np_orig, preview_source, overlay_opacity)
elif view_mode == "最終選擇性上色預覽":
    preview = composite_selective_color(rec.np_orig, preview_source, feather_radius)
else:
    preview = np.stack([preview_source * 255] * 3, axis=-1).astype(np.uint8)

st.image(preview, caption="即時預覽（依原始解析度運算，畫布顯示已縮放僅為互動用）", width="stretch")
st.caption(f"目前遮罩覆蓋率：{100.0 * rec.mask.sum() / rec.mask.size:.2f}%")

nav_col1, nav_col2, _ = st.columns([1, 1, 3])
names = list(st.session_state.images.keys())
idx = names.index(cur_name)
with nav_col1:
    if st.button("⬅️ 上一張", disabled=idx == 0, width="stretch"):
        st.session_state.current = names[idx - 1]
        st.rerun()
with nav_col2:
    if st.button("完成並下一張 ➡️", width="stretch"):
        rec.status = "done"
        if idx + 1 < len(names):
            st.session_state.current = names[idx + 1]
        st.rerun()

# =====================================================================================
# 批次匯出
# =====================================================================================

st.markdown("---")
st.header("📦 批次匯出")

exp_col1, exp_col2, exp_col3 = st.columns(3)
with exp_col1:
    export_scope = st.radio("匯出範圍", ["僅「已校正完成」", "全部影像"], horizontal=True)
with exp_col2:
    export_mask_toggle = st.checkbox("同步匯出二值化遮罩 PNG", value=True)
with exp_col3:
    export_feather = st.slider("匯出用羽化半徑 (px)", 0.0, 3.0, feather_radius, 0.5, key="export_feather")

target_names = [
    n for n, r in st.session_state.images.items()
    if export_scope == "全部影像" or r.status == "done"
]
st.caption(f"符合匯出範圍的影像：{len(target_names)} / {len(st.session_state.images)} 張")

out_dir_col, zip_col = st.columns(2)
with out_dir_col:
    out_dir = st.text_input("直接存至本機資料夾（此工具在本機執行，可直接寫入磁碟）", value="./output")
    if st.button("💾 匯出至本機資料夾", disabled=not target_names):
        records_and_files = [
            export_one(st.session_state.images[n], export_feather, export_mask_toggle)
            for n in target_names
        ]
        saved_path = save_to_local_folder(records_and_files, out_dir)
        st.success(f"已匯出 {len(target_names)} 張影像至：{saved_path}")

with zip_col:
    if st.button("⬇️ 產生 ZIP 供下載", disabled=not target_names):
        records_and_files = [
            export_one(st.session_state.images[n], export_feather, export_mask_toggle)
            for n in target_names
        ]
        zip_bytes = build_export_zip(records_and_files)
        st.download_button(
            "點此下載 ZIP 檔",
            data=zip_bytes,
            file_name=f"selective_color_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip",
        )
