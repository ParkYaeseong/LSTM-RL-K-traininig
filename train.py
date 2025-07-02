# ct_classification_train_with_xai_using_nifti.py
# 전처리된 NIfTI 영상 기반 암 분류 모델 학습 및 XAI 적용

# --- OpenMP 중복 라이브러리 로드 허용 ---
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import sys
import glob
import numpy as np
import pandas as pd
import monai
from monai.transforms import (
    LoadImageD, EnsureChannelFirstD,
    ScaleIntensityRangePercentilesD, ResizeD, Compose,
    EnsureTypeD, RandFlipd, RandRotate90d, LambdaD
)
from monai.data import Dataset, DataLoader, decollate_batch, list_data_collate
from monai.utils import set_determinism
from monai.visualize import GradCAM
from monai.networks.nets import resnet34

import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.data import Subset # 현재 직접 사용 안함

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix
)

from tqdm import tqdm
import logging
import traceback
import copy
import matplotlib.pyplot as plt
from collections import Counter


# --- 설정값 ---
# 입력 데이터 경로 (Manifest 파일 기준)
MANIFEST_FILE_PATH = os.path.join(os.getcwd(), "preprocessed_nifti_data", "preprocessed_manifest.csv")
PATIENT_ID_COL = 'bcr_patient_barcode'
LABEL_COL = 'original_label' # Manifest 파일 내 원본 (문자열) 레이블 컬럼명

# 출력 설정
BASE_OUTPUT_DIR = os.path.join(os.getcwd(), "classification_results_nifti_with_xai_v")
MODEL_CHECKPOINT_NAME = "best_ct_classification_nifti_model_v.pth"
XAI_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "xai_gradcam_outputs_nifti_v")
LOG_FILE_PATH = os.path.join(BASE_OUTPUT_DIR, "ct_classification_nifti_log_v.txt")
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
os.makedirs(XAI_OUTPUT_DIR, exist_ok=True)

# 전처리 및 모델 관련 설정
RESIZE_SHAPE = (96, 96, 96)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
set_determinism(RANDOM_SEED)

PRETRAINED_WEIGHTS_PATH = "G:/내 드라이브/2조/본프로젝트/resnet_34_23dataset.pth"

# 학습 파라미터
NUM_CLASSES = 5
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
NUM_EPOCHS = 300
K_FOLDS = 0
TEST_SPLIT_RATIO = 0.2
VAL_SPLIT_RATIO = 0.15
FREEZE_FEATURE_EXTRACTOR_EPOCHS = 5
NUM_WORKERS_DATALOADER = 0 # 디버깅 및 안정성을 위해 0으로 시작 권장

# XAI 설정
XAI_NUM_SAMPLES_TO_VISUALIZE = 5
XAI_TARGET_LAYER_NAME = "layer4"

# --- 로거 설정 ---
logger = logging.getLogger("train_nifti")
logger.setLevel(logging.INFO)
if logger.hasHandlers(): logger.handlers.clear()
file_handler = logging.FileHandler(LOG_FILE_PATH, mode='w', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)-8s - %(module)s - %(message)s'))
logger.addHandler(file_handler)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)-8s - %(message)s'))
logger.addHandler(stream_handler)

# --- 오류 처리 Dataset 및 Collate 함수 정의 ---
class ErrorTolerantDataset(Dataset):
    def __init__(self, data, transform, patient_id_col_name='bcr_patient_barcode'):
        super().__init__(data, transform)
        self.patient_id_col_name = patient_id_col_name

    def _transform(self, index):
        item_dict_original = self.data[index]
        try:
            if self.transform:
                return self.transform(item_dict_original.copy())
            return item_dict_original.copy()
        except Exception as e:
            pid = item_dict_original.get(self.patient_id_col_name, "N/A")
            image_path = item_dict_original.get("imageV", "N/A")
            error_msg_short = str(e).splitlines()[0]
            logger.warning(f"Error transforming item PID: {pid}, Path: {image_path} in ErrorTolerantDataset - returning None: {type(e).__name__} - {error_msg_short}")
            return None

def safe_list_data_collate(batch):
    valid_batch = [item for item in batch if item is not None]
    if not valid_batch:
        return None
    return list_data_collate(valid_batch)

# --- 데이터 로드 및 전처리 함수 (Manifest 기반) ---
def get_label_mapping_from_manifest(df, original_label_col_name_in_manifest):
    unique_labels_str = sorted(df[original_label_col_name_in_manifest].astype(str).unique())
    label_to_int_initial = {label_str: i for i, label_str in enumerate(unique_labels_str)}
    int_to_label_initial = {i: label_str for label_str, i in label_to_int_initial.items()}
    logger.info(f"Manifest 원본 문자열 레이블 기준 초기 매핑 생성: {label_to_int_initial}")
    
    global NUM_CLASSES
    actual_num_classes_initial = len(unique_labels_str)
    if actual_num_classes_initial != NUM_CLASSES:
        logger.warning(f"설정된 NUM_CLASSES({NUM_CLASSES})와 Manifest 원본 문자열 레이블 종류 수({actual_num_classes_initial})가 다릅니다. NUM_CLASSES를 초기값 {actual_num_classes_initial}로 업데이트합니다.")
        NUM_CLASSES = actual_num_classes_initial
    return label_to_int_initial, int_to_label_initial

def load_and_prepare_data_from_manifest(manifest_file_path, patient_id_col, original_label_col_for_mapping):
    try:
        manifest_df = pd.read_csv(manifest_file_path)
        logger.info(f"Manifest 파일 로드 완료: {manifest_file_path}, 총 {len(manifest_df)}개 항목.")
    except FileNotFoundError:
        logger.error(f"Manifest 파일({manifest_file_path})을 찾을 수 없습니다."); return [], None, None

    required_cols = ['image_nifti_path', patient_id_col, LABEL_COL, 'label_encoded']
    for col in required_cols:
        if col not in manifest_df.columns:
            logger.error(f"Manifest 파일에 필요한 컬럼 '{col}'이 없습니다."); return [], None, None
    
    label_to_int_map_ref_initial, int_to_label_map_for_initial_logging = get_label_mapping_from_manifest(manifest_df, original_label_col_for_mapping)
    all_data_dicts = []
    for _, row in manifest_df.iterrows():
        if not isinstance(row['image_nifti_path'], str) or not os.path.exists(row['image_nifti_path']):
            logger.warning(f"NIfTI 파일 경로가 유효하지 않거나 찾을 수 없습니다: {row['image_nifti_path']}. 이 항목은 건너뜁니다.")
            continue
        data_dict = {
            "image": row['image_nifti_path'],
            "label": torch.tensor(row['label_encoded'], dtype=torch.long),
            patient_id_col: row[patient_id_col],
            "original_label_str": str(row[LABEL_COL])
        }
        all_data_dicts.append(data_dict)
    if not all_data_dicts: logger.error("Manifest에서 유효한 데이터를 로드하지 못했습니다.")
    return all_data_dicts, label_to_int_map_ref_initial, int_to_label_map_for_initial_logging

# --- MONAI Transforms 정의 ---
train_transforms = Compose([
    LoadImageD(keys=["image"], reader="NibabelReader", image_only=True, ensure_channel_first=True),
    ScaleIntensityRangePercentilesD(keys=["image"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True),
    RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
    RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
    RandRotate90d(keys=["image"], prob=0.5, max_k=3, spatial_axes=(1, 2)),
    ResizeD(keys=["image"], spatial_size=RESIZE_SHAPE, mode="trilinear", align_corners=True),
    EnsureTypeD(keys=["image"], dtype=torch.float32),
    LambdaD(keys=["image"], func=lambda x: x.as_tensor() if hasattr(x, 'as_tensor') else torch.as_tensor(x))
])
val_test_transforms = Compose([
    LoadImageD(keys=["image"], reader="NibabelReader", image_only=True, ensure_channel_first=True),
    ScaleIntensityRangePercentilesD(keys=["image"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True),
    ResizeD(keys=["image"], spatial_size=RESIZE_SHAPE, mode="trilinear", align_corners=True),
    EnsureTypeD(keys=["image"], dtype=torch.float32),
    LambdaD(keys=["image"], func=lambda x: x.as_tensor() if hasattr(x, 'as_tensor') else torch.as_tensor(x))
])

# --- 모델 정의 ---
class CTClassifier(nn.Module):
    def __init__(self, num_classes_final, pretrained_model_arch=resnet34, pretrained_weights_path=None, freeze_feature_extractor=True):
        super().__init__()
        logger.info(f"모델 초기화: {pretrained_model_arch.__name__} 사용, 최종 클래스 수: {num_classes_final}")
        self.feature_extractor = pretrained_model_arch(n_input_channels=1, num_classes=1000) # ImageNet default classes
        if pretrained_weights_path and os.path.exists(pretrained_weights_path):
            try:
                state_dict = torch.load(pretrained_weights_path, map_location=DEVICE, weights_only=True)
                if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
                new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
                missing_keys, unexpected_keys = self.feature_extractor.load_state_dict(new_state_dict, strict=False)
                logger.info(f"모델({pretrained_model_arch.__name__})에 사전 훈련된 가중치 로드 완료: {pretrained_weights_path}")
                if missing_keys: logger.warning(f"누락된 키: {missing_keys}")
                if unexpected_keys: logger.warning(f"예상치 못한 키: {unexpected_keys}")
            except Exception as e: logger.error(f"사전 훈련된 가중치 로드 실패: {e}."); logger.debug(traceback.format_exc())
        elif pretrained_weights_path: logger.warning(f"사전 훈련된 가중치 파일을 찾을 수 없습니다: {pretrained_weights_path}.")
        else: logger.info("사전 훈련된 가중치 경로가 제공되지 않았습니다. 모델이 무작위로 초기화됩니다.") # 수정: Warning -> Info

        if freeze_feature_extractor:
            for param_name, param in self.feature_extractor.named_parameters():
                if "fc" not in param_name: param.requires_grad = False
            logger.info("특징 추출기 동결됨 (마지막 fc 레이어 제외).")
        try:
            original_num_ftrs = self.feature_extractor.fc.in_features
            self.feature_extractor.fc = nn.Identity()
            num_ftrs_for_head = original_num_ftrs
        except AttributeError:
            logger.warning(f"{pretrained_model_arch.__name__}에 'fc' 속성이 없습니다."); num_ftrs_for_head = 512
            logger.warning(f"분류기 헤드 입력 특징 수를 {num_ftrs_for_head}로 가정합니다.") # ResNet34의 경우 512가 맞음
        self.classifier_head = nn.Sequential(
            nn.Linear(num_ftrs_for_head, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(256, num_classes_final)
        )
        logger.info(f"새로운 분류기 헤드 구성: Linear({num_ftrs_for_head}, 256) -> ... -> Linear(256, {num_classes_final})")
    def forward(self, x): return self.classifier_head(self.feature_extractor(x))
    def unfreeze_feature_extractor(self):
        for param_name, param in self.feature_extractor.named_parameters():
            if "fc" not in param_name: param.requires_grad = True
        logger.info("특징 추출기 전체 동결 해제됨.")

# --- 학습 및 평가 함수 ---
def train_one_epoch(model, dataloader, criterion, optimizer, device, current_num_classes):
    model.train()
    running_loss = 0.0; correct_predictions = 0; total_predictions = 0
    all_labels_epoch = []; all_preds_proba_epoch = []
    
    prog_bar = tqdm(enumerate(dataloader), desc="Training", leave=False, total=len(dataloader))
    
    for batch_idx, batch_data in prog_bar:
        if batch_data is None:
            logger.warning(f"Skipping batch {batch_idx} in Training (all items failed transform or collate returned None).")
            continue

        if not batch_data: 
            logger.warning(f"Skipping batch {batch_idx} in Training because batch_data dictionary is empty.")
            continue

        inputs_tensor = batch_data.get("image")
        labels_tensor = batch_data.get("label")

        if inputs_tensor is None or (isinstance(inputs_tensor, torch.Tensor) and inputs_tensor.numel() == 0):
            logger.warning(f"Skipping batch {batch_idx} in Training: 'image' data is missing, None, or an empty tensor.")
            continue
        
        if labels_tensor is None:
            logger.warning(f"Skipping batch {batch_idx} in Training: 'label' data is missing or None.")
            continue
        
        if isinstance(inputs_tensor, torch.Tensor) and isinstance(labels_tensor, torch.Tensor):
            if inputs_tensor.size(0) == 0: 
                 logger.warning(f"Skipping batch {batch_idx} in Training: 'image' tensor batch size is 0.")
                 continue
            if inputs_tensor.size(0) != labels_tensor.size(0):
                logger.warning(f"Skipping batch {batch_idx} in Training: Mismatched batch sizes between images ({inputs_tensor.size(0)}) and labels ({labels_tensor.size(0)}).")
                continue
            if labels_tensor.numel() == 0 and inputs_tensor.numel() > 0 :
                 logger.warning(f"Skipping batch {batch_idx} in Training: 'label' tensor is empty while 'image' tensor is not.")
                 continue
        elif isinstance(inputs_tensor, torch.Tensor) and not isinstance(labels_tensor, torch.Tensor):
            logger.warning(f"Skipping batch {batch_idx} in Training: 'image' is a tensor, but 'label' is not (type: {type(labels_tensor)}).")
            continue

        inputs = inputs_tensor.to(device)
        labels = labels_tensor.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward(); optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted_classes = torch.max(outputs, 1)
        correct_predictions += (predicted_classes == labels).sum().item()
        total_predictions += labels.size(0)
        
        all_labels_epoch.extend(labels.cpu().numpy())
        all_preds_proba_epoch.extend(torch.softmax(outputs, dim=1).detach().cpu().numpy())
        
    prog_bar.close()
    
    epoch_loss = running_loss / total_predictions if total_predictions > 0 else float('inf')
    epoch_acc = correct_predictions / total_predictions if total_predictions > 0 else 0
    epoch_auc = 0.0
    
    if total_predictions > 0 and len(all_labels_epoch) > 0:
        try:
            unique_labels_in_epoch = np.unique(all_labels_epoch)
            roc_labels_range = list(range(current_num_classes)) if current_num_classes > 0 else None

            if current_num_classes > 2 and len(unique_labels_in_epoch) > 1:
                epoch_auc = roc_auc_score(all_labels_epoch, all_preds_proba_epoch, multi_class='ovr', average='macro', labels=roc_labels_range)
            elif current_num_classes == 2 and len(unique_labels_in_epoch) == 2:
                epoch_auc = roc_auc_score(all_labels_epoch, np.array(all_preds_proba_epoch)[:, 1])
            elif len(unique_labels_in_epoch) <= 1:
                logger.debug(f"훈련 중 AUC 계산 불가: 에포크 내 레이블 종류 부족 ({len(unique_labels_in_epoch)}개). 전체 클래스 수: {current_num_classes}")
            else:
                logger.debug(f"훈련 중 AUC 계산 조건 미충족. 레이블 종류: {len(unique_labels_in_epoch)}, 클래스 수: {current_num_classes}")
        except ValueError as e:
            logger.warning(f"훈련 중 AUC 계산 오류: {e}. AUC는 0.0으로 설정됩니다. Labels present: {np.unique(all_labels_epoch)}. Probas shape: {np.array(all_preds_proba_epoch).shape}")
            
    return epoch_loss, epoch_acc, epoch_auc

def evaluate_model(model, dataloader, criterion, device, phase="Validation", current_num_classes=None):
    if current_num_classes is None:
        try:
            current_num_classes = model.classifier_head[-1].out_features
            logger.debug(f"evaluate_model current_num_classes not provided, inferred: {current_num_classes}")
        except AttributeError: 
            logger.error(f"evaluate_model: Could not infer current_num_classes from model (classifier_head[-1].out_features not found). Using global NUM_CLASSES: {NUM_CLASSES} as fallback.")
            current_num_classes = NUM_CLASSES # Fallback to global NUM_CLASSES

    model.eval()
    running_loss = 0.0; correct_predictions = 0; total_predictions = 0
    all_labels_eval = []; all_predicted_classes_eval = []; all_preds_proba_eval = []
    
    prog_bar = tqdm(enumerate(dataloader), desc=phase, leave=False, total=len(dataloader))

    with torch.no_grad():
        for batch_idx, batch_data in prog_bar: 
            if batch_data is None:
                logger.warning(f"Skipping batch {batch_idx} in {phase} (all items failed transform or collate returned None).")
                continue

            if not batch_data: 
                logger.warning(f"Skipping batch {batch_idx} in {phase} because batch_data dictionary is empty.")
                continue

            inputs_tensor = batch_data.get("image")
            labels_tensor = batch_data.get("label")

            if inputs_tensor is None or (isinstance(inputs_tensor, torch.Tensor) and inputs_tensor.numel() == 0):
                logger.warning(f"Skipping batch {batch_idx} in {phase}: 'image' data is missing, None, or an empty tensor.")
                continue
            
            if labels_tensor is None:
                logger.warning(f"Skipping batch {batch_idx} in {phase}: 'label' data is missing or None.")
                continue
                
            if isinstance(inputs_tensor, torch.Tensor) and isinstance(labels_tensor, torch.Tensor):
                if inputs_tensor.size(0) == 0:
                     logger.warning(f"Skipping batch {batch_idx} in {phase}: 'image' tensor batch size is 0.")
                     continue
                if inputs_tensor.size(0) != labels_tensor.size(0):
                    logger.warning(f"Skipping batch {batch_idx} in {phase}: Mismatched batch sizes between images ({inputs_tensor.size(0)}) and labels ({labels_tensor.size(0)}).")
                    continue
                if labels_tensor.numel() == 0 and inputs_tensor.numel() > 0 :
                     logger.warning(f"Skipping batch {batch_idx} in {phase}: 'label' tensor is empty while 'image' tensor is not.")
                     continue
            elif isinstance(inputs_tensor, torch.Tensor) and not isinstance(labels_tensor, torch.Tensor):
                logger.warning(f"Skipping batch {batch_idx} in {phase}: 'image' is a tensor, but 'label' is not (type: {type(labels_tensor)}).")
                continue

            inputs = inputs_tensor.to(device)
            labels = labels_tensor.to(device)
            
            outputs = model(inputs); loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted_classes = torch.max(outputs, 1)
            correct_predictions += (predicted_classes == labels).sum().item()
            total_predictions += labels.size(0)
            
            all_labels_eval.extend(labels.cpu().numpy())
            all_predicted_classes_eval.extend(predicted_classes.cpu().numpy())
            all_preds_proba_eval.extend(torch.softmax(outputs, dim=1).cpu().numpy())
            
    prog_bar.close()
    
    eval_loss = running_loss / total_predictions if total_predictions > 0 else float('inf')
    eval_acc = correct_predictions / total_predictions if total_predictions > 0 else 0
    f1, precision, recall, auc_score_val = 0.0, 0.0, 0.0, 0.0
    
    # Define labels for sklearn metrics, using current_num_classes
    # This ensures metrics are calculated over all potential classes
    metric_labels_range = list(range(current_num_classes)) if current_num_classes is not None and current_num_classes > 0 else None

    if total_predictions > 0 and len(all_labels_eval) > 0:
        if current_num_classes is None or current_num_classes <= 0:
            logger.error(f"{phase}: Cannot calculate metrics due to invalid current_num_classes: {current_num_classes}")
            # Determine shape of empty confusion matrix based on current_num_classes
            cm_shape = (current_num_classes, current_num_classes) if current_num_classes is not None and current_num_classes > 0 else (0,0)
            return {"loss": eval_loss, "accuracy": eval_acc, "f1_score": 0.0, "precision": 0.0, "recall": 0.0, "auc": 0.0, "confusion_matrix": np.zeros(cm_shape, dtype=int)}

        # If metric_labels_range is None (e.g. current_num_classes=0), sklearn will determine labels from data.
        # This might be okay, but for consistency, it's better if current_num_classes is always valid.
        f1 = f1_score(all_labels_eval, all_predicted_classes_eval, average='macro', zero_division=0, labels=metric_labels_range)
        precision = precision_score(all_labels_eval, all_predicted_classes_eval, average='macro', zero_division=0, labels=metric_labels_range)
        recall = recall_score(all_labels_eval, all_predicted_classes_eval, average='macro', zero_division=0, labels=metric_labels_range)
        
        try:
            unique_labels_in_eval = np.unique(all_labels_eval)
            # Use metric_labels_range for roc_auc_score's labels parameter as well
            if current_num_classes > 2 and len(unique_labels_in_eval) > 1:
                auc_score_val = roc_auc_score(all_labels_eval, all_preds_proba_eval, multi_class='ovr', average='macro', labels=metric_labels_range)
            elif current_num_classes == 2 and len(unique_labels_in_eval) == 2:
                if np.array(all_preds_proba_eval).ndim == 2 and np.array(all_preds_proba_eval).shape[1] >= 2:
                    auc_score_val = roc_auc_score(all_labels_eval, np.array(all_preds_proba_eval)[:, 1])
                else:
                    logger.warning(f"{phase} 중 AUC 계산 불가: 2클래스 문제이나 확률 배열 형식이 부적절합니다. Shape: {np.array(all_preds_proba_eval).shape}")
            elif len(unique_labels_in_eval) <= 1 :
                logger.debug(f"{phase} 중 AUC 계산 불가: 레이블 종류 부족 ({len(unique_labels_in_eval)}개). 전체 클래스 수: {current_num_classes}")
            else:
                logger.debug(f"{phase} 중 AUC 계산 조건 미충족. 레이블 종류: {len(unique_labels_in_eval)}, 클래스 수: {current_num_classes}")
        except ValueError as e:
            logger.warning(f"{phase} 중 AUC 계산 오류: {e}. AUC는 0.0으로 설정됩니다. Labels present: {np.unique(all_labels_eval)}. Probas shape: {np.array(all_preds_proba_eval).shape}")
            
    # Generate confusion matrix
    if total_predictions > 0 and metric_labels_range:
        conf_matrix_val = confusion_matrix(all_labels_eval, all_predicted_classes_eval, labels=metric_labels_range)
    elif current_num_classes is not None and current_num_classes > 0: # No predictions, but we know class count
        conf_matrix_val = np.zeros((current_num_classes, current_num_classes), dtype=int)
    elif current_num_classes == 0 : # NUM_CLASSES is 0
        conf_matrix_val = np.zeros((0,0), dtype=int)
    else: # Default, e.g. current_num_classes is None
        conf_matrix_val = np.array([]) # Or np.zeros((0,0), dtype=int) if preferred for consistency
    
    return {"loss": eval_loss, "accuracy": eval_acc, "f1_score": f1, "precision": precision, "recall": recall, "auc": auc_score_val, "confusion_matrix": conf_matrix_val}

# --- XAI: Grad-CAM 시각화 함수 ---
def save_gradcam_slices(original_image_np, cam_map_np, patient_id, pred_class_name_str, true_class_name_str, output_dir, filename_prefix="gradcam"):
    if original_image_np.ndim == 4 and original_image_np.shape[0] == 1: original_image_np = original_image_np.squeeze(0)
    elif original_image_np.ndim != 3: logger.error(f"XAI 시각화 오류: 원본 이미지 차원({original_image_np.shape})이 (D,H,W)가 아님."); return
    
    if cam_map_np.ndim == 4 and cam_map_np.shape[0] == 1: cam_map_np = cam_map_np.squeeze(0)
    # Ensure cam_map_np has 3 dimensions for comparison
    if cam_map_np.ndim != 3:
        logger.error(f"XAI 시각화 오류: CAM 맵 차원({cam_map_np.shape})이 3이 아님."); return

    if original_image_np.shape != cam_map_np.shape:
        logger.warning(f"XAI 시각화: 원본({original_image_np.shape})과 CAM({cam_map_np.shape}) 차원이 불일치. CAM 리사이징 시도.")
        try:
            # Add channel dim for ResizeD if not present, assuming cam_map_np is (D,H,W)
            cam_map_tensor = torch.tensor(cam_map_np[np.newaxis, ...]).float() # (1, D, H, W)
            resizer = ResizeD(keys=["img"], spatial_size=original_image_np.shape, mode="trilinear", align_corners=True)
            # resizer expects a dict, and key should match 'keys'
            cam_map_np_resized = resizer({"img": cam_map_tensor})["img"].squeeze().numpy()
            if cam_map_np_resized.shape == original_image_np.shape:
                cam_map_np = cam_map_np_resized
            else:
                logger.error(f"CAM 맵 리사이즈 후에도 차원 불일치: {cam_map_np_resized.shape} vs {original_image_np.shape}. 시각화를 건너뜁니다."); return
        except Exception as e_resize: logger.error(f"CAM 맵 리사이즈 실패: {e_resize}. 시각화를 건너뜁니다."); return

    depth, height, width = original_image_np.shape
    slices_to_show = {
        "axial": (original_image_np[depth // 2, :, :], cam_map_np[depth // 2, :, :]),
        "coronal": (original_image_np[:, height // 2, :], cam_map_np[:, height // 2, :]),
        "sagittal": (original_image_np[:, :, width // 2], cam_map_np[:, :, width // 2])}
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    title_str = f"Grad-CAM: Patient {patient_id} (Predicted: {pred_class_name_str}, True: {true_class_name_str})"
    # Truncate title if too long to prevent display issues
    fig.suptitle(title_str[:150] + ('...' if len(title_str) > 150 else ''), fontsize=16)

    for i, (view_name, (img_slice, cam_slice)) in enumerate(slices_to_show.items()):
        if img_slice.ndim != 2 or cam_slice.ndim != 2: 
            logger.error(f"XAI 시각화 오류: {view_name} 뷰 슬라이스가 2D 아님 (Img: {img_slice.shape}, CAM: {cam_slice.shape})."); continue
        axes[i].imshow(np.rot90(img_slice), cmap="gray"); axes[i].imshow(np.rot90(cam_slice), cmap="jet", alpha=0.5)
        axes[i].set_title(f"{view_name.capitalize()} View"); axes[i].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust rect to prevent suptitle overlap
    safe_pred = "".join(c if c.isalnum() else '_' for c in str(pred_class_name_str))[:30] # Limit length
    safe_true = "".join(c if c.isalnum() else '_' for c in str(true_class_name_str))[:30] # Limit length
    safe_pid = "".join(c if c.isalnum() else '_' for c in str(patient_id))[:50] # Limit length
    
    save_path = os.path.join(output_dir, f"{filename_prefix}_pid_{safe_pid}_pred_{safe_pred}_true_{safe_true}.png")
    try: 
        plt.savefig(save_path); logger.info(f"Grad-CAM 시각화 저장 완료: {save_path}")
    except Exception as e_save: logger.error(f"Grad-CAM 시각화 저장 실패 ({save_path}): {e_save}")
    finally: plt.close(fig)

# --- 메인 실행 로직 ---
if __name__ == "__main__":
    logger.info("--- NIfTI 영상 기반 암 분류 모델 학습 및 XAI 적용 시작 ---")
    logger.info(f"사용 디바이스: {DEVICE}"); logger.info(f"결과 저장 폴더: {BASE_OUTPUT_DIR}");
    logger.info(f"XAI 결과 저장 폴더: {XAI_OUTPUT_DIR}"); logger.info(f"데이터 로더 워커 수: {NUM_WORKERS_DATALOADER}")
    
    all_data_dicts_initial_load, label_to_int_map_ref_initial, int_to_label_map_for_initial_logging = load_and_prepare_data_from_manifest(
        MANIFEST_FILE_PATH, PATIENT_ID_COL, LABEL_COL)
    if not all_data_dicts_initial_load: logger.error("Manifest에서 데이터를 로드하지 못했습니다. 시스템을 종료합니다."); sys.exit(1) # Add exit code
    
    logger.info(f"Manifest 로드 후 초기 NUM_CLASSES (original_label 기준): {NUM_CLASSES}")
    logger.info(f"참고용 초기 (original_label 문자열 -> 정수) 맵: {label_to_int_map_ref_initial}")
    logger.info(f"참고용 초기 (정수 -> original_label 문자열) 맵: {int_to_label_map_for_initial_logging}")

    logger.info("--- 데이터 유효성 검사 및 변환 테스트 시작 ---")
    valid_data_dicts_after_transform_test = []
    problematic_items_log = []; transforms_to_test_validity = val_test_transforms
    prog_bar_val_test = tqdm(all_data_dicts_initial_load, desc="Testing Data Item Transformations", leave=False)
    for i, item_dict_from_initial_load in enumerate(prog_bar_val_test):
        try:
            # Need to copy original dict as transforms can modify it in place
            transformed_item = transforms_to_test_validity(item_dict_from_initial_load.copy())
            # Ensure transformation did not return None or something unexpected
            if transformed_item is None or not isinstance(transformed_item, dict) or "image" not in transformed_item:
                 raise ValueError("Transformation returned None or invalid dictionary.")
            valid_data_dicts_after_transform_test.append(item_dict_from_initial_load) # append original if transform is successful
        except Exception as e_item_transform:
            pid = item_dict_from_initial_load.get(PATIENT_ID_COL, "N/A"); image_path = item_dict_from_initial_load.get("image", "N/A")
            log_msg = f"항목 {i} (PID: {pid}, Path: {image_path}) 변환 중 오류( 제외됨): {type(e_item_transform).__name__} - {str(e_item_transform).splitlines()[0]}"
            logger.warning(log_msg); problematic_items_log.append(log_msg)
    prog_bar_val_test.close()
    logger.info(f"데이터 유효성 검사 완료. 총 {len(all_data_dicts_initial_load)}개 중 {len(valid_data_dicts_after_transform_test)}개 유효.")
    if problematic_items_log:
        logger.warning("--- 문제 발생 항목 요약 (학습에서 제외됨) ---")
        for log_entry in problematic_items_log[:20]: logger.warning(log_entry)
        if len(problematic_items_log) > 20: logger.warning(f"... 외 {len(problematic_items_log) - 20}개의 추가 문제 항목이 있습니다.")
    all_data_dicts = valid_data_dicts_after_transform_test
    if not all_data_dicts: logger.error("변환 가능한 유효한 데이터가 없어 학습을 진행할 수 없습니다."); sys.exit(1)

    current_encoded_labels = [d['label'].item() for d in all_data_dicts]
    
    current_encoded_to_str_map = {}
    for d_item in all_data_dicts:
        label_val = d_item['label'].item()
        if label_val not in current_encoded_to_str_map:
            current_encoded_to_str_map[label_val] = d_item['original_label_str']
            
    current_label_counts = Counter(current_encoded_labels)
    logger.info(f"변환 테스트 후, 레이블 처리 전 분포 ('label_encoded' 기준): {current_label_counts}")
    logger.info(f"각 'label_encoded' 값의 원본 문자열 레이블 (변환 테스트 후 데이터 기준):")
    for encoded_val, count in sorted(current_label_counts.items()): # Sort for consistent logging
        logger.info(f"  - '{current_encoded_to_str_map.get(encoded_val, f'Unknown_Str_{encoded_val}')}' (초기 encoded: {encoded_val}): {count} 개")

    MIN_SAMPLES_PER_CLASS_FOR_SPLIT = 2 # Should be at least K_FOLDS if K_FOLDS > 1 for stratified k-fold
    if K_FOLDS > 1: MIN_SAMPLES_PER_CLASS_FOR_SPLIT = max(MIN_SAMPLES_PER_CLASS_FOR_SPLIT, K_FOLDS)

    encoded_labels_to_remove_due_to_rarity = {label for label, count in current_label_counts.items() if count < MIN_SAMPLES_PER_CLASS_FOR_SPLIT}
    if encoded_labels_to_remove_due_to_rarity:
        logger.warning(f"다음 초기 인코딩된 레이블들은 샘플 수가 {MIN_SAMPLES_PER_CLASS_FOR_SPLIT}개 미만이어서 추가로 제외됩니다: {sorted(list(encoded_labels_to_remove_due_to_rarity))}")
        for removed_enc_label in sorted(list(encoded_labels_to_remove_due_to_rarity)): 
            logger.warning(f"  - 희소 클래스 제거 대상: '{current_encoded_to_str_map.get(removed_enc_label, f'Unknown_Str_{removed_enc_label}')}' (초기 encoded: {removed_enc_label})")
        len_before_rare_filter = len(all_data_dicts)
        all_data_dicts = [d for d in all_data_dicts if d['label'].item() not in encoded_labels_to_remove_due_to_rarity]
        logger.info(f"희소 클래스 샘플 제거 후: 총 {len(all_data_dicts)}개 항목 (이전: {len_before_rare_filter}개)")
    if not all_data_dicts: logger.error("희소 클래스 필터링 후 유효한 데이터가 없어 학습을 진행할 수 없습니다."); sys.exit(1)

    logger.info("--- 최종 레이블 재매핑 수행 (0부터 시작하는 연속적인 정수로) ---")
    final_map_current_encoded_to_original_str = {}
    for d_current in all_data_dicts:
        current_enc = d_current['label'].item()
        original_str = d_current['original_label_str']
        if current_enc not in final_map_current_encoded_to_original_str:
            final_map_current_encoded_to_original_str[current_enc] = original_str
            
    # Ensure original string labels are unique if their current_encoded values are different
    # This should already be true if current_encoded_to_str_map was built correctly
    
    # sorted_final_unique_original_str_labels = sorted(list(set(d['original_label_str'] for d in all_data_dicts)))
    # This re-derives unique strings from the filtered data
    
    # The goal is to map the *remaining unique* current_encoded_labels to 0..N-1
    remaining_current_encoded_labels = sorted(list(set(d['label'].item() for d in all_data_dicts)))

    NUM_CLASSES = len(remaining_current_encoded_labels) # This is the true number of classes for the model
    logger.info(f"최종 레이블 재매핑 후 실제 클래스 수 (NUM_CLASSES 업데이트): {NUM_CLASSES}")

    if NUM_CLASSES == 0: logger.error("최종적으로 유효한 클래스가 없습니다. 학습을 진행할 수 없습니다."); sys.exit(1)
    
    current_encoded_to_final_cont_int_map = {old_label: new_label for new_label, old_label in enumerate(remaining_current_encoded_labels)}
    final_int_to_label_map = {
        new_label: final_map_current_encoded_to_original_str[old_label]
        for old_label, new_label in current_encoded_to_final_cont_int_map.items()
    }
    
    logger.info(f"최종 연속 레이블 매핑 (현재 남아있는 초기 인코딩 -> 최종 새 인코딩): {current_encoded_to_final_cont_int_map}")
    logger.info(f"최종 역 레이블 매핑 (최종 새 인코딩 -> 원본 문자열): {final_int_to_label_map}")
    
    for item_dict in all_data_dicts: 
        item_dict['label'] = torch.tensor(current_encoded_to_final_cont_int_map[item_dict['label'].item()], dtype=torch.long)
    
    labels_for_stratify = [d['label'].item() for d in all_data_dicts]
    logger.info(f"최종 레이블 재매핑 완료. 최종 레이블 분포 (최종 새 인코딩 기준): {Counter(labels_for_stratify)}")

    if not labels_for_stratify: logger.error("학습/분할에 사용할 데이터가 없습니다 (최종 필터링/재매핑 후)."); sys.exit(1)
    if any(l < 0 or l >= NUM_CLASSES for l in labels_for_stratify): 
        logger.error(f"오류 FATAL: 최종 재매핑 후에도 레이블이 [0, {NUM_CLASSES-1}] 범위를 벗어납니다! Labels: {Counter(labels_for_stratify)}"); sys.exit("레이블 범위 오류로 중단합니다.")
    if len(np.unique(labels_for_stratify)) != NUM_CLASSES: 
        logger.error(f"오류 FATAL: 최종 고유 레이블 수({len(np.unique(labels_for_stratify))})와 NUM_CLASSES({NUM_CLASSES}) 불일치!"); sys.exit("NUM_CLASSES 설정 오류로 중단합니다.")

    # Ensure enough classes for splitting if K_FOLDS > 1 or validation/test set is desired
    if NUM_CLASSES < 2 and (K_FOLDS > 1 or TEST_SPLIT_RATIO > 0 or VAL_SPLIT_RATIO > 0):
        logger.warning(f"데이터 분할에 필요한 최소 레이블 종류(2개) 미만 (현재: {NUM_CLASSES}개). 전체 데이터 학습으로 전환합니다 (K_FOLDS=0, TEST_SPLIT_RATIO=0, VAL_SPLIT_RATIO=0).")
        K_FOLDS = 0; TEST_SPLIT_RATIO = 0; VAL_SPLIT_RATIO = 0
    
    criterion = nn.CrossEntropyLoss(); logger.info(f"손실 함수: CrossEntropyLoss, 최종 NUM_CLASSES для модели: {NUM_CLASSES}")
    final_trained_model = None

    if K_FOLDS > 1:
        min_samples_in_class_for_kfold = min(Counter(labels_for_stratify).values()) if labels_for_stratify and Counter(labels_for_stratify) else 0
        if min_samples_in_class_for_kfold < K_FOLDS:
            logger.warning(f"K_FOLDS({K_FOLDS})가 최소 클래스 샘플 수({min_samples_in_class_for_kfold})보다 큽니다. Hold-out으로 전환합니다 (K_FOLDS=0).")
            K_FOLDS = 0
        
        if K_FOLDS > 1 : # Re-check K_FOLDS after potential modification
            logger.info(f"--- {K_FOLDS}-Fold 교차 검증 시작 ---")
            skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
            fold_results = []
            for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(all_data_dicts)), labels_for_stratify)):
                logger.info(f"--- Fold {fold + 1}/{K_FOLDS} ---")
                train_data_fold = [all_data_dicts[i] for i in train_idx]; val_data_fold = [all_data_dicts[i] for i in val_idx]
                
                train_dataset_fold = ErrorTolerantDataset(data=train_data_fold, transform=train_transforms, patient_id_col_name=PATIENT_ID_COL)
                val_dataset_fold = ErrorTolerantDataset(data=val_data_fold, transform=val_test_transforms, patient_id_col_name=PATIENT_ID_COL)
                train_loader_fold = DataLoader(train_dataset_fold, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS_DATALOADER, pin_memory=torch.cuda.is_available(), collate_fn=safe_list_data_collate)
                val_loader_fold = DataLoader(val_dataset_fold, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS_DATALOADER, pin_memory=torch.cuda.is_available(), collate_fn=safe_list_data_collate)
                
                model = CTClassifier(num_classes_final=NUM_CLASSES, pretrained_weights_path=PRETRAINED_WEIGHTS_PATH, freeze_feature_extractor=True).to(DEVICE)
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
                best_val_auc_fold = -1.0; best_model_state_fold = None
                
                for epoch in range(NUM_EPOCHS):
                    logger.info(f"Fold {fold+1}, Epoch {epoch+1}/{NUM_EPOCHS}")
                    if epoch == FREEZE_FEATURE_EXTRACTOR_EPOCHS and PRETRAINED_WEIGHTS_PATH: # Check PRETRAINED_WEIGHTS_PATH to ensure unfreezing is relevant
                        logger.info("특징 추출기 동결 해제 및 옵티마이저 재설정."); model.unfreeze_feature_extractor()
                        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE / 10) # Re-init optimizer for all params
                    
                    train_loss, train_acc, train_auc = train_one_epoch(model, train_loader_fold, criterion, optimizer, DEVICE, current_num_classes=NUM_CLASSES)
                    logger.info(f"Fold {fold+1} Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AUC: {train_auc:.4f}")
                    
                    val_metrics = evaluate_model(model, val_loader_fold, criterion, DEVICE, phase=f"Fold {fold+1} Validation", current_num_classes=NUM_CLASSES)
                    logger.info(f"Fold {fold+1} Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, AUC: {val_metrics['auc']:.4f}, F1: {val_metrics['f1_score']:.4f}")
                    
                    if val_metrics['auc'] > best_val_auc_fold: 
                        best_val_auc_fold = val_metrics['auc']; best_model_state_fold = copy.deepcopy(model.state_dict())
                        logger.info(f"Fold {fold+1} - 새로운 최적 검증 AUC: {best_val_auc_fold:.4f} (Epoch {epoch+1})")
                
                fold_results.append({"fold": fold + 1, "best_val_auc": best_val_auc_fold, "best_model_state": best_model_state_fold})
                logger.info(f"Fold {fold+1} 완료. 해당 Fold 최고 검증 AUC: {best_val_auc_fold:.4f}")
            
            if fold_results: # Ensure fold_results is not empty
                best_overall_fold_result = max(fold_results, key=lambda x: x['best_val_auc'])
                logger.info("\n--- 교차 검증 결과 요약 ---")
                for res in fold_results: logger.info(f"Fold {res['fold']}: Best Val AUC = {res['best_val_auc']:.4f}")
                logger.info(f"전체 Fold 중 최적 성능 Fold: {best_overall_fold_result['fold']} (Val AUC: {best_overall_fold_result['best_val_auc']:.4f})")
                
                if best_overall_fold_result['best_model_state']:
                    final_model_path = os.path.join(BASE_OUTPUT_DIR, MODEL_CHECKPOINT_NAME)
                    torch.save(best_overall_fold_result['best_model_state'], final_model_path); logger.info(f"최종 최적 모델(K-Fold) 저장 완료: {final_model_path}")
                    final_trained_model = CTClassifier(num_classes_final=NUM_CLASSES, pretrained_weights_path=None, freeze_feature_extractor=False).to(DEVICE) # Init new model instance
                    final_trained_model.load_state_dict(best_overall_fold_result['best_model_state'])
                else: logger.error("교차 검증 후 최적 모델 상태(best_model_state)를 찾지 못했습니다 (None).")
            else: logger.error("교차 검증 결과가 없어 (fold_results 비어 있음) 최적 Fold를 선택할 수 없습니다.")
    
    # Hold-out validation or training on full data if K_FOLDS <= 1 (or was set to 0)
    if K_FOLDS <= 1:
        logger.info("--- Hold-out 검증 또는 전체 데이터 학습 시작 ---")
        train_indices, val_indices, test_indices = list(range(len(all_data_dicts))), [], [] # Default to all train
        
        if not all_data_dicts: logger.error("분할할 데이터가 없습니다."); sys.exit(1)

        # Stratification requires at least 2 samples per class for the smallest split.
        # And at least 2 classes. NUM_CLASSES check already done.
        can_stratify = NUM_CLASSES >= 2 and all(c >= 1 for c in Counter(labels_for_stratify).values()) # Basic check

        if TEST_SPLIT_RATIO > 0:
            # Stratify if possible, otherwise split without it
            stratify_logic_test = labels_for_stratify if can_stratify and all(c >= (2 if VAL_SPLIT_RATIO > 0 else 1) for c in Counter(labels_for_stratify).values()) else None # Stricter check for multi-split
            if stratify_logic_test is None and TEST_SPLIT_RATIO > 0: logger.warning(f"Test 분할 위한 stratify 불가 (클래스 수 또는 샘플 수 부족). Stratify 없이 분할합니다.")
            try:
                # Ensure test_size doesn't lead to empty train_val set if VAL_SPLIT_RATIO is also high
                actual_test_size = TEST_SPLIT_RATIO
                if VAL_SPLIT_RATIO > 0 and (1 - TEST_SPLIT_RATIO) * VAL_SPLIT_RATIO / (1 - TEST_SPLIT_RATIO) == 0 : # Avoid 0 val if test is too large
                     pass # This logic is complex, simplifying for now. train_test_split handles small sets by potentially returning empty ones.

                train_val_indices, test_indices = train_test_split(
                    list(range(len(all_data_dicts))), 
                    test_size=actual_test_size, 
                    stratify=stratify_logic_test, 
                    random_state=RANDOM_SEED
                )
                train_indices = train_val_indices # Initially, all non-test are train_val
            except ValueError as e_split_test:
                logger.error(f"Test 분할 중 오류 (stratify 시도: {stratify_logic_test is not None}): {e_split_test}. 분할 재시도 (stratify=None).")
                try:
                    train_val_indices, test_indices = train_test_split(list(range(len(all_data_dicts))), test_size=TEST_SPLIT_RATIO, stratify=None, random_state=RANDOM_SEED)
                    train_indices = train_val_indices
                except ValueError as e_split_test_no_strat: # If still fails (e.g. dataset too small for split)
                    logger.error(f"Test 분할 최종 실패 (stratify=None): {e_split_test_no_strat}. 전체 데이터를 훈련용으로 사용하고 테스트셋 없음.")
                    train_indices = list(range(len(all_data_dicts)))
                    test_indices = []


        if VAL_SPLIT_RATIO > 0 and len(train_indices) > 0 : # train_indices is now effectively train_val_indices
            # Calculate val_split_ratio relative to the current train_val set
            # If TEST_SPLIT_RATIO was 0, then (1.0 - TEST_SPLIT_RATIO) is 1.0
            # If TEST_SPLIT_RATIO was > 0, train_indices is smaller.
            # The ratio should apply to the remaining data after test split.
            
            # Check if train_indices is large enough for validation split
            if len(train_indices) < 2 : # Need at least 2 samples to split into train and val
                logger.warning(f"검증셋 분할에 필요한 최소 샘플 수(2) 미만 ({len(train_indices)}개). 검증셋 없이 훈련 진행.")
                val_indices = [] # train_indices remains as is (becomes final training set)
            else:
                # Stratify validation split if possible from the train_val set
                train_val_labels_for_split = [labels_for_stratify[i] for i in train_indices]
                can_stratify_val = NUM_CLASSES >=2 and all(c >= 1 for c in Counter(train_val_labels_for_split).values()) # Simplified check
                
                stratify_logic_val = train_val_labels_for_split if can_stratify_val and all(c >= 1 for c in Counter(train_val_labels_for_split).values()) else None # Ensure at least 1 sample per class for smaller set
                if stratify_logic_val is None and VAL_SPLIT_RATIO > 0 : logger.warning(f"Train/Val 분할 위한 stratify 불가. Stratify 없이 분할합니다.")

                # Relative validation ratio: VAL_SPLIT_RATIO is of the original total.
                # We need to find how much of the current train_val set this corresponds to.
                # Example: Total 100, Test 0.2 (20), Val 0.1 (10). Train_val is 80. Val is 10/80 = 0.125 of train_val.
                # If Test_Split_Ratio was 0, Val_Split_Ratio is directly applied.
                if (1.0 - TEST_SPLIT_RATIO) <= 0 : # Avoid division by zero if TEST_SPLIT_RATIO is 1 or more
                    relative_val_split_ratio_of_train_val = 0 if len(train_indices) > 0 else 1.0 # effectively no validation or all validation
                else:
                    relative_val_split_ratio_of_train_val = VAL_SPLIT_RATIO / (1.0 - TEST_SPLIT_RATIO)
                
                if relative_val_split_ratio_of_train_val >= 1.0 : # e.g. VAL_SPLIT > remaining data
                    logger.warning(f"요청된 검증셋 비율({VAL_SPLIT_RATIO:.2f})이 테스트셋 제외 후 남은 데이터 비율({1.0 - TEST_SPLIT_RATIO:.2f})보다 크거나 같아, 훈련 데이터가 없을 수 있습니다. 검증셋 없이 진행합니다.")
                    val_indices = [] # train_indices remains for training
                elif relative_val_split_ratio_of_train_val > 0 :
                    try:
                        train_indices_final, val_indices_temp = train_test_split(
                            train_indices, # This is train_val_indices
                            test_size=relative_val_split_ratio_of_train_val,
                            stratify=stratify_logic_val,
                            random_state=RANDOM_SEED
                        )
                        train_indices = train_indices_final
                        val_indices = val_indices_temp
                    except ValueError as e_split_val:
                        logger.error(f"Train/Val 분할 중 오류 (stratify 시도: {stratify_logic_val is not None}): {e_split_val}. 분할 재시도 (stratify=None).")
                        try:
                            train_indices_final, val_indices_temp = train_test_split(train_indices, test_size=relative_val_split_ratio_of_train_val, stratify=None, random_state=RANDOM_SEED)
                            train_indices = train_indices_final
                            val_indices = val_indices_temp
                        except ValueError as e_split_val_no_strat:
                             logger.error(f"Train/Val 분할 최종 실패 (stratify=None): {e_split_val_no_strat}. 검증셋 없이 진행합니다.")
                             val_indices = [] # train_indices remains as is
                else: # relative_val_split_ratio is 0 or negative
                    val_indices = [] # No validation set
        
        # If only VAL_SPLIT_RATIO is set (TEST_SPLIT_RATIO = 0)
        elif VAL_SPLIT_RATIO > 0 and not test_indices : # and len(train_indices) == len(all_data_dicts)
            if len(all_data_dicts) < 2:
                 logger.warning(f"검증셋 분할에 필요한 최소 샘플 수(2) 미만 ({len(all_data_dicts)}개). 검증셋 없이 훈련 진행.")
                 val_indices = []
            else:
                stratify_logic_val_only = labels_for_stratify if can_stratify and all(c >= 1 for c in Counter(labels_for_stratify).values()) else None
                if stratify_logic_val_only is None and VAL_SPLIT_RATIO > 0: logger.warning("Validation 분할 위한 stratify 불가. Stratify 없이 분할합니다.")
                try:
                    train_indices_final, val_indices_temp = train_test_split(
                        list(range(len(all_data_dicts))), # Use all_data_dicts as base
                        test_size=VAL_SPLIT_RATIO, 
                        stratify=stratify_logic_val_only, 
                        random_state=RANDOM_SEED
                    )
                    train_indices = train_indices_final
                    val_indices = val_indices_temp
                except ValueError as e_split_val_only:
                    logger.error(f"Validation 분할 중 오류 (stratify 시도: {stratify_logic_val_only is not None}): {e_split_val_only}. 분할 재시도 (stratify=None).")
                    try:
                        train_indices_final, val_indices_temp = train_test_split(list(range(len(all_data_dicts))), test_size=VAL_SPLIT_RATIO, stratify=None, random_state=RANDOM_SEED)
                        train_indices = train_indices_final
                        val_indices = val_indices_temp
                    except ValueError as e_split_val_only_no_strat:
                        logger.error(f"Validation 분할 최종 실패 (stratify=None): {e_split_val_only_no_strat}. 전체를 Train으로 사용합니다.")
                        train_indices = list(range(len(all_data_dicts)))
                        val_indices = []
        
        train_data = [all_data_dicts[i] for i in train_indices] if train_indices else []
        val_data = [all_data_dicts[i] for i in val_indices] if val_indices else []
        test_data = [all_data_dicts[i] for i in test_indices] if test_indices else []
        
        logger.info(f"데이터 분할: 훈련 {len(train_data)}개, 검증 {len(val_data)}개, 테스트 {len(test_data)}개")
        if not train_data: logger.error("훈련 데이터가 없습니다. 종료합니다."); sys.exit(1)

        train_dataset = ErrorTolerantDataset(data=train_data, transform=train_transforms, patient_id_col_name=PATIENT_ID_COL)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS_DATALOADER, pin_memory=torch.cuda.is_available(), collate_fn=safe_list_data_collate)
        
        val_loader = None
        if val_data:
            val_dataset = ErrorTolerantDataset(data=val_data, transform=val_test_transforms, patient_id_col_name=PATIENT_ID_COL)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS_DATALOADER, pin_memory=torch.cuda.is_available(), collate_fn=safe_list_data_collate)
        
        test_loader = None
        if test_data:
            test_dataset = ErrorTolerantDataset(data=test_data, transform=val_test_transforms, patient_id_col_name=PATIENT_ID_COL)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS_DATALOADER, pin_memory=torch.cuda.is_available(), collate_fn=safe_list_data_collate)

        model = CTClassifier(num_classes_final=NUM_CLASSES, pretrained_weights_path=PRETRAINED_WEIGHTS_PATH, freeze_feature_extractor=True).to(DEVICE)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
        best_val_metric_holdout = -1.0; best_model_state_holdout = None
        
        for epoch in range(NUM_EPOCHS):
            logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS}")
            if epoch == FREEZE_FEATURE_EXTRACTOR_EPOCHS and PRETRAINED_WEIGHTS_PATH:
                logger.info("특징 추출기 동결 해제 및 옵티마이저 재설정."); model.unfreeze_feature_extractor()
                optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE / 10)
            
            train_loss, train_acc, train_auc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, current_num_classes=NUM_CLASSES)
            logger.info(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AUC: {train_auc:.4f}")
            
            current_epoch_val_metric_for_best_model = train_auc # Default to train_auc if no val_loader
            
            if val_loader:
                val_metrics = evaluate_model(model, val_loader, criterion, DEVICE, phase="Validation", current_num_classes=NUM_CLASSES)
                logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, AUC: {val_metrics['auc']:.4f}, F1: {val_metrics['f1_score']:.4f}")
                current_epoch_val_metric_for_best_model = val_metrics['auc'] # Use val_auc if available
            else: logger.info("검증 데이터로더(val_loader)가 없습니다. 훈련 AUC를 기준으로 최적 모델을 저장합니다.")
            
            if current_epoch_val_metric_for_best_model >= best_val_metric_holdout: # Use '>=' to prefer later epochs in case of tie
                best_val_metric_holdout = current_epoch_val_metric_for_best_model
                best_model_state_holdout = copy.deepcopy(model.state_dict())
                logger.info(f"새로운 최적 {'검증 AUC' if val_loader else '훈련 AUC'}: {best_val_metric_holdout:.4f} (Epoch {epoch+1})")
        
        if best_model_state_holdout:
            final_model_path = os.path.join(BASE_OUTPUT_DIR, MODEL_CHECKPOINT_NAME)
            torch.save(best_model_state_holdout, final_model_path); logger.info(f"최종 최적 모델(Hold-out/Full Train) 저장 완료: {final_model_path}")
            final_trained_model = CTClassifier(num_classes_final=NUM_CLASSES, pretrained_weights_path=None, freeze_feature_extractor=False).to(DEVICE)
            final_trained_model.load_state_dict(best_model_state_holdout)
        elif model: # Fallback to the last epoch model if no improvement was seen (best_model_state_holdout is None)
            final_trained_model = model 
            logger.warning("저장된 최적 모델 상태(best_model_state_holdout)가 없습니다 (예: 성능 향상 없음). 마지막 에폭의 모델을 사용하고 저장합니다.")
            final_model_path = os.path.join(BASE_OUTPUT_DIR, MODEL_CHECKPOINT_NAME)
            torch.save(model.state_dict(), final_model_path); logger.info(f"마지막 에폭 모델 저장 완료: {final_model_path}")
        else: logger.error("훈련된 모델을 찾을 수 없습니다."); final_trained_model = None


    if final_trained_model and test_loader:
        logger.info("\n--- 테스트셋 최종 평가 ---"); final_trained_model.eval()
        test_metrics = evaluate_model(final_trained_model, test_loader, criterion, DEVICE, phase="Test", current_num_classes=NUM_CLASSES)
        logger.info(f"Test Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['accuracy']:.4f}, AUC: {test_metrics['auc']:.4f}, F1: {test_metrics['f1_score']:.4f}")
        logger.info(f"Test Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}")
        # Ensure confusion matrix is numpy array before printing
        cm_to_print = test_metrics['confusion_matrix']
        if not isinstance(cm_to_print, np.ndarray): cm_to_print = np.array([]) # or handle as error
        logger.info(f"Test Confusion Matrix (rows: true, cols: pred):\n{cm_to_print}")
        
        if XAI_NUM_SAMPLES_TO_VISUALIZE > 0 and test_data:
            logger.info("\n--- XAI (Grad-CAM) 결과 생성 시작 ---")
            try:
                grad_cam_target_layer_str = f"feature_extractor.{XAI_TARGET_LAYER_NAME}"
                grad_cam_obj = GradCAM(nn_module=final_trained_model, target_layers=grad_cam_target_layer_str)
                logger.info(f"Grad-CAM 대상 레이어 설정: {grad_cam_target_layer_str}"); visualized_count = 0
                
                for i, item_dict_for_xai_original in enumerate(test_data):
                    if visualized_count >= XAI_NUM_SAMPLES_TO_VISUALIZE: break
                    try:
                        # Apply val_test_transforms to the original item dictionary for XAI
                        # Make a copy to avoid in-place modification of test_data items
                        data_for_xai_transformed = val_test_transforms(item_dict_for_xai_original.copy())
                        if data_for_xai_transformed is None or "image" not in data_for_xai_transformed:
                            logger.warning(f"XAI 샘플 {i} (PID: {item_dict_for_xai_original.get(PATIENT_ID_COL,'N/A')}) 변환 실패 또는 이미지 없음. 건너뜁니다.")
                            continue

                        input_tensor_gpu = data_for_xai_transformed["image"].unsqueeze(0).to(DEVICE)
                        # For visualization, the original image might be before ResizeD if we want full res, 
                        # but here we use the same input as the model for consistency.
                        original_image_for_vis_np = input_tensor_gpu.squeeze(0).cpu().numpy() # This is the resized input
                        
                        final_trained_model.eval() # Ensure model is in eval mode
                        with torch.no_grad(): logits = final_trained_model(input_tensor_gpu)
                        predicted_class_idx_final = torch.argmax(logits, dim=1).item()
                        
                        # GradCAM expects class_idx to be scalar if only one sample in batch
                        cam_map = grad_cam_obj(x=input_tensor_gpu, class_idx=predicted_class_idx_final) # class_idx can be None to use predicted class
                        cam_map_np = cam_map.squeeze().cpu().detach().numpy() # Squeeze batch and channel
                        
                        predicted_class_name_str = final_int_to_label_map.get(predicted_class_idx_final, f"Unknown_Pred_Class_{predicted_class_idx_final}")
                        true_class_name_str_from_dict = item_dict_for_xai_original["original_label_str"] # From original manifest data
                        patient_id_for_xai = item_dict_for_xai_original[PATIENT_ID_COL]
                        
                        logger.info(f"XAI 샘플 {i+1}/{len(test_data)} (Vis {visualized_count+1}): PID {patient_id_for_xai}, True_Str: '{true_class_name_str_from_dict}', Pred_Str: '{predicted_class_name_str}' (Pred_final_idx: {predicted_class_idx_final})")
                        save_gradcam_slices(original_image_for_vis_np, cam_map_np, patient_id_for_xai, predicted_class_name_str, true_class_name_str_from_dict, XAI_OUTPUT_DIR, filename_prefix=f"xai_test_sample_{visualized_count}")
                        visualized_count += 1
                    except Exception as e_xai_sample_item:
                        pid_err = item_dict_for_xai_original.get(PATIENT_ID_COL,'N/A')
                        logger.error(f"XAI 샘플 {i} (PID: {pid_err}) 개별 처리 중 오류: {type(e_xai_sample_item).__name__} - {e_xai_sample_item}")
                        logger.debug(traceback.format_exc())
                logger.info(f"XAI (Grad-CAM) 시각화 {visualized_count}개 완료. 저장 폴더: {XAI_OUTPUT_DIR}")
            except Exception as e_gradcam_setup: 
                logger.error(f"Grad-CAM 설정 또는 실행 중 오류 발생: {type(e_gradcam_setup).__name__} - {e_gradcam_setup}")
                logger.error(traceback.format_exc())
        elif not test_data: logger.info("테스트 데이터가 없어 XAI 시각화를 건너뜁니다.")
        elif XAI_NUM_SAMPLES_TO_VISUALIZE == 0: logger.info("XAI_NUM_SAMPLES_TO_VISUALIZE=0. XAI 시각화 건너뜁니다.")
    elif not final_trained_model: logger.error("훈련된 최종 모델이 없어 테스트 및 XAI를 진행할 수 없습니다.")
    elif not test_loader: logger.warning("테스트 로더가 없어 테스트 및 XAI를 진행할 수 없습니다 (아마도 테스트 데이터가 없는 경우).")
    
    logger.info("--- 모든 과정 완료 ---")