"""
Sistema avanzado de detección, segmentación y conteo de frutas en imágenes
Arquitectura modular y escalable para agricultura de precisión

Requisitos: Python 3.10+, torch, ultralytics, segment-anything, opencv-python, numpy, pandas, tqdm

Autor: [Tu Nombre]
Fecha: 2025-05-13
"""

import os
import cv2
import numpy as np
import torch
import json
import pandas as pd
from tqdm import tqdm
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

# --- Configuración global ---
CONFIG = {
    'detector': 'yolov8',  # Cambiado a 'yolov8' para reflejar el modelo usado
    'detector_weights': 'yolov8n.pt',  # Usar el modelo oficial descargado
    'sam_model_type': 'vit_h',
    'sam_checkpoint': 'sam_vit_h_4b8939.pth',
    'confidence_threshold': 0.4,
    'min_instance_area': 200,  # píxeles mínimos para considerar una fruta
    'output_format': 'json',  # o 'csv'
    'output_path': 'detections_output',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

# --- 1. Carga de modelos ---
def load_detector(config):
    if config['detector'] in ['yolov8', 'yolov9']:
        from ultralytics import YOLO
        model = YOLO(config['detector_weights'])
        return model
    elif config['detector'] == 'yolact':
        # Implementar carga de YOLACT si es necesario
        raise NotImplementedError('YOLACT no implementado aún')
    elif config['detector'] == 'maskrcnn':
        # Implementar carga de Mask R-CNN si es necesario
        raise NotImplementedError('Mask R-CNN no implementado aún')
    else:
        raise ValueError('Modelo de detector no soportado')

def load_sam(config):
    from segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry[config['sam_model_type']](checkpoint=config['sam_checkpoint'])
    sam.to(config['device'])
    predictor = SamPredictor(sam)
    return predictor

# --- 2. Procesamiento de imagen ---
def detect_fruits(detector, image, config):
    results = detector(image)
    detections = []
    for r in results[0].boxes:
        if r.conf < config['confidence_threshold']:
            continue
        bbox = r.xyxy.cpu().numpy().astype(int).tolist()[0]
        detections.append({
            'bbox': bbox,
            'conf': float(r.conf),
            'class': int(r.cls),
        })
    return detections

def is_suspect_box(bbox, image_shape, config):
    # Heurística: si la box es muy grande respecto al tamaño típico de fruta
    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)
    img_area = image_shape[0] * image_shape[1]
    if area > 0.15 * img_area:
        return True
    # Otras heurísticas pueden añadirse aquí
    return False

def segment_and_split(predictor, image, bbox, config):
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2]
    predictor.set_image(crop)
    # SAM2: segmentación automática de instancias
    masks, scores, _ = predictor.predict(
        multimask_output=True
    )
    # Postprocesado morfológico para separar instancias solapadas
    separated_masks = []
    for mask in masks:
        separated = split_mask_morphology(mask, config)
        separated_masks.extend(separated)
    # Ajustar coordenadas al sistema global
    global_masks = [offset_mask(m, x1, y1, image.shape) for m in separated_masks]
    return global_masks

# --- Watershed más agresivo + clusterización de picos ---
def split_mask_with_peaks(mask, config):
    # Calcula el distance transform
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    # Detecta picos locales (centros de frutas)
    coordinates = peak_local_max(dist, min_distance=10, threshold_abs=0.2*dist.max(), labels=mask)
    # Crea marcadores para watershed
    markers = np.zeros_like(mask, dtype=np.int32)
    for i, (y, x) in enumerate(coordinates, 1):
        markers[y, x] = i
    # Aplica watershed
    labels = cv2.watershed(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), markers.copy())
    separated = []
    for label in range(1, np.max(labels)+1):
        region = (labels == label).astype(np.uint8)
        if cv2.countNonZero(region) > config['min_instance_area']:
            separated.append(region)
    return separated

def split_mask_morphology(mask, config):
    mask = (mask * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.3*dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(opening, sure_fg)
    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown==255] = 0
    mask_color = cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR)
    cv2.watershed(mask_color, markers)
    separated = []
    for label in range(2, num_labels+2):
        region = (markers == label).astype(np.uint8)
        if cv2.countNonZero(region) > config['min_instance_area']:
            separated.append(region)
    # Si solo hay una región grande, intenta clusterizar por picos
    if len(separated) <= 1 and cv2.countNonZero(opening) > 2*config['min_instance_area']:
        separated = split_mask_with_peaks(opening, config)
    return separated

def offset_mask(mask, x_offset, y_offset, image_shape):
    h, w = mask.shape
    full_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    full_mask[y_offset:y_offset+h, x_offset:x_offset+w] = mask
    return full_mask

def filter_duplicates(masks, iou_threshold=0.5):
    # Elimina máscaras duplicadas usando IoU
    filtered = []
    for i, m1 in enumerate(masks):
        keep = True
        for j, m2 in enumerate(filtered):
            inter = np.logical_and(m1, m2).sum()
            union = np.logical_or(m1, m2).sum()
            iou = inter / union if union > 0 else 0
            if iou > iou_threshold:
                keep = False
                break
        if keep:
            filtered.append(m1)
    return filtered

def draw_instances_on_image(image, instances):
    import hashlib
    out_img = image.copy()
    color = (255, 0, 0)  # Azul fijo para todas las frutas
    for idx, inst in enumerate(instances):
        mask = inst['mask'].astype(bool)
        # Superponer máscara con transparencia
        out_img[mask] = cv2.addWeighted(out_img, 0.5, np.full_like(out_img, color, dtype=np.uint8), 0.5, 0)[mask]
        # Dibujar bounding box
        x1, y1, x2, y2 = inst['bbox']
        cv2.rectangle(out_img, (x1, y1), (x2, y2), color, 2)
        # Calcular el hash persistente igual que en export_results
        mask_bytes = inst['mask'].tobytes()
        bbox_bytes = np.array(inst['bbox']).tobytes()
        unique_hash = hashlib.sha1(mask_bytes + bbox_bytes).hexdigest()[:8]
        # Etiqueta con número y hash
        label = f"{idx+1} | {unique_hash}"
        cv2.putText(out_img, label, (x1, max(y1-10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out_img

def process_image(image_path, detector, predictor, config):
    image = cv2.imread(image_path)
    detections = detect_fruits(detector, image, config)
    all_instances = []
    for det in detections:
        bbox = det['bbox']
        # FORZAR segmentación SAM2 en todas las detecciones
        masks = segment_and_split(predictor, image, bbox, config)
        masks = filter_duplicates(masks)
        for mask in masks:
            x, y, w, h = cv2.boundingRect(mask)
            all_instances.append({
                'mask': mask,
                'bbox': [x, y, x+w, y+h],
                'class': det['class'],
                'conf': det['conf'],
            })
    # Filtrado final
    all_instances = [inst for inst in all_instances if cv2.countNonZero(inst['mask']) > config['min_instance_area']]
    # Guardar imagen procesada
    processed_img = draw_instances_on_image(image, all_instances)
    base, ext = os.path.splitext(image_path)
    processed_path = f"{base}_processed{ext}"
    cv2.imwrite(processed_path, processed_img)
    return all_instances

# --- Proceso detallado para mostrar progreso ---
def process_image_verbose(image_path, detector, predictor, config):
    image = cv2.imread(image_path)
    detections = detect_fruits(detector, image, config)
    print(f"Detecciones YOLO: {len(detections)}")
    all_instances = []
    for i, det in enumerate(detections):
        bbox = det['bbox']
        print(f"  Detección {i+1}: bbox={bbox}, conf={det['conf']:.2f}, clase={det['class']}")
        if is_suspect_box(bbox, image.shape, config):
            print("    Sospecha de múltiples frutas, aplicando SAM2...")
            masks = segment_and_split(predictor, image, bbox, config)
            masks = filter_duplicates(masks)
            print(f"    SAM2 generó {len(masks)} instancias")
            for j, mask in enumerate(masks):
                x, y, w, h = cv2.boundingRect(mask)
                print(f"      Instancia {j+1}: bbox={[x, y, x+w, y+h]}")
                all_instances.append({
                    'mask': mask,
                    'bbox': [x, y, x+w, y+h],
                    'class': det['class'],
                    'conf': det['conf'],
                })
        else:
            x1, y1, x2, y2 = bbox
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask[y1:y2, x1:x2] = 1
            print("    Caja normal, 1 fruta")
            all_instances.append({
                'mask': mask,
                'bbox': bbox,
                'class': det['class'],
                'conf': det['conf'],
            })
    # Filtrado final
    all_instances = [inst for inst in all_instances if cv2.countNonZero(inst['mask']) > config['min_instance_area']]
    print(f"Total instancias finales: {len(all_instances)}")
    # Guardar imagen procesada
    processed_img = draw_instances_on_image(image, all_instances)
    base, ext = os.path.splitext(image_path)
    processed_path = f"{base}_processed{ext}"
    cv2.imwrite(processed_path, processed_img)
    return all_instances

# --- 3. Exportación de resultados ---
def export_results(instances, image_path, config):
    import hashlib
    os.makedirs(config['output_path'], exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    results = []
    for idx, inst in enumerate(instances):
        # ID persistente: hash de la máscara y bbox
        mask_bytes = inst['mask'].tobytes()
        bbox_bytes = np.array(inst['bbox']).tobytes()
        unique_hash = hashlib.sha1(mask_bytes + bbox_bytes).hexdigest()[:16]
        tag = f"{base}_fruta_{idx+1}"
        results.append({
            'id': unique_hash,
            'tag': tag,
            'bbox': inst['bbox'],
            'class': inst['class'],
            'conf': inst['conf'],
        })
    # Añadir el conteo total de frutas en la imagen
    export_data = {
        'image': os.path.basename(image_path),
        'image_fruit_count': len(results),
        'fruits': results
    }
    if config['output_format'] == 'json':
        with open(os.path.join(config['output_path'], base + '.json'), 'w') as f:
            json.dump(export_data, f, indent=2)
    else:
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(config['output_path'], base + '.csv'), index=False)
    print(f"Imagen: {image_path} | Frutas detectadas: {len(results)}")

# --- 4. Main ---
def main(image_dir):
    # Si existe test_image.jpg en el directorio, solo procesa esa imagen
    test_img_path = os.path.join(image_dir, 'test_image.jpg')
    if os.path.exists(test_img_path):
        image_files = [test_img_path]
        print(f"Procesando solo: {test_img_path}")
    else:
        image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if not image_files:
            print("No se encontraron imágenes para procesar.")
            return
    detector = load_detector(CONFIG)
    predictor = load_sam(CONFIG)
    for img_path in image_files:
        print(f"\nProcesando imagen: {os.path.basename(img_path)}")
        instances = process_image_verbose(img_path, detector, predictor, CONFIG)
        export_results(instances, img_path, CONFIG)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Conteo avanzado de frutas en imágenes')
    parser.add_argument('--images', type=str, required=False, help='Directorio con imágenes a procesar (por defecto: el directorio actual)')
    args = parser.parse_args()
    image_dir = args.images if args.images else os.path.dirname(os.path.abspath(__file__))
    main(image_dir)

# --- 5. Estrategias avanzadas (comentarios para futuras mejoras) ---
# - Mejorar la separación de instancias solapadas usando watershed + markers automáticos (p.ej. con detección de picos en distance transform)
# - Aplicar graph-cuts para refinar bordes de segmentación
# - Integrar clusterización de centroides para casos de agrupamiento extremo
# - Añadir tracking en vídeo con Norfair, DeepSORT o ByteTrack (ver documentación de cada librería)
# - Implementar evaluación automática con ground truth y ajuste de hiperparámetros
# - Permitir entrenamiento/fine-tuning de modelos desde el mismo pipeline
