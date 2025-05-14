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
from skimage.segmentation import morphological_geodesic_active_contour, inverse_gaussian_gradient
from sklearn.cluster import KMeans
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

# --- Estrategias avanzadas activables ---
CONFIG_ADVANCED = {
    'clahe': True,                # Aumento de contraste adaptativo
    'hsv_clustering': True,       # Clusterización de color en HSV
    'active_contours': True,      # Refinamiento de máscaras con active contours
    'watershed_peaks': True,      # Clusterización de picos en distance transform
    'hough_circles': True,        # Detección de círculos por Hough Transform
    'edge_detection': True,       # Detección adaptativa de bordes
    'multi_scale': True           # Enfoque multi-escala para clusters complejos
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

def preprocess_image(image, config_adv):
    img = image.copy()
    if config_adv.get('clahe', False):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return img

def hsv_clustering_mask(crop, n_clusters=2):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=0).fit(pixels)
    labels = kmeans.labels_.reshape(hsv.shape[:2])
    # Asumimos que la fruta es el cluster más grande
    unique, counts = np.unique(labels, return_counts=True)
    fruit_label = unique[np.argmax(counts)]
    mask = (labels == fruit_label).astype(np.uint8)
    return mask

def refine_mask_active_contour(mask, crop):
    # Refinar bordes con active contours (Morphological Snakes)
    gimg = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gimg = gimg.astype(float) / 255.0
    gimg = inverse_gaussian_gradient(gimg)
    init_ls = np.zeros_like(mask, dtype=float)
    init_ls[mask > 0] = 1.0
    snake = morphological_geodesic_active_contour(gimg, 40, init_ls, smoothing=2, balloon=1, threshold=0.3)
    return (snake > 0.5).astype(np.uint8)

def split_mask_with_peaks(mask, config):
    # Calcula el distance transform con mayor precisión
    mask_8bit = (mask * 255).astype(np.uint8)
    # Aplica un suave suavizado para mejorar el distance transform
    mask_smooth = cv2.GaussianBlur(mask_8bit, (3, 3), 0)
    dist = cv2.distanceTransform(mask_smooth, cv2.DIST_L2, 5)
    
    # Normaliza distance transform para mejor visualización y detección
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
    
    # Detecta picos locales con parámetros adaptativos según el tamaño del cluster
    mask_area = np.count_nonzero(mask)
    img_area = mask.shape[0] * mask.shape[1]
    ratio = mask_area / img_area
    
    # Parámetros adaptativos basados en densidad del cluster
    if ratio > 0.5:  # Cluster muy grande
        min_distance = max(5, int(np.sqrt(mask_area) / 15))
        threshold_rel = 0.3
    else:  # Cluster normal o pequeño
        min_distance = max(8, int(np.sqrt(mask_area) / 10))
        threshold_rel = 0.5
    
    # Detecta más picos en clusters densos
    coordinates = peak_local_max(
        dist_norm, 
        min_distance=min_distance,
        threshold_rel=threshold_rel,  # Umbral relativo para detectar picos
        exclude_border=False,
        indices=True,
        labels=mask
    )
    
    # Si solo encontramos 0 o 1 pico en un área grande, bajamos el umbral para detectar más
    if len(coordinates) <= 1 and mask_area > 2 * config['min_instance_area']:
        coordinates = peak_local_max(
            dist_norm, 
            min_distance=max(3, int(min_distance/2)),
            threshold_rel=0.25,
            exclude_border=False,
            indices=True,
            labels=mask
        )
    
    # Crea marcadores para watershed
    markers = np.zeros_like(mask, dtype=np.int32)
    for i, (y, x) in enumerate(coordinates, 1):
        markers[y, x] = i
    
    # Si no hay picos detectados, devuelve la máscara original como una región
    if len(coordinates) == 0:
        return [mask]
    
    # Aplica watershed con gradiente óptimo
    # Convertir a color y aplicar un filtro de gradiente para mejorar los bordes
    mask_color = cv2.cvtColor(mask_8bit, cv2.COLOR_GRAY2BGR)
    
    # Aplica watershed
    labels = cv2.watershed(mask_color, markers.copy())
    
    # Procesa regiones resultantes
    separated = []
    for label in range(1, np.max(labels)+1):
        region = (labels == label).astype(np.uint8)
        if cv2.countNonZero(region) > config['min_instance_area']:
            # Aplica un cierre morfológico para suavizar bordes
            kernel = np.ones((3,3), np.uint8)
            region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, kernel)
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

def detect_hough_circles(crop, mask=None, min_radius=10, max_radius=None):
    """
    Utiliza la transformada circular de Hough para detectar círculos en zonas con frutas.
    Funciona mejor para frutas con forma redondeada.
    
    Args:
        crop: Imagen recortada donde buscar círculos
        mask: Máscara opcional para restringir la búsqueda
        min_radius: Radio mínimo de círculos a buscar
        max_radius: Radio máximo de círculos a buscar
    
    Returns:
        Lista de máscaras para cada círculo detectado
    """
    # Si no se especifica radio máximo, calcular basado en tamaño de imagen
    if max_radius is None:
        max_radius = min(crop.shape[0], crop.shape[1]) // 3
    
    # Preparar la imagen
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # Si hay máscara, aplicarla
    if mask is not None:
        gray = cv2.bitwise_and(gray, gray, mask=mask)
    
    # Aplicar un suavizado para reducir ruido
    gray = cv2.medianBlur(gray, 5)
    
    # Parámetros para HoughCircles - ajustados para ser más sensibles en clusters
    dp = 1.2  # Resolución inversa del acumulador
    minDist = min_radius * 1.5  # Distancia mínima entre círculos
    param1 = 50  # Umbral superior para el detector de bordes Canny interno
    param2 = 30  # Umbral del acumulador (menor = más círculos falsos positivos)
    
    # Detectar círculos
    circles = cv2.HoughCircles(
        gray, 
        cv2.HOUGH_GRADIENT, 
        dp=dp, 
        minDist=minDist,
        param1=param1, 
        param2=param2, 
        minRadius=min_radius, 
        maxRadius=max_radius
    )
    
    # Crear máscaras para cada círculo detectado
    masks = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        for circle in circles[0, :]:
            center_x, center_y, radius = circle
            
            # Crear máscara para este círculo
            circle_mask = np.zeros_like(gray, dtype=np.uint8)
            cv2.circle(circle_mask, (center_x, center_y), radius, 255, -1)
            
            # Si hay máscara original, intersectarla con la del círculo
            if mask is not None:
                circle_mask = cv2.bitwise_and(circle_mask, mask)
            
            masks.append(circle_mask)
    
    return masks

def detect_edge_adaptive(crop, mask=None):
    """
    Detecta bordes de forma adaptativa para separar frutas en clusters densos
    
    Args:
        crop: Imagen recortada donde buscar bordes
        mask: Máscara opcional para restringir la búsqueda
    
    Returns:
        Imagen con bordes detectados
    """
    # Convertir a escala de grises
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # Si hay máscara, aplicarla
    if mask is not None:
        gray = cv2.bitwise_and(gray, gray, mask=mask)
    
    # Aplicar un suavizado para reducir ruido
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detección adaptativa de bordes (Canny con umbral automático)
    v = np.median(blur)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    
    # Aplicar Canny detector
    edges = cv2.Canny(blur, lower, upper)
    
    # Dilatar bordes para hacerlos más visibles
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    return edges

def analyze_multi_scale(crop, mask=None):
    """
    Analiza una imagen a múltiples escalas para detectar mejor las frutas en clusters
    
    Args:
        crop: Imagen recortada a analizar
        mask: Máscara opcional para restringir el análisis
    
    Returns:
        Lista de máscaras de frutas detectadas
    """
    # Escalas a analizar (factores de escalado)
    scales = [0.5, 1.0, 1.5]
    
    all_masks = []
    height, width = crop.shape[:2]
    
    for scale in scales:
        # Redimensionar imagen
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Evitar tamaños muy pequeños
        if new_width < 20 or new_height < 20:
            continue
            
        resized = cv2.resize(crop, (new_width, new_height))
        
        # Si hay máscara, escalarla también
        resized_mask = None
        if mask is not None:
            resized_mask = cv2.resize(mask, (new_width, new_height), 
                                    interpolation=cv2.INTER_NEAREST)
        
        # Aplicar detección de círculos en esta escala
        circles_masks = detect_hough_circles(
            resized, 
            resized_mask,
            min_radius=max(5, int(10 * scale)),
            max_radius=max(20, int(50 * scale))
        )
        
        # Escalar las máscaras de vuelta al tamaño original
        for cmask in circles_masks:
            orig_mask = cv2.resize(cmask, (width, height), 
                                interpolation=cv2.INTER_NEAREST)
            all_masks.append(orig_mask)
    
    return all_masks

def segment_and_split(predictor, image, bbox, config):
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2]
    crop_proc = preprocess_image(crop, CONFIG_ADVANCED)
    if CONFIG_ADVANCED.get('hsv_clustering', False):
        mask = hsv_clustering_mask(crop_proc)
        masks = [mask]
    else:
        predictor.set_image(crop_proc)
        masks, scores, _ = predictor.predict(multimask_output=True)
    separated_masks = []
    for mask in masks:
        mask_bin = (mask > 0.5).astype(np.uint8)
        # Active contours opcional
        if CONFIG_ADVANCED.get('active_contours', False):
            try:
                mask_bin = refine_mask_active_contour(mask_bin, crop_proc)
            except Exception as e:
                print(f"[ActiveContours] Error: {e}")
        # Watershed+peaks opcional
        if CONFIG_ADVANCED.get('watershed_peaks', True):
            separated = split_mask_morphology(mask_bin, config)
        else:
            separated = [mask_bin]
        separated_masks.extend(separated)
    global_masks = [offset_mask(m, x1, y1, image.shape) for m in separated_masks]
    return global_masks

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