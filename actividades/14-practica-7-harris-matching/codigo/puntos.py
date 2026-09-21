"""Evaluación de detectores y descriptores de puntos clave con una transformación conocida.

Como no se dispone de pares de fotos reales con verdad de terreno, se generan vistas sintéticas de una foto real:
una rotación pura de la cámara (yaw, pitch, roll) equivale a una homografía H = K·R·K⁻¹ (sin paralaje), y se añaden
cambios fotométricos (brillo, gamma, ruido). Como H es exacta, se puede medir la repetibilidad del detector y la
corrección de cada correspondencia.
"""
import time

import cv2
import numpy as np


def matriz_K(ancho, alto, foco):
    return np.array([[foco, 0, ancho / 2], [0, foco, alto / 2], [0, 0, 1]], np.float64)


def rotacion(yaw=0.0, pitch=0.0, roll=0.0):
    y, p, r = np.radians([yaw, pitch, roll])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def homografia_rotacion(ancho, alto, foco, yaw=0.0, pitch=0.0, roll=0.0, escala=1.0):
    K = matriz_K(ancho, alto, foco)
    S = np.diag([escala, escala, 1.0])
    S[0, 2] = (1 - escala) * ancho / 2
    S[1, 2] = (1 - escala) * alto / 2
    return S @ K @ rotacion(yaw, pitch, roll) @ np.linalg.inv(K)

def homografia_plano(ancho, alto, foco, angulo_deg, eje="y"):
    """Homografía de un plano (el objeto) girado `angulo_deg` sobre su propio eje central, con la cámara fija frente a él.

    A diferencia de girar la cámara, el centro del objeto permanece en el centro de la imagen y solo aparece el escorzo
    (un lado se aleja y se ve más pequeño). Se obtiene proyectando las cuatro esquinas del plano.
    """
    a = np.radians(angulo_deg)
    cx, cy = ancho / 2, alto / 2
    src = np.float32([[0, 0], [ancho, 0], [ancho, alto], [0, alto]])
    dst = []
    for x, y in src:
        X, Y = x - cx, y - cy
        if eje == "y":
            Xr, Yr, Zr = X * np.cos(a), Y, X * np.sin(a)
        else:
            Xr, Yr, Zr = X, Y * np.cos(a), Y * np.sin(a)
        prof = foco + Zr
        dst.append([foco * Xr / prof + cx, foco * Yr / prof + cy])
    return cv2.getPerspectiveTransform(src, np.float32(dst)).astype(np.float64)


def vista(img, H, tam=None, brillo=1.0, offset=0.0, gamma=1.0, ruido=0.0, desenfoque=0.0, semilla=0):
    """Aplica la homografía H y luego cambios fotométricos: g = clip(brillo·f^gamma·255 + offset) + ruido."""
    h, w = img.shape[:2]
    tam = tam or (w, h)
    out = cv2.warpPerspective(img, H, tam, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    f = out.astype(np.float32) / 255.0
    f = np.clip(brillo * (f ** gamma) * 255.0 + offset, 0, 255)
    if desenfoque > 0:
        f = cv2.GaussianBlur(f, (0, 0), desenfoque)
    if ruido > 0:
        f = f + np.random.default_rng(semilla).normal(0, ruido, f.shape)
    return np.clip(f, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- detectores y descriptores
def gris(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def detectar_harris(g, n=1500, bloque=3, apertura=3, k=0.04, distancia=6):
    """Esquinas de Harris con supresión de no máximos; devuelve KeyPoints (size=1) ordenados por respuesta."""
    R = cv2.cornerHarris(np.float32(g), bloque, apertura, k)
    dil = cv2.dilate(R, np.ones((2 * distancia + 1, 2 * distancia + 1), np.uint8))
    ys, xs = np.nonzero((R == dil) & (R > 0.01 * R.max()))
    orden = np.argsort(-R[ys, xs])[:n]
    return [cv2.KeyPoint(float(xs[i]), float(ys[i]), 16, -1, float(R[ys[i], xs[i]])) for i in orden]


def descriptor_parche(g, kps, lado=17):
    """Descriptor de parche normalizado (media 0, norma 1): el emparejamiento «con Harris por sí solo»."""
    r = lado // 2
    gp = np.pad(g.astype(np.float32), r, mode="reflect")
    keep, desc = [], []
    for kp in kps:
        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
        p = gp[y:y + lado, x:x + lado].ravel()
        p = p - p.mean()
        n = np.linalg.norm(p)
        if n > 1e-3:
            keep.append(kp); desc.append(p / n)
    return keep, (np.array(desc, np.float32) if desc else np.zeros((0, lado * lado), np.float32))


def detectar(g, metodo, n=1500):
    """Devuelve (keypoints, descriptores, norma) para 'harris', 'orb' o 'sift'."""
    if metodo == "harris":
        kps, d = descriptor_parche(g, detectar_harris(g, n))
        return kps, d, cv2.NORM_L2
    if metodo == "orb":
        kps, d = cv2.ORB_create(nfeatures=n).detectAndCompute(g, None)
        return kps, d, cv2.NORM_HAMMING
    if metodo == "sift":
        kps, d = cv2.SIFT_create(nfeatures=n).detectAndCompute(g, None)
        return kps, d, cv2.NORM_L2
    raise ValueError(metodo)


def emparejar(d1, d2, norma, ratio=0.8):
    """Correspondencias con prueba de razón de Lowe."""
    if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
        return []
    m = cv2.BFMatcher(norma).knnMatch(d1, d2, k=2)
    return [a for a, b in (x for x in m if len(x) == 2) if a.distance < ratio * b.distance]


def proyectar(pts, H):
    p = np.hstack([np.asarray(pts, np.float64).reshape(-1, 2), np.ones((len(pts), 1))]) @ H.T
    return p[:, :2] / p[:, 2:3]


def repetibilidad(kps1, kps2, H, forma2, tol=3.0):
    """Fracción de puntos de la imagen 1 que, proyectados con la verdad H, caen dentro de la imagen 2 y a ≤ tol px de un punto detectado allí."""
    if not kps1 or not kps2:
        return 0.0, 0
    p1 = proyectar([k.pt for k in kps1], H)
    h, w = forma2[:2]
    dentro = (p1[:, 0] >= 0) & (p1[:, 0] < w) & (p1[:, 1] >= 0) & (p1[:, 1] < h)
    if not dentro.any():
        return 0.0, 0
    p2 = np.array([k.pt for k in kps2])
    d = np.linalg.norm(p1[dentro][:, None, :] - p2[None, :, :], axis=2).min(axis=1)
    return float((d <= tol).mean()), int(dentro.sum())


def evaluar(img1, img2, H, metodo, n=1500, ratio=0.8, tol=3.0):
    """Detecta, describe, empareja y compara con la verdad H. Devuelve un diccionario de métricas."""
    g1, g2 = gris(img1), gris(img2)
    t0 = time.perf_counter()
    k1, d1, norma = detectar(g1, metodo, n)
    k2, d2, _ = detectar(g2, metodo, n)
    t_det = time.perf_counter() - t0
    t0 = time.perf_counter()
    m = emparejar(d1, d2, norma, ratio)
    t_emp = time.perf_counter() - t0
    rep, visibles = repetibilidad(k1, k2, H, g2.shape, tol)
    out = {"metodo": metodo, "kp1": len(k1), "kp2": len(k2), "repetibilidad": rep, "matches": len(m), "correctos": 0, "precision": 0.0,
           "inliers_ransac": 0, "error_H_px": float("nan"), "t_deteccion_ms": t_det * 1000, "t_matching_ms": t_emp * 1000}
    if m:
        p1 = np.float32([k1[a.queryIdx].pt for a in m]); p2 = np.float32([k2[a.trainIdx].pt for a in m])
        err = np.linalg.norm(proyectar(p1, H) - p2, axis=1)
        out["correctos"] = int((err <= tol).sum()); out["precision"] = float((err <= tol).mean())
        if len(m) >= 4:
            Hest, mask = cv2.findHomography(p1, p2, cv2.RANSAC, tol)
            if Hest is not None:
                out["inliers_ransac"] = int(mask.sum())
                h, w = g1.shape
                esq = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
                out["error_H_px"] = float(np.linalg.norm(proyectar(esq, Hest) - proyectar(esq, H), axis=1).mean())
    return out, (k1, k2, m)


def dibujar_matches(img1, k1, img2, k2, matches, H=None, tol=3.0, max_n=80):
    """Verde = correspondencia correcta según la verdad H; rojo = errónea."""
    h1, w1 = img1.shape[:2]; h2, w2 = img2.shape[:2]
    lienzo = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    lienzo[:h1, :w1] = img1 if img1.ndim == 3 else cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    lienzo[:h2, w1:] = img2 if img2.ndim == 3 else cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    ms = sorted(matches, key=lambda m: m.distance)[:max_n]
    for m in ms:
        p1 = np.array(k1[m.queryIdx].pt); p2 = np.array(k2[m.trainIdx].pt)
        ok = True if H is None else np.linalg.norm(proyectar([p1], H)[0] - p2) <= tol
        col = (60, 220, 60) if ok else (60, 60, 240)
        cv2.line(lienzo, tuple(int(v) for v in p1), (int(p2[0] + w1), int(p2[1])), col, 1, cv2.LINE_AA)
        cv2.circle(lienzo, tuple(int(v) for v in p1), 3, col, -1); cv2.circle(lienzo, (int(p2[0] + w1), int(p2[1])), 3, col, -1)
    return lienzo
