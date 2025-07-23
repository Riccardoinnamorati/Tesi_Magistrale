import numpy as np
from typing import List, Tuple, Any
from choirbot_examples.FALKO.CCDAMatcher import CCDAMatcher

class MixedMatcher:
    """
    Matching misto: usa CCDA per trovare le corrispondenze iniziali tra due scansioni,
    poi stima la matrice di rototraslazione ottimale (rigida) con l'algoritmo di Kabsch.
    """
    def __init__(self, dist_tol: float = 0.1, dist_min: float = 0.0):
        self.ccda = CCDAMatcher(dist_tol=dist_tol, dist_min=dist_min)

    def match(self, v1: List[Any], v2: List[Any], match: List[Tuple[int, int]]) -> Tuple[int, np.ndarray]:
        """
        Trova le corrispondenze tra v1 e v2 usando CCDA, poi stima la matrice di rototraslazione (2D).
        Args:
            v1: lista di keypoint (con attributo .point)
            v2: lista di keypoint (con attributo .point)
            match: lista di tuple (output)
        Returns:
            num_match: numero di match trovati
            H: matrice 3x3 di rototraslazione (omogenea)
        """
        match.clear()
        num_match = self.ccda.match(v1, v2, match)
        #print (f"match trovati: {match}")
        if num_match < 3:
            # Non si può stimare una trasformazione
            return num_match, np.eye(3)
        # Estrai le corrispondenze trovate
        src_pts = []
        dst_pts = []
        for i1, i2 in match:
            if i1 >= 0 and i2 >= 0:
                p1 = v1[i1].position
                p2 = v2[i2].position
                arr1 = np.array([p1.x, p1.y])
                arr2 = np.array([p2.x, p2.y])
                src_pts.append(arr1)
                dst_pts.append(arr2)
        P = np.array(src_pts, dtype=np.float64)
        Q = np.array(dst_pts, dtype=np.float64)

        # Kabsch 2D
        centroid_P = np.mean(P, axis=0)
        centroid_Q = np.mean(Q, axis=0)
        P_centered = P - centroid_P
        Q_centered = Q - centroid_Q
        H = P_centered.T @ Q_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[1, :] *= -1
            R = Vt.T @ U.T
        t = centroid_Q - R @ centroid_P
        # Costruisci matrice omogenea 3x3
        H_mat = np.eye(3)
        H_mat[:2, :2] = R
        H_mat[:2, 2] = t
        # get angle from rotation matrix
        angle = np.arctan2(R[1, 0], R[0, 0])
        #print(f"Angle of rotation: {angle} radians")
        
        #print(f"H_mat:\n{H_mat}")
        return num_match, H_mat
