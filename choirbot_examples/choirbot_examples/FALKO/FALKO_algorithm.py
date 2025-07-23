import numpy as np
import math
from scipy.signal import find_peaks
from choirbot_interfaces.msg import KeypointArray, Keypoint

class FALKOKeypointDetector:
    def __init__(self):
        self.minScoreTh = 8 # % of the maximum score
        self.subbeam = True
        self.NMSRadius = 0.2
        self.neighA = 0.28   # 0.1
        self.neighB = 0.29   # 0.07
        self.neighMinPoint = 2
        self.bRatio = 3  #2
        self.gridSectors = 16
        self.num_triangles = 4  # Number of triangles to check for each corner
        self.base_calculation_method = 'DIRECT'  # 'DIRECT' or 'PROJECTED'
        self.linear_threshold = 0.05  # Threshold for linearity check
        self.linear_check = True  # Whether to check for linear neighborhoods        

    def extract_keypoints(self, scan_data):

        """
        Main keypoint extraction method
        """
        
        self.minExtractionRange = scan_data.range_min
        self.maxExtractionRange = scan_data.range_max

        self.angle_min = scan_data.angle_max
        self.angle_max = scan_data.angle_min
        # Note: angle_max and angle_min are inverted in the scan data because the scan is in clockwise direction   
        self.angle_increment = - scan_data.angle_increment # negative because the scan is in clockwise direction
        self.ranges = np.array(scan_data.ranges[::-1])  # Reverse the order of ranges to match angles
        self.num_beams = len(self.ranges)
        self.angles = np.linspace(self.angle_min, self.angle_max, self.num_beams)
        self.x = self.ranges * np.cos(self.angles)
        self.y = self.ranges * np.sin(self.angles)
        self.points = np.column_stack((self.x, self.y))
        
        keypoint_array = KeypointArray()
        neigh = []
        scores = np.full(self.num_beams, -10.0)
        radius = np.zeros(self.num_beams)
        thetaCorner = np.zeros(self.num_beams)
        neighs = [None] * self.num_beams
        neighL = []
        neighR = []
        peaks = []
        scoreMax = 0.0

        for ind in range(self.num_beams):
            # Skip points outside extraction range
            if self.ranges[ind] < self.minExtractionRange or self.ranges[ind] > self.maxExtractionRange:
                scores[ind] = -10
                continue
                
            neigh.clear()
                     
            radius[ind] = self.get_neigh_radius(self.ranges[ind])
            
            midIndex, neighL, neighR = self.get_neigh_points(ind, radius[ind], neigh)
            
            neighs[ind] = neigh.copy()
            #print (f"neigh : {neighs[ind]} ")
            neighSizeL = len(neighL)
            neighSizeR = len(neighR)
            
            neighSize = len(neigh)
            
            # Check if neighborhood has enough points on both sides
            if neighSizeL < self.neighMinPoint or neighSizeR < self.neighMinPoint:
                scores[ind] = -10
                continue

            # Triangle properties check
            if not self.check_triangle_properties(neigh, midIndex, radius[ind], ind):
                scores[ind] = -10
                continue
            
            neighSizeL = midIndex
            neighSizeR = len(neigh) - midIndex - 1
            
            # Calculate corner orientation
            thetaCorner[ind] = self.get_corner_orientation(neigh, midIndex)
            
            # Map neighborhood points to circular sectors            
            neighCircularIndexesL = [self.get_circular_sector_index(neigh[i], neigh[midIndex], thetaCorner[ind]) for i in range(neighSizeL)]
            neighCircularIndexesR = [
                self.get_circular_sector_index(neigh[midIndex + i], neigh[midIndex], thetaCorner[ind]) 
                for i in range(neighSizeR)
            ]
            
            # Calculate scores for left and right sides
            scoreL = 0
            scoreR = 0
            
            # Calculate left side scores
            for i in range(midIndex - 1, 0, -1):
                for j in range(i, 0, -1):
                    scoreL += self.circular_sector_distance(
                        neighCircularIndexesL[i - 1], neighCircularIndexesL[j - 1], self.gridSectors)
            
            # Calculate right side scores
            for i in range(midIndex + 1, neighSize):
                for j in range(i, neighSize):
                    scoreR += self.circular_sector_distance(
                        neighCircularIndexesR[i - midIndex - 1], 
                        neighCircularIndexesR[j - midIndex - 1], 
                        self.gridSectors)
            
            # Total score is sum of left and right scores
            scores[ind] = scoreL + scoreR
            
            # Track maximum score
            if scores[ind] > scoreMax:
                scoreMax = scores[ind]
        
        # Normalize scores
        for ind in range(self.num_beams):
            if scores[ind] < 0:
                scores[ind] = scoreMax
            scores[ind] = scoreMax - scores[ind]
 
        # Non-Maximum Suppression and keypoint creation
        ibeg = 0
        iend = self.num_beams
        minval = self.minScoreTh * scoreMax / 100.0
        # Apply Non-Maximum Suppression to find peaks
        self.NMS_keypoint(scores, scan_data, ibeg, iend, self.NMSRadius, minval, peaks)
        
        # Create final keypoints
        for i in range(len(peaks)):
            kp = Keypoint()

            # Position
            kp.position.x = float(self.x[peaks[i]])
            kp.position.y = float(self.y[peaks[i]])
            kp.position.z = 0.0
            
            # Attributes
            kp.score = float(scores[peaks[i]])
            kp.orientation = float(thetaCorner[peaks[i]])
            kp.radius = float(radius[peaks[i]])
            kp.index = int(peaks[i])

            if self.linear_check:
                if self.is_linear_neighborhood(neighs[peaks[i]]):
                    continue
            
            if self.subbeam:
                refined_point = self.sub_beam_corner(peaks[i], radius[peaks[i]])
                kp.position.x = float(refined_point[0])
                kp.position.y = float(refined_point[1])

            keypoint_array.header.frame_id = scan_data.header.frame_id
            keypoint_array.keypoints.append(kp)
        
        return keypoint_array, scores
    
    # Sposta qui tutti i metodi helper dal nodo ROS
    def get_neigh_radius(self, rho):
        """Calculate neighborhood radius based on distance"""
        radius = self.neighA * math.exp(self.neighB * rho)
        if radius >= rho:
            return rho * 0.8
        return radius
    
    def get_neigh_points(self, ind, radius, neigh):
        """Trova i punti vicini entro il raggio specificato"""
        neigh.clear()
        current_point = self.points[ind]
                
        # La formula usata è quella per calcolare l'angolo tra le tangenti a una circonferenza
        # da un punto esterno a essa. Definiti quindi A il punto esterno (il nostro sensore),
        # B il punto di interesse (il nostro punto ind) e C il punto di intersezione tra il raggio
        # e la circonferenza nel punto di tangenza, si ha che:
        # AC = raggio (distanza tra il sensore e il punto di tangenza)
        # AB = distanza tra il sensore e il punto di interesse (il nostro ind)
        # BC = distanza tra il punto di interesse e il punto di tangenza (raggio)
        # Così, usando il teorema dei seni, si ha che:
        # sin(alpha) = AC / AB (ossia il seno dell'angolo alpha è uguale al rapporto tra il cateto opposto e l'ipotenusa)
        # Così, per calcolare alpha, si usa la funzione arcsin:
        # alpha = arcsin(AC / AB) = arcsin(raggio / distanza tra il sensore e il punto di interesse)
        # Si divide poi per l'angolo incrementale per ottenere il numero di punti da considerare 
        alpha = int(np.ceil(np.arcsin(radius / self.ranges[ind]) / self.angle_increment))

        # Calcola gli indici di inizio e fine senza troncamento
        beg_index = ind - alpha
        end_index = ind + alpha + 1
        
        # Raccogli i punti entro il raggio
        mid_index_value = 0
        
        for i_raw in range(beg_index, end_index ):

            # Modulo per evitare overflow
            i= i_raw % len(self.points)
            
            dist = np.linalg.norm(np.array(self.points[i]) - np.array(current_point))
            
            if dist <= radius:
                if i == ind:
                    mid_index_value = len(neigh)
                neigh.append(self.points[i])
        
        # Separa i punti a sinistra e a destra
        left_neigh = neigh[:mid_index_value]
        right_neigh = neigh[mid_index_value+1:]
        
        return mid_index_value, left_neigh, right_neigh
    
    def check_triangle_properties(self, neigh, midIndex, radius, ind):
        """Check triangle geometry properties"""
        """
        Verifica le proprietà dei triangoli basandosi sulla configurazione.
        
        Args:
            neigh: Lista dei punti nel neighborhood
            midIndex: Indice del punto centrale
            radius: Raggio del neighborhood
            ind: Indice del punto corrente
            
        Returns:
            True se le proprietà dei triangoli sono valide, False altrimenti
        """
        if len(neigh) < 3:
            return False
            
        # Calcola tutte le basi dei triangoli richiesti
        triangle_props = []
        
        # Scegli il metodo di calcolo della base
        if self.base_calculation_method == 'DIRECT':
            # Usa direttamente i punti agli estremi
            for i in range(self.num_triangles):
                left_idx = i
                right_idx = len(neigh) - 1 - i
                
                if left_idx >= midIndex or right_idx <= midIndex:
                    continue  # Triangolo non valido
                    
                base_length = self.points_distance(neigh[left_idx], neigh[right_idx])
                leftmost = neigh[left_idx]
                rightmost = neigh[right_idx]
                
                # H = 2*Area / B, where Area is the area of the triangle and B is the base length
                # the area is calculated using the ERONE formula:
                # Area = sqrt(p*(p-a)*(p-b)*(p-c)) where p is the semi-perimeter
                
                perimeter = self.points_distance(leftmost, neigh[midIndex]) + \
                            self.points_distance(neigh[midIndex], rightmost) + \
                            self.points_distance(leftmost, rightmost)
                semiperimeter = perimeter / 2
                
                radicando = semiperimeter * \
                            (semiperimeter - self.points_distance(leftmost, neigh[midIndex])) * \
                            (semiperimeter - self.points_distance(neigh[midIndex], rightmost)) * \
                            (semiperimeter - self.points_distance(leftmost, rightmost))
                            
                if radicando <= 0:
                    # Triangolo degenere, salta
                    continue
                    
                area = math.sqrt(radicando)
                height = 2 * area / base_length
                
                triangle_props.append((base_length, height))
                
        elif self.base_calculation_method == 'PROJECTED':
            # Usa la proiezione sulla direzione media
            
            for i in range(self.num_triangles):
                # Sottrai i da midIndex e aggiungi i alla lunghezza del neighborhood
                start_idx = max(0, i)
                end_idx = min(len(neigh) - 1, len(neigh) - 1 - i)
                
                if start_idx >= midIndex or end_idx <= midIndex:
                    continue  # Non possiamo creare un triangolo valido
                    
                sub_neigh = neigh[start_idx:end_idx + 1]
                sub_mid_idx = midIndex - start_idx
                
                base_length, left_dir, right_dir, leftmost, rightmost = \
                    self.points_base_calc(sub_neigh, sub_mid_idx, radius)
                    
                if base_length is None:
                    continue
                    
                perimeter = self.points_distance(leftmost, neigh[midIndex]) + \
                            self.points_distance(neigh[midIndex], rightmost) + \
                            self.points_distance(leftmost, rightmost)
                semiperimeter = perimeter / 2
                
                radicando = semiperimeter * \
                            (semiperimeter - self.points_distance(leftmost, neigh[midIndex])) * \
                            (semiperimeter - self.points_distance(neigh[midIndex], rightmost)) * \
                            (semiperimeter - self.points_distance(leftmost, rightmost))
                            
                if radicando <= 0:
                    continue
                    
                area = math.sqrt(radicando)
                height = 2 * area / base_length
                
                triangle_props.append((base_length, height))
                
                
        # Controllo adattivo in base alla distanza
        distance_scale = max(0.0, min(1.0, radius / self.maxExtractionRange))
        adaptive_bRatio = self.bRatio * (2.0 * math.exp(-13 * distance_scale) + 0.5)
        
        # Almeno un triangolo deve essere valido
        if not triangle_props:
            return False
        # if triangleBLength < (radius[ind] / adaptive_bRatio) or triangleHLength < (radius[ind] / adaptive_bRatio) or \
        #             (triangleBLength2 < (radius[ind] / self.bRatio) or triangleHLength2 < (radius[ind] / self.bRatio)):
        # Ogni triangolo deve soddisfare i criteri
        valid_triangles = 0
        for base_length, height in triangle_props:
            if base_length >= (radius / adaptive_bRatio) and height >= (radius / adaptive_bRatio):
                valid_triangles += 1
        
        # Richiedi che almeno metà dei triangoli controllati siano validi
        return valid_triangles >= max(1, self.num_triangles)
    
    def points_base_calc(self, neigh, midIndex, radius): # molto bene ma da tunare
        """Calculate the base of the triangle formed by the distance between the two most distant points
        projected on the average direction of the left and right neighbors, then return the base
        length, the left direction and the right direction
            Args:
                neigh (list): List of points in the neighborhood
                midIndex (int): Index of the mid point in the neighborhood
            Outputs:
                base_length (float): Length of the base of the triangle
                leftmost_proj (array): Leftmost projected point
                rightmost_proj (array): Rightmost projected point"""
        
        neighL = neigh[:midIndex]
        neighR = neigh[midIndex + 1:]
        
        neighsizeL = len(neighL)    
        neighsizeR = len(neighR)
        
        neighsize = len(neigh)
        
        # Calcola la direzione media dei vicini a sinistra
        left_dir = np.array([0.0, 0.0])
        for i in range(midIndex):
            dir_vec = (np.array(neigh[i]) - np.array(neigh[midIndex]))
            norm = np.linalg.norm(dir_vec)
            if norm > 0:  # Evita divisione per zero
                left_dir += dir_vec / norm
        
        # Normalizza la direzione media sinistra
        if midIndex > 0:
            left_dir /= midIndex
            norm = np.linalg.norm(left_dir)
            if norm > 0:
                left_dir = left_dir / norm
        
        # Calcola la direzione media dei vicini a destra
        right_dir = np.array([0.0, 0.0])
        right_count = len(neigh) - midIndex - 1
        for i in range(midIndex + 1, len(neigh)):
            dir_vec = (np.array(neigh[i]) - np.array(neigh[midIndex]))
            norm = np.linalg.norm(dir_vec)
            if norm > 0:  # Evita divisione per zero
                right_dir += dir_vec / norm
        
        # Normalizza la direzione media destra
        if right_count > 0:
            right_dir /= right_count
            norm = np.linalg.norm(right_dir)
            if norm > 0:
                right_dir = right_dir / norm
        
        # Proietta i punti a sinistra del midIndex sulla direzione sinistra
        projections = []
        for i in range(midIndex):
            vec = (np.array(neigh[i]) - np.array(neigh[midIndex]))
            left_proj_dist = np.dot(vec, left_dir)
            left_proj = neigh[midIndex] + left_dir * left_proj_dist
            projections.append(left_proj)
        
        # Proietta i punti a destra del midIndex sulla direzione destra
        for i in range(midIndex + 1, len(neigh)):
            vec = (np.array(neigh[i]) - np.array(neigh[midIndex]))
            right_proj_dist = np.dot(vec, right_dir)
            right_proj = neigh[midIndex] + right_dir * right_proj_dist
            projections.append(right_proj)
                
        # Trova il punto proiettato più lontano a sinistra e quello più lontano a destra
        leftmost_proj = projections[0]
        rightmost_proj = projections[-1]
        
        rightmost_point = rightmost_proj
        leftmost_point = leftmost_proj
        
        base_length = np.linalg.norm(leftmost_proj - rightmost_proj)
        
        if np.linalg.norm(neighsizeL - neighsizeR) < neighsize*0.3 and neighsizeL > 6 and neighsizeR > 6: 
            # calcola il punto più a destra come il punto sulla direzione destra che 
            # interseca il cerchio di raggio radius e lo stesso a sinistra
            
            leftmost_point = neigh[midIndex] + left_dir * radius
            rightmost_point = neigh[midIndex] + right_dir * radius
            
            # La lunghezza della base è la distanza tra questi due punti
            base_length = np.linalg.norm(leftmost_point - rightmost_point)
        
        return base_length, left_dir, right_dir, rightmost_point, leftmost_point

    def get_corner_orientation(self, neigh, midIndex):
        """Calculate orientation of corner feature"""
        oriL = np.zeros(2)
        oriR = np.zeros(2)
        size = len(neigh)
        TYPE = 'MEDIAN'  # 'MEAN' or 'MEDIAN'
        
        if TYPE == 'MEAN':
            # Calculate left and right orientations by averaging neighbors
            for i in range(size):
                if i < midIndex:
                    oriL += (np.array(neigh[i]) - np.array(neigh[midIndex]))
                elif i > midIndex:
                    oriR += (np.array(neigh[i]) - np.array(neigh[midIndex]))
            
            oriL /= midIndex
            oriR /= (size - (midIndex + 1))
        elif TYPE == 'MEDIAN': 
            # Calculate left and right orientations using median
            left_points = [np.array(neigh[i]) for i in range(midIndex)]
            right_points = [np.array(neigh[i]) for i in range(midIndex + 1, size)]
        
            oriL = np.median(left_points, axis=0) - np.array(neigh[midIndex])
            oriR = np.median(right_points, axis=0) - np.array(neigh[midIndex])
            
        # Normalize orientations (aggiunta mia)
        oriL /= np.linalg.norm(oriL)
        oriR /= np.linalg.norm(oriR)

        # Calculate angle between left and right orientations
        ori = oriL + oriR
        angle = math.atan2(ori[1], ori[0])
        
        return 0 if math.isnan(angle) else angle
    
    def get_circular_sector_index(self, p, pmid, theta):
        """Calculate circular distance between two sectors"""
        return int(math.floor((self.angle_between_points(p, pmid) 
                               - theta) / (2.0 * math.pi / self.gridSectors)))
    
    def circular_sector_distance(self, a1, a2, res):
        """Calculate distance between sectors"""
        r2 = res // 2
        return abs(((a1 - a2) + r2) % res - r2)
    
    def is_linear_neighborhood(self, neigh):
        """Check if neighborhood points form a line"""
        if len(neigh) < 3:
            return True
            
        points = np.array(neigh)
        x = points[:, 0]
        y = points[:, 1]
        
        # Simple linear regression
        A = np.vstack([x, np.ones(len(x))]).T
        result = np.linalg.lstsq(A, y, rcond=None)
        
        # Check R² - if high, points are likely on a line
        y_pred = A.dot(result[0])
        ss_total = np.sum((y - np.mean(y))**2)
        if ss_total < 1e-10:  # Prevent division by zero
            return True
            
        ss_res = np.sum((y - y_pred)**2)
        r_squared = 1 - (ss_res / ss_total)
        
        return ss_total < self.linear_threshold
    
    def points_distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return np.linalg.norm(np.array(p1) - np.array(p2))
    
    def signed_triangle_area(self, p1, p2, p3):
        """Calculate signed area of triangle formed by three points"""
        # return 0.5 * ((p2[0] - p1[0]) * (p3[1] - p1[1]) - 
        #               (p3[0] - p1[0]) * (p2[1] - p1[1]))
        return ((p3[1] - p2[1]) * p1[0] - (p3[0] - p1[0]) * p1[1] + p2[0] * p2[1] - p2[1] * p2[0])
    
    def angle_between_points(self, p, pmid):
        """Calculate angle between two points"""
        return math.atan2(p[1] - pmid[1], p[0] - pmid[0])

    def NMS_keypoint(self, scores, scan, ibeg, iend, radius, minval, peaks):            
        """Non-Maximum Suppression per trovare i picchi locali nei punteggi"""
        i, imax = ibeg, ibeg
        candidates = []
        peaks.clear()
        
        while i < iend:
            # Calcola la finestra di ricerca
            if radius >= self.ranges[i]: # se il raggio è maggiore della distanza del punto
                # si usa un valore fisso per la finestra di ricerca
                win = int(np.floor(np.arcsin(0.8) / (2*math.pi/self.num_beams)))
            else: # Calcola la finestra di ricerca in base alla distanza del raggio
                win = int(np.floor(np.arcsin(radius / self.ranges[i]) / (2*math.pi/self.num_beams)))
            # print('i:', i, 'win:', win)
            jmax = i
            
            if imax + win < i: # se il massimo precedente è più lontano della finestra di ricerca
                # calcola l'inizio della finestra di ricerca
                jbeg = max(i - win, 0)
                imax = i
            else:
                jbeg = i
                
            jend = min(i + win + 1, iend) # calcola la fine della finestra di ricerca, assicurandosi di non superare iend
            
            for j in range(jbeg, jend): # scorre nella finestra di ricerca e si aggiorna il massimo
                if scores[j] > scores[jmax]:
                    jmax = j
                    
            imax = jmax if scores[jmax] >= scores[imax] else imax # aggiorna il massimo globale??
            
            if i == imax and scores[i] > minval: 
                candidates.append(i)
                
            if jmax > i: # se il massimo locale è maggiore dell'indice corrente salta a jmax
                i = jmax
            else:
                i += 1
                
        # Filtraggio finale dei candidati
        i1, i2, counter = 0, 0, 0
        
        while i1 < len(candidates):
            if scores[candidates[i2]] == scores[candidates[i1]]:
                counter += 1
                if 2 * abs(i2 - i1) > counter:
                    i2 += 1
            else:
                peaks.append(candidates[i2])
                i2 = i1
                counter = 0
            i1 += 1
            
        if i2 != len(candidates):
            peaks.append(candidates[i2])
      
    def NMS_keypoint_scipy(self,scores, scan, ibeg, iend, radius, minval, peaks):
        """Non-Maximum Suppression usando scipy per trovare i picchi locali."""
        scores_section = scores[ibeg:iend]

        # Calcola la distanza minima media tra i picchi basata sul raggio
        avg_min_distance = int(np.floor(np.arcsin(radius) / (2 * np.pi / self.num_beams)))

        # Trova i picchi con altezza minima e distanza minima
        peak_indices, _ = find_peaks(scores_section, height=minval, distance=avg_min_distance)
        peaks.clear()
        peaks.extend(peak_indices + ibeg)
        
    def NMS_keypoint_numpy(self, scores, scan, ibeg, iend, radius, minval, peaks):
        """Non-Maximum Suppression usando numpy per trovare i picchi locali."""
        scores_section = scores[ibeg:iend]
        peaks.clear()

        # Calcola la distanza minima media tra i picchi basata sul raggio
        avg_min_distance = int(np.floor(np.arcsin(radius) / (2 * np.pi / self.num_beams)))

        for i in range(1, len(scores_section) - 1):
            if scores_section[i] > scores_section[i - 1] and scores_section[i] > scores_section[i + 1] and scores_section[i] > minval:
                # Verifica che il picco sia distante almeno avg_min_distance dai precedenti
                if len(peaks) == 0 or (i + ibeg) - peaks[-1] >= avg_min_distance:
                    peaks.append(i + ibeg)    

    def sub_beam_corner(self, index, radius):
        neigh = []
        midIndex, _, __ = self.get_neigh_points(index, radius, neigh)
        neighSize = len(neigh)
        
        leftSide = []
        for i in range(midIndex + 1):
            leftSide.append(neigh[i])
        
        rightSide = []
        for i in range(midIndex, neighSize):
            rightSide.append(neigh[i])
        
        leftLine = self.generate_line(leftSide)
        rightLine = self.generate_line(rightSide)
        
        A = [leftLine[0], leftLine[1], rightLine[0], rightLine[1]]
        b = [-leftLine[2], -rightLine[2]]
        
        valid, x = self.solve_system2x2(A, b)
        
        if not valid:
            return neigh[midIndex]
        
        temp = np.array([x[0], x[1]])
        
        if self.points_distance(neigh[midIndex], temp) < 0.20:
            return temp
        else:
            return neigh[midIndex]

    def generate_line(self, points):
        sxx = 0.0
        syy = 0.0
        sxy = 0.0
        sx = 0.0
        sy = 0.0
        num = 0
        
        for i in range(len(points)):
            sxx += points[i][0] * points[i][0]
            syy += points[i][1] * points[i][1]
            sxy += points[i][0] * points[i][1]
            sx += points[i][0]
            sy += points[i][1]
            num += 1
        
        msxx = (sxx - sx * sx / num) / num
        msyy = (syy - sy * sy / num) / num
        msxy = (sxy - sx * sy / num) / num
        b = 2.0 * msxy
        a = msxx
        c = msyy
        theta = 0.5 * (math.atan2(b, a - c) + math.pi)
        theta = math.atan2(math.sin(theta), math.cos(theta))
        rho = (sx * math.cos(theta) + sy * math.sin(theta)) / num
        
        if rho < 0:
            theta = math.atan2(math.sin(theta + math.pi), math.cos(theta + math.pi))
            rho = -rho
        
        model = np.zeros(3)
        model[0] = math.cos(theta)
        model[1] = math.sin(theta)
        model[2] = -rho
        
        return model

    def solve_system2x2(self, A, b):
        det = A[0] * A[3] - A[1] * A[2]
        if abs(det) < 0.0000001:
            return False, [0, 0]
        
        x = [(b[0] * A[3] - A[1] * b[1]) / det,
            (A[0] * b[1] - b[0] * A[2]) / det]
        
        return True, x    