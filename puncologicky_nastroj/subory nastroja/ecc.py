"""
SKRIPT POROVN8VANIA PUNCOV CEZ ROZŠÍRENÝ KORELAČNÝ KOEFICIENT
Účel: Tento modul poskytuje funkcionalitu pre porovnávanie kovových odtlačkov pomocou
metódy rozšíreného korelačného koeficientu (ECC) a analýzy hĺbkových charakteristík.
"""

"""
KNIŽNICE PRE SPRAVOVANIE OBRAZOV
- cv2: OpenCV pre počítačové videnie a spracovanie obrazu
- numpy: Numerické operácie a práca s poliami
"""
import cv2
import numpy as np

"""
TRIEDA: MetalprintMatcher
Účel: Hlavná trieda pre porovnávanie kovových odtlačkov
"""
class MetalprintMatcher:
    """
    METÓDA: __init__
    Účel: Inicializácia triedy a načítanie obrázkov
    Parametre:
        *image_path: Cesty k obrázkom alebo priamo obrazové dáta
    """
    def __init__(self, *image_path):
        self.image1 = image_path[0] if not isinstance(image_path[0], str) else cv2.imread(image_path[0], cv2.IMREAD_GRAYSCALE)
        self.image2 = image_path[1] if not isinstance(image_path[1], str) else cv2.imread(image_path[1], cv2.IMREAD_GRAYSCALE)
    
    """
    METÓDA: rotate_image
    Účel: Otočí obrázok o zadaný uhol
    Parametre:
        image: Vstupný obrázok
        angle: Uhol otočenia v stupňoch
    Návratová hodnota:
        Otočený obrázok
    """
    def rotate_image(self, image, angle):
        (h, w) = image.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    
    """
    METÓDA: calculate_centroid
    Účel: Vypočíta ťažisko (centroid) obrázka
    Parametre:
        image: Vstupný obrázok
    Návratová hodnota:
        tuple: Súradnice centroidu (x, y) alebo None ak sa nedá vypočítať
    """
    def calculate_centroid(self, image):
        moments = cv2.moments(image)
        if moments["m00"] == 0:
            return None
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        return (cx, cy)
    
    """
    METÓDA: center_pad
    Účel: Vytvorí obrázok s vycentrovaným obsahom
    Parametre:
        mask: Vstupná maska
        target_size: Požadovaná veľkosť výstupu
    Návratová hodnota:
        Nový obrázok s vycentrovaným obsahom
    """
    def center_pad(self, mask, target_size):
        h, w = mask.shape
        size = target_size + 10
        canvas = np.zeros((size, size), dtype=np.float32)
        x_offset = (size - w) // 2
        y_offset = (size - h) // 2
        canvas[y_offset:y_offset+h, x_offset:x_offset+w] = mask
        return canvas
    
    """
    METÓDA: show_mask_with_centroid
    Účel: Zobrazí masku s vyznačeným centroidom
    Parametre:
        mask: Vstupná maska
        centroid: Súradnice centroidu
    """
    def show_mask_with_centroid(self, mask, centroid):
        if centroid is None:
            print("❗ Centroid not found.")
            return
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.circle(mask_bgr, centroid, 5, (0, 0, 255), -1)
        cv2.imshow("Mask with Centroid", mask_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    """
    METÓDA: depth_score
    Účel: Vypočíta hĺbkové charakteristiky obrázka
    Parametre:
        image: Vstupný obrázok
    Návratová hodnota:
        tuple: (priemerná hodnota, smerodajná odchýlka) alebo (None, None) pre prázdny obrázok
    """
    def depth_score(self, image):
        # Prijímame obraz s hodnotami vo formáte float32 alebo uint8
        # Ignoruj nulové hĺbky
        valid_pixels = image[image > 0]

        if len(valid_pixels) == 0:
            return None, None

        mean_val = np.mean(valid_pixels)
        std_val = np.std(valid_pixels)

        # Predpoklad: čím menšia hodnota, tým väčšia hĺbka
        # Teda chceme vysoký "score", keď sú hodnoty nízke
        #score = 1.0 - mean_val if image.dtype in [np.float32, np.float64] else 255 - mean_val

        return mean_val, std_val
    
    """
    METÓDA: match_loop
    Účel: Hlavná metóda pre porovnávanie obrázkov pomocou ECC
    Návratová hodnota:
        float: Najvyšší nájdený ECC koeficient (0-1)
    Popis algoritmu:
        1. Predspracovanie obrázkov
        2. Nájdenie kontúr a vytvorenie masiek
        3. Vycentrovanie obrázkov
        4. Testovanie rôznych uhlov otočenia
        5. Výpočet ECC pre každú rotáciu
        6. Návrat najlepšieho výsledku
    """
    def match_loop(self):
        max_ecc = 0.0
        best_warp = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

        # Predspracovanie prvého obrázka
        gray1 = self._preprocess_image(self.image1)
        mask1 = self._create_mask(gray1)
        target_size = self._calculate_target_size()
        padded_img1 = self.center_pad(self.image1, target_size)
        padded_mask1 = self.center_pad(mask1, target_size)
        centroid1 = self.calculate_centroid(padded_mask1)

        if centroid1 is None:
            return 0.0

        # Testovanie rôznych uhlov natočenia pre druhý obrázok
        for angle in range(0, 360, 20):
            # Predspracovanie druhého obrázka pre aktuálny uhol
            gray2 = self._preprocess_image(self.image2)
            mask2 = self._create_mask(gray2)
            padded_img2 = self.center_pad(self.image2, target_size)
            padded_img2_rotated = self.rotate_image(padded_img2, angle)
            padded_mask2 = self.center_pad(mask2, target_size)
            padded_mask2_rotated = self.rotate_image(padded_mask2, angle)
            centroid2 = self.calculate_centroid(padded_mask2_rotated)

            if centroid2 is None:
                continue

            # Výpočet posunu medzi centroidmi
            shift_x, shift_y = self._calculate_shift(centroid1, centroid2)
            warp_matrix = np.array([[1, 0, shift_x], [0, 1, shift_y]], dtype=np.float32)

            # Výpočet ECC
            current_ecc = self._compute_ecc(padded_img1, padded_img2_rotated, warp_matrix)
            
            # Aktualizácia maxima
            if current_ecc > max_ecc:
                max_ecc = current_ecc
                best_warp = warp_matrix

        return max_ecc
    
    """
    METÓDA: _preprocess_image
    Účel: Konvertuje obrázok do odtieňov sivej a normalizuje ho.
    """
    def _preprocess_image(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return gray
    
    """
    METÓDA: _create_mask
    Účel: Vytvorí binárnu masku z obrázka v odtieňoch sivej.
    """
    def _create_mask(self, gray_image):
        _, thresh = cv2.threshold(gray_image, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray_image)
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
        return mask
    
    """
    METÓDA: _calculate_target_size
    Účel: Vypočíta cieľovú veľkosť pre vycentrovanie obrázkov.
    """
    def _calculate_target_size(self):
        return max(self.image1.shape[0], self.image1.shape[1], 
                   self.image2.shape[0], self.image2.shape[1])
    
    """
    METÓDA: _calculate_shift
    Účel: Vypočíta posun medzi dvoma ťažiskami.
    """
    def _calculate_shift(self, centroid1, centroid2):
        return centroid1[0] - centroid2[0], centroid1[1] - centroid2[1]
    
    """
    METÓDA: _compute_ecc
    Účel: Vypočíta ECC (Enhanced Correlation Coefficient) medzi dvoma obrázkami.
    
    Parametre:
        template_image: Referenčný obrázok
        input_image: Vstupný obrázok pre porovnanie
        warp_matrix: Počiatočná odhadovaná transformácia
    
    Návratová hodnota:
        float: Hodnota ECC alebo 0.0 ak výpočet zlyhá
    """
    def _compute_ecc(self, template_image, input_image, warp_matrix):
        try:
            ecc_value, _ = cv2.findTransformECC(
                template_image, 
                input_image, 
                warp_matrix,
                cv2.MOTION_EUCLIDEAN,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 2000, 1e-15),
                None, 
                3
            )
            return ecc_value
        except cv2.error:
            return 0.0
    