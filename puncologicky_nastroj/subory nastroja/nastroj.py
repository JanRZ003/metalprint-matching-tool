"""
HLAVNÝ SKRIPT PRE INTERAKTÍVNU ANALÝZU ZHODY PUNCOV
Účel: Tento skript poskytuje nástroje na porovnávanie kovových odtlačkov pomocou segmentácie obrázkov
a metódy rozšíreného korelačného koeficientu (ECC). Obsahuje:
- Interaktívne GUI pre výber dátovej sady
- Integráciu modelu SAM (Segment Anything Model)
- Porovnávanie oblastí segmentačných masiek
- Vizualizáciu výsledkov
- Možnosti ukladania výsledkov
"""

"""
SYSTÉMOVÉ KNIZNICE
Pre základné operácie so systémom a prácu so súbormi
"""
import os
import glob

"""
KNIZNICE PRE SPRAVOVANIE OBRAZOV
Pre načítavanie, úpravu a spracovanie obrazových dát
"""

import cv2
import numpy as np

"""
GRAFICKÉ UZIVATEĽSKÉ ROZHRANIE
Komponenty pre vytvorenie použivateľského rozhrania
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog, Tk

"""
KNIZNICE NASTROJOV STROJOVEHO UCENIA
PyTorch a komponenty modelu SAM pre segmentáciu
"""
import torch
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

"""
VIZUALIZÁCIA DÁT
Nástroje pre zobrazovanie obrázkov a výsledkov
"""
import matplotlib.pyplot as plt

"""
VLASTNÉ ALGORITMY
Implementácia špecializovaných algoritmov pre porovnávanie odtlačkov
"""
from ecc import MetalprintMatcher

"""
Ukladanie a načítavanie natrénovaných modelov logistickej regresie
"""
import joblib

"""
FUNKCIA: get_image_files_in_dataset
Účel: Získa všetky obrazové súbory (PNG/TIFF) zo zadaného priečinka
Parametre:
    dataset_name (str): Cesta k priečinku s obrázkami
Návratová hodnota:
    list: Zoznam ciest k obrazovým súborom s normalizovanými oddelovačmi
"""
def get_image_files_in_dataset(dataset_name):
    image_files = []
    for ext in ['*.png', '*.tif']:
        image_files.extend(glob.glob(os.path.join(dataset_name, ext)))
    for i in range(len(image_files)):
        image_files[i] = image_files[i].replace("\\", "/")
    return image_files

"""
FUNKCIA: run_selector
Účel: Riadi celý proces segmentácie a porovnávania
Parametre:
    image_path_1 (str): Cesta k prvému obrázku
    image_path_2 (str): Cesta k druhému obrázku
    mode_var (tuple): Režimy segmentácie pre oba obrázky
    nms_thresh_var (tuple): Prahové hodnoty pre potlačenie ne-maximálnych masiek
    min_area_var (tuple): Minimálne plochy masiek
    model: Natrénovaný klasifikačný model
"""
def run_selector(image_path_1, image_path_2, mode_var, nms_thresh_var, min_area_var, model):
    
    print(f"Inicializuje sa segmentačný nástroj pre prvý obrázok")
    
    selector1 = SegmentSelector(image_path_1, mode_var[0], nms_thresh_var[0], min_area_var[0])
    
    print(f"Inicializovanie segmentačného nástroja pre prvý obrázok dokončené, spúšťa sa segmentácia vo vybranej oblasti prvého obrázku")
    
    masks1 = selector1.get_mask()  # uložené masky z prvého obrázka
    
    print("segmentácia pre prvý obrázok dokončená")
    

    print(f"Inicializuje sa segmentačný nástroj pre druhý obrázok")
    
    selector1.reset(image_path_2, mode_var[1], nms_thresh_var[1], min_area_var[1])
    
    print(f"Inicializovanie segmentačného nástroja pre druhý obrázok dokončené, spúšťa sa segmentovanie vo vybranej oblasti druhého obrázku")
    
    masks2 = selector1.get_mask()  # uložené masky z druhého obrázka
    
    print("segmentácia pre druhý obrázok dokončená")
    print("spúšťa sa porovnávanie dvojíc vysegmentovaných oblastí medzi obrázkami pomocou rozšíreného korelačného koeficientu")

    # masks1, masks2 teraz použijeme na ďalšie spracovanie (ECC)
    compare_masks_ecc(image_path_1, image_path_2, masks1, masks2, model)

"""
FUNKCIA: get_save_file
Účel: Spravuje dialóg pre ukladanie výsledkov
Návratová hodnota:
    str: Názov súboru pre uloženie alebo None ak používateľ zrušil
"""
# Funkcia na ziskanie potvrdenia a nazvu priecinka od pouzivatela
def get_save_file():
    root = Tk()
    root.withdraw()  # Skryje hlavne okno

    # Spytaj sa, ci chce ulozit
    if not messagebox.askyesno("Uloženie výsledkov", "Chceš uložiť výsledky porovnania?"):
        return None  # Nechce ukladat

    # Spytaj sa na nazov priecinka
    folder_name = simpledialog.askstring("Názov súboru", "Zadaj názov súboru pre uloženie:")
    if not folder_name:
        return None

    return folder_name

"""
FUNKCIA: extract_roi
Účel: Extrahuje oblasť záujmu z obrázka pomocou masky a ohraničujúceho rámca
Parametre:
    image: Vstupný obrázok
    mask: Segmentačná maska
    bbox: Ohraničujúci rámec (x,y,šírka,výška)
Návratová hodnota:
    numpy.ndarray: Extrahovaná oblasť záujmu
"""
def extract_roi(image, mask, bbox):
    x, y, w, h = bbox
    cropped_image = image[y:y+h, x:x+w]
    cropped_mask = mask[y:y+h, x:x+w]
    
    roi = cv2.bitwise_and(cropped_image, cropped_image, mask=cropped_mask.astype(np.uint8))
    return roi

"""
FUNKCIA: compare_masks_ecc
Účel: Porovnáva segmentované oblasti medzi obrázkami pomocou ECC
Parametre:
    image_path1 (str): Cesta k prvému obrázku
    image_path2 (str): Cesta k druhému obrázku
    masks1 (list): Masky z prvého obrázka
    masks2 (list): Masky z druhého obrázka
    model: Natrénovaný model pre výpočet pravdepodobnosti
"""
def compare_masks_ecc(image_path1, image_path2, masks1, masks2, model):
    image1 = cv2.imread(image_path1, cv2.IMREAD_UNCHANGED)
    image2 = cv2.imread(image_path2, cv2.IMREAD_UNCHANGED)
    
    #konvertovanie snímok skenera CRUSE na šedotónový obrázok pomocou rovnice luminancie
    image_type=len(image1.shape)
    
    if len(image1.shape) == 3 and image1.shape[2] == 3:
        img = image1.astype(np.float32)
        image1 = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
        image1 = image1.astype(np.float32)
        img = image2.astype(np.float32)
        image2 = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
        image2 = image1.astype(np.float32)

    for i, (m1, bbox1) in enumerate(masks1):
        roi1 = extract_roi(image1, m1, bbox1)
        for j, (m2, bbox2) in enumerate(masks2):
            roi2 = extract_roi(image2, m2, bbox2)

            matcher = MetalprintMatcher(roi1, roi2)
            ecc_score = matcher.match_loop()
            
            def pad_to_same_height(img1, img2):        
                h1, w1 = img1.shape[:2]
                h2, w2 = img2.shape[:2]
                max_h = max(h1, h2)

                def pad_img(img, target_h):
                    h, w = img.shape[:2]
                    if h < target_h:
                        pad = np.zeros((target_h - h, w), dtype=img.dtype)
                        return np.vstack([img, pad])
                    return img

                img1_padded = pad_img(img1, max_h)
                img2_padded = pad_img(img2, max_h)
                return img1_padded, img2_padded

            image3, image4 = pad_to_same_height(roi1, roi2)
            
            def normalize_to_uint8(img):
                img_norm = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                return img_norm.astype(np.uint8)

            image3 = normalize_to_uint8(image3)
            image4 = normalize_to_uint8(image4)
            
            gap = np.zeros((image3.shape[0]*2, 300), dtype=np.uint8)
            
            combined = np.hstack([
                gap,
                cv2.resize(image3, (image3.shape[1]*2, image3.shape[0]*2)),
                gap,
                cv2.resize(image4, (image4.shape[1]*2, image4.shape[0]*2)),
                gap
            ])
            
            p=0
            if image_type < 3:
                p1,r1=matcher.depth_score(roi1)
                p2,r2=matcher.depth_score(roi2)
                p=1

            h, w = combined.shape[:2]
            
            # Vytvorím čierny pás pod obrázkom na text (napr. 40 pixelov vysoký)
            text_area = np.zeros((300, w), dtype=np.uint8)

            # Napíšem text do textovej oblasti (červená farba)
            text = f"korelacny koeficient: {ecc_score:.3f}"
            cv2.putText(text_area, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            
            
            prob = model.predict_proba(np.array([[ecc_score]]))[0, 1]
            text0 = f"pravdepodobnost zhody: {prob:.3f}"
            cv2.putText(text_area, text0, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            
            if p==1:
                # Napíšem text do textovej oblasti (červená farba)
                text1 = f"hlbkova povaha prveho puncu:"
                cv2.putText(text_area, text1, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
                text2 = f"priemerna intenzita - {p1:.3f}, rozptyl intenzit - {r1:.3f}"
                cv2.putText(text_area, text2, (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
                text3 = f"hlbkova povaha druheho puncu:"
                cv2.putText(text_area, text3, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
                text4 = f"priemerna intenzita - {p2:.3f}, rozptyl intenzit - {r2:.3f}"
                cv2.putText(text_area, text4, (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
                
            # Spojím obrázok a textovú oblasť vertikálne
            combined = np.vstack([combined, text_area])
            
            cv2.imshow(f'Porovnanie Punc0{i} a Punc1{j} - ECC: {ecc_score:.3f}', combined)
            
            save_file = get_save_file()

            if save_file:
                cv2.imwrite(save_file+".png", combined)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

"""
TRIEDA: SegmentSelector
Účel: Poskytuje interaktívnu segmentáciu obrázkov pomocou modelu SAM
Hlavné funkcie:
- Manuálny výber oblasti
- Automatické generovanie masiek
- Viacero režimov segmentácie
"""
class SegmentSelector:
    """
    METÓDA: __init__
    Účel: Inicializácia segmentátora
    Parametre:
        image_path_1 (str): Cesta k vstupnému obrázku
        mode_var: Režim segmentácie
        nms_thresh_var: Prah pre potlačenie ne-maximálnych masiek
        min_area_var: Minimálna plocha masky
        sam_checkpoint (str): Cesta k modelu SAM
        model_type (str): Typ modelu
        device: Zariadenie pre výpočty (CPU/GPU)
    """
    def __init__(self, image_path_1, mode_var, nms_thresh_var, min_area_var,
                 sam_checkpoint="sam_vit_h_4b8939.pth", model_type="vit_h", device=None):
        self.image_path_1 = image_path_1
        
        # Načítanie obrázkov
        self.image_1 = cv2.imread(image_path_1, cv2.IMREAD_UNCHANGED)
        if len(self.image_1.shape)==3:
            self.image_1 = self.image_1.astype(np.float32) / 65535.
            mantiuk = cv2.createTonemapMantiuk(gamma=1.0, scale=0.85, saturation=1.0)
            self.image_1 = mantiuk.process(self.image_1)
        
        if self.image_1 is None:
            raise ValueError(f"Obrázky '{image_path_1}' sa nepodarilo načítať.")
        
        # Konverzia do RGB formátu pre obidva obrázky
        self.image_1_rgb = cv2.cvtColor(self.image_1, cv2.COLOR_BGR2RGB)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mode_var = mode_var
        self.nms_thresh_var = nms_thresh_var
        self.min_area_var = min_area_var

        # Iniciovanie modelu
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(self.device)
        self.predictor = SamPredictor(self.sam)

        # Nastavenie obrázkov do modelu
        self.predictor.set_image(self.image_1_rgb)

        # Parametre pre úpravu zobrazenia
        self.max_display_height = 650
        self.scale = self.max_display_height / self.image_1.shape[0]
        self.resized_1 = cv2.resize(self.image_1, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        self.clone=self.resized_1.copy()
        self.box = []
        self.drawing = False
    
    """
    METÓDA: reset
    Účel: Resetuje segmentátor pre nový obrázok
    Parametre:
        image_path_1 (str): Nový obrázok
        mode_var: Nový režim segmentácie
        nms_thresh_var: Nový prah NMS
        min_area_var: Nové minimálna plocha
    """
    def reset(self, image_path_1, mode_var, nms_thresh_var, min_area_var):
        self.image_path_1 = image_path_1
        
        # Načítanie obrázkov
        self.image_1 = cv2.imread(image_path_1, cv2.IMREAD_UNCHANGED)
        if len(self.image_1.shape)==3:
            self.image_1 = self.image_1.astype(np.float32) / 65535.
            mantiuk = cv2.createTonemapMantiuk(gamma=1.0, scale=0.85, saturation=1.0)
            self.image_1 = mantiuk.process(self.image_1)
        
        if self.image_1 is None:
            raise ValueError(f"Obrázky '{image_path_1}' sa nepodarilo načítať.")
        
        # Konverzia do RGB formátu
        self.image_1_rgb = cv2.cvtColor(self.image_1, cv2.COLOR_BGR2RGB)
        
        self.mode_var = mode_var
        self.nms_thresh_var = nms_thresh_var
        self.min_area_var = min_area_var

        self.predictor.set_image(self.image_1_rgb)

        # Parametre pre úpravu zobrazenia
        self.max_display_height = 650
        self.scale = self.max_display_height / self.image_1.shape[0]
        self.resized_1 = cv2.resize(self.image_1, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        self.clone=self.resized_1.copy()
        self.box = []
        self.drawing = False
    
    """
    METÓDA: draw_rectangle
    Účel: Callback pre kreslenie obdĺžnika myšou
    Parametre:
        event: Typ udalosti
        x, y: Súradnice
        flags: Príznaky
        param: Doplnkové parametre
    """
    def draw_rectangle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.box = [(x, y)]
            self.drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            img = self.clone.copy()
            cv2.rectangle(img, self.box[0], (x, y), (0, 255, 0), 2)
            cv2.imshow("Vyber bounding box", img)
        elif event == cv2.EVENT_LBUTTONUP:
            self.box.append((x, y))
            self.drawing = False
            cv2.rectangle(self.clone, self.box[0], self.box[1], (0, 255, 0), 2)
            cv2.imshow("Vyber bounding box", self.clone)
    
    """
    METÓDA: get_mask
    Účel: Hlavná metóda na získanie masiek
    Návratová hodnota:
        list: Zoznam masiek a ich ohraničujúcich rámcov
    """
    def get_mask(self):
        win_name = "Vyber bounding box"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, self.clone.shape[1], self.clone.shape[0])
        cv2.setMouseCallback(win_name, self.draw_rectangle)

        print("Klikni a ťahaj myšou na výber oblasti. Stlač Enter keď skončíš.")
        while True:
            cv2.imshow(win_name, self.clone)
            key = cv2.waitKey(0) & 0xFF
            if key == 13:
                break
        cv2.destroyAllWindows()

        if len(self.box) != 2:
            raise RuntimeError("Nebola zvolená žiadna platná oblasť.")

        x0, y0 = self.box[0]
        x1, y1 = self.box[1]
        x_min, x_max = sorted([x0, x1])
        y_min, y_max = sorted([y0, y1])

        x_min_orig = int(x_min / self.scale)
        x_max_orig = int(x_max / self.scale)
        y_min_orig = int(y_min / self.scale)
        y_max_orig = int(y_max / self.scale)

        if self.mode_var.get() == 0:
            input_box = torch.tensor([[x_min_orig, y_min_orig, x_max_orig, y_max_orig]], device=self.device)
            transformed_box = self.predictor.transform.apply_boxes_torch(input_box, self.image_1_rgb.shape[:2])
            masks, scores, logits = self.predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_box,
                multimask_output=False
            )
            mask = masks[0][0].cpu().numpy().astype(np.uint8)

            # Urč bounding box podľa masky
            x, y, w, h = cv2.boundingRect(mask)

            # Zobrazenie
            output = self.image_1_rgb.copy()
            output[mask > 0] = [0, 255, 0]
            if output.dtype!=np.uint8:
                output = np.clip(output, 0.0, 1.0)
            plt.imshow(output)
            plt.axis("off")
            plt.title("Výsledná maska")
            plt.show()

            return [(mask, (x, y, w, h))]  # zabalené do zoznamu

        else:
            generator = SamAutomaticMaskGenerator(
                model=self.sam,
                box_nms_thresh=float(float(self.nms_thresh_var.get())/100),
                min_mask_region_area=int(self.min_area_var.get())
            )
            masks = generator.generate(self.image_1_rgb)

            output = self.image_1_rgb.copy()
            selected_masks = []
            for m in masks:
                mask = m["segmentation"].astype(np.uint8)
                x, y, w, h = cv2.boundingRect(mask)
                if x >= x_min_orig and x + w <= x_max_orig and y >= y_min_orig and y + h <= y_max_orig:
                    # Vizualizácia
                    colored = output.copy()
                    colored[mask > 0] = [255, 0, 0]
                    if colored.dtype!=np.uint8:
                        colored = np.clip(colored, 0.0, 1.0)
                    plt.imshow(colored)
                    plt.axis("off")
                    plt.title("Maska objektu")
                    plt.show()

                    # Ukladáme: (maska, bounding box)
                    selected_masks.append((mask, (x, y, w, h)))

            return selected_masks

"""
TRIEDA: DatasetSelector
Účel: Hlavná GUI aplikácia pre výber dátovej sady a spracovanie
Hlavné funkcie:
- Výber typu dátovej sady
- Prechádzanie obrázkami
- Konfigurácia parametrov
- Spustenie segmentácie
"""
class DatasetSelector(tk.Tk):
    
    """
    METÓDA: __init__
    Účel: Inicializácia hlavného okna aplikácie
    Parametre:
        Žiadne (dedí od tk.Tk)
    """
    def __init__(self):
        super().__init__()
        self.title("Výber Datasetu")
        
        self.selected_dataset = ""
        self.selected_light_mode = ""
        self.image_files = []
        self.current_image_idx1 = 0
        self.current_image_idx2 = 0
        
        self.create_widgets()
    
    """
    METÓDA: create_widgets
    Účel: Vytvára základné GUI komponenty podľa aktuálneho stavu
    """
    def create_widgets(self):
        # Ak sme ešte nevybrali dataset, zobrazíme základné tlačidlá
        if not self.selected_dataset:
            ttk.Label(self, text="Vyberte dataset:").grid(row=0, column=0, padx=20, pady=20)

            button_frame = ttk.Frame(self)
            button_frame.grid(row=1, column=0, padx=10, pady=10)

            self.skener_button = ttk.Button(button_frame, text="Skener Cruse", command=self.select_skener_cruse)
            self.skener_button.grid(row=0, column=0, padx=10, pady=10)

            self.surfacecontrol_button = ttk.Button(button_frame, text="Senzor SurfaceControl3D", command=lambda: self.load_images("../predspracovany dataset senzor surfacecontrol3D"))
            self.surfacecontrol_button.grid(row=0, column=1, padx=10, pady=10)

            self.scancontrol_button = ttk.Button(button_frame, text="Senzor ScanControl", command=lambda: self.load_images("../predspracovany dataset senzor scancontrol"))
            self.scancontrol_button.grid(row=0, column=2, padx=10, pady=10)

        # Ak sme vybrali Skener Cruse, ale ešte nie svetelný režim, zobrazíme výber podpriečinka
        elif self.selected_dataset == "../predspracovany dataset skener cruse" and not self.selected_light_mode:
            ttk.Label(self, text="Vyberte svetelný režim:").grid(row=0, column=0, padx=20, pady=20)

            button_frame = ttk.Frame(self)
            button_frame.grid(row=1, column=0, padx=10, pady=10)

            self.lla_button = ttk.Button(button_frame, text="LLa", command=lambda: self.load_images_for_light_mode("LLa"))
            self.lla_button.grid(row=0, column=0, padx=10, pady=10)

            self.lrfb_button = ttk.Button(button_frame, text="LRFB", command=lambda: self.load_images_for_light_mode("LRFB"))
            self.lrfb_button.grid(row=0, column=1, padx=10, pady=10)

        else:
            # Ak sme už vybrali dataset aj svetelný režim, alebo iný dataset, len načítame obrázky
            self.load_images(self.selected_dataset)
    
    """
    METÓDA: select_skener_cruse
    Účel: Nastaví výber skeneru Cruse ako aktuálny dataset
    """
    def select_skener_cruse(self):
        self.selected_dataset = "../predspracovany dataset skener cruse"
        self.create_widgets()

    """
    METÓDA: load_images_for_light_mode
    Účel: Načíta obrázky pre konkrétny svetelný režim
    Parametre:
        light_mode (str): Názov svetelného režimu (LLa/LRFB)
    """
    def load_images_for_light_mode(self, light_mode):
        self.selected_light_mode = light_mode
        # Cesta do podpriečinka so svetelným režimom v rámci Skener Cruse datasetu
        path = os.path.join(self.selected_dataset, light_mode)
        self.load_images(path)
    
    """
    METÓDA: load_images
    Účel: Načíta všetky obrázky zo zadaného priečinka
    Parametre:
        dataset_name (str): Cesta k priečinku s obrázkami
    """
    def load_images(self, dataset_name):
        self.selected_dataset = dataset_name
        self.image_files = get_image_files_in_dataset(dataset_name)
        if not self.image_files:
            messagebox.showerror("Chyba", "V danom dataset-e nie sú žiadne obrázky.")
            return
        
        for widget in self.winfo_children():
            widget.destroy()
            
        self.show_segmentation_options()
        self.show_images(self.image_files[self.current_image_idx1],self.image_files[self.current_image_idx2])
        
        self.withdraw()
    
    """
    METÓDA: show_images
    Účel: Zobrazí dvojicu obrázkov pre porovnanie
    Parametre:
        image_path1 (str): Cesta k prvému obrázku
        image_path2 (str): Cesta k druhému obrázku
    """
    def show_images(self, image_path1,image_path2):
        img1 = cv2.imread(image_path1, cv2.IMREAD_UNCHANGED)
        img2 = cv2.imread(image_path2, cv2.IMREAD_UNCHANGED)

        target_height = 270
        scale1 = target_height / img1.shape[0]
        scale2 = target_height / img2.shape[0]

        resized_1 = cv2.resize(img1, (0, 0), fx=scale1, fy=scale1)
        resized_2 = cv2.resize(img2, (0, 0), fx=scale2, fy=scale2)

        gap_width = 20
        gap_color = (255, 255, 255)
        
        if len(resized_1.shape)==2:
            gap = np.ones((resized_1.shape[0], gap_width), dtype=np.uint8) * 255
        else:
            gap = np.ones((resized_1.shape[0], gap_width,resized_1.shape[2]), dtype=np.uint8) * 255

        combined = np.hstack((resized_1, gap, resized_2))
        cv2.imshow("Zobrazenie snimok", combined)
        cv2.waitKey(1)
    
    """
    METÓDA: next_image
    Účel: Prepne na ďalší obrázok v poradí
    Parametre:
        number (int): 1 pre prvý obrázok, 2 pre druhý obrázok
    """
    def next_image(self,number):
        if number==1:
            self.current_image_idx1 = (self.current_image_idx1 + 1) % len(self.image_files)
            self.show_images(self.image_files[self.current_image_idx1],self.image_files[self.current_image_idx2])
            
        if number==2:
            self.current_image_idx2 = (self.current_image_idx2 + 1) % len(self.image_files)
            self.show_images(self.image_files[self.current_image_idx1],self.image_files[self.current_image_idx2])
    
    """
    METÓDA: show_segmentation_options
    Účel: Zobrazí okno s možnosťami segmentácie
    """
    def show_segmentation_options(self):
        options_window = tk.Toplevel(self)
        options_window.title("Nastavenia Segmentácie")
        
        next_button = ttk.Button(options_window, text="Prepnúť prvý obrázok", command=lambda:self.next_image(1))
        next_button.grid(row=0, column=0, padx=10, pady=5)
        
        next_button = ttk.Button(options_window, text="Prepnúť druhý obrázok", command=lambda:self.next_image(2))
        next_button.grid(row=1, column=0, padx=10, pady=(5,20))

        mode_var1 = tk.IntVar(value=0)  # Jedna maska
        nms_thresh_var1 = tk.StringVar(value="70")
        min_area_var1 = tk.StringVar(value="100")

        ttk.Label(options_window, text="Režim segmentácie pre prvý obrázok:").grid(row=2, column=0, padx=10, pady=5)
        ttk.Radiobutton(options_window, text="Jedna maska", variable=mode_var1, value=0).grid(row=3, column=0, padx=10, pady=5)
        ttk.Radiobutton(options_window, text="Viac masiek", variable=mode_var1, value=1).grid(row=4, column=0, padx=10, pady=5)
        
        ttk.Label(options_window, text="Nastavenie parametrov pre režim 'Viac masiek':").grid(row=5, column=0, padx=10, pady=5)
        ttk.Label(options_window, text="Dovolený prah prekrytia masiek v percentách:").grid(row=6, column=0, padx=10, pady=5)
        ttk.Entry(options_window, textvariable=nms_thresh_var1).grid(row=7, column=0, padx=10, pady=5)

        ttk.Label(options_window, text="Požadovaná minimálna plocha masky v pixeloch:").grid(row=8, column=0, padx=10, pady=5)
        ttk.Entry(options_window, textvariable=min_area_var1).grid(row=9, column=0, padx=10, pady=(5,50))
        
        mode_var2 = tk.IntVar(value=0)  # Jedna maska
        nms_thresh_var2 = tk.StringVar(value="70")
        min_area_var2 = tk.StringVar(value="100")

        ttk.Label(options_window, text="Režim segmentácie pre druhý obrázok:").grid(row=10, column=0, padx=10, pady=5)
        ttk.Radiobutton(options_window, text="Jedna maska", variable=mode_var2, value=0).grid(row=11, column=0, padx=10, pady=5)
        ttk.Radiobutton(options_window, text="Viac masiek", variable=mode_var2, value=1).grid(row=12, column=0, padx=10, pady=5)
        
        ttk.Label(options_window, text="Nastavenie parametrov pre režim 'Viac masiek':").grid(row=13, column=0, padx=10, pady=5)
        ttk.Label(options_window, text="Dovolený prah prekrytia masiek v percentách:").grid(row=14, column=0, padx=10, pady=5)
        ttk.Entry(options_window, textvariable=nms_thresh_var2).grid(row=15, column=0, padx=10, pady=5)

        ttk.Label(options_window, text="Požadovaná minimálna plocha masky v pixeloch:").grid(row=16, column=0, padx=10, pady=5)
        ttk.Entry(options_window, textvariable=min_area_var2).grid(row=17, column=0, padx=10, pady=(5,50))

        def on_segment_button_click():
            options_window.destroy()
            try:
                cv2.destroyWindow("Zobrazenie snimok")
            except cv2.error as e:
                if "NULL window" in str(e):
                    pass
                else:
                    raise
            s={"../predspracovany dataset senzor surfacecontrol3D":"lr_surface_control3D.joblib",
               "../predspracovany dataset senzor scancontrol":"lr_scan_control.joblib",
               "../predspracovany dataset skener cruse\LLa":"lr_lla_cruse.joblib",
               "../predspracovany dataset skener cruse\LRFB":"lr_lrfb_cruse.joblib"}
            model = joblib.load(s.get(self.selected_dataset))
            run_selector(self.image_files[self.current_image_idx1], self.image_files[self.current_image_idx2], (mode_var1,mode_var2), (nms_thresh_var1,nms_thresh_var2), (min_area_var1,min_area_var2),model)
            self.create_widgets()
        
        def back_step():
            self.selected_dataset = ""
            self.selected_light_mode = ""
            options_window.destroy()
            try:
                cv2.destroyWindow("Zobrazenie snimok")
            except cv2.error as e:
                if "NULL window" in str(e):
                    pass
                else:
                    raise
            self.deiconify()
            self.create_widgets()
            self.current_image_idx1 = 0
            self.current_image_idx2 = 0

        ttk.Button(options_window, text="Spustiť segmentáciu", command=on_segment_button_click).grid(row=18, column=0, columnspan=2, pady=10)
        
        ttk.Button(options_window, text="Späť na výber datasetu", command=back_step).grid(row=19, column=0, columnspan=2, pady=10)

if __name__ == "__main__":
    app = DatasetSelector()
    app.mainloop()
