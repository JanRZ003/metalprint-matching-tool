Tento projekt slúži na porovnávanie kovových odtlačkov (puncov) pomocou:
- segmentácie obrazu pomocou modelu **Segment Anything (SAM)** od Meta AI.
- rozšíreného korelačného koeficientu (ECC) a hĺbkovej analýzy na základe štatistických formúl.

Obsahuje interaktívne GUI na výber datasetu, načítanie obrázkov, segmentáciu a výpočet miery zhody vrátane hĺbkovej analýzy.

---

## Štruktúra repozitára (dôležité!)

Pre správne fungovanie projektu je **nevyhnutné zachovať túto adresárovú štruktúru**, keďže skripty pracujú s pevne očakávanými cestami:

```
../puncologicky_nastroj/
├── predspracovany dataset skener cruse/
│   ├── LLa/
│   └── LRFB/
├── predspracovany dataset senzor surfacecontrol3D/
├── predspracovany dataset senzor scancontrol/
├── subory nastroja/
│   ├── ecc.py
│   ├── nastroj.py
│   ├── sam_vit_h_4b8939.pth        # ← sem vlož checkpoint SAM modelu
│   ├── lr_lla_cruse.joblib         # Trénované modely
│   ├── lr_lrfb_cruse.joblib
│   ├── lr_lla_cruse.joblib
│   └── lr_lla_cruse.joblib
└──   
```

---

## Inštalácia a závislosti

Odporúčané prostredie:  
Python **3.8 až 3.11**

### Inštalácia knižníc cez systémový príkazový riadok:

```bash
pip install opencv-python numpy matplotlib torch torchvision joblib segment_anything
```

> `tkinter` je vo väčšine prípadov súčasťou Pythona. Ak nie, na Ubuntu alebo v systémovo príkazovom riadku nainštalujte pomocou:
```bash
sudo apt install python3-tk
```

---

## Segment Anything (SAM) – nastavenie

Súbor v ktorom sú uložené natrénované váhy pre model SAM je použitý na segmentáciu obrázkov. Tento súbor **nie je súčasťou repozitára kvôli veľkosti**.

### 1. Stiahnite súbor váh modelu ViT-H

- Oficiálny zdroj:  
  https://github.com/facebookresearch/segment-anything

- Priamy link:
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

Uložte do rovnakého priečinka ako `nastroj.py`.

## Spustenie aplikácie

1. Skontrolujte, že máte stiahnutý checkpoint `sam_vit_h_4b8939.pth`.
2. Pripravte si dataset obrázkov vo formáte `.png` alebo `.tif`.
3. Spustite aplikáciu:

```bash
python nastroj.py
```

4. V GUI:
   - Zvoľte typ datasetu (Cruse / senzor),
   - Vyber dvojicu obrázkov,
   - Zvoľte režim segmentácie (jedna / viac masiek),
   - Spustite segmentáciu a analýzu.

---

## Použité knižnice

Zoznam knižníc použitých v projekte:

- `opencv-python` – spracovanie obrazu a ECC
- `numpy` – numerické výpočty
- `matplotlib` – vizualizácia masiek
- `torch` + `torchvision` – model SAM
- `joblib` – načítavanie modelov pravdepodobnosti
- `tkinter` – GUI (grafické používateľské rozhranie)
- `segment-anything` – model SAM (lokálne inštalovaný)

---

## Dôležité upozornenia

- **Veľké súbory (SAM checkpoint)**: nie sú súčasťou GitHub repozitára kvôli limitu 100 MB.
- **Modely logistických regresií(`*.joblib`)** musíte mať lokálne v priečinku `../subory nastroja/` spolu so skriptom nastroj.py. Skript ich načíta automaticky podľa zvoleného datasetu.
- **Nepremiestňujte súbory**, pokiaľ neupravíte cesty v kóde — závisia od adresárovej štruktúry.

---

## Videonávod

Súčasťou repozitára je aj videonávod, demonštrujúca spúšťanie a fungovanie puncologického nástroja

## Autor

Tento projekt vznikol ako súčasť bakalárskej práce so zameraním na analýzu zhody reliéfnych kovových puncov pomocou metód počítačového videnia.

V prípade otázok alebo spolupráce neváhaj kontaktovať autora repozitára na maily zjanradovan@gmail.com.
