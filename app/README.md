# Aftermath Streamlit App

This folder contains the first Streamlit prototype for visualizing CrisisMap AI
U-Net predictions on xBD/xView2 train, validation, or test pairs. The app is
branded as Aftermath and presents a French UI for the damage mapping demo.

Expected local files:

- xBD training data: `data/raw/xbd/train`
- split CSVs: `data/processed/splits`
- checkpoint: `outputs/checkpoints/unet_512_ce_dice_w005_1_4_50epochs/best_unet.pt`

The current default model is a 3-class U-Net trained with weighted CE + Dice loss
using class weights `[0.05, 1.0, 4.0]`.

Run from the project root:

```powershell
streamlit run app/streamlit_app.py
```
