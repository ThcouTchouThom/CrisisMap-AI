# Aftermath Streamlit App

This folder contains the first Streamlit prototype for visualizing CrisisMap AI
U-Net predictions on xBD/xView2 train, validation, or test pairs. The app is
branded as Aftermath and presents a French UI for the damage mapping demo.

Expected local files:

- xBD training data: `data/raw/xbd/train`
- split CSVs: `data/processed/splits`
- checkpoint: `outputs/checkpoints/unet_1024_ce_dice_w005_1_4_noleak_match_hist1000_bs2_250epochs/best_unet_portable.pt`

The current default model is a 3-class U-Net trained with CE + Dice loss using
class weights `[0.05, 1.0, 4.0]`.

The app supports two demo modes:

- Dataset mode: select an xBD split and pair id, with ground-truth mask display.
- Upload mode: provide a real pre/post image pair and run inference without
  ground truth.

Recommended Jalon 3 demo pairs:

- `hurricane-florence_00000070`
- `hurricane-florence_00000217`
- `hurricane-florence_00000153`

Run from the project root:

```powershell
streamlit run app/streamlit_app.py
```
