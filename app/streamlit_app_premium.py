"""Aftermath - interface Streamlit premium expérimentale pour démonstration."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
import torch

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from streamlit_app import (  # noqa: E402
    AppError,
    BUILDING_CHECKPOINT,
    BUILDING_MODULE_AVAILABLE,
    DAMAGE_MODELS,
    DATA_ROOT,
    PIPELINES,
    PROJECT_ROOT,
    RECOMMENDED_PAIR_IDS,
    TARGET_MODE,
    build_result_record,
    colorize,
    colorize_building,
    compute_metrics,
    entropy_map,
    load_sample,
    load_split,
    make_overlay,
    prediction_stats,
    prepare_models,
    read_upload,
    render_comparison_slider,
    render_download,
    run_inference,
    tensor_to_rgb,
)


PREMIUM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --bg: #06080d;
  --bg-2: #0b111b;
  --panel: rgba(17, 24, 39, 0.72);
  --panel-strong: rgba(15, 23, 42, 0.94);
  --border: rgba(148, 163, 184, 0.18);
  --text: #f8fafc;
  --muted: #94a3b8;
  --subtle: #64748b;
  --red: #ef4444;
  --green: #22c55e;
  --cyan: #22d3ee;
  --blue: #3b82f6;
}

html, body, .stApp {
  background:
    radial-gradient(circle at 18% 5%, rgba(239, 68, 68, 0.18), transparent 32%),
    radial-gradient(circle at 85% 15%, rgba(34, 211, 238, 0.16), transparent 30%),
    linear-gradient(135deg, #05070c 0%, #08111f 45%, #05070c 100%) !important;
  color: var(--text) !important;
  font-family: 'Inter', sans-serif !important;
}

#MainMenu, footer {
  visibility: hidden !important;
}

section[data-testid="stSidebar"] {
  background: rgba(5, 8, 13, 0.93) !important;
  border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * {
  color: var(--text) !important;
}

.block-container {
  padding-top: 1.3rem !important;
  max-width: 1480px !important;
}

.premium-hero {
  position: relative;
  overflow: hidden;
  padding: 2.1rem 2.2rem;
  border: 1px solid rgba(148, 163, 184, 0.22);
  border-radius: 22px;
  background:
    linear-gradient(135deg, rgba(15, 23, 42, 0.88), rgba(8, 13, 23, 0.66)),
    radial-gradient(circle at 80% 30%, rgba(34, 211, 238, 0.16), transparent 35%);
  box-shadow: 0 30px 100px rgba(0, 0, 0, 0.38);
  backdrop-filter: blur(18px);
}

.premium-hero::after {
  content: "";
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px);
  background-size: 46px 46px;
  mask-image: linear-gradient(90deg, rgba(0,0,0,.65), transparent 80%);
  pointer-events: none;
}

.hero-kicker {
  font-family: 'JetBrains Mono', monospace;
  color: var(--cyan);
  font-size: 0.76rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  margin-bottom: 0.4rem;
}

.hero-title {
  font-family: 'Syne', sans-serif;
  font-size: clamp(3.2rem, 7vw, 6.5rem);
  line-height: 0.88;
  font-weight: 800;
  letter-spacing: -0.07em;
  color: var(--text);
}

.hero-subtitle {
  color: #cbd5e1;
  max-width: 760px;
  font-size: 1.05rem;
  line-height: 1.65;
  margin-top: 1rem;
}

.hero-pill-row, .timeline {
  display: flex;
  gap: 0.6rem;
  flex-wrap: wrap;
  margin-top: 1.25rem;
}

.pill {
  border: 1px solid rgba(148, 163, 184, 0.26);
  background: rgba(15, 23, 42, 0.72);
  color: #dbeafe;
  padding: 0.42rem 0.72rem;
  border-radius: 999px;
  font-size: 0.78rem;
  font-family: 'JetBrains Mono', monospace;
}

.pill.red { color: #fecaca; border-color: rgba(239,68,68,.42); }
.pill.green { color: #bbf7d0; border-color: rgba(34,197,94,.42); }
.pill.cyan { color: #a5f3fc; border-color: rgba(34,211,238,.42); }

.glass {
  border: 1px solid var(--border);
  border-radius: 18px;
  background: var(--panel);
  box-shadow: 0 20px 80px rgba(0, 0, 0, 0.26);
  backdrop-filter: blur(14px);
  padding: 1.15rem;
}

.section-title {
  font-family: 'Syne', sans-serif;
  color: var(--text);
  font-size: 1.25rem;
  font-weight: 700;
  margin: 1.4rem 0 0.8rem;
}

.metric-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 0.85rem;
  margin: 0.8rem 0 1rem;
}

.metric-card {
  border: 1px solid rgba(148, 163, 184, 0.20);
  background: linear-gradient(180deg, rgba(15, 23, 42, .92), rgba(15, 23, 42, .62));
  border-radius: 16px;
  padding: 1rem;
}

.metric-value {
  font-family: 'Syne', sans-serif;
  font-size: 2rem;
  font-weight: 800;
  letter-spacing: -0.04em;
}

.metric-label {
  color: var(--muted);
  font-size: 0.74rem;
  text-transform: uppercase;
  letter-spacing: 0.10em;
}

.pipeline-step {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  padding: 0.52rem 0.78rem;
  border: 1px solid rgba(148, 163, 184, 0.24);
  background: rgba(15, 23, 42, 0.70);
  border-radius: 12px;
  color: #e2e8f0;
  font-size: 0.82rem;
}

.pipeline-step span {
  color: var(--cyan);
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem;
}

.stButton > button, .stDownloadButton > button {
  border-radius: 12px !important;
  border: 1px solid rgba(148, 163, 184, 0.26) !important;
  background: linear-gradient(135deg, rgba(239,68,68,.96), rgba(185,28,28,.92)) !important;
  color: white !important;
  font-weight: 700 !important;
}

.stTabs [data-baseweb="tab-list"] {
  gap: 0.4rem;
  border-bottom: 1px solid rgba(148, 163, 184, 0.18);
}

.stTabs [data-baseweb="tab"] {
  border-radius: 12px 12px 0 0;
  color: #cbd5e1 !important;
}

.stTabs [aria-selected="true"] {
  color: #f8fafc !important;
  border-bottom: 2px solid var(--red) !important;
}

.caption-muted {
  color: var(--muted);
  font-size: 0.86rem;
}
</style>
"""


def rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def building_available() -> bool:
    return BUILDING_MODULE_AVAILABLE and BUILDING_CHECKPOINT.exists()


def metric_card(label: str, value: str, color: str = "#f8fafc") -> str:
    return (
        '<div class="metric-card">'
        f'<div class="metric-value" style="color:{color};">{value}</div>'
        f'<div class="metric-label">{label}</div>'
        "</div>"
    )


def render_hero(device: str, checkpoint_label: str) -> None:
    st.markdown(
        f"""
<div class="premium-hero">
  <div class="hero-kicker">CrisisMap AI · Prototype expérimental</div>
  <div class="hero-title">AFTERMATH</div>
  <div class="hero-subtitle">
    Voir les dégâts pour agir plus vite. Une interface de démonstration premium pour transformer
    une paire satellite avant/après catastrophe en carte visuelle des bâtiments intacts et endommagés.
  </div>
  <div class="hero-pill-row">
    <span class="pill red">Damage champion v2 · F1 0.7013</span>
    <span class="pill green">Building b400 · F1 0.8504</span>
    <span class="pill cyan">TTA d4 + component majority</span>
    <span class="pill">{device.upper()}</span>
    <span class="pill">{checkpoint_label}</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_pipeline_timeline(use_building: bool, damage_tta: str) -> None:
    steps = [
        ("01", "Image pré"),
        ("02", "Image post"),
        ("03", "Damage model"),
    ]
    if damage_tta == "d4":
        steps.append(("04", "TTA d4"))
    if use_building:
        steps.extend([("05", "Building mask"), ("06", "Component majority")])
    steps.append(("07", "Carte finale"))
    html = '<div class="timeline">' + "".join(
        f'<div class="pipeline-step"><span>{idx}</span>{label}</div>' for idx, label in steps
    ) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_sidebar() -> dict[str, Any]:
    st.sidebar.markdown("### Mission Control")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cuda_available = device == "cuda"
    b_available = building_available()

    model_id = st.sidebar.selectbox(
        "Modèle damage",
        list(DAMAGE_MODELS),
        index=0,
        format_func=lambda key: DAMAGE_MODELS[key]["label"],
    )
    model_cfg = DAMAGE_MODELS[model_id]
    st.sidebar.caption(model_cfg["summary"])
    st.sidebar.caption(f"Checkpoint : {rel_path(model_cfg['checkpoint'])}")

    pipeline_ids = list(PIPELINES)
    default_pipeline = "damage_tta_building" if cuda_available and b_available else "damage_tta"
    pipeline_id = st.sidebar.selectbox(
        "Pipeline",
        pipeline_ids,
        index=pipeline_ids.index(default_pipeline),
        format_func=lambda key: PIPELINES[key]["label"],
    )
    pipeline_cfg = PIPELINES[pipeline_id]

    threshold = 0.60
    building_tta = "none"
    if pipeline_cfg["use_building"]:
        threshold = st.sidebar.slider("Seuil bâtiment", 0.10, 0.90, 0.60, 0.05)
        building_tta = "d4" if st.sidebar.toggle("TTA d4 building", value=cuda_available) else "none"

    source_mode = st.sidebar.radio(
        "Source",
        ["upload", "dataset"],
        format_func=lambda value: "Upload paire réelle" if value == "upload" else "Exemples dataset",
    )
    demo_mode = st.sidebar.toggle("Mode présentation", value=True)

    st.sidebar.divider()
    st.sidebar.markdown("### Statut")
    st.sidebar.caption(f"Device : {device.upper()}")
    st.sidebar.caption(f"Damage checkpoint : {'OK' if model_cfg['checkpoint'].exists() else 'MANQUANT'}")
    st.sidebar.caption(f"Building checkpoint : {'OK' if b_available else 'MANQUANT'}")
    if b_available:
        st.sidebar.caption(f"Building utilisé : {rel_path(BUILDING_CHECKPOINT)}")
    if pipeline_cfg["use_building"] and not b_available:
        st.sidebar.error("Le pipeline building est indisponible : checkpoint ou module manquant.")

    return {
        "device": device,
        "model_id": model_id,
        "model_cfg": model_cfg,
        "model_label": model_cfg["label"],
        "pipeline_id": pipeline_id,
        "pipeline_label": pipeline_cfg["label"],
        "damage_tta": pipeline_cfg["damage_tta"],
        "use_building": bool(pipeline_cfg["use_building"] and b_available),
        "pipeline_requested_building": bool(pipeline_cfg["use_building"]),
        "building_threshold": threshold,
        "building_tta": building_tta,
        "source_mode": source_mode,
        "demo_mode": demo_mode,
    }


def load_models(cfg: dict[str, Any]):
    safe_cfg = {
        "device": cfg["device"],
        "model_cfg": cfg["model_cfg"],
        "use_building": cfg["use_building"],
    }
    return prepare_models(safe_cfg)


def available_dataset_ids() -> list[str]:
    ids: list[str] = []
    for split in ("test", "val", "train"):
        try:
            ids.extend(load_split(split)["pair_id"].astype(str).tolist())
        except AppError:
            continue
    seen: set[str] = set()
    unique = [pair_id for pair_id in ids if not (pair_id in seen or seen.add(pair_id))]
    recommended = [pair_id for pair_id in RECOMMENDED_PAIR_IDS if pair_id in set(unique)]
    return recommended or unique


def split_for_pair(pair_id: str) -> str:
    for split in ("test", "val", "train"):
        try:
            if pair_id in load_split(split)["pair_id"].astype(str).values:
                return split
        except AppError:
            continue
    return "test"


def infer_from_dataset(cfg: dict[str, Any], pair_id: str) -> dict[str, Any]:
    sample = load_sample(split_for_pair(pair_id), pair_id, int(cfg["model_cfg"]["image_size"]))
    damage_model, building_model = load_models(cfg)
    inference = run_inference(
        damage_model,
        sample["image"],
        cfg["device"],
        cfg["damage_tta"],
        cfg["use_building"],
        building_model,
        cfg["building_threshold"],
        cfg["building_tta"],
    )
    target = sample["target"].detach().cpu().numpy()
    return build_result_record(
        pre=tensor_to_rgb(sample["image"][:3]),
        post=tensor_to_rgb(sample["image"][3:6]),
        inference=inference,
        target=target,
        pair_id=pair_id,
        model_name=cfg["model_label"],
        pipeline_label=cfg["pipeline_label"],
    )


def infer_from_upload(cfg: dict[str, Any], pre_file, post_file) -> dict[str, Any]:
    size = int(cfg["model_cfg"]["image_size"])
    pre = read_upload(pre_file, size)
    post = read_upload(post_file, size)
    image_np = np.concatenate([pre, post], axis=2).transpose(2, 0, 1)
    image = torch.from_numpy(image_np.copy()).float().div(255.0)
    damage_model, building_model = load_models(cfg)
    inference = run_inference(
        damage_model,
        image,
        cfg["device"],
        cfg["damage_tta"],
        cfg["use_building"],
        building_model,
        cfg["building_threshold"],
        cfg["building_tta"],
    )
    return build_result_record(
        pre=pre,
        post=post,
        inference=inference,
        target=None,
        pair_id=f"upload_{datetime.now():%Y%m%d_%H%M%S}",
        model_name=cfg["model_label"],
        pipeline_label=cfg["pipeline_label"],
    )


def render_source_panel(cfg: dict[str, Any]) -> dict[str, Any] | None:
    st.markdown('<div class="section-title">Entrée satellite</div>', unsafe_allow_html=True)
    render_pipeline_timeline(cfg["use_building"], cfg["damage_tta"])

    if cfg["pipeline_requested_building"] and not cfg["use_building"]:
        st.error("Pipeline qualité maximale demandé, mais le modèle building est indisponible.")
        return None

    if cfg["source_mode"] == "dataset":
        if not DATA_ROOT.exists():
            st.error(f"Données xBD introuvables : `{DATA_ROOT}`")
            return None
        ids = available_dataset_ids()
        if not ids:
            st.error("Aucun exemple dataset disponible.")
            return None
        left, right = st.columns([4, 1])
        pair_id = left.selectbox("Exemple xBD", ids, label_visibility="collapsed")
        current_key = (
            "dataset",
            pair_id,
            cfg["model_id"],
            cfg["pipeline_id"],
            cfg["damage_tta"],
            cfg["building_threshold"],
            cfg["building_tta"],
        )
        if right.button("Analyser", type="primary", width="stretch"):
            with st.spinner("Inférence premium en cours..."):
                try:
                    record = infer_from_dataset(cfg, pair_id)
                except Exception as exc:
                    st.error(str(exc))
                    return None
            st.session_state["premium_record"] = record
            st.session_state["premium_key"] = current_key
    else:
        left, right = st.columns(2)
        pre_file = left.file_uploader("Image avant catastrophe", type=["png", "jpg", "jpeg", "tif", "tiff"])
        post_file = right.file_uploader("Image après catastrophe", type=["png", "jpg", "jpeg", "tif", "tiff"])
        if pre_file:
            left.image(read_upload(pre_file, int(cfg["model_cfg"]["image_size"])), caption="Avant", width="stretch")
        if post_file:
            right.image(read_upload(post_file, int(cfg["model_cfg"]["image_size"])), caption="Après", width="stretch")
        ready = pre_file is not None and post_file is not None
        current_key = (
            "upload",
            pre_file.name if pre_file else None,
            getattr(pre_file, "size", None) if pre_file else None,
            post_file.name if post_file else None,
            getattr(post_file, "size", None) if post_file else None,
            cfg["model_id"],
            cfg["pipeline_id"],
            cfg["damage_tta"],
            cfg["building_threshold"],
            cfg["building_tta"],
        )
        if st.button("Lancer l'inférence premium", type="primary", disabled=not ready, width="stretch"):
            with st.spinner("Inférence premium en cours..."):
                try:
                    record = infer_from_upload(cfg, pre_file, post_file)
                except Exception as exc:
                    st.error(str(exc))
                    return None
            st.session_state["premium_record"] = record
            st.session_state["premium_key"] = current_key

    if st.session_state.get("premium_key") == current_key:
        return st.session_state.get("premium_record")
    return None


def metric_html(record: dict[str, Any]) -> str:
    target = record["target"]
    final_pred = record["final_pred"]
    if target is not None:
        metrics = compute_metrics(final_pred, target)
        return '<div class="metric-row">' + "".join(
            [
                metric_card("F1 damaged", f"{metrics['f1_damaged']:.3f}", "#fca5a5"),
                metric_card("IoU damaged", f"{metrics['iou_damaged']:.3f}", "#fca5a5"),
                metric_card("Mean IoU", f"{metrics['mean_iou']:.3f}", "#bfdbfe"),
                metric_card("Taux dommage", f"{metrics['damage_ratio']:.1%}", "#fca5a5"),
            ]
        ) + "</div>"
    stats = prediction_stats(final_pred)
    return '<div class="metric-row">' + "".join(
        [
            metric_card("Taux dommage", f"{stats['damage_ratio']:.1%}", "#fca5a5"),
            metric_card("Pixels bâtiment", f"{stats['building_pixels']:,}", "#bbf7d0"),
            metric_card("Pixels endommagés", f"{stats['damaged_pixels']:,}", "#fca5a5"),
        ]
    ) + "</div>"


def render_record(record: dict[str, Any], cfg: dict[str, Any]) -> None:
    pre = record["pre"]
    post = record["post"]
    raw_pred = record["raw_pred"]
    final_pred = record["final_pred"]
    building_mask = record["building_mask"]
    target = record["target"]
    overlay = make_overlay(post, final_pred)

    st.markdown('<div class="section-title">Résultat principal</div>', unsafe_allow_html=True)
    st.markdown(metric_html(record), unsafe_allow_html=True)
    render_comparison_slider(post, overlay, "Après catastrophe", "Overlay final")

    tabs = st.tabs(["Overview", "Model output", "Metrics", "Uncertainty", "Export"])

    with tabs[0]:
        st.markdown('<div class="section-title">Vue de démonstration</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        cols[0].image(pre, caption="Avant catastrophe", width="stretch")
        cols[1].image(post, caption="Après catastrophe", width="stretch")
        cols[2].image(overlay, caption="Overlay final", width="stretch")
        st.markdown(
            f"""
<div class="glass">
  <strong>Pipeline :</strong> {record['pipeline_label']}<br>
  <span class="caption-muted">Modèle : {record['model_name']} · Target : {TARGET_MODE}</span>
</div>
""",
            unsafe_allow_html=True,
        )

    with tabs[1]:
        st.markdown('<div class="section-title">Sorties intermédiaires</div>', unsafe_allow_html=True)
        row1 = st.columns(3)
        row1[0].image(colorize(raw_pred), caption="Damage brut", width="stretch")
        if building_mask is not None:
            row1[1].image(colorize_building(building_mask), caption="Masque bâtiment", width="stretch")
        else:
            row1[1].image(np.zeros_like(post), caption="Masque bâtiment non utilisé", width="stretch")
        row1[2].image(colorize(final_pred), caption="Damage final", width="stretch")
        if target is not None:
            st.image(colorize(target), caption="Vérité terrain", width="stretch")

    with tabs[2]:
        st.markdown('<div class="section-title">Métriques</div>', unsafe_allow_html=True)
        st.markdown(metric_html(record), unsafe_allow_html=True)
        if target is None:
            st.info("Aucune vérité terrain fournie : statistiques de prédiction uniquement.")
        else:
            st.json(compute_metrics(final_pred, target))

    with tabs[3]:
        st.markdown('<div class="section-title">Incertitude</div>', unsafe_allow_html=True)
        st.image(entropy_map(record["probs"]), caption="Entropie du modèle", width="stretch")
        prob_cols = st.columns(3)
        for class_id, label in enumerate(["Fond", "Bâtiment intact", "Bâtiment endommagé"]):
            prob_cols[class_id].image(
                (record["probs"][:, :, class_id] * 255).astype(np.uint8),
                caption=label,
                width="stretch",
            )

    with tabs[4]:
        st.markdown('<div class="section-title">Exports</div>', unsafe_allow_html=True)
        metrics = compute_metrics(final_pred, target) if target is not None else None
        render_download(final_pred, post, metrics, record["pair_id"], record["model_name"])
        report = {
            "pair_id": record["pair_id"],
            "model": record["model_name"],
            "pipeline": record["pipeline_label"],
            "created_at": datetime.now().isoformat(),
            "checkpoint_damage": rel_path(cfg["model_cfg"]["checkpoint"]),
            "checkpoint_building": rel_path(BUILDING_CHECKPOINT) if cfg["use_building"] else None,
        }
        st.download_button(
            "Rapport premium JSON",
            json.dumps(report, indent=2, ensure_ascii=False),
            file_name=f"premium_report_{record['pair_id']}.json",
            mime="application/json",
            width="stretch",
        )


def main() -> None:
    st.set_page_config(
        page_title="Aftermath Premium",
        page_icon="satellite",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(PREMIUM_CSS, unsafe_allow_html=True)
    cfg = render_sidebar()

    checkpoint_label = "portable" if "portable" in cfg["model_cfg"]["checkpoint"].name else "standard"
    render_hero(cfg["device"], checkpoint_label)

    if not cfg["model_cfg"]["checkpoint"].exists():
        st.error(f"Checkpoint damage manquant : `{cfg['model_cfg']['checkpoint']}`")
        st.stop()

    record = render_source_panel(cfg)
    if record is None:
        st.markdown(
            """
<div class="glass">
  <strong>Prêt pour la démonstration.</strong><br>
  Sélectionnez une source, choisissez un pipeline, puis lancez l'inférence pour afficher
  l'overlay final, les masques intermédiaires, les métriques et les exports.
</div>
""",
            unsafe_allow_html=True,
        )
        return

    render_record(record, cfg)


if __name__ == "__main__":
    main()
