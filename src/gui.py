"""Interactive Streamlit GUI for XGBoost vessel-route inference.

Run with::

    streamlit run shipping_route_predictor/src/gui.py
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Vessel Route Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

import plotly.graph_objects as go

from shipping_route_predictor.config import (
    CATEGORY_COLORS,
    KNOT_TO_KMH,
    GridResolution,
    InferenceConfig,
    WELL_KNOWN_PORTS,
    PORT_ALIASES,
)
from shipping_route_predictor.inference import Inference

log = logging.getLogger("gui")


def _build_port_display() -> Tuple[Dict[str, str], List[str]]:
    """Return (code→display_label, sorted_codes) for port selectors."""
    alias_rev: Dict[str, str] = {}
    for alias, code in PORT_ALIASES.items():
        if code not in alias_rev:
            alias_rev[code] = alias.title()
    display: Dict[str, str] = {}
    for code in sorted(WELL_KNOWN_PORTS):
        friendly = alias_rev.get(code, "")
        display[code] = f"{code} — {friendly}" if friendly else code
    return display, list(display.keys())


class RouteGUI:
    """Streamlit-based GUI wrapping :class:`Inference`."""

    MODEL_COLORS = InferenceConfig.RUNNER_COLORS
    RUNNER_LABELS = InferenceConfig.RUNNER_LABELS
    OPERATOR_OPTIONS = ["(auto)"] + list(InferenceConfig.KNOWN_OPERATORS)
    CATEGORY_COLORS = CATEGORY_COLORS

    def __init__(self) -> None:
        self._port_display, self._port_codes = _build_port_display()

    # -- figure builders ----------------------------------------------------

    @staticmethod
    def _split_at_dateline(
        lats: List[float], lons: List[float],
    ) -> List[Tuple[List[float], List[float]]]:
        """Split a path into segments that don't jump across ±180° lon."""
        if not lats:
            return []
        segs: List[Tuple[List[float], List[float]]] = []
        s_lat, s_lon = [lats[0]], [lons[0]]
        for i in range(1, len(lats)):
            if abs(lons[i] - lons[i - 1]) > 180:
                segs.append((s_lat, s_lon))
                s_lat, s_lon = [lats[i]], [lons[i]]
            else:
                s_lat.append(lats[i])
                s_lon.append(lons[i])
        segs.append((s_lat, s_lon))
        return segs

    @staticmethod
    def _detect_pacific(results: Dict[str, Dict[str, Any]]) -> bool:
        """Return True if any result has a dateline crossing."""
        for rec in results.values():
            wps = rec.get("waypoints", [])
            for i in range(1, len(wps)):
                if abs(wps[i]["lon"] - wps[i - 1]["lon"]) > 180:
                    return True
        return False

    def build_route_map(
        self,
        results: Dict[str, Dict[str, Any]],
        start_latlon: Tuple[float, float],
        goal_latlon: Tuple[float, float],
        start_label: str,
        goal_label: str,
        map_height: int = 650,
    ) -> go.Figure:
        """Build a Plotly Scattergeo figure with all predicted routes."""
        fig = go.Figure()
        all_lats = [start_latlon[0], goal_latlon[0]]
        all_lons = [start_latlon[1], goal_latlon[1]]

        for name, record in results.items():
            wps = record.get("waypoints", [])
            if not wps:
                continue
            r_lats = [w["lat"] for w in wps]
            r_lons = [w["lon"] for w in wps]
            all_lats.extend(r_lats)
            all_lons.extend(r_lons)
            color = self.MODEL_COLORS.get(name, "#17becf")
            label = self.RUNNER_LABELS.get(name, name)
            steps = record.get("num_steps", len(wps))
            reached = record.get("reached_goal", False)
            suffix = f" ({steps} steps)" if reached else f" ({steps} steps, ✗)"

            segments = self._split_at_dateline(r_lats, r_lons)
            for seg_i, (seg_lat, seg_lon) in enumerate(segments):
                fig.add_trace(go.Scattergeo(
                    lat=seg_lat, lon=seg_lon,
                    mode="lines+markers",
                    name=label + suffix if seg_i == 0 else None,
                    showlegend=(seg_i == 0),
                    legendgroup=name,
                    line=dict(width=3.5, color=color),
                    marker=dict(size=4, color=color),
                    hovertemplate=(
                        f"{label}<br>Lat: %{{lat:.2f}}<br>"
                        f"Lon: %{{lon:.2f}}<extra></extra>"
                    ),
                ))

        fig.add_trace(go.Scattergeo(
            lat=[start_latlon[0]], lon=[start_latlon[1]],
            mode="markers+text", name=f"Origin ({start_label})",
            marker=dict(size=16, color="green", symbol="circle",
                        line=dict(width=2, color="white")),
            text=[start_label], textposition="top center",
            textfont=dict(size=12, color="black"),
        ))
        fig.add_trace(go.Scattergeo(
            lat=[goal_latlon[0]], lon=[goal_latlon[1]],
            mode="markers+text", name=f"Destination ({goal_label})",
            marker=dict(size=16, color="red", symbol="star",
                        line=dict(width=2, color="white")),
            text=[goal_label], textposition="top center",
            textfont=dict(size=12, color="black"),
        ))

        pacific = self._detect_pacific(results)
        center_lat = float(np.mean(all_lats))
        center_lon = 180.0 if pacific else float(np.mean(all_lons))

        fig.update_layout(
            title=dict(text=f"Route: {start_label} → {goal_label}",
                       font=dict(size=16)),
            geo=dict(
                projection_type="equirectangular",
                showland=True, landcolor="rgb(243,243,243)",
                showocean=True, oceancolor="rgb(204,229,255)",
                showcoastlines=True, coastlinecolor="rgb(100,100,100)",
                coastlinewidth=1,
                showcountries=True, countrycolor="rgb(180,180,180)",
                countrywidth=0.5,
                showlakes=True, lakecolor="rgb(204,229,255)",
                center=dict(lat=center_lat, lon=center_lon),
                showframe=True, framecolor="rgb(100,100,100)", framewidth=1,
                lonaxis=dict(
                    showgrid=True, gridwidth=0.5,
                    gridcolor="rgb(180,180,180)", dtick=10,
                    range=([center_lon - 180, center_lon + 180]
                           if pacific else None),
                ),
                lataxis=dict(showgrid=True, gridwidth=0.5,
                             gridcolor="rgb(180,180,180)", dtick=10),
            ),
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                        bgcolor="rgba(255,255,255,0.9)", font=dict(size=11)),
            margin=dict(l=10, r=10, t=40, b=10),
            height=map_height, autosize=True,
        )
        return fig

    def build_shap_timeline_fig(
        self, shap_res: Dict[str, Any], run_name: str,
    ) -> go.Figure:
        """Stacked-area chart of per-step category SHAP importance."""
        step_cat = shap_res["step_category_shap"]
        cat_names = shap_res["category_names"]
        steps = list(range(step_cat.shape[0]))

        fig = go.Figure()
        for j, cat in enumerate(cat_names):
            color = self.CATEGORY_COLORS.get(cat, "#333333")
            fig.add_trace(go.Scatter(
                x=steps, y=step_cat[:, j].tolist(),
                mode="lines", name=cat, stackgroup="one",
                line=dict(width=0.5, color=color), fillcolor=color,
                hovertemplate=(
                    f"{cat}<br>Step %{{x}}<br>|SHAP|: %{{y:.4f}}"
                    "<extra></extra>"
                ),
            ))
        fig.update_layout(
            title=dict(
                text=f"Category SHAP Importance Along Route — {run_name}",
                font=dict(size=14),
            ),
            xaxis_title="Step along predicted route",
            yaxis_title="Mean |SHAP value| by category",
            height=350, margin=dict(l=50, r=20, t=40, b=40),
            legend=dict(font=dict(size=10)),
        )
        return fig

    @staticmethod
    def build_shap_bar_fig(
        shap_res: Dict[str, Any], run_name: str,
    ) -> go.Figure:
        """Horizontal bar chart of feature importance (top-25)."""
        imp = shap_res["importance_df"].head(25)
        fig = go.Figure(go.Bar(
            x=imp["mean_abs_shap"].values[::-1],
            y=imp["feature"].values[::-1],
            orientation="h", marker_color="#1f77b4",
        ))
        fig.update_layout(
            title=dict(text=f"Feature Importance (SHAP) — {run_name}",
                       font=dict(size=14)),
            xaxis_title="Mean |SHAP value|",
            height=max(350, 18 * len(imp)),
            margin=dict(l=200, r=20, t=40, b=40),
        )
        return fig

    # -- cached engine loader -----------------------------------------------

    @staticmethod
    @st.cache_resource(show_spinner="Loading models …")
    def _get_inference(
        grid: str, runners_key: str, runners: List[str],
    ) -> Inference:
        """Return a cached Inference engine for *(grid, runners)*."""
        cfg = InferenceConfig(grid=GridResolution(grid), runners=runners)
        return Inference(cfg)

    # -- sidebar widgets ----------------------------------------------------

    def _render_sidebar(self) -> Dict[str, Any]:
        """Draw sidebar widgets and return the collected settings dict."""
        with st.sidebar:
            st.header("⚙️ Settings")

            grid_choice = st.selectbox(
                "Grid resolution", ["coarse", "fine"], index=0,
                help="Coarse: 32×64 (faster). Fine: 128×256 (detailed).",
            )
            st.divider()

            st.subheader("🤖 Models")
            all_runners = list(InferenceConfig.VALID_RUNNERS)
            selected_runners: List[str] = []
            for r in all_runners:
                default_on = r in ("simple_greedy", "shortest_path")
                if st.checkbox(self.RUNNER_LABELS.get(r, r),
                               value=default_on, key=f"cb_{r}"):
                    selected_runners.append(r)

            if any("full" in r for r in selected_runners):
                st.info("⚠️ Full models need weather data (2023-2024).")
            if any("cisc" in r for r in selected_runners):
                st.info("ℹ️ CISC models are slower (stochastic rollouts).")
            st.divider()

            st.subheader("📅 Departure")
            dep_date = st.date_input(
                "Date", value=datetime(2024, 3, 30),
                min_value=datetime(2023, 1, 1),
                max_value=datetime(2024, 12, 31),
                help="Weather data available for 2023-2024.",
            )
            dep_hour = st.slider("Hour (UTC)", 0, 23, 7)
            st.divider()

            st.subheader("🚢 Vessel")
            vessel_loa = st.number_input(
                "LOA (m)", min_value=50.0, max_value=500.0,
                value=399.0, step=10.0,
            )
            vessel_company = st.selectbox(
                "Operator", self.OPERATOR_OPTIONS, index=0,
            )
            if vessel_company == "(auto)":
                vessel_company = "msc"
            speed_kn = st.number_input(
                "Speed (kn)", min_value=5.0, max_value=30.0,
                value=14.5, step=0.5,
                help="Vessel speed for ETA estimation.",
            )
            st.divider()

            st.subheader("📊 SHAP Explainability")
            run_shap = st.checkbox(
                "Run SHAP analysis", value=False,
                help="Compute SHAP values along the predicted route.",
            )
            shap_model: Optional[str] = None
            if run_shap:
                xgb_runners = [
                    r for r in selected_runners
                    if r not in ("shortest_path", "company_baseline")
                ]
                if xgb_runners:
                    shap_model = st.selectbox("SHAP model", xgb_runners,
                                              index=0)
                else:
                    st.warning("Select an XGBoost model for SHAP.")
                    run_shap = False

        return dict(
            grid=grid_choice, runners=selected_runners,
            dep_date=dep_date, dep_hour=dep_hour,
            vessel_loa=vessel_loa, vessel_company=vessel_company,
            speed_kn=speed_kn, run_shap=run_shap, shap_model=shap_model,
        )

    # -- endpoint selectors -------------------------------------------------

    def _render_endpoint_selector(
        self, *, role: str, default_code: str, default_lat: float,
        default_lon: float,
    ) -> Tuple[Tuple[float, float], str]:
        """Render start/goal port-or-latlon selector, return (latlon, label)."""
        icon = "🟢" if role == "start" else "🔴"
        st.markdown(f"**{icon} {role.title()}**")
        mode = st.radio(
            "", ["Port", "Lat/Lon"], horizontal=True,
            key=f"{role}_mode", label_visibility="collapsed",
        )
        if mode == "Port":
            idx = (self._port_codes.index(default_code)
                   if default_code in self._port_codes else 0)
            code = st.selectbox(
                f"{role.title()} port", self._port_codes, index=idx,
                format_func=lambda c: self._port_display[c],
                key=f"{role}_port",
            )
            return WELL_KNOWN_PORTS[code], code
        c1, c2 = st.columns(2)
        with c1:
            lat = st.number_input("Lat", value=default_lat,
                                  key=f"{role}_lat",
                                  label_visibility="collapsed")
        with c2:
            lon = st.number_input("Lon", value=default_lon,
                                  key=f"{role}_lon",
                                  label_visibility="collapsed")
        return (lat, lon), f"({lat:.2f}, {lon:.2f})"

    # -- result panels ------------------------------------------------------

    def _render_metrics(self, results: Dict[str, Dict[str, Any]]) -> None:
        """Render per-model metric cards + expandable detail table."""
        st.divider()
        st.subheader("📊 Results")
        cols = st.columns(min(len(results), 4))
        for i, (name, rec) in enumerate(results.items()):
            with cols[i % 4]:
                label = self.RUNNER_LABELS.get(name, name)
                st.markdown(f"**{label}**")
                reached = rec.get("reached_goal", False)
                n_steps = rec.get("num_steps", 0)
                dist_km = rec.get("total_distance_km", 0)
                eta = rec.get("eta", "—")
                ms = rec.get("elapsed_ms", 0)

                if reached:
                    st.success(f"✅ Goal reached ({n_steps} steps)")
                else:
                    st.warning(f"⚠️ Partial ({n_steps} steps)")
                st.metric("Distance", f"{dist_km:.0f} km")
                st.metric("ETA",
                          eta[:16] if isinstance(eta, str) and len(eta) > 16
                          else eta)
                st.caption(f"Inference: {ms:.0f} ms")

        with st.expander("📋 Detailed results"):
            for name, rec in results.items():
                st.markdown(f"### {self.RUNNER_LABELS.get(name, name)}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("📏 Distance",
                          f"{rec.get('total_distance_km', 0):.0f} km")
                c2.metric("🔢 Steps", rec.get("num_steps", 0))
                c3.metric("📅 ETA", rec.get("eta", "—"))
                c4.metric("⏱ Inference",
                          f"{rec.get('elapsed_ms', 0):.0f} ms")
                st.write(
                    f"Reached goal: **{rec.get('reached_goal', False)}**")
                st.divider()

    def _render_cached_metrics(
        self, results: Dict[str, Dict[str, Any]],
    ) -> None:
        """Lighter metric display for previously-cached results."""
        st.divider()
        st.subheader("📊 Results (cached)")
        cols = st.columns(min(len(results), 4))
        for i, (name, rec) in enumerate(results.items()):
            with cols[i % 4]:
                label = self.RUNNER_LABELS.get(name, name)
                st.markdown(f"**{label}**")
                reached = rec.get("reached_goal", False)
                n_steps = rec.get("num_steps", 0)
                if reached:
                    st.success(f"✅ Goal ({n_steps} steps)")
                else:
                    st.warning(f"⚠️ Partial ({n_steps} steps)")
                st.metric("Distance",
                          f"{rec.get('total_distance_km', 0):.0f} km")
                st.metric("ETA", rec.get("eta", "—"))

    def _render_shap(
        self, results: Dict[str, Dict[str, Any]],
        engine: Inference, shap_model: str,
        start_label: str, goal_label: str,
    ) -> None:
        """Run SHAP analysis and render timeline + bar charts."""
        shap_subset = {
            k: v for k, v in results.items() if shap_model in k
        }
        if not shap_subset:
            st.warning(f"No results for SHAP model *{shap_model}*.")
            return

        with st.spinner(f"Running SHAP on **{shap_model}** …"):
            try:
                from shipping_route_predictor.explainability_shap import ShapAnalysis
                from shipping_route_predictor.config import SHAPConfig

                shap_cfg = SHAPConfig(
                    grid=engine.grid, background_samples=200,
                )
                analyser = ShapAnalysis(
                    engine._envs, engine.grid, shap_cfg,
                )
                shap_data = analyser.run(
                    shap_subset,
                    trajectory=engine.cfg.trajectory,
                    start_label=start_label,
                    goal_label=goal_label,
                )
                shap_out = analyser.save()

                st.divider()
                rlabel = self.RUNNER_LABELS.get(shap_model, shap_model)
                st.subheader(f"📊 SHAP — {rlabel}")

                for sname, sres in shap_data.items():
                    st.plotly_chart(
                        self.build_shap_timeline_fig(sres, sname),
                        use_container_width=True,
                        key=f"shap_timeline_{sname}",
                    )
                    st.plotly_chart(
                        self.build_shap_bar_fig(sres, sname),
                        use_container_width=True,
                        key=f"shap_bar_{sname}",
                    )
                st.caption(f"SHAP results also saved to: `{shap_out}`")
            except Exception as exc:
                st.error(f"SHAP analysis failed: {exc}")
                log.exception("SHAP error")

    # -- default (pre-prediction) map ---------------------------------------

    @staticmethod
    def _render_empty_map(
        placeholder,
        start_latlon: Tuple[float, float],
        goal_latlon: Tuple[float, float],
        start_label: str,
        goal_label: str,
    ) -> None:
        """Show an empty Scattergeo with start/goal markers."""
        fig = go.Figure()
        fig.add_trace(go.Scattergeo(
            lat=[start_latlon[0]], lon=[start_latlon[1]],
            mode="markers+text", name=f"Origin ({start_label})",
            marker=dict(size=16, color="green", symbol="circle"),
            text=[start_label], textposition="top center",
        ))
        fig.add_trace(go.Scattergeo(
            lat=[goal_latlon[0]], lon=[goal_latlon[1]],
            mode="markers+text", name=f"Destination ({goal_label})",
            marker=dict(size=16, color="red", symbol="star"),
            text=[goal_label], textposition="top center",
        ))
        c_lat = (start_latlon[0] + goal_latlon[0]) / 2
        c_lon = (start_latlon[1] + goal_latlon[1]) / 2
        fig.update_layout(
            geo=dict(
                projection_type="equirectangular",
                showland=True, landcolor="rgb(243,243,243)",
                showocean=True, oceancolor="rgb(204,229,255)",
                showcoastlines=True, coastlinecolor="rgb(100,100,100)",
                showcountries=True, countrycolor="rgb(180,180,180)",
                center=dict(lat=c_lat, lon=c_lon),
                lonaxis=dict(showgrid=True, dtick=10,
                             gridcolor="rgb(180,180,180)"),
                lataxis=dict(showgrid=True, dtick=10,
                             gridcolor="rgb(180,180,180)"),
            ),
            showlegend=True,
            margin=dict(l=10, r=10, t=10, b=10),
            height=650, autosize=True,
        )
        placeholder.plotly_chart(fig, use_container_width=True,
                                 key="init_map")

    # -- main entry point ---------------------------------------------------

    def run(self) -> None:
        """Top-level Streamlit render loop."""
        st.title("🚢 Vessel Route Predictor")
        st.caption("XGBoost route inference with optional SHAP explainability")

        settings = self._render_sidebar()

        ctrl_col, map_col = st.columns([1, 3])

        with ctrl_col:
            start_latlon, start_label = self._render_endpoint_selector(
                role="start", default_code="CNSHA",
                default_lat=31.4, default_lon=121.5,
            )
            goal_latlon, goal_label = self._render_endpoint_selector(
                role="goal", default_code="USLAX",
                default_lat=33.7, default_lon=-118.3,
            )
            st.markdown("---")
            predict_clicked = st.button(
                "🚀 Predict", type="primary", use_container_width=True,
                disabled=(not settings["runners"]),
            )
            if not settings["runners"]:
                st.caption("⚠️ Select at least one model in the sidebar.")

        with map_col:
            map_placeholder = st.empty()
            if "results" not in st.session_state or not st.session_state.results:
                self._render_empty_map(
                    map_placeholder, start_latlon, goal_latlon,
                    start_label, goal_label,
                )

        if predict_clicked:
            self._handle_prediction(
                settings, map_col, map_placeholder,
                start_latlon, goal_latlon, start_label, goal_label,
            )
        elif "results" in st.session_state and st.session_state.results:
            self._handle_cached(
                map_col, map_placeholder, start_latlon, goal_latlon,
                start_label, goal_label,
            )

    # -- prediction handler -------------------------------------------------

    def _handle_prediction(
        self,
        settings: Dict[str, Any],
        map_col,
        map_placeholder,
        start_latlon: Tuple[float, float],
        goal_latlon: Tuple[float, float],
        start_label: str,
        goal_label: str,
    ) -> None:
        dep = settings["dep_date"]
        departure_iso = datetime(
            dep.year, dep.month, dep.day, settings["dep_hour"], 0, 0,
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        runners_key = "_".join(sorted(settings["runners"]))

        with st.spinner("Loading models …"):
            engine = self._get_inference(
                settings["grid"], runners_key, list(settings["runners"]),
            )

        with st.spinner("Running inference …"):
            t0 = time.perf_counter()
            results = engine.predict(
                start_latlon=start_latlon,
                goal_latlon=goal_latlon,
                start_label=start_label,
                goal_label=goal_label,
                start_time=departure_iso,
                vessel_loa=settings["vessel_loa"],
                vessel_company=settings["vessel_company"],
            )
            elapsed = time.perf_counter() - t0

        st.session_state.results = results
        st.session_state.start_latlon = start_latlon
        st.session_state.goal_latlon = goal_latlon
        st.session_state.start_label = start_label
        st.session_state.goal_label = goal_label

        if not results:
            st.error("No model returned a result. Check logs for details.")
        else:
            st.success(
                f"Inference complete — {len(results)} model(s), "
                f"{elapsed:.1f} s total.",
            )

        with map_col:
            if results:
                fig = self.build_route_map(
                    results, start_latlon, goal_latlon,
                    start_label, goal_label,
                )
                map_placeholder.plotly_chart(
                    fig, use_container_width=True, key="result_map",
                )

        if results:
            self._render_metrics(results)

        if settings["run_shap"] and settings["shap_model"] and results:
            self._render_shap(
                results, engine, settings["shap_model"],
                start_label, goal_label,
            )

    # -- cached-result handler ----------------------------------------------

    def _handle_cached(
        self,
        map_col,
        map_placeholder,
        start_latlon: Tuple[float, float],
        goal_latlon: Tuple[float, float],
        start_label: str,
        goal_label: str,
    ) -> None:
        results = st.session_state.results
        s_ll = st.session_state.get("start_latlon", start_latlon)
        g_ll = st.session_state.get("goal_latlon", goal_latlon)
        s_lbl = st.session_state.get("start_label", start_label)
        g_lbl = st.session_state.get("goal_label", goal_label)

        with map_col:
            fig = self.build_route_map(results, s_ll, g_ll, s_lbl, g_lbl)
            map_placeholder.plotly_chart(
                fig, use_container_width=True, key="cached_map",
            )
        self._render_cached_metrics(results)


if __name__ == "__main__":
    RouteGUI().run()
