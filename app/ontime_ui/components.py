"""Small HTML building blocks so every page looks and reads the same."""

from __future__ import annotations

from html import escape

import streamlit as st

from ontime_ui.theme import BAND_STYLE, CRITICAL, GOOD


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="ot-hero"><h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )


def kpi(label: str, value: str, sub: str = "", help_text: str | None = None) -> None:
    tip = f' title="{escape(help_text)}"' if help_text else ""
    st.markdown(
        f'<div class="ot-card"{tip}><div class="ot-kpi-label">{escape(label)}</div>'
        f'<div class="ot-kpi-value">{escape(value)}</div>'
        f'<div class="ot-kpi-sub">{escape(sub)}</div></div>',
        unsafe_allow_html=True,
    )


def band_badge(band: str) -> str:
    s = BAND_STYLE[band]
    return (
        f'<span class="ot-badge" style="background:{s["bg"]};color:{s["color"]}">'
        f"{s['icon']} {escape(s['label'])}</span>"
    )


def reasons(items: list[str], kind: str) -> None:
    """kind: 'risk' (adds risk) or 'protect' (reduces risk). Icon + label + colour."""
    color, tag, icon = (
        (CRITICAL, "Adds risk", "▲") if kind == "risk" else (GOOD, "Reduces risk", "▼")
    )
    if not items:
        st.markdown(
            '<div class="ot-cap">Nothing above the display threshold.</div>', unsafe_allow_html=True
        )
        return
    for text in items:
        clean = text.rsplit(" (", 1)[0]  # the tag replaces the "(adds risk)" suffix
        st.markdown(
            f'<div class="ot-reason" style="border-left-color:{color}">'
            f'<span class="tag" style="color:{color}">{icon} {tag}</span>{escape(clean)}</div>',
            unsafe_allow_html=True,
        )


def callout(text: str, kind: str = "info") -> None:
    cls = "ot-callout warn" if kind == "warn" else "ot-callout"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


def caption(text: str) -> None:
    """One-line 'what to look at' under every chart (BUILD_PLAN Phase 9)."""
    st.markdown(f'<div class="ot-cap">{escape(text)}</div>', unsafe_allow_html=True)


def steps(items: list[tuple[str, str]]) -> None:
    cols = st.columns(len(items))
    for i, (col, (title, body)) in enumerate(zip(cols, items, strict=True), start=1):
        with col:
            st.markdown(
                f'<div class="ot-step"><span class="n">{i}</span><h4>{escape(title)}</h4>'
                f"<p>{escape(body)}</p></div>",
                unsafe_allow_html=True,
            )


def pct(v: float, nd: int = 1) -> str:
    return f"{v * 100:.{nd}f}%"
