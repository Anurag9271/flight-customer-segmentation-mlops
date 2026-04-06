"""
app.py — Airline Customer Segmentation Explorer
------------------------------------------------
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import joblib
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="✈️ Flight Customer Segmentation",
    page_icon="✈️",
    layout="wide"
)

# ── Colour map for segments ───────────────────────────────────────────────────
SEGMENT_MAP = {
    0: 'Loyal Regulars',
    1: 'Occasional Leisure Flyers',
    2: 'Champions',
    3: 'At-Risk Customers'
}

SEGMENT_COLORS = {
    'Champions'               : '#0F6E56',
    'Loyal Regulars'          : '#185FA5',
    'At-Risk Customers'       : '#D85A30',
    'Occasional Leisure Flyers': '#993556'
}

SEGMENT_ICONS = {
    'Champions'               : '🏆',
    'Loyal Regulars'          : '✈️',
    'At-Risk Customers'       : '⚠️',
    'Occasional Leisure Flyers': '🎯'
}

SEGMENT_ACTIONS = {
    'Champions': "**Retain at all costs.** Priority boarding, lounge access, exclusive tier rewards. These customers generate the most revenue — losing one is costly.",
    'Loyal Regulars': "**Push toward next tier.** Milestone rewards (e.g. '10 flights to Gold'), co-brand credit card offers, early upgrade access.",
    'At-Risk Customers': "**Win-back campaign.** 'We miss you' email with targeted points bonus or discount on their most-flown route. Act before they fully churn.",
    'Occasional Leisure Flyers': "**Seasonal promotions.** Flash sales around holiday periods. Price-led offers only — deep loyalty investment rarely converts this segment."
}

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scaler   = joblib.load(os.path.join(base, 'outputs', 'models', 'scaler.pkl'))
    model    = joblib.load(os.path.join(base, 'outputs', 'models', 'kmeans_model.pkl'))
    return scaler, model

@st.cache_data
def load_data():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, 'data', 'processed', 'dataset_with_clusters.csv')
    return pd.read_csv(path)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("✈️ Airline Customer Segmentation Dashboard")
st.markdown("Unsupervised ML project — LRFMC clustering of 62,936 frequent flyer members")
st.divider()

# ── Load ──────────────────────────────────────────────────────────────────────
try:
    scaler, km_model = load_models()
    df = load_data()
    models_loaded = True
except Exception as e:
    st.warning(f"Models not found. Run the notebook first to generate them. ({e})")
    models_loaded = False

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Segment Overview", "🔍 Explore Segments", "🧮 Predict New Customer"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — SEGMENT OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Segment Distribution")

    if models_loaded:
        df['Segment'] = df['Cluster'].map(SEGMENT_MAP)

        # Metric cards
        col1, col2, col3, col4 = st.columns(4)
        counts = df['Segment'].value_counts()
        cols_list = [col1, col2, col3, col4]

        for col_ui, (seg, icon) in zip(cols_list, SEGMENT_ICONS.items()):
            count = counts.get(seg, 0)
            pct   = count / len(df) * 100
            with col_ui:
                st.metric(
                    label=f"{icon} {seg}",
                    value=f"{count:,}",
                    delta=f"{pct:.1f}% of customers"
                )

        st.divider()

        # Bar chart
        fig, ax = plt.subplots(figsize=(10, 4))
        seg_counts = df['Segment'].value_counts()
        bar_colors = [SEGMENT_COLORS.get(s, '#888888') for s in seg_counts.index]
        bars = ax.bar(seg_counts.index, seg_counts.values,
                      color=bar_colors, edgecolor='white')
        ax.set_title('Customer Count by Segment', fontsize=13, fontweight='bold')
        ax.set_ylabel('Customers')
        for bar, val in zip(bars, seg_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                    f'{val:,}\n({val/len(df)*100:.1f}%)', ha='center', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.divider()

        # LRFMC mean profiles
        st.subheader("Average LRFMC Profile per Segment")
        profile = df.groupby('Segment')[['L','R','F','M','C']].mean().round(3)
        st.dataframe(profile.style.background_gradient(cmap='RdYlGn', axis=0), width='stretch')

        st.caption("R is inverted — lower R means flew more recently (better). All other dimensions: higher = more active/valuable.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — EXPLORE SEGMENTS
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Explore a Specific Segment")

    if models_loaded:
        df['Segment'] = df['Cluster'].map(SEGMENT_MAP)
        selected = st.selectbox("Choose a segment:", list(SEGMENT_MAP.values()))

        seg_df = df[df['Segment'] == selected]
        icon   = SEGMENT_ICONS[selected]
        color  = SEGMENT_COLORS[selected]

        col_a, col_b = st.columns([1, 2])

        with col_a:
            st.markdown(f"### {icon} {selected}")
            st.metric("Customers", f"{len(seg_df):,}")
            st.metric("Share", f"{len(seg_df)/len(df)*100:.1f}%")
            st.divider()
            st.markdown("**Recommended Action:**")
            st.info(SEGMENT_ACTIONS[selected])

        with col_b:
            # Radar-style bar chart for this segment vs overall
            features = ['L','R','F','M','C']
            seg_means   = seg_df[features].mean()
            all_means   = df[features].mean()
            all_std     = df[features].std()

            # Normalise
            seg_norm = (seg_means - df[features].min()) / (df[features].max() - df[features].min())
            all_norm = (all_means - df[features].min()) / (df[features].max() - df[features].min())

            x = np.arange(len(features))
            width = 0.35

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(x - width/2, seg_norm, width, label=selected,
                   color=color, alpha=0.85, edgecolor='white')
            ax.bar(x + width/2, all_norm, width, label='Overall average',
                   color='#AAAAAA', alpha=0.6, edgecolor='white')
            ax.set_xticks(x)
            ax.set_xticklabels(features, fontsize=12)
            ax.set_ylabel('Normalised Mean (0=lowest, 1=highest)')
            ax.set_title(f'{selected} vs Overall Average', fontweight='bold')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Distribution plots
        st.divider()
        st.markdown("**Feature Distributions — this segment vs full dataset**")
        fig, axes = plt.subplots(1, 5, figsize=(18, 3))
        feat_labels = ['L — Length','R — Recency','F — Frequency','M — Monetary','C — Discount']
        for ax, feat, label in zip(axes, features, feat_labels):
            ax.hist(df[feat], bins=40, color='#CCCCCC', alpha=0.5, label='All customers', density=True)
            ax.hist(seg_df[feat], bins=40, color=color, alpha=0.7, label=selected, density=True)
            ax.set_title(label, fontsize=9, fontweight='bold')
            ax.legend(fontsize=7)
            ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — PREDICT NEW CUSTOMER
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🧮 Predict Segment for a New Customer")
    st.markdown("Enter a customer's LRFMC values to instantly see which segment they belong to.")

    if models_loaded:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Customer LRFMC Values**")
            L = st.slider("L — Membership length (days since joining)",
                          min_value=30, max_value=3500, value=1500,
                          help="How many days since this customer enrolled in the FFP")
            R = st.slider("R — Recency (days since last flight)",
                          min_value=1, max_value=730, value=100,
                          help="Lower = flew more recently = more active")
            F = st.slider("F — Frequency (total flights taken)",
                          min_value=1, max_value=220, value=20,
                          help="Total number of flights ever taken")
            M = st.slider("M — Monetary (total km flown)",
                          min_value=1000, max_value=600000, value=30000, step=1000,
                          help="Total kilometers flown across all flights")
            C = st.slider("C — Discount coefficient",
                          min_value=0.1, max_value=1.5, value=0.8, step=0.01,
                          help="Average discount rate: ~1.0 = full fare, <0.7 = heavy discount")

        with col2:
            st.markdown("**Prediction Result**")

            # Apply same transformations as feature engineering
            L_t = np.log1p(L)
            R_t = np.log1p(R)
            F_t = np.log1p(F)
            M_t = np.log1p(M)
            C_t = C  # C was not log-transformed

            # REPLACE with this:
            features_raw = pd.DataFrame(
                [[L_t, R_t, F_t, M_t, C_t]],
                columns=['L', 'R', 'F', 'M', 'C']
            )
            features_scaled = scaler.transform(features_raw)

            cluster_id  = int(km_model.predict(features_scaled)[0])
            segment     = SEGMENT_MAP[cluster_id]
            icon        = SEGMENT_ICONS[segment]
            color       = SEGMENT_COLORS[segment]

            st.markdown(f"### {icon} {segment}")

            # Colour-coded result box
            st.markdown(
                f"""<div style="background-color:{color}22; border-left: 5px solid {color};
                padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                <strong style="color:{color}">Cluster {cluster_id} — {segment}</strong><br>
                </div>""",
                unsafe_allow_html=True
            )

            st.markdown("**What this means:**")
            if segment == 'Champions':
                st.success("This is a high-value customer. Frequent, recent, high km, pays full fare. Treat as VIP.")
            elif segment == 'Loyal Regulars':
                st.info("Active, consistent flyer. A tier upgrade push could convert them to Champions.")
            elif segment == 'At-Risk Customers':
                st.warning("Long-term member but not flying recently. Send a win-back offer before they churn fully.")
            elif segment == 'Occasional Leisure Flyers':
                st.error("Infrequent, price-sensitive, not recent. Target with seasonal promotions only.")

            st.divider()
            st.markdown("**Recommended Action:**")
            st.write(SEGMENT_ACTIONS[segment])

            st.divider()

            # Show where this customer sits vs segment averages
            st.markdown("**Your customer vs segment averages:**")
            if 'Segment' in df.columns:
                seg_avg = df[df['Segment']==segment][['L','R','F','M','C']].mean()
                compare_df = pd.DataFrame({
                    'Your Customer': {
                        'L': round(L,0), 'R': round(R,0),
                        'F': round(F,0), 'M': round(M,0), 'C': round(C,3)
                    },
                    f'{segment} Average': seg_avg.round(2)
                })
                st.dataframe(compare_df, width='stretch')
    else:
        st.info("Run the notebook first to generate models, then restart the app.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Unsupervised ML Project · LRFMC Customer Segmentation · K-Means K=4 · 62,936 customers")