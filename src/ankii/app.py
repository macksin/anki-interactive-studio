"""Streamlit app for visualizing Anki card clusters."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from ankii.anki_connect import AnkiConnect, AnkiConnectError
from ankii.evaluator import CardInfo
from ankii.embeddings import EmbeddingService, EmbeddingError
from ankii.llm import OpenRouterClient, LLMError
from ankii.clustering import (
    reduce_dimensions,
    cluster_embeddings,
    get_cluster_info,
    find_similar_pairs,
    group_similar_cards,
    find_similar_to_reference,
    classify_cards_in_groups,
)


# Page config
st.set_page_config(
    page_title="Anki Card Clusters",
    page_icon="🎴",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .card-box {
        background-color: #1e1e1e;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
        border-left: 4px solid #4CAF50;
    }
    .card-front {
        font-size: 1.1em;
        margin-bottom: 8px;
    }
    .card-back {
        color: #888;
        font-size: 0.95em;
    }
    .cluster-header {
        font-size: 1.3em;
        margin-bottom: 16px;
    }
    .delete-card {
        border-left: 4px solid #f44336 !important;
        background-color: #2d1a1a !important;
    }
    .keep-card {
        border-left: 4px solid #4CAF50 !important;
        background-color: #1a2d1a !important;
    }
    .unique-card {
        border-left: 4px solid #2196F3 !important;
        background-color: #1a1a2d !important;
    }
    .reference-card {
        border-left: 4px solid #ff9800 !important;
        background-color: #2d2a1a !important;
    }
    .similarity-badge {
        background-color: #333;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        margin-left: 8px;
    }
</style>
""", unsafe_allow_html=True)


def init_anki_connection():
    """Initialize AnkiConnect and handle errors."""
    try:
        anki = AnkiConnect()
        # Test connection
        anki.get_decks()
        return anki
    except AnkiConnectError as e:
        st.error(f"❌ Could not connect to Anki: {e}")
        st.info("Make sure Anki is running with AnkiConnect installed.")
        st.stop()


def load_cards(anki: AnkiConnect, deck: str, filter_query: str) -> list[CardInfo]:
    """Load cards from Anki."""
    query = f'deck:"{deck}"'
    if filter_query:
        query = f"{query} {filter_query}"
    
    card_ids = anki.find_cards(query)
    if not card_ids:
        return []
    
    cards_info = anki.get_cards_info(card_ids)
    return [CardInfo.from_anki_info(info) for info in cards_info]


def get_card_text(card: CardInfo) -> str:
    """Get combined text for embedding."""
    return f"{card.front} {card.back}"


@st.cache_data(ttl=3600)
def compute_embeddings_cached(texts: tuple[str, ...], model: str) -> np.ndarray:
    """Compute embeddings with caching."""
    service = EmbeddingService(model=model)
    return service.get_embeddings_batch(list(texts))


def main():
    st.title("🎴 Anki Card Clusters")
    st.markdown("Visualize similar cards using AI embeddings")
    
    # Initialize connection
    anki = init_anki_connection()
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Deck selection
        decks = anki.get_decks()
        deck = st.selectbox("Deck", sorted(decks), index=0)
        
        # Filter options with card counts
        filter_queries = {
            "All cards": "",
            "Leech cards": "tag:leech",
            "Suspended": "is:suspended",
            "Red flag": "flag:1",
            "Orange flag": "flag:2",
            "Green flag": "flag:3",
            "Blue flag": "flag:4",
            "Learning": "is:learn",
            "Due": "is:due",
            "New": "is:new",
            "High lapses (>3)": "prop:lapses>3",
            "High lapses (>5)": "prop:lapses>5",
        }
        
        # Get counts for each filter (cached per deck)
        @st.cache_data(ttl=60)
        def get_filter_counts(deck_name: str) -> dict[str, int]:
            counts = {}
            for label, query in filter_queries.items():
                full_query = f'deck:"{deck_name}"'
                if query:
                    full_query = f"{full_query} {query}"
                try:
                    card_ids = anki.find_cards(full_query)
                    counts[label] = len(card_ids)
                except:
                    counts[label] = 0
            return counts
        
        counts = get_filter_counts(deck)
        
        # Build display labels with counts
        filter_options = {
            f"{label} ({counts.get(label, 0)})": query 
            for label, query in filter_queries.items()
        }
        
        filter_label = st.selectbox("Filter", list(filter_options.keys()))
        filter_query = filter_options[filter_label]
        
        # Limit
        max_cards = st.slider("Max cards", 10, 2000, 100)
        
        st.divider()
        
        # Clustering settings
        st.subheader("Clustering")
        cluster_method = st.selectbox(
            "Algorithm", 
            ["hdbscan", "kmeans"],
            help="HDBSCAN auto-detects clusters; KMeans needs a fixed number"
        )
        
        if cluster_method == "kmeans":
            n_clusters = st.slider("Number of clusters", 2, 20, 5)
        else:
            min_cluster_size = st.slider("Min cluster size", 2, 10, 3)
        
        st.divider()
        
        # Embedding model
        embedding_model = st.selectbox(
            "Embedding model",
            ["openai/text-embedding-3-small", "openai/text-embedding-3-large"],
        )
        
        # Load button
        load_btn = st.button("🔄 Load & Cluster", type="primary", use_container_width=True)
    
    # Main content
    if load_btn or "cards_df" in st.session_state:
        if load_btn:
            with st.spinner("Loading cards from Anki..."):
                cards = load_cards(anki, deck, filter_query)
                
                if not cards:
                    st.warning("No cards found matching the query.")
                    return
                
                # Limit cards
                if len(cards) > max_cards:
                    st.info(f"Limiting to {max_cards} cards (found {len(cards)})")
                    cards = cards[:max_cards]
                
                st.session_state["cards"] = cards
            
            # Compute embeddings
            with st.spinner(f"Computing embeddings for {len(cards)} cards..."):
                try:
                    texts = tuple(get_card_text(c) for c in cards)
                    embeddings = compute_embeddings_cached(texts, embedding_model)
                except EmbeddingError as e:
                    st.error(f"Embedding error: {e}")
                    return
            
            # Reduce dimensions
            with st.spinner("Reducing dimensions with t-SNE..."):
                coords_2d = reduce_dimensions(embeddings, method="tsne")
            
            # Cluster
            with st.spinner("Clustering cards..."):
                if cluster_method == "kmeans":
                    labels = cluster_embeddings(embeddings, method="kmeans", n_clusters=n_clusters)
                else:
                    labels = cluster_embeddings(embeddings, method="hdbscan", min_cluster_size=min_cluster_size)
            
            # Create dataframe
            df = pd.DataFrame({
                "x": coords_2d[:, 0],
                "y": coords_2d[:, 1],
                "cluster": labels,
                "front": [c.front[:100] + "..." if len(c.front) > 100 else c.front for c in cards],
                "back": [c.back[:100] + "..." if len(c.back) > 100 else c.back for c in cards],
                "lapses": [c.lapses for c in cards],
                "card_idx": range(len(cards)),
            })
            
            # Mark noise as "Unclustered"
            df["cluster_label"] = df["cluster"].apply(
                lambda x: "Unclustered" if x == -1 else f"Cluster {x}"
            )
            
            st.session_state["cards_df"] = df
            st.session_state["cluster_info"] = get_cluster_info(labels)
        
        # Get from session state
        df = st.session_state["cards_df"]
        cards = st.session_state["cards"]
        cluster_info = st.session_state["cluster_info"]
        
        # Stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Cards", len(df))
        col2.metric("Clusters", cluster_info["n_clusters"])
        col3.metric("Unclustered", cluster_info["n_noise"])
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Cluster View", "🔍 Similar Pairs", "🎯 Suspended Card Matcher"])
        
        with tab1:
            # Plot and details side by side
            plot_col, detail_col = st.columns([2, 1])
            
            with plot_col:
                # Create scatter plot
                fig = px.scatter(
                    df,
                    x="x",
                    y="y",
                    color="cluster_label",
                    hover_data=["front", "lapses"],
                    title="Card Embeddings (click to select)",
                    height=600,
                )
                
                fig.update_layout(
                    showlegend=True,
                    legend_title="Clusters",
                    xaxis_title="",
                    yaxis_title="",
                    xaxis=dict(showticklabels=False, showgrid=False),
                    yaxis=dict(showticklabels=False, showgrid=False),
                )
                
                fig.update_traces(marker=dict(size=10, opacity=0.8))
                
                # Display plot with selection
                selected = st.plotly_chart(
                    fig, 
                    use_container_width=True, 
                    on_select="rerun",
                    key="scatter"
                )
            
            with detail_col:
                st.subheader("Card Details")
                
                # Check if points were selected
                if selected and selected.selection and selected.selection.point_indices:
                    indices = selected.selection.point_indices
                    st.info(f"Selected {len(indices)} card(s)")
                    
                    for idx in indices[:10]:  # Limit display
                        card = cards[idx]
                        with st.expander(f"🃏 {card.front[:50]}...", expanded=True):
                            st.markdown(f"**Front:** {card.front}")
                            st.markdown(f"**Back:** {card.back}")
                            st.caption(f"Lapses: {card.lapses} | Reviews: {card.reps}")
                else:
                    # Show cluster selector
                    cluster_options = sorted(df["cluster_label"].unique())
                    selected_cluster = st.selectbox("View cluster", cluster_options)
                    
                    cluster_cards = df[df["cluster_label"] == selected_cluster]
                    st.caption(f"{len(cluster_cards)} cards in this cluster")
                    
                    for _, row in cluster_cards.head(10).iterrows():
                        card = cards[row["card_idx"]]
                        with st.expander(f"🃏 {card.front[:50]}..."):
                            st.markdown(f"**Front:** {card.front}")
                            st.markdown(f"**Back:** {card.back}")
                            st.caption(f"Lapses: {card.lapses} | Reviews: {card.reps}")
                    
                    # Analyze cluster button
                    if st.button("🤖 Analyze with LLM", key="analyze_cluster"):
                        cluster_card_list = [
                            {"front": cards[row["card_idx"]].front, "back": cards[row["card_idx"]].back}
                            for _, row in cluster_cards.iterrows()
                        ]
                        with st.spinner("Analyzing cluster with LLM..."):
                            try:
                                llm = OpenRouterClient()
                                # Extract filter name from the label
                                current_filter = filter_label.split(" (")[0] if " (" in filter_label else filter_label
                                analysis = llm.analyze_card_group(
                                    cluster_card_list[:20],  # Limit to 20 cards for LLM
                                    filter_applied=current_filter,
                                    group_type="cluster"
                                )
                                st.session_state["last_analysis"] = analysis
                                st.session_state["last_analysis_cards"] = cluster_card_list
                                st.session_state["last_analysis_filter"] = current_filter
                                st.session_state["last_analysis_cluster"] = selected_cluster
                                st.markdown("---")
                                st.markdown("### 🤖 LLM Analysis")
                                st.markdown(analysis)
                            except LLMError as e:
                                st.error(f"LLM Error: {e}")
                    
                    # Show copy button if we have analysis
                    if "last_analysis" in st.session_state and st.session_state.get("last_analysis_cluster") == selected_cluster:
                        # Format for clipboard
                        clipboard_text = f"""# Cluster Analysis: {st.session_state.get("last_analysis_cluster", "Unknown")}
**Filter:** {st.session_state.get("last_analysis_filter", "All cards")}
**Cards:** {len(st.session_state.get("last_analysis_cards", []))}

---

## Cards in this cluster

"""
                        for i, c in enumerate(st.session_state.get("last_analysis_cards", []), 1):
                            clipboard_text += f"""### Card {i}
**Front:** {c['front']}
**Back:** {c['back']}

"""
                        clipboard_text += f"""---

## LLM Analysis

{st.session_state["last_analysis"]}
"""
                        st.code(clipboard_text, language="markdown")
                        st.caption("👆 Select all and copy the text above")
        
        with tab2:
            st.subheader("Find Similar Cards (Cosine Similarity)")
            
            # Get embeddings from cache
            texts = tuple(get_card_text(c) for c in cards)
            embeddings = compute_embeddings_cached(texts, st.session_state.get("embedding_model", "openai/text-embedding-3-small"))
            
            # Similarity threshold slider
            threshold = st.slider(
                "Similarity Threshold", 
                min_value=0.7, 
                max_value=0.99, 
                value=0.85,
                step=0.01,
                help="Higher = more similar. 0.9+ usually means near-duplicates"
            )
            
            # Find similar groups
            similar_groups = group_similar_cards(embeddings, threshold=threshold)
            
            if not similar_groups:
                st.info(f"No similar card pairs found at threshold {threshold:.2f}. Try lowering the threshold.")
            else:
                st.success(f"Found **{len(similar_groups)} groups** of similar cards")
                
                for group_idx, group in enumerate(similar_groups[:20]):  # Limit to 20 groups
                    # Calculate average similarity within group
                    group_cards_list = [cards[i] for i in group]
                    
                    with st.expander(f"📚 Group {group_idx + 1} ({len(group)} cards)", expanded=(group_idx < 3)):
                        for i, card_idx in enumerate(group):
                            card = cards[card_idx]
                            st.markdown(f"**Card {i+1}:**")
                            st.markdown(f"> **Front:** {card.front}")
                            st.markdown(f"> **Back:** {card.back}")
                            st.caption(f"Lapses: {card.lapses} | Reviews: {card.reps}")
                            if i < len(group) - 1:
                                st.divider()
                        
                        # Analyze group button
                        if st.button("🤖 Analyze with LLM", key=f"analyze_similar_{group_idx}"):
                            card_dicts = [
                                {"front": c.front, "back": c.back}
                                for c in group_cards_list
                            ]
                            with st.spinner("Analyzing similar cards..."):
                                try:
                                    llm = OpenRouterClient()
                                    current_filter = filter_label.split(" (")[0] if " (" in filter_label else filter_label
                                    analysis = llm.analyze_card_group(
                                        card_dicts,
                                        filter_applied=current_filter,
                                        group_type="similar"
                                    )
                                    # Store for copy
                                    st.session_state[f"similar_analysis_{group_idx}"] = analysis
                                    st.session_state[f"similar_cards_{group_idx}"] = card_dicts
                                    st.session_state[f"similar_filter_{group_idx}"] = current_filter
                                    
                                    st.markdown("---")
                                    st.markdown("### 🤖 LLM Analysis")
                                    st.markdown(analysis)
                                except LLMError as e:
                                    st.error(f"LLM Error: {e}")
                        
                        # Show copy text if analysis exists
                        if f"similar_analysis_{group_idx}" in st.session_state:
                            analysis = st.session_state[f"similar_analysis_{group_idx}"]
                            card_dicts = st.session_state[f"similar_cards_{group_idx}"]
                            current_filter = st.session_state[f"similar_filter_{group_idx}"]
                            
                            clipboard_text = f"""# Similar Cards Analysis - Group {group_idx + 1}
**Filter:** {current_filter}
**Similarity Threshold:** {threshold}
**Cards:** {len(card_dicts)}

---

## Cards in this group

"""
                            for i, c in enumerate(card_dicts, 1):
                                clipboard_text += f"""### Card {i}
**Front:** {c['front']}
**Back:** {c['back']}

"""
                            clipboard_text += f"""---

## LLM Analysis

{analysis}
"""
                            st.code(clipboard_text, language="markdown")
                            st.caption("👆 Select all and copy the text above")

        with tab3:
            st.subheader("Suspended Card Matcher")
            st.markdown("Find similar cards to suspended cards and manage duplicates")

            # Get embeddings from cache
            texts = tuple(get_card_text(c) for c in cards)
            embeddings = compute_embeddings_cached(texts, st.session_state.get("embedding_model", "openai/text-embedding-3-small"))

            # Settings
            col_settings1, col_settings2 = st.columns(2)
            with col_settings1:
                similarity_threshold = st.slider(
                    "Similarity Threshold",
                    min_value=0.5,
                    max_value=0.99,
                    value=0.80,
                    step=0.01,
                    key="suspended_threshold",
                    help="Cards above this similarity will be grouped together"
                )
            with col_settings2:
                deletion_tag = st.text_input(
                    "Deletion Tag",
                    value="to-delete",
                    key="deletion_tag",
                    help="Tag to apply to cards marked for deletion"
                )

            # Find suspended cards in the current set
            suspended_indices = [i for i, c in enumerate(cards) if "suspended" in str(c.fields).lower() or any("is:suspended" in filter_query for filter_query in [filter_query])]

            # Mode selection
            mode = st.radio(
                "Mode",
                ["🔍 Reference Card Mode", "🗂️ Group All Cards"],
                horizontal=True,
                key="matcher_mode"
            )

            if mode == "🔍 Reference Card Mode":
                # Reference Card Mode: Select a card and find similar ones
                st.markdown("---")
                st.markdown("### Select a Reference Card")

                # Allow selecting any card as reference
                card_options = {f"{i}: {c.front[:60]}..." if len(c.front) > 60 else f"{i}: {c.front}": i for i, c in enumerate(cards)}
                selected_card_label = st.selectbox(
                    "Reference Card",
                    list(card_options.keys()),
                    key="reference_card"
                )

                if selected_card_label:
                    ref_idx = card_options[selected_card_label]
                    ref_card = cards[ref_idx]

                    # Show reference card
                    st.markdown("#### 🎯 Reference Card")
                    with st.container():
                        st.markdown(f"**Front:** {ref_card.front}")
                        st.markdown(f"**Back:** {ref_card.back}")
                        st.caption(f"Card ID: {ref_card.card_id} | Lapses: {ref_card.lapses} | Reviews: {ref_card.reps}")

                    # Find similar cards
                    similar = find_similar_to_reference(
                        embeddings[ref_idx],
                        embeddings,
                        threshold=similarity_threshold,
                        exclude_indices=[ref_idx]
                    )

                    if similar:
                        st.markdown(f"#### Found {len(similar)} similar cards")

                        # Initialize session state for selections
                        if "selected_for_deletion" not in st.session_state:
                            st.session_state["selected_for_deletion"] = set()

                        # Display similar cards with checkboxes
                        for card_idx, sim_score in similar:
                            card = cards[card_idx]
                            col1, col2 = st.columns([0.1, 0.9])

                            with col1:
                                is_selected = st.checkbox(
                                    "🗑️",
                                    key=f"del_{card_idx}",
                                    value=card_idx in st.session_state["selected_for_deletion"],
                                    help="Mark for deletion"
                                )
                                if is_selected:
                                    st.session_state["selected_for_deletion"].add(card_idx)
                                elif card_idx in st.session_state["selected_for_deletion"]:
                                    st.session_state["selected_for_deletion"].discard(card_idx)

                            with col2:
                                status = "🔴 Delete" if card_idx in st.session_state["selected_for_deletion"] else "⚪ Keep"
                                with st.expander(f"{status} | Similarity: {sim_score:.2%} | {card.front[:50]}..."):
                                    st.markdown(f"**Front:** {card.front}")
                                    st.markdown(f"**Back:** {card.back}")
                                    st.caption(f"Card ID: {card.card_id} | Lapses: {card.lapses} | Reviews: {card.reps}")

                        # Action buttons
                        st.markdown("---")
                        col_action1, col_action2, col_action3 = st.columns(3)

                        with col_action1:
                            if st.button("🏷️ Tag Selected for Deletion", key="tag_delete_ref"):
                                if st.session_state["selected_for_deletion"]:
                                    st.session_state["confirm_action"] = "tag_selected_ref"
                                else:
                                    st.warning("No cards selected")

                        with col_action2:
                            if st.button("🔄 Clear Selection", key="clear_ref"):
                                st.session_state["selected_for_deletion"] = set()
                                st.rerun()

                        with col_action3:
                            if st.button("✅ Select All Similar", key="select_all_ref"):
                                for card_idx, _ in similar:
                                    st.session_state["selected_for_deletion"].add(card_idx)
                                st.rerun()

                        # Confirmation dialog for tagging selected cards
                        if st.session_state.get("confirm_action") == "tag_selected_ref":
                            selected_cards = [cards[i] for i in st.session_state["selected_for_deletion"]]
                            st.warning("⚠️ **Confirm Action**")
                            st.markdown(f"""
**You are about to add the tag `{deletion_tag}` to {len(selected_cards)} cards:**

| # | Front (preview) |
|---|-----------------|
""" + "\n".join([f"| {i+1} | {c.front[:50]}... |" for i, c in enumerate(selected_cards[:10])]) +
(f"\n| ... | *and {len(selected_cards) - 10} more cards* |" if len(selected_cards) > 10 else ""))

                            col_yes, col_no = st.columns(2)
                            with col_yes:
                                if st.button("✅ Yes, Tag Them", key="confirm_tag_ref", type="primary"):
                                    note_ids = [cards[i].note_id for i in st.session_state["selected_for_deletion"]]
                                    try:
                                        anki.add_tags(note_ids, deletion_tag)
                                        st.success(f"Tagged {len(note_ids)} cards with '{deletion_tag}'")
                                        st.session_state["confirm_action"] = None
                                        st.rerun()
                                    except AnkiConnectError as e:
                                        st.error(f"Error: {e}")
                            with col_no:
                                if st.button("❌ No, Cancel", key="cancel_tag_ref"):
                                    st.session_state["confirm_action"] = None
                                    st.rerun()
                    else:
                        st.info(f"No similar cards found above {similarity_threshold:.0%} threshold")

            else:
                # Group All Cards Mode
                st.markdown("---")
                st.markdown("### Card Grouping Analysis")

                # Group similar cards
                groups = group_similar_cards(embeddings, threshold=similarity_threshold)
                classification = classify_cards_in_groups(groups, len(cards))

                # Stats
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                col_stat1.metric("🟢 Keep (First in Group)", len(classification["keep"]))
                col_stat2.metric("🔴 Duplicates to Delete", len(classification["delete"]))
                col_stat3.metric("🔵 Unique (No Duplicates)", len(classification["unique"]))

                # Initialize session state for group selections
                if "group_selections" not in st.session_state:
                    st.session_state["group_selections"] = {}

                # View mode
                view_mode = st.radio(
                    "View",
                    ["📚 By Groups", "📋 By Classification"],
                    horizontal=True,
                    key="group_view_mode"
                )

                if view_mode == "📚 By Groups":
                    if groups:
                        st.markdown(f"#### Found {len(groups)} groups of similar cards")

                        for group_idx, group in enumerate(groups[:30]):  # Limit to 30 groups
                            with st.expander(f"📚 Group {group_idx + 1} ({len(group)} cards)", expanded=(group_idx < 3)):
                                # Track which card to keep in this group
                                group_key = f"group_{group_idx}_keep"
                                if group_key not in st.session_state["group_selections"]:
                                    st.session_state["group_selections"][group_key] = group[0]

                                for i, card_idx in enumerate(group):
                                    card = cards[card_idx]
                                    keep_this = st.session_state["group_selections"][group_key] == card_idx

                                    col1, col2 = st.columns([0.15, 0.85])
                                    with col1:
                                        if st.button(
                                            "✅" if keep_this else "⬜",
                                            key=f"keep_{group_idx}_{card_idx}",
                                            help="Keep this card"
                                        ):
                                            st.session_state["group_selections"][group_key] = card_idx
                                            st.rerun()

                                    with col2:
                                        status_icon = "🟢 KEEP" if keep_this else "🔴 DELETE"
                                        st.markdown(f"**{status_icon}** | {card.front[:80]}...")
                                        st.markdown(f"> {card.back[:100]}..." if len(card.back) > 100 else f"> {card.back}")
                                        st.caption(f"ID: {card.card_id} | Lapses: {card.lapses}")

                                    if i < len(group) - 1:
                                        st.divider()
                    else:
                        st.info("No duplicate groups found at this threshold. Try lowering it.")

                else:
                    # By Classification view
                    sub_tab1, sub_tab2, sub_tab3 = st.tabs(["🔴 To Delete", "🟢 To Keep", "🔵 Unique"])

                    with sub_tab1:
                        if classification["delete"]:
                            st.markdown(f"**{len(classification['delete'])} cards marked as duplicates**")
                            for idx in classification["delete"][:50]:
                                card = cards[idx]
                                with st.expander(f"🔴 {card.front[:60]}..."):
                                    st.markdown(f"**Front:** {card.front}")
                                    st.markdown(f"**Back:** {card.back}")
                                    st.caption(f"Card ID: {card.card_id}")
                        else:
                            st.info("No duplicates found")

                    with sub_tab2:
                        if classification["keep"]:
                            st.markdown(f"**{len(classification['keep'])} cards to keep (first in each group)**")
                            for idx in classification["keep"][:50]:
                                card = cards[idx]
                                with st.expander(f"🟢 {card.front[:60]}..."):
                                    st.markdown(f"**Front:** {card.front}")
                                    st.markdown(f"**Back:** {card.back}")
                                    st.caption(f"Card ID: {card.card_id}")
                        else:
                            st.info("No grouped cards found")

                    with sub_tab3:
                        if classification["unique"]:
                            st.markdown(f"**{len(classification['unique'])} unique cards (no duplicates)**")
                            for idx in classification["unique"][:50]:
                                card = cards[idx]
                                with st.expander(f"🔵 {card.front[:60]}..."):
                                    st.markdown(f"**Front:** {card.front}")
                                    st.markdown(f"**Back:** {card.back}")
                                    st.caption(f"Card ID: {card.card_id}")
                        else:
                            st.info("All cards have duplicates")

                # Batch Actions
                st.markdown("---")
                st.markdown("### Batch Actions")

                # Calculate delete indices for confirmation
                delete_indices = []
                for group_idx, group in enumerate(groups):
                    group_key = f"group_{group_idx}_keep"
                    keep_idx = st.session_state["group_selections"].get(group_key, group[0])
                    delete_indices.extend([idx for idx in group if idx != keep_idx])

                col_batch1, col_batch2 = st.columns(2)

                with col_batch1:
                    if st.button("🏷️ Tag All Duplicates for Deletion", type="primary", key="tag_all_duplicates"):
                        if delete_indices:
                            st.session_state["confirm_action"] = "tag_all_duplicates"
                            st.session_state["pending_delete_indices"] = delete_indices
                        else:
                            st.info("No duplicates to tag")

                with col_batch2:
                    if st.button("🔄 Remove Deletion Tags", key="remove_tags"):
                        st.session_state["confirm_action"] = "remove_all_tags"

                # Confirmation dialog for tagging all duplicates
                if st.session_state.get("confirm_action") == "tag_all_duplicates":
                    pending_indices = st.session_state.get("pending_delete_indices", [])
                    pending_cards = [cards[i] for i in pending_indices]
                    note_ids = list(set([c.note_id for c in pending_cards]))

                    st.warning("⚠️ **Confirm Batch Tagging**")
                    st.markdown(f"""
**You are about to add the tag `{deletion_tag}` to {len(note_ids)} notes ({len(pending_cards)} cards):**

This action will mark the following cards for deletion:

| # | Front (preview) | Lapses |
|---|-----------------|--------|
""" + "\n".join([f"| {i+1} | {c.front[:40]}... | {c.lapses} |" for i, c in enumerate(pending_cards[:15])]) +
(f"\n| ... | *and {len(pending_cards) - 15} more cards* | |" if len(pending_cards) > 15 else ""))

                    col_yes, col_no = st.columns(2)
                    with col_yes:
                        if st.button("✅ Yes, Tag All Duplicates", key="confirm_tag_all", type="primary"):
                            try:
                                anki.add_tags(note_ids, deletion_tag)
                                st.success(f"Tagged {len(note_ids)} notes with '{deletion_tag}'")
                                st.session_state["confirm_action"] = None
                                st.session_state["pending_delete_indices"] = None
                                st.rerun()
                            except AnkiConnectError as e:
                                st.error(f"Error: {e}")
                    with col_no:
                        if st.button("❌ No, Cancel", key="cancel_tag_all"):
                            st.session_state["confirm_action"] = None
                            st.session_state["pending_delete_indices"] = None
                            st.rerun()

                # Confirmation dialog for removing tags
                if st.session_state.get("confirm_action") == "remove_all_tags":
                    all_note_ids = list(set([c.note_id for c in cards]))

                    st.warning("⚠️ **Confirm Tag Removal**")
                    st.markdown(f"""
**You are about to remove the tag `{deletion_tag}` from {len(all_note_ids)} notes:**

This will clear the deletion tag from all loaded cards.
""")

                    col_yes, col_no = st.columns(2)
                    with col_yes:
                        if st.button("✅ Yes, Remove Tags", key="confirm_remove_tags", type="primary"):
                            try:
                                anki.remove_tags(all_note_ids, deletion_tag)
                                st.success(f"Removed '{deletion_tag}' tag from all cards")
                                st.session_state["confirm_action"] = None
                                st.rerun()
                            except AnkiConnectError as e:
                                st.error(f"Error: {e}")
                    with col_no:
                        if st.button("❌ No, Cancel", key="cancel_remove_tags"):
                            st.session_state["confirm_action"] = None
                            st.rerun()

                # Summary section
                st.markdown("---")
                st.markdown("### Summary")

                summary_text = f"""## Card Grouping Summary
**Threshold:** {similarity_threshold:.0%}
**Total Cards:** {len(cards)}
**Groups Found:** {len(groups)}
**Cards to Keep:** {len(classification['keep'])}
**Cards to Delete:** {len(classification['delete'])}
**Unique Cards:** {len(classification['unique'])}

### Groups Detail
"""
                for group_idx, group in enumerate(groups[:20]):
                    group_key = f"group_{group_idx}_keep"
                    keep_idx = st.session_state["group_selections"].get(group_key, group[0])
                    keep_card = cards[keep_idx]
                    summary_text += f"\n**Group {group_idx + 1}** ({len(group)} cards)\n"
                    summary_text += f"- KEEP: {keep_card.front[:50]}...\n"
                    for idx in group:
                        if idx != keep_idx:
                            summary_text += f"- DELETE: {cards[idx].front[:50]}...\n"

                st.code(summary_text, language="markdown")
                st.caption("👆 Copy the summary above")

    else:
        st.info("👈 Select a deck and click **Load & Cluster** to start")


if __name__ == "__main__":
    main()

