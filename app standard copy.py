import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np

# Set full-width layout
st.set_page_config(layout="wide", page_title="Price Sensitivity Dashboard")

st.markdown(
    """
    <style>
    body, .stApp {
        background-color: #E8F0FF;
    }
    .stSidebar {
        background-color: #3C37FF !important;
    }
    .stSidebar div {
        color: white !important;
    }
    /* Restore main content text color to original (dark) */
    .stMarkdown, .stText, .stSubheader, .stMetric, .stTitle, .stHeader, .stTable {
        color: #12123B !important;
    }
    /* Restore selectbox, multiselect, and expander label text color */
    label, .stSelectbox label, .stMultiSelect label, .streamlit-expanderHeader, details > summary {
        color: #12123B !important;
        font-size: 16px;
    }
    /* Style buttons */
    .stButton>button {
        background-color: #3C37FF !important;
        color: white !important;
        border-radius: 8px;
        border: none;
    }
    .stMetric {
        color: #E8F0FF !important;
    }
    .metric-box {
        width: 250px;
        height: 250px;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        background-color: #F9F6F0;
        margin-bottom: 10px;
    }
    .main-title, .column-title {
        color: #12123B !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Load Aggregated Data
st.sidebar.markdown("### 📁 Upload per-product CSV")
uploaded = st.sidebar.file_uploader(
    "Upload instructed CSV",
    type="csv",
)

if not uploaded:
    st.sidebar.info("Please upload your per-product CSV to enable the dashboard")
    st.stop()

@st.cache_data
def load_product_df(csv) -> pd.DataFrame:
    return pd.read_csv(csv, dtype={"ProductId": str})

product_df = load_product_df(uploaded)


# --- Auto-map columns if not already mapped ---
if "column_mappings" not in st.session_state:
    required_columns = [
        "Revenue After", "Revenue Before",
        "Margin After", "Margin Before",
        "Quantity After", "Quantity Before"
    ]
    sensitivity_col = "Sensitivity" if "Sensitivity" in product_df.columns else None

    st.session_state.column_mappings = {}
    if sensitivity_col:
        st.session_state.column_mappings["Sensitivity"] = sensitivity_col
    for col in required_columns:
        st.session_state.column_mappings[col] = col


# ─── Build aggregated revenue, margin & quantity DataFrames ───────────────────
def make_agg_df(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    raw_group = st.session_state.column_mappings.get("Sensitivity", None)
    if raw_group is not None and raw_group not in df.columns:
        raw_group = None

    raw_after = st.session_state.column_mappings[f"{metric} After"]
    raw_before = st.session_state.column_mappings[f"{metric} Before"]

    group_cols = [raw_group] if raw_group else []
    grp = (
      df
      .groupby(group_cols, as_index=False) if group_cols else df
    )
    if group_cols:
        grp = grp.agg({
            raw_after: "sum",
            raw_before: "sum",
        })
    else:
        grp = pd.DataFrame([{
            raw_after: df[raw_after].sum(),
            raw_before: df[raw_before].sum(),
        }])

    # Rename columns
    rename_map = {
      raw_after: "After",
      raw_before: "Before",
    }
    if raw_group:
        rename_map[raw_group] = "Sensitivity"

    grp.rename(columns=rename_map, inplace=True)

    # Compute deltas & pct changes
    grp["Change"]    = grp["After"]  - grp["Before"]
    grp["%Change"]   = ((grp["Change"]    / grp["Before"])  * 100).round(2)

    return grp


def calculate_aggregated_performance(df, test_after_col, test_before_col, control_after_col, control_before_col):
    test_after = df[test_after_col].sum()
    test_before = df[test_before_col].sum()
    control_after = df[control_after_col].sum()
    control_before = df[control_before_col].sum()

    test_pct = round(((test_after - test_before) / test_before) * 100, 2) if test_before != 0 else 0.0
    control_pct = round(((control_after - control_before) / control_before) * 100, 2) if control_before != 0 else 0.0
    perf_diff = round(test_pct - control_pct, 2)
    return test_pct, control_pct, perf_diff, test_after, test_before, control_after, control_before



# Correct Test % Change Calculation
def compute_percentage_change(df, column_after, column_before):
    return round(((df[column_after].sum() - df[column_before].sum()) / df[column_before].sum()) * 100, 2)


# Function to display arrows based on performance
def performance_arrow(perf_diff):
    if perf_diff > 0:
        return f"<span style='color: green;'>{perf_diff:.2f}% better than Control</span>"
    elif perf_diff < 0:
        return f"<span style='color: red;'>{abs(perf_diff):.2f}% worse than Control</span>"
    else:
        return f"<span style='color: #12123B;'>No difference from Control</span>"

def rename_columns(df: pd.DataFrame, column_mappings: dict) -> pd.DataFrame:
    """
    Rename the columns of the DataFrame based on the user-provided mappings.
    """
    # Filter out any empty mappings
    valid_mappings = {key: value for key, value in column_mappings.items() if value}
    
    # Reverse the mapping to rename columns
    rename_mapping = {value: key for key, value in valid_mappings.items()}
    
    # Rename the columns in the DataFrame
    return df.rename(columns=rename_mapping) 

def style_pct_change(pct_change):
    color = "green" if pct_change >= 0 else "red"
    return f'<span style="color: {color};">{pct_change}%</span>'

# Sidebar Navigation
st.sidebar.title("🔍 Select a View")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Per Product Performance"])



# Function to create bar charts with rounded values
def create_bar_chart(df, column, title):
    df = df.copy()
    df[column] = df[column].round(2)  # Ensure values are rounded before plotting
    color_col = "Sensitivity" if "Sensitivity" in df.columns else None
    fig = px.bar(
        df, 
        x="Sensitivity" if color_col else df.index, 
        y=column, 
        color=color_col, 
        title=title, 
        text=df[column].astype(str) + '%'
    )
    return fig

# --- HOME PAGE ---
if page == "🏠 Home":
    st.markdown("""
    <script>
    window.scrollTo(0, 0);
    </script>
    """, unsafe_allow_html=True)
    
    if "column_mappings" not in st.session_state or any(value == "" for key, value in st.session_state.column_mappings.items() if key not in ["Price change"]):
        st.error("Please complete the Data Setup page and map all required columns (except 'Price change') before proceeding.")
        st.stop()

    # Create aggregated DataFrames
    revenue_df = make_agg_df(product_df, "Revenue")
    margin_df = make_agg_df(product_df, "Margin")
    quantity_df = make_agg_df(product_df, "Quantity")
    
    # Correct Percentage Change Calculation for Home Page using the same method as in Per Product Performance
    revenue_pct = compute_percentage_change(revenue_df, 'After', 'Before')
    margin_pct = compute_percentage_change(margin_df, 'After', 'Before')
    quantity_pct = compute_percentage_change(quantity_df, 'After', 'Before')

    # Proceed with the rest of the Home Page logic
    st.markdown("<h1 class='main-title';'>Before/After Dashboard</h1>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])  

    # --- COLUMN 1: REVENUE ---
    with col1:
        st.markdown("<h2 class='column-title'>Revenue</h2>", unsafe_allow_html=True)
        with st.container():
            st.markdown(
                f"""
                <div style=\"background-color:#F9F6F0; padding: 15px; border-radius: 0px; margin-bottom: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);\">
                    <h5 style=\"margin: 0; color: #414168;\">After</h5>
                    <div style=\"display: flex; justify-content: space-between; align-items: center;\">
                        <h3 style=\"margin: 0; color: #12123B;\">€{round(revenue_df['After'].sum()):,}</h3>
                        <p style=\"margin: 0; display: inline; color: #414168;\"> % Change: {style_pct_change(revenue_pct)}</p>
                    </div>
                    <p style=\"margin: 0; color: #414168;\">Before: €{round(revenue_df['Before'].sum()):,}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    # --- COLUMN 2: MARGIN ---
    with col2:
        st.markdown("<h2 class='column-title' style='text-align: left;'>Margin</h2>", unsafe_allow_html=True)
        with st.container():
            st.markdown(
                f"""
                <div style=\"background-color:#F9F6F0; padding: 15px; border-radius: 0px; margin-bottom: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);\">
                    <h5 style=\"margin: 0; color: #414168;\">After</h5>
                    <div style=\"display: flex; justify-content: space-between; align-items: center;\">
                        <h3 style=\"margin: 0; color: #12123B;\">€{round(margin_df['After'].sum()):,}</h3>
                        <p style=\"margin: 0; display: inline; color: #414168;\"> % Change: {style_pct_change(margin_pct)}</p>
                    </div>
                    <p style=\"margin: 0; color: #414168;\">Before: €{round(margin_df['Before'].sum()):,}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    # --- COLUMN 3: QUANTITY ---
    with col3:
        st.markdown("<h2 class='column-title' style='text-align: left;'>Quantity</h2>", unsafe_allow_html=True)
        with st.container():
            st.markdown(
                f"""
                <div style=\"background-color:#F9F6F0; padding: 15px; border-radius: 0px; margin-bottom: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);\">
                    <h5 style=\"margin: 0; color: #414168;\">After</h5>
                    <div style=\"display: flex; justify-content: space-between; align-items: center;\">
                        <h3 style=\"margin: 0; color: #12123B;\">{round(quantity_df['After'].sum()):,}</h3>
                        <p style=\"margin: 0; display: inline; color: #414168;\"> % Change: {style_pct_change(quantity_pct)}</p>
                    </div>
                    <p style=\"margin: 0; color: #414168;\">Before: {round(quantity_df['Before'].sum()):,}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    # --- Add the Matplotlib Figure ---
    # Data for the three categories: Revenue, Margin, and Quantity (in percentage)
    categories = ['Revenue Change', 'Margin Change', 'Quantity Change']
    values = [revenue_pct, margin_pct, quantity_pct]

    # Create a bar plot for the data
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.tight_layout()
    fig.patch.set_facecolor('#F9F6F0')
    ax.set_facecolor('#F9F6F0')
    bar_width = 0.5
    index = range(len(categories))
    bars = ax.bar(index, values, bar_width, color='#3C37FF')
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, f'{bar.get_height():.2f}%', 
                ha='center', va='center', fontsize=10, color='white')
    ax.set_xlabel('Category', color='black')
    ax.set_ylabel('Percentage Change (%)', color='black')
    ax.set_title('Percentage Changes in Revenue, Margin, and Quantity', color='black')
    ax.set_xticks([i for i in index])
    ax.set_xticklabels(categories, color='black')
    st.pyplot(fig)




if page == "📊 Per Product Performance":
    perf_df = rename_columns(product_df, st.session_state.column_mappings)
    perf_df["ProductId"] = perf_df["ProductId"].astype(str)
    st.markdown("<h1 style='color: #12123B; text-align: left;'>Per Product Performance</h1>", unsafe_allow_html=True)

    # --------------------- FILTER SECTION ---------------------
    with st.expander("🔍 Filter Products by Attributes", expanded=True):
        filter_columns = st.multiselect("Select attributes to filter on", perf_df.columns.tolist())
        filters = {}
        for col in filter_columns:
            unique_vals = perf_df[col].dropna().unique()
            selected_vals = st.multiselect(f"Select values for '{col}'", sorted(unique_vals.astype(str)), key=col)
            if selected_vals:
                filters[col] = selected_vals
        filtered_df = perf_df.copy()
        for col, vals in filters.items():
            filtered_df = filtered_df[filtered_df[col].astype(str).isin(vals)]
    st.dataframe(filtered_df, use_container_width=True)

    # --------------------- PERFORMANCE METRICS ---------------------
    if not filtered_df.empty:
        def calculate_aggregated_performance(df, after_col, before_col):
            after = df[after_col].sum()
            before = df[before_col].sum()
            pct = round(((after - before) / before) * 100, 2) if before != 0 else 0.0
            return pct, after, before

        # Revenue
        rev_pct, rev_after, rev_before = calculate_aggregated_performance(
            filtered_df, 'Revenue After', 'Revenue Before')
        # Margin
        mar_pct, mar_after, mar_before = calculate_aggregated_performance(
            filtered_df, 'Margin After', 'Margin Before')
        # Quantity
        qty_pct, qty_after, qty_before = calculate_aggregated_performance(
            filtered_df, 'Quantity After', 'Quantity Before')

        col1, col2, col3 = st.columns(3)
        # --- REVENUE METRIC ---
        with col1:
            st.markdown("<h2 class='column-title'>Revenue</h2>", unsafe_allow_html=True)
            with st.container():
                st.markdown(f"""
                    <div style=\"background-color:#F9F6F0; padding: 15px; border-radius: 0px; margin-bottom: 10px;\">
                        <h5 style=\"margin: 0; color: #414168;\">After</h5>
                        <div style=\"display: flex; justify-content: space-between;\">
                            <h3 style=\"margin: 0; color: #12123B;\">€{rev_after:,.0f}</h3>
                            <p style=\"margin: 0; color: #414168;\">% Change: {style_pct_change(rev_pct)}</p>
                        </div>
                        <p style=\"margin: 0; color: #414168;\">Before: €{rev_before:,.0f}</p>
                    </div>
                """, unsafe_allow_html=True)
        # --- MARGIN METRIC ---
        with col2:
            st.markdown("<h2 class='column-title'>Margin</h2>", unsafe_allow_html=True)
            with st.container():
                st.markdown(f"""
                    <div style=\"background-color:#F9F6F0; padding: 15px; border-radius: 0px; margin-bottom: 10px;\">
                        <h5 style=\"margin: 0; color: #414168;\">After</h5>
                        <div style=\"display: flex; justify-content: space-between;\">
                            <h3 style=\"margin: 0; color: #12123B;\">€{mar_after:,.0f}</h3>
                            <p style=\"margin: 0; color: #414168;\">% Change: {style_pct_change(mar_pct)}</p>
                        </div>
                        <p style=\"margin: 0; color: #414168;\">Before: €{mar_before:,.0f}</p>
                    </div>
                """, unsafe_allow_html=True)
        # --- QUANTITY METRIC ---
        with col3:
            st.markdown("<h2 class='column-title'>Quantity</h2>", unsafe_allow_html=True)
            with st.container():
                st.markdown(f"""
                    <div style=\"background-color:#F9F6F0; padding: 15px; border-radius: 0px; margin-bottom: 10px;\">
                        <h5 style=\"margin: 0; color: #414168;\">After</h5>
                        <div style=\"display: flex; justify-content: space-between;\">
                            <h3 style=\"margin: 0; color: #12123B;\">{qty_after:,.0f}</h3>
                            <p style=\"margin: 0; color: #414168;\">% Change: {style_pct_change(qty_pct)}</p>
                        </div>
                        <p style=\"margin: 0; color: #414168;\">Before: {qty_before:,.0f}</p>
                    </div>
                """, unsafe_allow_html=True)


    # --------------------- VISUAL COMPARISON SECTION ---------------------
    st.markdown("<br><h2 class='column-title'>Create a Visual Comparison</h2>", unsafe_allow_html=True)
    with st.container():
        agg_method = st.selectbox("Select Aggregation Method", ["Sum", "Average"])
        metric = st.selectbox("Select Metric for Visual", ["Quantity", "Revenue", "Margin"])
        # --- Group 1: Multi-column filter ---
        group1_cols = st.multiselect("Select Group 1 Columns", product_df.columns.tolist(), key="group1_cols")
        filters_group1 = {}
        for col in group1_cols:
            filter_vals = st.multiselect(f"Select values for Group 1 ({col})", sorted(product_df[col].dropna().unique().astype(str)), key=f"group1_{col}")
            if filter_vals:
                filters_group1[col] = filter_vals
        # --- Group 2: Multi-column filter ---
        group2_cols = st.multiselect("Select Group 2 Columns", product_df.columns.tolist(), key="group2_cols")
        filters_group2 = {}
        for col in group2_cols:
            filter_vals = st.multiselect(f"Select values for Group 2 ({col})", sorted(product_df[col].dropna().unique().astype(str)), key=f"group2_{col}")
            if filter_vals:
                filters_group2[col] = filter_vals
        agg_func = np.sum if agg_method == "Sum" else np.mean
        col_map = st.session_state.column_mappings
        before_col = col_map.get(f"{metric} Before")
        after_col = col_map.get(f"{metric} After")
        missing_cols = [c for c in [before_col, after_col] if c is None]
        if missing_cols:
            st.error(f"Missing column mapping(s) for: {', '.join(missing_cols)}. Please check Data Setup.")
        else:
            df_group1 = product_df.copy()
            for col, vals in filters_group1.items():
                df_group1 = df_group1[df_group1[col].astype(str).isin(vals)]
            if filters_group2:
                df_group2 = product_df.copy()
                for col, vals in filters_group2.items():
                    df_group2 = df_group2[df_group2[col].astype(str).isin(vals)]
            else:
                df_group2 = pd.DataFrame()
            g1_before = agg_func(df_group1[before_col])
            g1_after = agg_func(df_group1[after_col])
            g2_before = agg_func(df_group2[before_col]) if not df_group2.empty else None
            g2_after = agg_func(df_group2[after_col]) if not df_group2.empty else None
            plot_df = []
            plot_df.append({"Group": "Group 1", "Period": "Before", "Value": g1_before})
            plot_df.append({"Group": "Group 1", "Period": "After",  "Value": g1_after})
            if not df_group2.empty:
                plot_df.append({"Group": "Group 2", "Period": "Before", "Value": g2_before})
                plot_df.append({"Group": "Group 2", "Period": "After", "Value": g2_after})
            plot_df = pd.DataFrame(plot_df)
            fig = px.line(
                plot_df,
                x="Period",
                y="Value",
                color="Group",
                markers=True,
                title=f"{agg_method} {metric} Comparison Before and After"
            )
            fig.update_layout(
                xaxis_title="Period",
                yaxis_title=metric,
                legend_title="Group"
            )
            st.plotly_chart(fig, use_container_width=True)

    # --------------------- TOP/BOTTOM PRODUCTS SECTION ---------------------
    st.markdown("<br><h2 class='column-title'>Top and Bottom Products</h2>", unsafe_allow_html=True)
    with st.container():
        # --- Top Performers Filter Section ---
        with st.expander("🔍 Filter Top/Bottom Products by Attributes", expanded=False):
            filter_columns_top = st.multiselect("Select attributes to filter on", perf_df.columns.tolist(), key="top_perf_filter_cols")
            filters_top = {}
            for col in filter_columns_top:
                unique_vals = perf_df[col].dropna().unique()
                selected_vals = st.multiselect(f"Select values for '{col}'", sorted(unique_vals.astype(str)), key=f"top_perf_{col}")
                if selected_vals:
                    filters_top[col] = selected_vals
            filtered_perf_df = perf_df.copy()
            for col, vals in filters_top.items():
                filtered_perf_df = filtered_perf_df[filtered_perf_df[col].astype(str).isin(vals)]
        df_for_top = filtered_perf_df if 'filtered_perf_df' in locals() else perf_df

        # Metric selector for Top/Bottom products
        selected_metric = st.selectbox(
            "Select a Metric:",
            ["Revenue", "Margin", "Quantity"],
            index=0,
            help="Choose the metric to analyze top and bottom products"
        )
        # Top-X dropdown (this already persists by default)
        top_x = st.selectbox(
            "Select the Number of Top Products to Display:",
            [5, 10, 15, 20, 30],
            index=0
        )
        # Map to the right column in product_df
        column_map = {
            "Revenue":  "Revenue After",
            "Margin":   "Margin After",
            "Quantity": "Quantity After"
        }
        selected_column = column_map[selected_metric]
        # Build & round the top/bottom tables
        top_products    = df_for_top[df_for_top[selected_column] > 0].nlargest(top_x, selected_column).copy()
        bottom_products = df_for_top[df_for_top[selected_column] > 0].nsmallest(top_x, selected_column).copy()
        top_products[selected_column]    = top_products[selected_column].round(2)
        bottom_products[selected_column] = bottom_products[selected_column].round(2)
        top_products["ProductId"] = top_products["ProductId"].astype(str)
        bottom_products["ProductId"] = bottom_products["ProductId"].astype(str)
        # Plot Top X
        fig_top = px.bar(
            top_products,
            x="ProductId",
            y=selected_column,
            title=f"Top {top_x} Products by {selected_metric}",
            text=selected_column
        )
        fig_top.update_xaxes(type="category")
        st.plotly_chart(fig_top, use_container_width=True)
        # Plot Bottom X
        fig_bottom = px.bar(
            bottom_products,
            x="ProductId",
            y=selected_column,
            title=f"Bottom {top_x} Products by {selected_metric}",
            text=selected_column
        )
        fig_bottom.update_xaxes(type="category");
        st.plotly_chart(fig_bottom, use_container_width=True)

    # --------------------- OUTLIER ANALYSIS SECTION ---------------------
    st.markdown("---")
    st.markdown("<h2 style='text-align: left; color: #12123B;'>Outlier Analysis</h2>", unsafe_allow_html=True)
    outlier_metric = st.selectbox("Select Metric for Outlier Analysis", ["Revenue", "Margin", "Quantity"])
    metric_columns = {
        "Revenue": ("Revenue Before", "Revenue After"),
        "Margin": ("Margin Before", "Margin After"),
        "Quantity": ("Quantity Before", "Quantity After")
    }
    before_col, after_col = metric_columns[outlier_metric]
    excluded_cols = set([col for cols in metric_columns.values() for col in cols])
    filterable_cols = [col for col in perf_df.columns if col not in excluded_cols and col != "ProductId"]
    with st.expander("🔍 Filter Outlier Table", expanded=False):
        filter_columns = st.multiselect("Select attributes to filter on", filterable_cols)
        filters = {}
        for col in filter_columns:
            unique_vals = perf_df[col].dropna().unique()
            selected_vals = st.multiselect(f"Select values for '{col}'", sorted(unique_vals.astype(str)), key=f"outlier_{col}")
            if selected_vals:
                filters[col] = selected_vals
        filtered_perf_df = perf_df.copy()
        for col, vals in filters.items():
            filtered_perf_df = filtered_perf_df[filtered_perf_df[col].astype(str).isin(vals)]
    attribute_cols = [col for col in filtered_perf_df.columns if col not in excluded_cols and col != "ProductId"]
    outlier_df = filtered_perf_df[["ProductId"] + attribute_cols + [before_col, after_col]].copy()
    outlier_df["Δ"] = outlier_df[after_col] - outlier_df[before_col]
    outlier_df[[before_col, after_col, "Δ"]] = outlier_df[[before_col, after_col, "Δ"]].round(2)
    outlier_df = outlier_df.sort_values(by="Δ", ascending=True)
    styled_df = outlier_df.style.background_gradient(
        subset=["Δ"], cmap='RdYlGn', low=0.2, high=0.8
    ).format(precision=2)
    st.dataframe(styled_df, use_container_width=True)













