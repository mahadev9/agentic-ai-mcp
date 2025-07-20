import os

from typing import List, Optional
import pandas as pd
from mcp.server.fastmcp import FastMCP, Context

mcp = FastMCP(name="Vehicle Insurance Claims")

FILE_PATH = "./data/insurance_claims.csv"


def load_data():
    """Load the insurance claims data"""
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"Data file not found: {FILE_PATH}")
    return pd.read_csv(FILE_PATH)


@mcp.tool(
    name="get_data_overview",
    description="Get an overview of the insurance claims dataset including shape, columns, and data types",
)
def get_data_overview(ctx: Context):
    """
    Provides a comprehensive overview of the vehicle insurance claims dataset.
    Returns information about dataset shape, columns, data types, and basic statistics.
    """
    try:
        df = load_data()

        overview = {
            "dataset_shape": {"rows": len(df), "columns": len(df.columns)},
            "columns": list(df.columns),
            "data_types": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
        }

        return overview
    except Exception as e:
        return {"error": str(e)}


@mcp.tool(
    name="get_column_info",
    description="Get detailed information about specific columns including unique values and statistics",
)
def get_column_info(ctx: Context, column_names: List[str]):
    """
    Get detailed information about specific columns.

    Args:
        column_names: List of column names to analyze
    """
    try:
        df = load_data()

        info = {}
        for col in column_names:
            if col not in df.columns:
                info[col] = {"error": f"Column '{col}' not found"}
                continue

            col_info = {
                "data_type": str(df[col].dtype),
                "unique_values_count": df[col].nunique(),
                "missing_values": df[col].isnull().sum(),
                "missing_percentage": f"{(df[col].isnull().sum() / len(df)) * 100:.2f}%",
            }

            # Add statistics for numeric columns
            if df[col].dtype in ["int64", "float64"]:
                col_info["statistics"] = {
                    "mean": float(df[col].mean())
                    if not df[col].isnull().all()
                    else None,
                    "median": float(df[col].median())
                    if not df[col].isnull().all()
                    else None,
                    "std": float(df[col].std()) if not df[col].isnull().all() else None,
                    "min": float(df[col].min()) if not df[col].isnull().all() else None,
                    "max": float(df[col].max()) if not df[col].isnull().all() else None,
                }

            # Add unique values for categorical columns (if reasonable number)
            if df[col].nunique() <= 50:
                col_info["unique_values"] = df[col].value_counts().to_dict()
            else:
                col_info["sample_values"] = df[col].dropna().unique()[:20].tolist()

            info[col] = col_info

        return info
    except Exception as e:
        return {"error": str(e)}


@mcp.tool(
    name="filter_claims",
    description="Filter insurance claims based on multiple criteria",
)
def filter_claims(
    ctx: Context,
    state: Optional[str] = None,
    incident_type: Optional[str] = None,
    incident_severity: Optional[str] = None,
    fraud_reported: Optional[str] = None,
    min_claim_amount: Optional[float] = None,
    max_claim_amount: Optional[float] = None,
    min_age: Optional[int] = None,
    max_age: Optional[int] = None,
    auto_make: Optional[str] = None,
    limit: Optional[int] = 100,
):
    """
    Filter insurance claims based on various criteria.

    Args:
        state: Policy state (e.g., 'OH', 'IN', 'IL')
        incident_type: Type of incident (e.g., 'Single Vehicle Collision', 'Multi-vehicle Collision')
        incident_severity: Severity level (e.g., 'Minor Damage', 'Major Damage', 'Total Loss')
        fraud_reported: Whether fraud was reported ('Y' or 'N')
        min_claim_amount: Minimum total claim amount
        max_claim_amount: Maximum total claim amount
        min_age: Minimum age of insured
        max_age: Maximum age of insured
        auto_make: Vehicle manufacturer
        limit: Maximum number of records to return (default 100)
    """
    try:
        df = load_data()

        # Apply filters
        if state:
            df = df[df["policy_state"] == state]
        if incident_type:
            df = df[df["incident_type"] == incident_type]
        if incident_severity:
            df = df[df["incident_severity"] == incident_severity]
        if fraud_reported:
            df = df[df["fraud_reported"] == fraud_reported]
        if min_claim_amount:
            df = df[df["total_claim_amount"] >= min_claim_amount]
        if max_claim_amount:
            df = df[df["total_claim_amount"] <= max_claim_amount]
        if min_age:
            df = df[df["age"] >= min_age]
        if max_age:
            df = df[df["age"] <= max_age]
        if auto_make:
            df = df[df["auto_make"] == auto_make]

        # Limit results
        if limit:
            df = df.head(limit)

        return {"filtered_count": len(df), "records": df.to_dict("records")}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool(
    name="aggregate_by_columns",
    description="Aggregate insurance claims data by one or more columns with various aggregation functions",
)
def aggregate_by_columns(
    ctx: Context,
    group_by_columns: List[str],
    aggregation_columns: List[str],
    aggregation_functions: List[str] = ["sum", "mean", "count"],
):
    """
    Aggregate insurance claims data by specified columns.

    Args:
        group_by_columns: Columns to group by
        aggregation_columns: Columns to aggregate
        aggregation_functions: Functions to apply ('sum', 'mean', 'count', 'min', 'max', 'std')
    """
    try:
        df = load_data()

        # Validate columns
        invalid_group_cols = [col for col in group_by_columns if col not in df.columns]
        invalid_agg_cols = [col for col in aggregation_columns if col not in df.columns]

        if invalid_group_cols:
            return {"error": f"Group by columns not found: {invalid_group_cols}"}
        if invalid_agg_cols:
            return {"error": f"Aggregation columns not found: {invalid_agg_cols}"}

        # Create aggregation dictionary
        agg_dict = {}
        for col in aggregation_columns:
            agg_dict[col] = aggregation_functions

        # Perform aggregation
        result = df.groupby(group_by_columns).agg(agg_dict).reset_index()

        # Flatten column names
        result.columns = [
            f"{col[0]}_{col[1]}" if col[1] else col[0] for col in result.columns
        ]

        return {
            "aggregation_summary": {
                "grouped_by": group_by_columns,
                "aggregated_columns": aggregation_columns,
                "functions": aggregation_functions,
                "result_count": len(result),
            },
            "results": result.to_dict("records"),
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool(
    name="get_fraud_analysis", description="Analyze fraud patterns in insurance claims"
)
def get_fraud_analysis(ctx: Context):
    """
    Analyze fraud patterns in the insurance claims data.
    Provides insights into fraud distribution, common characteristics, and statistics.
    """
    try:
        df = load_data()

        # Basic fraud statistics
        fraud_counts = df["fraud_reported"].value_counts().to_dict()
        fraud_percentage = (
            df["fraud_reported"].value_counts(normalize=True) * 100
        ).to_dict()

        # Fraud by state
        fraud_by_state = (
            df.groupby(["policy_state", "fraud_reported"]).size().unstack(fill_value=0)
        )

        # Fraud by incident type
        fraud_by_incident = (
            df.groupby(["incident_type", "fraud_reported"]).size().unstack(fill_value=0)
        )

        # Average claim amounts for fraud vs non-fraud
        avg_claims = (
            df.groupby("fraud_reported")["total_claim_amount"]
            .agg(["mean", "median", "std"])
            .to_dict()
        )

        # Fraud by vehicle make
        fraud_by_make = (
            df[df["fraud_reported"] == "Y"]["auto_make"]
            .value_counts()
            .head(10)
            .to_dict()
        )

        return {
            "fraud_distribution": {
                "counts": fraud_counts,
                "percentages": fraud_percentage,
            },
            "fraud_by_state": fraud_by_state.to_dict(),
            "fraud_by_incident_type": fraud_by_incident.to_dict(),
            "claim_amounts_by_fraud": avg_claims,
            "top_fraud_vehicle_makes": fraud_by_make,
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool(
    name="get_claim_statistics",
    description="Get comprehensive statistics about claim amounts and patterns",
)
def get_claim_statistics(ctx: Context, group_by: Optional[str] = None):
    """
    Get comprehensive statistics about insurance claim amounts and patterns.

    Args:
        group_by: Optional column to group statistics by (e.g., 'incident_type', 'policy_state')
    """
    try:
        df = load_data()

        # Overall statistics
        claim_cols = [
            "total_claim_amount",
            "injury_claim",
            "property_claim",
            "vehicle_claim",
        ]
        overall_stats = df[claim_cols].describe().to_dict()

        if group_by and group_by in df.columns:
            # Grouped statistics
            grouped_stats = (
                df.groupby(group_by)[claim_cols]
                .agg(["mean", "median", "sum", "count"])
                .to_dict()
            )

            return {
                "overall_statistics": overall_stats,
                "grouped_statistics": {"grouped_by": group_by, "stats": grouped_stats},
            }
        else:
            # Additional insights without grouping
            insights = {
                "highest_claims": df.nlargest(10, "total_claim_amount")[
                    [
                        "policy_number",
                        "total_claim_amount",
                        "incident_type",
                        "fraud_reported",
                    ]
                ].to_dict("records"),
                "claims_by_severity": df.groupby("incident_severity")[
                    "total_claim_amount"
                ]
                .agg(["mean", "sum", "count"])
                .to_dict(),
                "claims_by_incident_hour": df.groupby("incident_hour_of_the_day")[
                    "total_claim_amount"
                ]
                .agg(["mean", "count"])
                .to_dict(),
            }

            return {"overall_statistics": overall_stats, "insights": insights}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool(
    name="search_claims",
    description="Search for specific claims using text search across multiple fields",
)
def search_claims(
    ctx: Context,
    search_term: str,
    search_columns: Optional[List[str]] = None,
    limit: Optional[int] = 50,
):
    """
    Search for insurance claims using text search across specified columns.

    Args:
        search_term: Term to search for
        search_columns: Columns to search in (default: text columns)
        limit: Maximum number of results to return
    """
    try:
        df = load_data()

        # Default search columns if not specified
        if not search_columns:
            search_columns = [
                "incident_type",
                "collision_type",
                "incident_city",
                "auto_make",
                "auto_model",
                "insured_occupation",
                "insured_hobbies",
            ]

        # Filter to existing columns
        search_columns = [col for col in search_columns if col in df.columns]

        # Perform search
        mask = (
            df[search_columns]
            .astype(str)
            .apply(lambda x: x.str.contains(search_term, case=False, na=False))
            .any(axis=1)
        )

        results = df[mask]

        if limit:
            results = results.head(limit)

        return {
            "search_term": search_term,
            "searched_columns": search_columns,
            "total_matches": len(results),
            "results": results.to_dict("records"),
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool(
    name="get_vehicle_analysis",
    description="Analyze vehicle-related patterns in insurance claims",
)
def get_vehicle_analysis(ctx: Context):
    """
    Analyze vehicle-related patterns including make, model, year, and their relationship with claims.
    """
    try:
        df = load_data()

        # Vehicle make analysis
        make_stats = (
            df.groupby("auto_make")
            .agg(
                {
                    "total_claim_amount": ["mean", "sum", "count"],
                    "fraud_reported": lambda x: (x == "Y").sum(),
                }
            )
            .round(2)
        )

        # Vehicle age analysis
        current_year = 2015  # Based on incident dates in data
        df["vehicle_age"] = current_year - df["auto_year"]
        age_stats = (
            df.groupby("vehicle_age")
            .agg(
                {
                    "total_claim_amount": ["mean", "count"],
                    "fraud_reported": lambda x: (x == "Y").sum(),
                }
            )
            .round(2)
        )

        # Most expensive claims by vehicle
        top_expensive = df.nlargest(10, "total_claim_amount")[
            [
                "auto_make",
                "auto_model",
                "auto_year",
                "total_claim_amount",
                "incident_type",
            ]
        ].to_dict("records")

        return {
            "vehicle_make_analysis": make_stats.to_dict(),
            "vehicle_age_analysis": age_stats.to_dict(),
            "most_expensive_claims": top_expensive,
            "vehicle_summary": {
                "unique_makes": df["auto_make"].nunique(),
                "unique_models": df["auto_model"].nunique(),
                "year_range": f"{df['auto_year'].min()}-{df['auto_year'].max()}",
            },
        }
    except Exception as e:
        return {"error": str(e)}
