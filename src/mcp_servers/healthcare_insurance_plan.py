import os

from mcp.server.fastmcp import FastMCP, Context
import pandas as pd
from typing import List, Optional

mcp = FastMCP(name="Healthcare Insurance Plan")

FILE_PATH = "./data/insurance_dataset.csv"


def load_data():
    """Load the healthcare insurance dataset"""
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"Data file not found: {FILE_PATH}")
    return pd.read_csv(FILE_PATH)


@mcp.tool(
    name="get_dataset_overview",
    description="Get an overview of the healthcare insurance dataset including shape, columns, and basic statistics",
)
def get_dataset_overview(ctx: Context):
    """
    Provides a comprehensive overview of the healthcare insurance dataset.
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
            "charges_summary": {
                "mean": float(df["charges"].mean()),
                "median": float(df["charges"].median()),
                "min": float(df["charges"].min()),
                "max": float(df["charges"].max()),
                "std": float(df["charges"].std()),
            },
        }

        return overview
    except Exception as e:
        return {"error": str(e)}


@mcp.tool(
    name="get_demographic_analysis",
    description="Analyze demographic patterns in healthcare insurance data",
)
def get_demographic_analysis(ctx: Context):
    """
    Analyze demographic patterns including age, gender, BMI, and their relationship with insurance charges.
    """
    try:
        df = load_data()

        # Age analysis
        age_groups = pd.cut(
            df["age"],
            bins=[0, 25, 35, 50, 65, 100],
            labels=["18-25", "26-35", "36-50", "51-65", "65+"],
        )
        age_stats = (
            df.groupby(age_groups)
            .agg({"charges": ["mean", "median", "count"], "bmi": "mean"})
            .round(2)
        )

        # Gender analysis
        gender_stats = (
            df.groupby("gender")
            .agg({"charges": ["mean", "median", "count"], "bmi": "mean", "age": "mean"})
            .round(2)
        )

        # BMI categories analysis
        bmi_categories = pd.cut(
            df["bmi"],
            bins=[0, 18.5, 25, 30, float("inf")],
            labels=["Underweight", "Normal", "Overweight", "Obese"],
        )
        bmi_stats = (
            df.groupby(bmi_categories)
            .agg({"charges": ["mean", "median", "count"]})
            .round(2)
        )

        return {
            "age_group_analysis": age_stats.to_dict(),
            "gender_analysis": gender_stats.to_dict(),
            "bmi_category_analysis": bmi_stats.to_dict(),
            "summary": {
                "average_age": float(df["age"].mean()),
                "average_bmi": float(df["bmi"].mean()),
                "gender_distribution": df["gender"].value_counts().to_dict(),
            },
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool(
    name="analyze_risk_factors",
    description="Analyze risk factors and their impact on insurance charges",
)
def analyze_risk_factors(ctx: Context):
    """
    Analyze various risk factors including smoking, medical history, family history, and their impact on charges.
    """
    try:
        df = load_data()

        # Smoking analysis
        smoking_stats = (
            df.groupby("smoker")
            .agg({"charges": ["mean", "median", "count"], "bmi": "mean", "age": "mean"})
            .round(2)
        )

        # Medical history analysis
        medical_history_stats = (
            df.groupby("medical_history")
            .agg({"charges": ["mean", "median", "count"]})
            .round(2)
        )

        # Family medical history analysis
        family_history_stats = (
            df.groupby("family_medical_history")
            .agg({"charges": ["mean", "median", "count"]})
            .round(2)
        )

        # Exercise frequency analysis
        exercise_stats = (
            df.groupby("exercise_frequency")
            .agg({"charges": ["mean", "median", "count"], "bmi": "mean"})
            .round(2)
        )

        # Combined risk factors
        risk_combinations = (
            df.groupby(["smoker", "medical_history"])
            .agg({"charges": ["mean", "count"]})
            .round(2)
        )

        return {
            "smoking_impact": smoking_stats.to_dict(),
            "medical_history_impact": medical_history_stats.to_dict(),
            "family_history_impact": family_history_stats.to_dict(),
            "exercise_frequency_impact": exercise_stats.to_dict(),
            "combined_risk_analysis": risk_combinations.to_dict(),
            "risk_factor_summary": {
                "smokers_percentage": (df["smoker"] == "yes").mean() * 100,
                "people_with_medical_history": df["medical_history"].notna().sum(),
                "people_with_family_history": df["family_medical_history"]
                .notna()
                .sum(),
            },
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool(
    name="filter_insurance_records",
    description="Filter healthcare insurance records based on multiple criteria",
)
def filter_insurance_records(
    ctx: Context,
    min_age: Optional[int] = None,
    max_age: Optional[int] = None,
    gender: Optional[str] = None,
    smoker: Optional[str] = None,
    region: Optional[str] = None,
    min_bmi: Optional[float] = None,
    max_bmi: Optional[float] = None,
    medical_history: Optional[str] = None,
    coverage_level: Optional[str] = None,
    min_charges: Optional[float] = None,
    max_charges: Optional[float] = None,
    limit: Optional[int] = 100,
):
    """
    Filter healthcare insurance records based on various criteria.

    Args:
        min_age: Minimum age
        max_age: Maximum age
        gender: Gender ('male' or 'female')
        smoker: Smoking status ('yes' or 'no')
        region: Region ('northeast', 'northwest', 'southeast', 'southwest')
        min_bmi: Minimum BMI
        max_bmi: Maximum BMI
        medical_history: Medical history condition
        coverage_level: Coverage level ('Basic', 'Standard', 'Premium')
        min_charges: Minimum insurance charges
        max_charges: Maximum insurance charges
        limit: Maximum number of records to return (default 100)
    """
    try:
        df = load_data()

        # Apply filters
        if min_age:
            df = df[df["age"] >= min_age]
        if max_age:
            df = df[df["age"] <= max_age]
        if gender:
            df = df[df["gender"] == gender]
        if smoker:
            df = df[df["smoker"] == smoker]
        if region:
            df = df[df["region"] == region]
        if min_bmi:
            df = df[df["bmi"] >= min_bmi]
        if max_bmi:
            df = df[df["bmi"] <= max_bmi]
        if medical_history:
            df = df[df["medical_history"] == medical_history]
        if coverage_level:
            df = df[df["coverage_level"] == coverage_level]
        if min_charges:
            df = df[df["charges"] >= min_charges]
        if max_charges:
            df = df[df["charges"] <= max_charges]

        # Limit results
        if limit:
            df = df.head(limit)

        return {"filtered_count": len(df), "records": df.to_dict("records")}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool(
    name="aggregate_by_attributes",
    description="Aggregate healthcare insurance data by one or more attributes with various aggregation functions",
)
def aggregate_by_attributes(
    ctx: Context,
    group_by_columns: List[str],
    aggregation_columns: List[str],
    aggregation_functions: List[str] = ["sum", "mean", "count"],
):
    """
    Aggregate healthcare insurance data by specified attributes.

    Args:
        group_by_columns: Columns to group by (e.g., ['region', 'smoker'])
        aggregation_columns: Columns to aggregate (e.g., ['charges', 'bmi'])
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
    name="get_regional_analysis",
    description="Analyze healthcare insurance patterns by region",
)
def get_regional_analysis(ctx: Context):
    """
    Analyze healthcare insurance patterns across different regions.
    """
    try:
        df = load_data()

        # Regional statistics
        regional_stats = (
            df.groupby("region")
            .agg(
                {
                    "charges": ["mean", "median", "std", "count"],
                    "age": "mean",
                    "bmi": "mean",
                    "children": "mean",
                }
            )
            .round(2)
        )

        # Regional smoking rates
        regional_smoking = df.groupby(["region", "smoker"]).size().unstack(fill_value=0)
        regional_smoking_pct = (
            regional_smoking.div(regional_smoking.sum(axis=1), axis=0) * 100
        )

        # Regional coverage distribution
        regional_coverage = (
            df.groupby(["region", "coverage_level"]).size().unstack(fill_value=0)
        )

        # Regional medical conditions
        regional_medical = (
            df.groupby(["region", "medical_history"]).size().unstack(fill_value=0)
        )

        return {
            "regional_statistics": regional_stats.to_dict(),
            "smoking_rates_by_region": regional_smoking_pct.to_dict(),
            "coverage_distribution_by_region": regional_coverage.to_dict(),
            "medical_conditions_by_region": regional_medical.to_dict(),
            "regional_summary": {
                "highest_avg_charges": regional_stats["charges"]["mean"].idxmax(),
                "lowest_avg_charges": regional_stats["charges"]["mean"].idxmin(),
                "most_populated_region": regional_stats["charges"]["count"].idxmax(),
            },
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool(
    name="predict_premium_factors",
    description="Analyze factors that most influence insurance premiums",
)
def predict_premium_factors(ctx: Context):
    """
    Analyze which factors have the strongest correlation with insurance premiums.
    """
    try:
        df = load_data()

        # Create dummy variables for categorical columns
        df_encoded = pd.get_dummies(
            df,
            columns=[
                "gender",
                "smoker",
                "region",
                "medical_history",
                "family_medical_history",
                "exercise_frequency",
                "occupation",
                "coverage_level",
            ],
        )

        # Calculate correlations with charges
        correlations = df_encoded.corr()["charges"].abs().sort_values(ascending=False)

        # Top factors affecting charges
        top_factors = correlations.head(15).to_dict()

        # Age vs charges analysis
        age_charges_corr = df["age"].corr(df["charges"])

        # BMI vs charges analysis
        bmi_charges_corr = df["bmi"].corr(df["charges"])

        # Children vs charges analysis
        children_charges_corr = df["children"].corr(df["charges"])

        return {
            "top_factors_affecting_premiums": top_factors,
            "correlation_analysis": {
                "age_charges_correlation": float(age_charges_corr),
                "bmi_charges_correlation": float(bmi_charges_corr),
                "children_charges_correlation": float(children_charges_corr),
            },
            "premium_insights": {
                "highest_risk_profile": "Smokers with medical history",
                "most_expensive_region": df.groupby("region")["charges"]
                .mean()
                .idxmax(),
                "optimal_bmi_range": "18.5-25 (Normal BMI)",
                "exercise_benefit": "Regular exercise correlates with lower premiums",
            },
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool(
    name="search_similar_profiles",
    description="Find insurance records with similar demographic and risk profiles",
)
def search_similar_profiles(
    ctx: Context,
    target_age: int,
    target_bmi: float,
    target_gender: str,
    target_smoker: str,
    age_tolerance: Optional[int] = 5,
    bmi_tolerance: Optional[float] = 2.0,
    limit: Optional[int] = 50,
):
    """
    Find insurance records with similar demographic and risk profiles to a target profile.

    Args:
        target_age: Target age to match
        target_bmi: Target BMI to match
        target_gender: Target gender ('male' or 'female')
        target_smoker: Target smoking status ('yes' or 'no')
        age_tolerance: Age tolerance range (default 5 years)
        bmi_tolerance: BMI tolerance range (default 2.0)
        limit: Maximum number of similar profiles to return
    """
    try:
        df = load_data()

        # Apply similarity filters
        similar_profiles = df[
            (df["gender"] == target_gender)
            & (df["smoker"] == target_smoker)
            & (df["age"] >= target_age - age_tolerance)
            & (df["age"] <= target_age + age_tolerance)
            & (df["bmi"] >= target_bmi - bmi_tolerance)
            & (df["bmi"] <= target_bmi + bmi_tolerance)
        ]

        # Calculate similarity scores (simple distance-based)
        similar_profiles["similarity_score"] = (
            abs(similar_profiles["age"] - target_age) / age_tolerance
            + abs(similar_profiles["bmi"] - target_bmi) / bmi_tolerance
        )

        # Sort by similarity and limit results
        similar_profiles = similar_profiles.sort_values("similarity_score").head(limit)

        # Calculate statistics for similar profiles
        stats = {
            "average_charges": float(similar_profiles["charges"].mean()),
            "median_charges": float(similar_profiles["charges"].median()),
            "charge_range": {
                "min": float(similar_profiles["charges"].min()),
                "max": float(similar_profiles["charges"].max()),
            },
            "most_common_coverage": similar_profiles["coverage_level"].mode().iloc[0]
            if len(similar_profiles) > 0
            else None,
        }

        return {
            "target_profile": {
                "age": target_age,
                "bmi": target_bmi,
                "gender": target_gender,
                "smoker": target_smoker,
            },
            "similar_profiles_found": len(similar_profiles),
            "statistics": stats,
            "sample_profiles": similar_profiles.drop(
                "similarity_score", axis=1
            ).to_dict("records"),
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool(
    name="get_coverage_analysis",
    description="Analyze insurance coverage levels and their characteristics",
)
def get_coverage_analysis(ctx: Context):
    """
    Analyze different insurance coverage levels and their characteristics.
    """
    try:
        df = load_data()

        # Coverage level statistics
        coverage_stats = (
            df.groupby("coverage_level")
            .agg(
                {
                    "charges": ["mean", "median", "std", "count"],
                    "age": "mean",
                    "bmi": "mean",
                    "children": "mean",
                }
            )
            .round(2)
        )

        # Coverage by demographics
        coverage_by_gender = (
            df.groupby(["coverage_level", "gender"]).size().unstack(fill_value=0)
        )
        coverage_by_smoking = (
            df.groupby(["coverage_level", "smoker"]).size().unstack(fill_value=0)
        )
        coverage_by_region = (
            df.groupby(["coverage_level", "region"]).size().unstack(fill_value=0)
        )

        # Medical conditions by coverage
        coverage_medical = (
            df.groupby(["coverage_level", "medical_history"])
            .size()
            .unstack(fill_value=0)
        )

        return {
            "coverage_level_statistics": coverage_stats.to_dict(),
            "coverage_by_gender": coverage_by_gender.to_dict(),
            "coverage_by_smoking_status": coverage_by_smoking.to_dict(),
            "coverage_by_region": coverage_by_region.to_dict(),
            "medical_conditions_by_coverage": coverage_medical.to_dict(),
            "coverage_insights": {
                "most_expensive_coverage": coverage_stats["charges"]["mean"].idxmax(),
                "most_popular_coverage": coverage_stats["charges"]["count"].idxmax(),
                "premium_vs_basic_ratio": float(
                    coverage_stats.loc["Premium", ("charges", "mean")]
                    / coverage_stats.loc["Basic", ("charges", "mean")]
                ),
            },
        }
    except Exception as e:
        return {"error": str(e)}
