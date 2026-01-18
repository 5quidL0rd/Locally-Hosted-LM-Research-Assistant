"""
agents/data_profiler.py - Deep dataset analysis and profiling

Provides comprehensive dataset intelligence including:
- Statistical profiling (distributions, correlations, outliers)
- Data quality reports (missing values, duplicates, class imbalance)
- Feature analysis and recommendations
- Preprocessing suggestions
- Data leakage detection
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class DataProfiler:
    """
    Comprehensive dataset profiler for research workflows.

    Generates detailed reports including:
    - Basic statistics and data types
    - Missing value analysis
    - Distribution analysis
    - Correlation analysis
    - Outlier detection
    - Class imbalance detection (for classification)
    - Data quality score
    - Preprocessing recommendations
    """

    def __init__(self, output_dir: str = "data_profiles"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set visualization style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")

    def profile_dataset(self, data_path: str, target_column: str = None,
                       generate_report: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive profile of a dataset.

        Args:
            data_path: Path to CSV file
            target_column: Target column for ML (auto-detected if None)
            generate_report: Whether to generate HTML/markdown report

        Returns:
            Dictionary containing all profiling results
        """
        print(f"\n{'='*60}")
        print("DATASET PROFILER")
        print(f"{'='*60}")

        # Load data
        try:
            df = pd.read_csv(data_path)
            print(f"[Profiler] Loaded: {data_path}")
            print(f"[Profiler] Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        except Exception as e:
            return {"error": f"Failed to load dataset: {e}"}

        # Auto-detect target column
        if target_column is None:
            target_column = self._detect_target_column(df)
            print(f"[Profiler] Auto-detected target: {target_column}")

        # Generate profile sections
        profile = {
            "file_path": data_path,
            "file_name": Path(data_path).name,
            "generated_at": datetime.now().isoformat(),
            "target_column": target_column,
        }

        print("\n[1/7] Basic Statistics...")
        profile["basic_stats"] = self._basic_statistics(df)

        print("[2/7] Data Types Analysis...")
        profile["data_types"] = self._analyze_data_types(df)

        print("[3/7] Missing Values Analysis...")
        profile["missing_values"] = self._analyze_missing_values(df)

        print("[4/7] Distribution Analysis...")
        profile["distributions"] = self._analyze_distributions(df)

        print("[5/7] Correlation Analysis...")
        profile["correlations"] = self._analyze_correlations(df, target_column)

        print("[6/7] Outlier Detection...")
        profile["outliers"] = self._detect_outliers(df)

        print("[7/7] Target Analysis...")
        profile["target_analysis"] = self._analyze_target(df, target_column)

        # Calculate overall data quality score
        profile["quality_score"] = self._calculate_quality_score(profile)

        # Generate recommendations
        profile["recommendations"] = self._generate_recommendations(profile, df)

        # Detect potential data leakage
        profile["leakage_warnings"] = self._detect_data_leakage(df, target_column)

        # Generate visualizations
        if generate_report:
            print("\n[Profiler] Generating visualizations...")
            profile["visualizations"] = self._generate_visualizations(df, target_column, profile)

            print("[Profiler] Generating reports...")
            profile["report_path"] = self._generate_markdown_report(profile, df)

        print(f"\n{'='*60}")
        print(f"PROFILING COMPLETE - Quality Score: {profile['quality_score']:.1f}/100")
        print(f"{'='*60}\n")

        return profile

    def _detect_target_column(self, df: pd.DataFrame) -> Optional[str]:
        """Auto-detect likely target column"""
        candidates = ['target', 'label', 'class', 'y', 'outcome', 'result',
                     'Close', 'price', 'Price', 'value', 'output']

        for col in candidates:
            if col in df.columns:
                return col

        # Return last column as fallback
        return df.columns[-1]

    def _basic_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate basic dataset statistics"""
        return {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            "duplicates": int(df.duplicated().sum()),
            "duplicate_pct": round(df.duplicated().sum() / len(df) * 100, 2),
            "total_missing": int(df.isnull().sum().sum()),
            "total_missing_pct": round(df.isnull().sum().sum() / df.size * 100, 2),
        }

    def _analyze_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data types of columns"""
        type_counts = df.dtypes.value_counts()

        columns_by_type = {
            "numeric": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical": df.select_dtypes(include=['object', 'category']).columns.tolist(),
            "datetime": df.select_dtypes(include=['datetime64']).columns.tolist(),
            "boolean": df.select_dtypes(include=['bool']).columns.tolist(),
        }

        # Detect potential datetime columns stored as strings
        potential_dates = []
        for col in df.select_dtypes(include=['object']).columns:
            sample = df[col].dropna().head(100)
            try:
                pd.to_datetime(sample, errors='raise')
                potential_dates.append(col)
            except:
                pass

        return {
            "type_counts": {str(k): int(v) for k, v in type_counts.items()},
            "columns_by_type": columns_by_type,
            "potential_datetime_columns": potential_dates,
        }

    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detailed missing value analysis"""
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)

        # Columns with missing values
        cols_with_missing = missing[missing > 0].sort_values(ascending=False)

        missing_details = {}
        for col in cols_with_missing.index:
            missing_details[col] = {
                "count": int(cols_with_missing[col]),
                "percentage": float(missing_pct[col]),
            }

        # Categorize severity
        high_missing = [c for c, d in missing_details.items() if d["percentage"] > 50]
        moderate_missing = [c for c, d in missing_details.items() if 10 < d["percentage"] <= 50]
        low_missing = [c for c, d in missing_details.items() if d["percentage"] <= 10]

        return {
            "total_missing_cells": int(missing.sum()),
            "columns_with_missing": len(cols_with_missing),
            "missing_by_column": missing_details,
            "high_missing_cols": high_missing,
            "moderate_missing_cols": moderate_missing,
            "low_missing_cols": low_missing,
        }

    def _analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze distributions of numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        distributions = {}

        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) == 0:
                continue

            stats = {
                "mean": float(data.mean()),
                "median": float(data.median()),
                "std": float(data.std()),
                "min": float(data.min()),
                "max": float(data.max()),
                "skewness": float(data.skew()),
                "kurtosis": float(data.kurtosis()),
                "unique_values": int(data.nunique()),
                "unique_pct": round(data.nunique() / len(data) * 100, 2),
            }

            # Determine distribution type
            if abs(stats["skewness"]) < 0.5:
                stats["distribution_type"] = "approximately_normal"
            elif stats["skewness"] > 1:
                stats["distribution_type"] = "right_skewed"
            elif stats["skewness"] < -1:
                stats["distribution_type"] = "left_skewed"
            else:
                stats["distribution_type"] = "moderately_skewed"

            # Check if likely categorical despite numeric type
            if stats["unique_values"] <= 20 and stats["unique_pct"] < 5:
                stats["likely_categorical"] = True
            else:
                stats["likely_categorical"] = False

            distributions[col] = stats

        # Analyze categorical columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        categorical_stats = {}

        for col in cat_cols:
            value_counts = df[col].value_counts()
            categorical_stats[col] = {
                "unique_values": int(df[col].nunique()),
                "most_common": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                "most_common_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                "most_common_pct": round(value_counts.iloc[0] / len(df) * 100, 2) if len(value_counts) > 0 else 0,
            }

        return {
            "numeric": distributions,
            "categorical": categorical_stats,
        }

    def _analyze_correlations(self, df: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
        """Analyze correlations between features"""
        numeric_df = df.select_dtypes(include=[np.number])

        if len(numeric_df.columns) < 2:
            return {"message": "Not enough numeric columns for correlation analysis"}

        corr_matrix = numeric_df.corr()

        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append({
                        "feature1": corr_matrix.columns[i],
                        "feature2": corr_matrix.columns[j],
                        "correlation": round(corr_val, 3),
                    })

        # Correlations with target
        target_correlations = {}
        if target_column and target_column in numeric_df.columns:
            target_corr = corr_matrix[target_column].drop(target_column).sort_values(key=abs, ascending=False)
            target_correlations = {col: round(val, 3) for col, val in target_corr.items()}

        return {
            "high_correlation_pairs": sorted(high_corr_pairs, key=lambda x: abs(x["correlation"]), reverse=True),
            "target_correlations": target_correlations,
            "multicollinearity_risk": len(high_corr_pairs) > 0,
        }

    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers using IQR method"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_info = {}

        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) == 0:
                continue

            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = data[(data < lower_bound) | (data > upper_bound)]

            if len(outliers) > 0:
                outlier_info[col] = {
                    "count": int(len(outliers)),
                    "percentage": round(len(outliers) / len(data) * 100, 2),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                }

        total_outlier_cols = len(outlier_info)

        return {
            "columns_with_outliers": total_outlier_cols,
            "outlier_details": outlier_info,
        }

    def _analyze_target(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Analyze target variable"""
        if target_column not in df.columns:
            return {"error": f"Target column '{target_column}' not found"}

        target = df[target_column]

        # Determine if classification or regression
        unique_values = target.nunique()

        if unique_values <= 20 or target.dtype == 'object':
            # Classification
            task_type = "classification"
            value_counts = target.value_counts()

            class_distribution = {str(k): int(v) for k, v in value_counts.items()}

            # Check for imbalance
            if len(value_counts) > 1:
                imbalance_ratio = value_counts.max() / value_counts.min()
                is_imbalanced = imbalance_ratio > 3
            else:
                imbalance_ratio = 1.0
                is_imbalanced = False

            return {
                "task_type": task_type,
                "num_classes": unique_values,
                "class_distribution": class_distribution,
                "imbalance_ratio": round(imbalance_ratio, 2),
                "is_imbalanced": is_imbalanced,
                "minority_class": str(value_counts.idxmin()) if len(value_counts) > 1 else None,
                "majority_class": str(value_counts.idxmax()) if len(value_counts) > 0 else None,
            }
        else:
            # Regression
            task_type = "regression"
            return {
                "task_type": task_type,
                "mean": float(target.mean()),
                "median": float(target.median()),
                "std": float(target.std()),
                "min": float(target.min()),
                "max": float(target.max()),
                "skewness": float(target.skew()),
            }

    def _calculate_quality_score(self, profile: Dict[str, Any]) -> float:
        """Calculate overall data quality score (0-100)"""
        score = 100.0

        # Deduct for missing values
        missing_pct = profile["basic_stats"]["total_missing_pct"]
        score -= min(missing_pct * 0.5, 20)  # Max 20 point deduction

        # Deduct for duplicates
        dup_pct = profile["basic_stats"]["duplicate_pct"]
        score -= min(dup_pct * 0.3, 10)  # Max 10 point deduction

        # Deduct for high missing columns
        high_missing = len(profile["missing_values"].get("high_missing_cols", []))
        score -= min(high_missing * 5, 15)  # Max 15 point deduction

        # Deduct for multicollinearity
        if profile["correlations"].get("multicollinearity_risk"):
            score -= 5

        # Deduct for class imbalance
        if profile["target_analysis"].get("is_imbalanced"):
            score -= 10

        # Deduct for many outliers
        outlier_cols = profile["outliers"].get("columns_with_outliers", 0)
        total_cols = profile["basic_stats"]["columns"]
        outlier_ratio = outlier_cols / total_cols if total_cols > 0 else 0
        score -= min(outlier_ratio * 20, 10)  # Max 10 point deduction

        return max(0, min(100, score))

    def _generate_recommendations(self, profile: Dict[str, Any], df: pd.DataFrame) -> List[Dict[str, str]]:
        """Generate preprocessing recommendations"""
        recommendations = []

        # Missing value recommendations
        missing = profile["missing_values"]
        if missing["high_missing_cols"]:
            recommendations.append({
                "category": "Missing Values",
                "priority": "high",
                "issue": f"{len(missing['high_missing_cols'])} columns have >50% missing values",
                "suggestion": f"Consider dropping: {', '.join(missing['high_missing_cols'][:3])}",
            })

        if missing["moderate_missing_cols"]:
            recommendations.append({
                "category": "Missing Values",
                "priority": "medium",
                "issue": f"{len(missing['moderate_missing_cols'])} columns have 10-50% missing values",
                "suggestion": "Use imputation (mean/median for numeric, mode for categorical)",
            })

        # Duplicate recommendations
        if profile["basic_stats"]["duplicate_pct"] > 1:
            recommendations.append({
                "category": "Duplicates",
                "priority": "medium",
                "issue": f"{profile['basic_stats']['duplicates']} duplicate rows ({profile['basic_stats']['duplicate_pct']}%)",
                "suggestion": "Remove duplicate rows with df.drop_duplicates()",
            })

        # Correlation recommendations
        high_corr = profile["correlations"].get("high_correlation_pairs", [])
        if high_corr:
            recommendations.append({
                "category": "Multicollinearity",
                "priority": "medium",
                "issue": f"{len(high_corr)} highly correlated feature pairs detected",
                "suggestion": f"Consider removing one from: {high_corr[0]['feature1']} / {high_corr[0]['feature2']} (r={high_corr[0]['correlation']})",
            })

        # Class imbalance recommendations
        target_analysis = profile["target_analysis"]
        if target_analysis.get("is_imbalanced"):
            recommendations.append({
                "category": "Class Imbalance",
                "priority": "high",
                "issue": f"Imbalance ratio: {target_analysis['imbalance_ratio']:.1f}x",
                "suggestion": "Use SMOTE, class weights, or stratified sampling",
            })

        # Outlier recommendations
        outliers = profile["outliers"].get("outlier_details", {})
        severe_outliers = [c for c, d in outliers.items() if d["percentage"] > 5]
        if severe_outliers:
            recommendations.append({
                "category": "Outliers",
                "priority": "medium",
                "issue": f"{len(severe_outliers)} columns have >5% outliers",
                "suggestion": f"Review: {', '.join(severe_outliers[:3])} - consider clipping or transformation",
            })

        # Skewness recommendations
        distributions = profile["distributions"].get("numeric", {})
        highly_skewed = [c for c, d in distributions.items() if abs(d.get("skewness", 0)) > 2]
        if highly_skewed:
            recommendations.append({
                "category": "Skewness",
                "priority": "low",
                "issue": f"{len(highly_skewed)} columns are highly skewed",
                "suggestion": f"Apply log/sqrt transformation to: {', '.join(highly_skewed[:3])}",
            })

        # Potential categorical columns
        likely_cat = [c for c, d in distributions.items() if d.get("likely_categorical")]
        if likely_cat:
            recommendations.append({
                "category": "Data Types",
                "priority": "low",
                "issue": f"{len(likely_cat)} numeric columns may actually be categorical",
                "suggestion": f"Convert to categorical: {', '.join(likely_cat[:3])}",
            })

        return recommendations

    def _detect_data_leakage(self, df: pd.DataFrame, target_column: str) -> List[Dict[str, str]]:
        """Detect potential data leakage issues"""
        warnings = []

        if target_column not in df.columns:
            return warnings

        numeric_df = df.select_dtypes(include=[np.number])

        if target_column in numeric_df.columns:
            # Check for suspiciously high correlations
            corr = numeric_df.corr()
            if target_column in corr.columns:
                target_corr = corr[target_column].drop(target_column)

                for col, val in target_corr.items():
                    if abs(val) > 0.95:
                        warnings.append({
                            "type": "high_correlation",
                            "column": col,
                            "correlation": round(val, 3),
                            "message": f"'{col}' has suspiciously high correlation ({val:.3f}) with target - possible leakage",
                        })

        # Check for columns that might contain target information
        leaky_patterns = ['future', 'tomorrow', 'next', 'result', 'outcome', 'prediction', 'pred_']
        for col in df.columns:
            col_lower = col.lower()
            for pattern in leaky_patterns:
                if pattern in col_lower and col != target_column:
                    warnings.append({
                        "type": "suspicious_name",
                        "column": col,
                        "message": f"'{col}' name suggests it may contain future information",
                    })

        return warnings

    def _generate_visualizations(self, df: pd.DataFrame, target_column: str,
                                 profile: Dict[str, Any]) -> Dict[str, str]:
        """Generate visualization plots"""
        viz_dir = self.output_dir / f"viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        viz_dir.mkdir(exist_ok=True)

        paths = {}

        # 1. Missing values heatmap
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)[:20]

            if len(missing_data) > 0:
                missing_data.plot(kind='bar', ax=ax, color='coral')
                ax.set_title('Missing Values by Column (Top 20)')
                ax.set_ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                path = viz_dir / "missing_values.png"
                plt.savefig(path, dpi=100, bbox_inches='tight')
                paths["missing_values"] = str(path)
            plt.close()
        except Exception as e:
            print(f"  [Warning] Missing values plot failed: {e}")

        # 2. Correlation heatmap
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) >= 2:
                fig, ax = plt.subplots(figsize=(12, 10))
                corr = numeric_df.corr()

                # Limit to top correlated features if too many
                if len(corr.columns) > 20:
                    if target_column in corr.columns:
                        top_cols = corr[target_column].abs().sort_values(ascending=False)[:20].index.tolist()
                        corr = corr.loc[top_cols, top_cols]

                sns.heatmap(corr, annot=len(corr.columns) <= 15, fmt='.2f',
                           cmap='coolwarm', center=0, ax=ax)
                ax.set_title('Feature Correlations')
                plt.tight_layout()
                path = viz_dir / "correlations.png"
                plt.savefig(path, dpi=100, bbox_inches='tight')
                paths["correlations"] = str(path)
            plt.close()
        except Exception as e:
            print(f"  [Warning] Correlation plot failed: {e}")

        # 3. Target distribution
        try:
            if target_column in df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))

                target_analysis = profile["target_analysis"]
                if target_analysis.get("task_type") == "classification":
                    df[target_column].value_counts().plot(kind='bar', ax=ax, color='steelblue')
                    ax.set_title(f'Target Distribution: {target_column}')
                    ax.set_ylabel('Count')
                    plt.xticks(rotation=45, ha='right')
                else:
                    df[target_column].hist(bins=50, ax=ax, color='steelblue', edgecolor='black')
                    ax.set_title(f'Target Distribution: {target_column}')
                    ax.set_xlabel(target_column)
                    ax.set_ylabel('Frequency')

                plt.tight_layout()
                path = viz_dir / "target_distribution.png"
                plt.savefig(path, dpi=100, bbox_inches='tight')
                paths["target_distribution"] = str(path)
            plt.close()
        except Exception as e:
            print(f"  [Warning] Target distribution plot failed: {e}")

        # 4. Numeric feature distributions (top 9)
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:9]
            if len(numeric_cols) > 0:
                n_cols = min(3, len(numeric_cols))
                n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

                fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
                axes = np.array(axes).flatten() if n_rows * n_cols > 1 else [axes]

                for i, col in enumerate(numeric_cols):
                    df[col].hist(bins=30, ax=axes[i], color='steelblue', edgecolor='black')
                    axes[i].set_title(col, fontsize=10)
                    axes[i].tick_params(labelsize=8)

                # Hide empty subplots
                for j in range(len(numeric_cols), len(axes)):
                    axes[j].set_visible(False)

                plt.suptitle('Feature Distributions', fontsize=12)
                plt.tight_layout()
                path = viz_dir / "feature_distributions.png"
                plt.savefig(path, dpi=100, bbox_inches='tight')
                paths["feature_distributions"] = str(path)
            plt.close()
        except Exception as e:
            print(f"  [Warning] Feature distributions plot failed: {e}")

        # 5. Outlier boxplots
        try:
            outlier_cols = list(profile["outliers"].get("outlier_details", {}).keys())[:8]
            if outlier_cols:
                fig, axes = plt.subplots(2, 4, figsize=(14, 8))
                axes = axes.flatten()

                for i, col in enumerate(outlier_cols):
                    df.boxplot(column=col, ax=axes[i])
                    axes[i].set_title(col, fontsize=10)

                for j in range(len(outlier_cols), 8):
                    axes[j].set_visible(False)

                plt.suptitle('Columns with Outliers', fontsize=12)
                plt.tight_layout()
                path = viz_dir / "outliers.png"
                plt.savefig(path, dpi=100, bbox_inches='tight')
                paths["outliers"] = str(path)
            plt.close()
        except Exception as e:
            print(f"  [Warning] Outlier plot failed: {e}")

        return paths

    def _generate_markdown_report(self, profile: Dict[str, Any], df: pd.DataFrame) -> str:
        """Generate a markdown report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.output_dir / f"profile_report_{timestamp}.md"

        lines = []
        lines.append(f"# Dataset Profile Report")
        lines.append(f"\n**File:** `{profile['file_name']}`")
        lines.append(f"**Generated:** {profile['generated_at']}")
        lines.append(f"**Quality Score:** {profile['quality_score']:.1f}/100")

        # Basic stats
        lines.append("\n## Basic Statistics\n")
        stats = profile["basic_stats"]
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Rows | {stats['rows']:,} |")
        lines.append(f"| Columns | {stats['columns']} |")
        lines.append(f"| Memory Usage | {stats['memory_usage_mb']} MB |")
        lines.append(f"| Duplicate Rows | {stats['duplicates']:,} ({stats['duplicate_pct']}%) |")
        lines.append(f"| Total Missing | {stats['total_missing']:,} ({stats['total_missing_pct']}%) |")

        # Target analysis
        lines.append("\n## Target Variable Analysis\n")
        target = profile["target_analysis"]
        lines.append(f"**Target Column:** `{profile['target_column']}`")
        lines.append(f"**Task Type:** {target.get('task_type', 'unknown')}")

        if target.get("task_type") == "classification":
            lines.append(f"**Classes:** {target.get('num_classes')}")
            lines.append(f"**Imbalance Ratio:** {target.get('imbalance_ratio', 1):.2f}x")
            if target.get("is_imbalanced"):
                lines.append(f"**Warning:** Dataset is imbalanced!")

        # Data types
        lines.append("\n## Data Types\n")
        types = profile["data_types"]
        lines.append(f"| Type | Count |")
        lines.append(f"|------|-------|")
        for dtype, count in types["type_counts"].items():
            lines.append(f"| {dtype} | {count} |")

        # Missing values
        lines.append("\n## Missing Values\n")
        missing = profile["missing_values"]
        if missing["columns_with_missing"] > 0:
            lines.append(f"**Columns with Missing:** {missing['columns_with_missing']}")
            lines.append("\n| Column | Missing Count | Percentage |")
            lines.append("|--------|---------------|------------|")
            for col, details in list(missing["missing_by_column"].items())[:10]:
                lines.append(f"| {col} | {details['count']:,} | {details['percentage']}% |")
        else:
            lines.append("No missing values detected!")

        # Correlations
        lines.append("\n## Correlation Analysis\n")
        corr = profile["correlations"]
        if corr.get("target_correlations"):
            lines.append("### Top Correlations with Target\n")
            lines.append("| Feature | Correlation |")
            lines.append("|---------|-------------|")
            for feat, val in list(corr["target_correlations"].items())[:10]:
                lines.append(f"| {feat} | {val} |")

        if corr.get("high_correlation_pairs"):
            lines.append("\n### Highly Correlated Feature Pairs\n")
            for pair in corr["high_correlation_pairs"][:5]:
                lines.append(f"- `{pair['feature1']}` ↔ `{pair['feature2']}`: {pair['correlation']}")

        # Recommendations
        lines.append("\n## Recommendations\n")
        for rec in profile["recommendations"]:
            priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(rec["priority"], "⚪")
            lines.append(f"### {priority_emoji} {rec['category']}\n")
            lines.append(f"**Issue:** {rec['issue']}")
            lines.append(f"**Suggestion:** {rec['suggestion']}\n")

        # Data leakage warnings
        if profile["leakage_warnings"]:
            lines.append("\n## ⚠️ Data Leakage Warnings\n")
            for warning in profile["leakage_warnings"]:
                lines.append(f"- **{warning['column']}**: {warning['message']}")

        # Visualizations
        if profile.get("visualizations"):
            lines.append("\n## Visualizations\n")
            for name, path in profile["visualizations"].items():
                lines.append(f"- [{name}]({path})")

        # Write report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"[Profiler] Report saved: {report_path}")
        return str(report_path)

    def quick_profile(self, data_path: str) -> str:
        """Generate a quick text summary without full report"""
        profile = self.profile_dataset(data_path, generate_report=False)

        if "error" in profile:
            return f"Error: {profile['error']}"

        summary = []
        summary.append(f"Dataset: {profile['file_name']}")
        summary.append(f"Quality Score: {profile['quality_score']:.1f}/100")
        summary.append(f"Shape: {profile['basic_stats']['rows']:,} rows x {profile['basic_stats']['columns']} columns")
        summary.append(f"Task Type: {profile['target_analysis'].get('task_type', 'unknown')}")
        summary.append(f"Missing: {profile['basic_stats']['total_missing_pct']}%")
        summary.append(f"Duplicates: {profile['basic_stats']['duplicate_pct']}%")

        if profile["recommendations"]:
            summary.append("\nTop Recommendations:")
            for rec in profile["recommendations"][:3]:
                summary.append(f"  - [{rec['priority'].upper()}] {rec['issue']}")

        return '\n'.join(summary)


def create_data_profiler(output_dir: str = "data_profiles") -> DataProfiler:
    """Factory function to create DataProfiler"""
    return DataProfiler(output_dir)
