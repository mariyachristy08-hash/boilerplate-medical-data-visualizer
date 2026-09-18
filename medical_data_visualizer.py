import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv("medical_examination.csv")

# Add overweight column
df["overweight"] = (
    df["weight"] / ((df["height"] / 100) ** 2) > 25
).astype(int)

# Normalize cholesterol and gluc
df["cholesterol"] = (df["cholesterol"] > 1).astype(int)
df["gluc"] = (df["gluc"] > 1).astype(int)


# Draw Categorical Plot
def draw_cat_plot():

    # Create DataFrame for cat plot
    df_cat = pd.melt(
        df,
        id_vars=["cardio"],
        value_vars=[
            "cholesterol",
            "gluc",
            "smoke",
            "alco",
            "active",
            "overweight"
        ]
    )

    # Group and count
    df_cat = (
        df_cat
        .groupby(["cardio", "variable", "value"])["value"]
        .count()
        .rename("total")
        .reset_index()
    )

    # Draw catplot
    graphic = sns.catplot(
        x="variable",
        y="total",
        hue="value",
        col="cardio",
        data=df_cat,
        kind="bar"
    )

    # Get figure
    fig = graphic.figure

    fig.savefig("catplot.png")
    return fig


# Draw Heat Map
def draw_heat_map():

    # Clean the data
    df_heat = df[
        (df["ap_lo"] <= df["ap_hi"])
        & (df["height"] >= df["height"].quantile(0.025))
        & (df["height"] <= df["height"].quantile(0.975))
        & (df["weight"] >= df["weight"].quantile(0.025))
        & (df["weight"] <= df["weight"].quantile(0.975))
    ]

    # Calculate correlation matrix
    corr = df_heat.corr()

    # Generate mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw heatmap
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        square=True,
        linewidths=0.8,
        fmt="0.1f"
    )

    fig.savefig("heatmap.png")
    return fig