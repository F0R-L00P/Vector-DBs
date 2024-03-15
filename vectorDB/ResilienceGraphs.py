# imprting libraries
import numpy as np # for numerical operations
import pandas as pd # for data manipulation
import seaborn as sns # for data visualization
import matplotlib.pyplot as plt # for data visualization

######################################################################################
#########################PLOT WITH WHITE BACKGROUND###################################
######################################################################################
# Time points and resilience values
time = np.linspace(0, 10, 100)
t1 = 2  # Time of disruption
t2 = 8  # Time of recovery
resilience = np.piecewise(
    time,
    [time < t1, (time >= t1) & (time <= t2), time > t2],
    [1, lambda t: 1 - (0.5 * (t - t1) / (t2 - t1)), 1],
)

# Data prep using dataframe
data = pd.DataFrame({"Time": time, "Resilience Indicator (%)": resilience * 100})

# Seaborn settings & plot
sns.set_theme(
    style="whitegrid"
)
plt.figure(figsize=[8, 4])
ax = sns.lineplot(
    x="Time",
    y="Resilience Indicator (%)",
    data=data,
    linestyle="--",
    color="k",
    label="Expected functionality",
)

# filling between lines and adding lines
ax.fill_between(
    time,
    0,
    resilience * 100,
    where=(time >= t1) & (time <= t2),
    color="grey",
    alpha=0.4,
    label="Resilience Loss",
)
ax.axhline(y=100, color="r", linestyle="-", linewidth=2.5, label="Normal Functionality")
ax.axvline(x=t1, color="b", linestyle=":", label="Event Occurs")
ax.axvline(x=t2, color="g", linestyle=":", label="Full Recovery")

# Plot settings
ax.set(title="System Resiliency", xlabel="Time", ylabel="Resilience Indicator (%)")
ax.set_xticks([0, t1, t2, 10])
ax.set_xticklabels(["0", "$t_1$ (Disruption)", "$t_2$ (Recovery)", "10"])
ax.set_ylim(0, 110)
ax.set_xlim(0, 10)
ax.legend()
sns.despine()  # Remove the top and right border
plt.tight_layout()  # Optimize layout
plt.show()

######################################################################################
#########################PLOT WITH DARK BACKGROUND####################################
######################################################################################

# sns.set_theme(style="whitegrid")
plt.style.use("seaborn-v0_8-darkgrid")

# Time points and resilience values
time = np.linspace(0, 10, 100)
t1 = 2  # Time of disruption
t2 = 8  # Time of recovery
resilience = np.piecewise(
    time,
    [time < t1, (time >= t1) & (time <= t2), time > t2],
    [1, lambda t: 1 - (0.5 * (t - t1) / (t2 - t1)), 1],
)

data = pd.DataFrame({"Time": time, "Resilience Indicator (%)": resilience * 100})

# Plotting
plt.figure(figsize=[10, 6])
ax = sns.lineplot(
    x="Time",
    y="Resilience Indicator (%)",
    data=data,
    linestyle="--",
    color="k",
    linewidth=2.5,  # Enhance line visibility
    label="Expected functionality",
)

ax.fill_between(
    time,
    0,
    100,
    where=(time >= t1) & (time <= t2),
    color="white",
    alpha=0.8,  # Make degradation loss more visible
    label="Operational Degradation",
)

ax.fill_between(
    time,
    0,
    resilience * 100,
    where=(time >= t1) & (time <= t2),
    color="grey",
    alpha=0.5,
    label="Resilience Loss",
)

ax.axhline(y=100, color="r", linestyle="-", linewidth=2.5, label="Normal Functionality")
ax.axvline(x=t1, color="b", linestyle=":", linewidth=2.5, label="Event Occurs")
ax.axvline(x=t2, color="g", linestyle=":", linewidth=2.5, label="Full Recovery")

# Enhancing the visualization
ax.set(title="System Resiliency", xlabel="Time", ylabel="Resilience Indicator (%)")
ax.legend(
    frameon=True, framealpha=0.9, edgecolor="black"
)  # Make legend more pronounced
ax.set_xticks([0, t1, t2, 10])
ax.set_xticklabels(["0", "$t_1$ (Disruption)", "$t_2$ (Recovery)", "10"], fontsize=12)
ax.set_ylim(0, 110)
ax.set_xlim(0, 10)
sns.despine()
plt.tight_layout()
plt.show()


######################################################################################
#########################RESILIENCY COST TRADE-OFF####################################
######################################################################################

# Setting style
plt.style.use("seaborn-v0_8-darkgrid")


# Adjusted Sigmoid function for smooth transition
def adjusted_sigmoid(x, stretch=1, shift_x=0):
    return 100 / (1 + np.exp(-stretch * (x - shift_x)))


# Mirrored Adjusted Sigmoid function
def mirrored_adjusted_sigmoid(x, stretch=1, shift_x=0):
    return 100 - (100 / (1 + np.exp(-stretch * (x - shift_x))))


# Modified new sigmoid curve function
def modified_new_sigmoid_curve(x, start_phase=35, end_value=10, stretch=1, shift_x=75):
    if x <= start_phase:
        return 100
    else:
        offset = 100
        return (
            offset
            - (110 / (1 + np.exp(-stretch * (x - shift_x))))
            + (end_value - (offset - 91))
        )


# setting figure and subplots
fig, ax1 = plt.subplots(figsize=(10, 6)) 

# Enhance labels and set limits for the primary y-axis
ax1.set_ylabel("Max Theoretical R()", color="#333333", fontsize=12)
ax1.set_ylim([0, 120])
y_ticks = np.arange(0, 111, 20)  # Exclude 120 from y-axis labels
ax1.set_yticks(y_ticks)
ax1.tick_params(
    axis="y", labelcolor="#333333", labelsize=10
)  # Adjusting y-axis tick parameters

# X-axis settings
# NOTE: ANY MODIFICATION TO X-AXIS REQUIRES CHANGING ALL VERTICAL LINES & IDEAL SIGMOID CURVES
ax1.set_xticks([25, 55, 85])  # Set positions for 3 proposed phases 
ax1.set_xticklabels(
    ["Design", "Construction", "Operation"], fontsize=10, color="#333333"
)
ax1.set_xlim([0, 100])

# Draw vertical lines for each phase
ax1.axvline(x=35, color="gray", linestyle="--", linewidth=1)
ax1.axvline(x=75, color="gray", linestyle="--", linewidth=1)

# Secondary y-axis for cost
ax2 = ax1.twinx()
ax2.set_ylabel("Cost ($)", color="#333333", fontsize=12)
ax2.set_ylim([0, 120])
ax2.set_yticks([])
ax2.set_yticks(y_ticks)  # Match primary y-axis ticks to exclude 120
ax2.set_yticks([])       # remove all tick values
ax2.tick_params(axis="y", labelcolor="#333333", labelsize=10)

# Draw a straight horizontal line at y=100 across the graph
ax1.axhline(y=100, color="gray", linestyle="--", linewidth=1)

# Generate values for the curves
x_values = np.linspace(0, 100, 300)
y_values = adjusted_sigmoid(x_values, stretch=0.11, shift_x=50)
y_values_mirrored = mirrored_adjusted_sigmoid(x_values, stretch=0.11, shift_x=50)
y_values_new_curve = [
    modified_new_sigmoid_curve(
        x, start_phase=35, end_value=20, stretch=0.11, shift_x=55
    )
    for x in x_values
]
y_values_new_curve = np.array(y_values_new_curve, dtype=float)  # Ensure proper dtype

# Plot the curves with improved aesthetics
ax2.plot(x_values, y_values, color="black", label="cost of resolution", linewidth=2)
ax2.plot(x_values, y_values_mirrored, color="red", label="actual", linewidth=2)
valid_indices_new_curve = ~np.isnan(y_values_new_curve)
ax2.plot(
    x_values[valid_indices_new_curve],
    y_values_new_curve[valid_indices_new_curve],
    color="blue",
    label="ideal",
    linewidth=2,
)

# Improve the legend
ax2.legend(frameon=True, framealpha=0.9, shadow=True, borderpad=1)

# Draw a straight line at y=100 from the left to the 'Design' phase
ax1.axhline(y=100, color="blue", linestyle="--", linewidth=1, xmax=0.35)

# plot title
ax1.set_title(
    "P(System Resiliency | Change Made in Phase)", fontsize=14, color="#333333"
)

plt.show()