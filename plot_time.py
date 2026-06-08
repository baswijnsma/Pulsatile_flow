import pandas as pd
import matplotlib.pyplot as plt

# --- load data ---
file_name = "results_50pct/flow_monitor.csv"   # change this

# your data looks whitespace-separated, so use:
df = pd.read_csv(file_name, sep=",")
print(df.columns)
print(df.head())
# ensure numeric (important because of scientific notation / strings)
df = df.apply(pd.to_numeric, errors="coerce")

# --- plot example: multiple signals vs time ---
plt.figure(figsize=(10, 5))

plt.plot(df["t"], df["Q_in"], label="Q_in")
plt.plot(df["t"], df["Q_out1"], label="Q_out1")
plt.plot(df["t"], df["Q_out2"], label="Q_out2")

plt.xlabel("Time (t)")
plt.ylabel("Flow")
plt.title("Time Series")
plt.legend()
plt.grid(True)

plt.show()