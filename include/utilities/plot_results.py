import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import io
import sys
import os
#Script Python to collect data and generate graphs.
# --- CONFIG ---
executable = "./benchmark_test"

# 1. CHECK ARGUMENTS
if len(sys.argv) < 3:
    print("ERROR: Please specify solver and precision.")
    print("Using: python3 plot_results.py <solver> <precision>")
    print("Es:  python3 plot_results.py naive float")
    sys.exit(1)

solver_name = sys.argv[1]    # es. "naive"
precision = sys.argv[2]      # es. "float"
full_name = f"{solver_name}_{precision}" # For file names

print(f"--- [PYTHON] Starting Benchmark: {solver_name.upper()} ({precision.upper()}) ---")

# 2. C++ EXECUTION
try:
    result = subprocess.run(
        [executable, solver_name, precision], 
        capture_output=True, 
        text=True, 
        check=True
    )
except subprocess.CalledProcessError as e:
    print("ERROR DURING C++ EXECUTION:")
    print(e.stderr)
    sys.exit(1)
except FileNotFoundError:
    print(f"ERROR: executable not found '{executable}'.")
    sys.exit(1)

raw_output = result.stdout

# 3. PARSING CSV
csv_lines = [] 

for line in raw_output.splitlines():
    # Accept header or rows which starts with a number
    if "Size,Time" in line or (line and line[0].isdigit() and "," in line):
        csv_lines.append(line)

clean_csv_data = "\n".join(csv_lines)

if not clean_csv_data:
    print("ERROR: No valid data received from C++.")
    print("Raw output received:\n", raw_output)
    sys.exit(1)

try:
    df = pd.read_csv(io.StringIO(clean_csv_data))
    df.columns = df.columns.str.strip()
except Exception as e:
    print(f"ERROR during CSV reading: {e}")
    sys.exit(1)

# 4. GENERATION GRAPHS
print(f"--- Generation graphs for {full_name} ---")

if not os.path.exists("plots"):
    os.makedirs("plots")

# --- TIME GRAPH ---
plt.figure(figsize=(10, 6))

if "Time_Mine" in df.columns:
    plt.plot(df["Size"], df["Time_Mine"], marker='o', linewidth=2, label=f"My Solver ({full_name})")
elif "Time_ms" in df.columns:
    plt.plot(df["Size"], df["Time_ms"], marker='o', linewidth=2, label=f"My Solver ({full_name})")

if "Time_Blas" in df.columns:
    plt.plot(df["Size"], df["Time_Blas"], linestyle='--', color='black', alpha=0.7, label="OpenBLAS")

plt.xlabel("Matrix Size (N)")
plt.ylabel("Time (ms)")
plt.title(f"Performance Analysis: {solver_name} ({precision})")
plt.legend()
plt.grid(True, alpha=0.3)

file_time = f"plots/time_{full_name}.png"
plt.savefig(file_time)
print(f"-> Time graph stored in: {file_time}")

# --- GFLOPS GRAPH ---
plt.figure(figsize=(10, 6))

if "GFLOPs_Mine" in df.columns:
    plt.plot(df["Size"], df["GFLOPs_Mine"], marker='s', linewidth=2, color='red', label=f"My Solver ({full_name})")
elif "GFLOPs" in df.columns:
    plt.plot(df["Size"], df["GFLOPs"], marker='s', linewidth=2, color='red', label=f"My Solver ({full_name})")

if "GFLOPs_Blas" in df.columns:
    plt.plot(df["Size"], df["GFLOPs_Blas"], linestyle='--', color='black', alpha=0.7, label="OpenBLAS")

plt.xlabel("Matrix Size (N)")
plt.ylabel("GFLOPs")
plt.title(f"Throughput Analysis: {solver_name} ({precision})")
plt.legend()
plt.grid(True, alpha=0.3)

file_gflops = f"plots/gflops_{full_name}.png"
plt.savefig(file_gflops)
print(f"-> GFLOPs graph stored in: {file_gflops}")