import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import io
import sys
import os

executable = "./benchmark_test" 
if len(sys.argv) < 3:
    print("ERRORE: Devi specificare solver e precisione.")
    print("Uso: python3 plot_results.py <solver> <precision>")
    print("Es:  python3 plot_results.py simd float")
    sys.exit(1)

solver_name = sys.argv[1]    # es. "simd"
precision = sys.argv[2]      # es. "float"
full_name = f"{solver_name}_{precision}" 

print(f"--- [PYTHON] Avvio Benchmark: {solver_name.upper()} ({precision.upper()}) ---")
try:
    result = subprocess.run(
        [executable, solver_name, precision], 
        capture_output=True, 
        text=True, 
        check=True 
    )
except subprocess.CalledProcessError as e:
    print("ERRORE DURANTE L'ESECUZIONE DEL C++:")
    print(e.stderr)
    sys.exit(1)
except FileNotFoundError:
    print(f"ERRORE: Non trovo l'eseguibile '{executable}'. Hai compilato?")
    sys.exit(1)

raw_output = result.stdout

for line in raw_output.splitlines():
    if "Size,Time" in line or (line and line[0].isdigit() and "," in line):
        csv_lines.append(line)

clean_csv_data = "\n".join(csv_lines)

if not clean_csv_data:
    print("ERRORE: Nessun dato valido ricevuto dal C++.")
    print("Output grezzo ricevuto:\n", raw_output)
    sys.exit(1)

try:
    df = pd.read_csv(io.StringIO(clean_csv_data))
    df.columns = df.columns.str.strip()
except Exception as e:
    print(f"ERRORE nella lettura del CSV: {e}")
    sys.exit(1)

print(f"--- Generazione grafici per {full_name} ---")

if not os.path.exists("plots"):
    os.makedirs("plots")

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
print(f"-> Grafico Tempo salvato in: {file_time}")

plt.figure(figsize=(10, 6))

if "GFLOPs_Mine" in df.columns:
    plt.plot(df["Size"], df["GFLOPs_Mine"], marker='s', linewidth=2, color='red', label=f"My Solver ({full_name})")
elif "GFLOPs" in df.columns:
    plt.plot(df["Size"], df["GFLOPs"], marker='s', linewidth=2, color='red', label=f"My Solver ({full_name})")

if "GFLOPs_Blas" in df.columns:
    plt.plot(df["Size"], df["GFLOPs_Blas"], linestyle='--', color='black', alpha=0.7, label="OpenBLAS")

plt.xlabel("Matrix Size (N)")
plt.ylabel("GFLOPs (Higher is better)")
plt.title(f"Throughput Analysis: {solver_name} ({precision})")
plt.legend()
plt.grid(True, alpha=0.3)

file_gflops = f"plots/gflops_{full_name}.png"
plt.savefig(file_gflops)
print(f"-> Grafico GFLOPs salvato in: {file_gflops}")








