import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import io
import sys

# Configurazione
solver_name = "Naive"
executable = "./benchmark_test"

# --- ESECUZIONE DEL C++ ---
print(f"Eseguendo benchmark per {solver_name}...")
result = subprocess.run([executable], capture_output=True, text=True)
raw_output = result.stdout

# --- DEBUG: Vediamo cosa ha stampato il C++ ---
#print("--- OUTPUT RICEVUTO DAL C++ ---")
#print(raw_output)
#print("-------------------------------")

# --- PULIZIA E LETTURA DATI ---
# Filtriamo le righe: teniamo solo quelle che contengono virgole e sembrano numeri o l'header
csv_lines = []
for line in raw_output.splitlines():
    # Cerca l'header o righe di dati (es: "128,45.2,1.5")
    if "Size,Time" in line or ("," in line and line[0].isdigit()):
        csv_lines.append(line)

clean_csv_data = "\n".join(csv_lines)

if not clean_csv_data:
    print("ERRORE: Non ho trovato dati validi nel CSV!")
    sys.exit(1)

try:
    df = pd.read_csv(io.StringIO(clean_csv_data))
    # Rimuoviamo eventuali spazi vuoti dai nomi delle colonne
    df.columns = df.columns.str.strip() 
except Exception as e:
    print(f"Errore nella lettura del CSV: {e}")
    sys.exit(1)

# Verifica che le colonne esistano
if "Size" not in df.columns:
    print(f"ERRORE: Colonne trovate: {df.columns}")
    print("Mi aspettavo 'Size' ma non c'è. Controlla l'output sopra.")
    sys.exit(1)

# --- GRAFICI (Il resto rimane uguale) ---
print("Generazione grafici...")

# Grafico Tempo
plt.figure(figsize=(10, 5))
plt.plot(df["Size"], df["Time_ms"], marker='o', label=f"{solver_name}")
plt.xlabel("Matrix Size (N)")
plt.ylabel("Time (ms)")
plt.title("Matrix Multiplication Performance")
plt.grid(True)
plt.legend()
plt.savefig("time_plot.png")
print("-> time_plot.png salvato")

# Grafico GFLOPs
plt.figure(figsize=(10, 5))
plt.plot(df["Size"], df["GFLOPs"], marker='s', color='red', label=f"{solver_name}")
plt.xlabel("Matrix Size (N)")
plt.ylabel("GFLOPs (Higher is better)")
plt.title("Computation Throughput")
plt.grid(True)
plt.legend()
plt.savefig("gflops_plot.png")
print("-> gflops_plot.png salvato")
