comandi per runnare il tb



1)
export LD_LIBRARY_PATH=/opt/OpenBLAS/lib:$LD_LIBRARY_PATH

2)
g++ -std=c++17 -O3 mainT.cpp 
-o benchmark_test 
-I./include  
-I/opt/OpenBLAS/include
-L/opt/OpenBLAS/lib
-lopenblas 
-lpthread
-fopenmp

3)
./benchmark_test


export OMP_NUM_THREADS= #numero di thread desiderato
./benchmark_test


### Possibili automazioni
#!/bin/bash

# Compila prima per sicurezza
echo "Compilazione..."
g++ -std=c++17 -O3 src/main.cpp -o benchmark_test \
    -I./include -I/opt/OpenBLAS/include \
    -L/opt/OpenBLAS/lib -lopenblas \
    -lpthread -fopenmp

echo "============================================"
echo "      TEST STRONG SCALABILITY (N=1024)      "
echo "============================================"
echo "Threads | Tempo (ms) | Speedup"
echo "--------------------------------------------"

# Variabile per salvare il tempo base (1 thread)
BASE_TIME=0

for t in 1 2 4 8; do
    # 1. Imposta i thread
    export OMP_NUM_THREADS=$t
    
    # 2. Esegui il programma e cattura l'output
    # Grep cerca la riga che inizia con "1024," (assumendo formato CSV: Size,Time,Gflops)
    OUTPUT=$(./benchmark_test | grep "^1024,")
    
    # 3. Estrai il tempo (assumendo che sia il secondo campo dopo la virgola)
    # Esempio riga: 1024,500.23,4.5
    TIME=$(echo $OUTPUT | cut -d',' -f2)
    
    # 4. Calcola Speedup (Tempo Base / Tempo Attuale)
    if [ "$t" -eq "1" ]; then
        BASE_TIME=$TIME
        SPEEDUP="1.00"
    else
        # Usa bc per fare calcoli con la virgola in bash
        SPEEDUP=$(echo "scale=2; $BASE_TIME / $TIME" | bc)
    fi
    
    echo "   $t    |   $TIME   |   ${SPEEDUP}x"
done
echo "============================================"
