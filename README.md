4. Filtrowanie obrazów (rozmycie, wyostrzanie, krawędzie)
Opis:
Projekt obejmuje zastosowanie klasycznych filtrów splotowych do obrazów (m.in. rozmycie
Gaussa, wyostrzanie, operator Sobela/Canny). Celem jest przyspieszenie wsadowego
przetwarzania wielu plików oraz porównanie jakości efektów.
Zakres implementacji:
OpenMP – równoległe przetwarzanie wierszy i/lub kafli obrazu. MPI – rozdzielanie plików
między procesy i scalanie wyników. CUDA/OpenCL – implementacja filtrów jako jąder z pamięcią
współdzieloną; rozważ separowalny Gauss (2×1D).
Dodatkowe wyniki do raportu:
Czasy działania dla różnych rozdzielczości, porównanie PSNR/SSIM, przykłady obrazów
przed/po.