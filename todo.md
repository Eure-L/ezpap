# Plan

PC salle 203 -- RTX 2070 + Intel 12c24T

## 1 - Hybride OCL + AVX
- equilibrage de charge dynamique
    - analyse de trame
    - partie du code
    - (Graph)comparaison Avec/Sans

- speedup variant le nb de threads
- comparaison temps vs single GPU
- comparaison temps vs best AVX


## 2 - Bit board CPU
- speedup variant le nombre de threads 
- comparaison vs version Hybride

## 3 - Bit board GPU


--------------------
# Liste des tests à faire



|   -a   |  -nt  |     -s      | -v          |
| :----: | :---: | :---------: | :---------- |
| Random | 1 -24 | 1024 - 8192 | ocl hybrid  |
| Random |   1   | 1024 - 8192 | ocl         |
| Random | 1 -24 | 1024 - 8192 | gottagofast |
| Random | 1 -24 | 1024 - 8192 | lazybtmpvec |