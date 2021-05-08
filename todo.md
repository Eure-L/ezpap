# Plan

PC salle 203 -- RTX 2070 + Intel 12c24T

## 1 - Hybride OCL + AVX
- equilibrage de charge dynamique
    - analyse de trame                  OK
    - partie du code                    OK
    - (Graph)comparaison Avec/Sans      OK

- comparaison temps vs single GPU       OK


## 2 - Bit board CPU
- trouver la meilleure tile
- speedup variant le nombre de threads  OK
- comparaison vs version Hybride        OK
### Explications
- Origine
- explication de l'algorithme


## 3 - Bit board GPU


--------------------
# Liste des tests à faire



|   -a   |  -nt  |     -s      | -v          |
| :----: | :---: | :---------: | :---------- |
| Random | 1 -24 | 1024 - 8192 | ocl hybrid  |
| Random |   1   | 1024 - 8192 | ocl         |
| Random | 1 -24 | 1024 - 8192 | gottagofast |
| Random | 1 -24 | 1024 - 8192 | lazybtmpvec |