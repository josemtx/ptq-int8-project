# PTQ INT8 — Memoria del Proyecto

Este repo contiene la **memoria LaTeX** para el proyecto: *Cuantización Post-Entrenamiento a INT8 con ejecución simulada en NumPy*.

## Estructura
```
memoria/
  main.tex
  resumen.tex
  intro.tex
  metodos_datos.tex
  implementacion.tex
  experimentos.tex
  conclusiones.tex
  refs.bib
results/
  figuras/        # coloca aquí las figuras generadas por tus notebooks
```

## Compilación local
Requisitos: `latexmk` y distribución LaTeX (TeX Live/MacTeX).
```bash
cd memoria
latexmk -pdf main.tex
```
El PDF quedará en `memoria/main.pdf`.

## Publicación en GitHub
1. Crea un repo nuevo en GitHub (público o privado).
2. En tu máquina:
   ```bash
   git init
   git add .
   git commit -m "Plantilla memoria LaTeX (PTQ INT8)"
   git branch -M main
   git remote add origin <URL_DE_TU_REPO>
   git push -u origin main
   ```

## Licencia
MIT — ver `LICENSE`.
